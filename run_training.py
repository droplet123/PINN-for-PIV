"""
run_training.py
===============
CLI entry point: load PIV data from a processed case, train the PINN,
and export the 3-D flow field as a VTR file.

Usage
-----
    python run_training.py --case_dir ./experimental_data/LargeView/Ethanol_drop \\
                           --R_e 1.5 --R_p 1.0 \\
                           --n_epochs 5000 --n_pde_pts 8000 \\
                           --out_dir ./output

The script automatically uses CUDA if available (RTX 4090), otherwise CPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.data_parser import FlowRegime, classify_directory, parse_vc7
from src.geometry.geometry import OblateSpheroid
from src.network.pinn_model import StreamingPINN
from src.postprocess.exporter import export_vtr, generate_3d_grid, predict_flow_field
from src.training.trainer import PINNTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_training")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_vc7_files(case_dir: Path) -> list[Path]:
    """Return all .vc7 files inside PIV_MP* sub-dirs (or recursively)."""
    piv_mp_dirs = sorted(d for d in case_dir.iterdir()
                         if d.is_dir() and d.name.startswith("PIV_MP"))
    files: list[Path] = []
    for d in piv_mp_dirs:
        files.extend(sorted(d.glob("*.vc7")))
    if not files:
        files = sorted(case_dir.rglob("*.vc7"))
    return files


def _load_piv_tensors(
    case_dir: Path,
    R_e_m: float,
    u_ref: float,
    cx_mm: float,
    cy_mm: float,
    max_snapshots: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse all VC7 snapshots in case_dir, shift coordinates to droplet-centred
    frame, non-dimensionalise, and stack into tensors for the trainer.

    Parameters
    ----------
    cx_mm, cy_mm : droplet centre in the same physical coordinate system as
                   x_phys / y_phys [mm].  Coordinates are shifted by
                   (x - cx, y - cy) before non-dimensionalisation so that
                   the network input is centred on the droplet.

    Returns
    -------
    xyz_piv : [M, 3] float32  — (x*, y*, 0) droplet-centred non-dim coords
    u_piv   : [M, 2] float32  — (u*, v*) non-dim velocities
    """
    regime = classify_directory(case_dir)
    vc7_files = _find_vc7_files(case_dir)
    if not vc7_files:
        raise FileNotFoundError(f"No .vc7 files found in {case_dir}")

    if max_snapshots is not None:
        vc7_files = vc7_files[:max_snapshots]

    log.info("Loading %d VC7 snapshot(s) from %s", len(vc7_files), case_dir.name)
    log.info("Droplet centre: cx=%.4f mm, cy=%.4f mm", cx_mm, cy_mm)

    xyz_list, u_list = [], []
    for vc7_path in vc7_files:
        snap = parse_vc7(vc7_path, regime=regime)

        # Shift to droplet-centred frame before non-dimensionalisation
        x_centred = snap.x_phys - cx_mm   # [mm], origin at droplet centre
        y_centred = snap.y_phys - cy_mm

        # Non-dimensionalise: x* = x / R_e,  u* = u / u_ref
        x_star = x_centred * 1e-3 / R_e_m   # mm → m → non-dim
        y_star = y_centred * 1e-3 / R_e_m
        u_star = snap.u_phys / u_ref
        v_star = snap.v_phys / u_ref

        xx, yy = np.meshgrid(x_star, y_star)   # (ny, nx)
        xyz = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
        uv  = np.stack([u_star.ravel(), v_star.ravel()], axis=1)

        valid = np.isfinite(uv).all(axis=1)
        xyz_list.append(xyz[valid])
        u_list.append(uv[valid])

    xyz_all = np.concatenate(xyz_list, axis=0).astype(np.float32)
    u_all   = np.concatenate(u_list,   axis=0).astype(np.float32)
    log.info("Total PIV points: %d  |  x* range: [%.2f, %.2f]  |  u* range: [%.2f, %.2f]",
             xyz_all.shape[0], xyz_all[:, 0].min(), xyz_all[:, 0].max(),
             u_all[:, 0].min(), u_all[:, 0].max())
    return torch.from_numpy(xyz_all), torch.from_numpy(u_all)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PINN on PIV data and export 3-D flow field."
    )
    p.add_argument("--case_dir", type=Path, required=True,
                   help="Path to the experimental case directory.")
    p.add_argument("--R_e", type=float, default=1.5, metavar="MM",
                   help="Equatorial radius [mm]. Default: 1.5")
    p.add_argument("--R_p", type=float, default=1.0, metavar="MM",
                   help="Polar radius [mm]. Default: 1.0")
    p.add_argument("--cx", type=float, default=None, metavar="MM",
                   help="Droplet centre x in physical coords [mm]. "
                        "LargeView≈10.6, SmallView≈6.0. Required.")
    p.add_argument("--cy", type=float, default=None, metavar="MM",
                   help="Droplet centre y in physical coords [mm]. "
                        "LargeView≈14.5, SmallView≈8.0. Required.")
    p.add_argument("--u_ref", type=float, default=0.01, metavar="M_S",
                   help="Reference velocity [m/s]. Default: 0.01")
    p.add_argument("--n_epochs", type=int, default=5000)
    p.add_argument("--n_pde_pts", type=int, default=8000,
                   help="PDE collocation points per epoch.")
    p.add_argument("--n_data_pts", type=int, default=4000,
                   help="PIV points randomly sampled per epoch (0=all). "
                        "Keep ≤10000 to saturate GPU without OOM.")
    p.add_argument("--mini_batch", type=int, default=2000,
                   help="Mini-batch size for gradient aggregation. "
                        "Increase until GPU-Util >80%%.")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_snapshots", type=int, default=None,
                   help="Limit number of VC7 snapshots loaded (default: all).")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Output directory. Default: <case_dir>/output/")
    p.add_argument("--grid_res", type=int, default=64,
                   help="3-D export grid resolution per axis. Default: 64")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    case_dir = args.case_dir.resolve()
    out_dir  = (args.out_dir or case_dir / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    R_e_m = args.R_e * 1e-3
    R_p_m = args.R_p * 1e-3

    # Droplet centre defaults: LargeView or SmallView heuristic
    cx_mm = args.cx if args.cx is not None else 10.6
    cy_mm = args.cy if args.cy is not None else 14.5
    if args.cx is None or args.cy is None:
        log.warning("--cx/--cy not provided, using defaults (%.1f, %.1f) mm. "
                    "Pass explicit values for accurate results.", cx_mm, cy_mm)

    # --- 1. Load PIV data ---
    xyz_piv, u_piv = _load_piv_tensors(
        case_dir, R_e_m=R_e_m, u_ref=args.u_ref,
        cx_mm=cx_mm, cy_mm=cy_mm,
        max_snapshots=args.max_snapshots,
    )

    # --- 2. Configure and run trainer ---
    cfg = TrainerConfig(
        R_e=R_e_m,
        R_p=R_p_m,
        n_epochs=args.n_epochs,
        n_pde_pts=args.n_pde_pts,
        n_data_pts_per_epoch=args.n_data_pts,
        mini_batch_size=args.mini_batch,
        hidden=args.hidden,
        layers=args.layers,
        lr=args.lr,
        log_every=args.log_every,
    )

    trainer = PINNTrainer(cfg)
    log.info(
        "Device: %s | Model params: %d | Epochs: %d",
        trainer.device,
        sum(p.numel() for p in trainer.model.parameters()),
        cfg.n_epochs,
    )

    history = trainer.train(xyz_piv, u_piv)

    # Save loss history
    loss_path = out_dir / "loss_history.txt"
    with loss_path.open("w") as f:
        f.write("epoch\tdata\tpde\ttotal\n")
        for row in history:
            f.write(f"{row['epoch']}\t{row['data']:.6e}\t"
                    f"{row['pde']:.6e}\t{row['total']:.6e}\n")
    log.info("Loss history saved: %s", loss_path)

    # Save model checkpoint
    ckpt_path = out_dir / "model.pt"
    torch.save(trainer.model.state_dict(), ckpt_path)
    log.info("Model checkpoint saved: %s", ckpt_path)

    # --- 3. Export 3-D flow field ---
    log.info("Exporting 3-D flow field (grid=%d³)...", args.grid_res)
    geom = OblateSpheroid(R_e=R_e_m, R_p=R_p_m)
    # Domain: ±3 R_e in x/y, ±3 R_p in z (non-dim: ±3)
    bound = 3.0
    xs, ys, zs, pts = generate_3d_grid(
        x_bounds=(-bound, bound),
        y_bounds=(-bound, bound),
        z_bounds=(-bound * R_p_m / R_e_m, bound * R_p_m / R_e_m),
        resolution=args.grid_res,
    )

    trainer.model.eval()
    fields = predict_flow_field(trainer.model, pts, batch_size=4096)

    vtr_path = out_dir / "flow_field.vtr"
    export_vtr(xs, ys, zs, fields, vtr_path, geom=geom)
    log.info("3-D flow field saved: %s", vtr_path)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
