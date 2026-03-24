"""
exporter.py
===========
Step 5 — Inference & VTK Export for 3D acoustic streaming post-processing.

Workflow
--------
1. generate_3d_grid   — build a dense Cartesian grid over the flow domain
2. predict_flow_field — batched inference through StreamingPINN (OOM-safe)
3. export_vtr         — write a pyvista RectilinearGrid (.vtr) readable by
                        ParaView, with optional droplet-interior masking

Physical note
-------------
The network operates in non-dimensionalised coordinates (x* = x/R_e).
All grid bounds and the geometry passed to export_vtr must use the same
coordinate system as the trained model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pyvista as pv

from src.geometry.geometry import OblateSpheroid, sdf_oblate_spheroid
from src.network.pinn_model import StreamingPINN, curl


# ---------------------------------------------------------------------------
# 1. Grid generation
# ---------------------------------------------------------------------------

def generate_3d_grid(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    resolution: int | tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Create a dense regular Cartesian grid.

    Parameters
    ----------
    x/y/z_bounds : (min, max) in the model's coordinate system
    resolution   : int (uniform) or (nx, ny, nz)

    Returns
    -------
    xs, ys, zs : 1-D coordinate arrays (for RectilinearGrid axes)
    pts        : [N, 3] float32 tensor of all grid points (C-order flattened)
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    nx, ny, nz = resolution

    xs = np.linspace(*x_bounds, nx, dtype=np.float32)
    ys = np.linspace(*y_bounds, ny, dtype=np.float32)
    zs = np.linspace(*z_bounds, nz, dtype=np.float32)

    # meshgrid with 'ij' indexing → shape (nx, ny, nz) each
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = torch.from_numpy(
        np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    )  # [nx*ny*nz, 3] float32
    return xs, ys, zs, pts


# ---------------------------------------------------------------------------
# 2. Batched inference
# ---------------------------------------------------------------------------

def predict_flow_field(
    model: StreamingPINN,
    pts: torch.Tensor,
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    """
    Run batched inference over a large point cloud without OOM.

    Parameters
    ----------
    model      : trained StreamingPINN (any device)
    pts        : [N, 3] float32 coordinate tensor
    batch_size : points per forward pass

    Returns
    -------
    dict with keys:
        'velocity'         [N, 3]  u = curl(Psi)
        'pressure'         [N, 1]  p
        'vector_potential' [N, 3]  Psi
    All arrays are float32 numpy on CPU.
    """
    device = next(model.parameters()).device
    model.eval()

    vel_chunks, pres_chunks, psi_chunks = [], [], []

    with torch.no_grad():
        for start in range(0, pts.shape[0], batch_size):
            chunk = pts[start:start + batch_size].to(device).requires_grad_(True)

            # Need grad for curl even inside no_grad context — use enable_grad
            with torch.enable_grad():
                psi, p = model(chunk)
                # create_graph=False: inference only needs 1st-order curl;
                # skipping the higher-order graph halves peak VRAM here.
                u = curl(psi, chunk, create_graph=False)

            vel_chunks.append(u.detach().cpu().float().numpy())
            pres_chunks.append(p.detach().cpu().float().numpy())
            psi_chunks.append(psi.detach().cpu().float().numpy())

    return {
        'velocity':         np.concatenate(vel_chunks,  axis=0),
        'pressure':         np.concatenate(pres_chunks, axis=0),
        'vector_potential': np.concatenate(psi_chunks,  axis=0),
    }


# ---------------------------------------------------------------------------
# 3. VTR export
# ---------------------------------------------------------------------------

def export_vtr(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    fields: dict[str, np.ndarray],
    output_path: str | Path,
    geom: Optional[OblateSpheroid] = None,
    mask_interior: bool = True,
) -> Path:
    """
    Write a pyvista RectilinearGrid (.vtr) file for ParaView.

    The grid axes (xs, ys, zs) define the rectilinear mesh.  Field arrays
    must have shape [nx*ny*nz, ...] in the same C-order as generate_3d_grid.

    Parameters
    ----------
    xs, ys, zs    : 1-D coordinate arrays from generate_3d_grid
    fields        : dict of arrays, e.g. {'velocity': ..., 'pressure': ...}
    output_path   : destination .vtr file path
    geom          : if provided and mask_interior=True, points with SDF < 0
                    (inside the droplet) are set to NaN in all field arrays
    mask_interior : apply droplet masking (requires geom)

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = pv.RectilinearGrid(xs, ys, zs)   # pyvista uses (x, y, z) axes
    N = grid.n_points                        # nx * ny * nz

    # --- optional droplet masking ---
    interior_mask = None
    if geom is not None and mask_interior:
        # Reconstruct grid points in the same order pyvista uses (Fortran order)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64)
        pts_t  = torch.from_numpy(pts_np)
        phi    = sdf_oblate_spheroid(pts_t, geom).numpy()
        interior_mask = phi < 0.0   # True where inside droplet

    # --- attach fields ---
    for name, arr in fields.items():
        arr = arr.copy()
        if interior_mask is not None:
            if arr.ndim == 1:
                arr[interior_mask] = np.nan
            else:
                arr[interior_mask, :] = np.nan
        grid.point_data[name] = arr

    grid.save(str(output_path))
    return output_path


# ===========================================================================
# Self-contained smoke test
# ===========================================================================
if __name__ == '__main__':
    import sys, tempfile, os
    torch.manual_seed(0)

    model = StreamingPINN(hidden=32, layers=3)
    geom  = OblateSpheroid(R_e=1.5e-3, R_p=1.0e-3)

    xs, ys, zs, pts = generate_3d_grid(
        (-5e-3, 5e-3), (-5e-3, 5e-3), (-5e-3, 5e-3), resolution=10
    )
    print(f"Grid: {len(xs)}x{len(ys)}x{len(zs)} = {pts.shape[0]} points")

    fields = predict_flow_field(model, pts, batch_size=200)
    for k, v in fields.items():
        print(f"  {k}: {v.shape}, finite={np.isfinite(v).all()}")

    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(xs, ys, zs, fields, os.path.join(d, 'test.vtr'),
                          geom=geom, mask_interior=True)
        size = os.path.getsize(path)
        print(f"Saved {path} ({size} bytes)")
        assert size > 0

    print("Smoke test passed.")
    sys.exit(0)
