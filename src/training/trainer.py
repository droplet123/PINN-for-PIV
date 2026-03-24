"""
trainer.py
==========
Step 4 — Training Pipeline: data assimilation + physics-informed optimisation.

Coordinates:
  * PIV data loading (mock or real)
  * Geometry-based collocation point sampling (volume + surface)
  * SDF-weighted Navier-Stokes momentum loss
  * Data assimilation loss (PIV MSE)
  * Gradient Aggregation: accumulate gradients over mini-batches before
    each optimizer.step(), enabling arbitrarily large effective batches
    on memory-constrained hardware (CPU or limited VRAM GPU).

Loss composition
----------------
  L_total = lambda_data * L_data  +  lambda_pde * L_pde_weighted

  L_data          = MSE(u_pred(x_piv), u_piv)          [data assimilation]
  L_pde_weighted  = mean( w(phi) * ||NS_residual||^2 )  [SDF-weighted PDE]

  w(phi) = exp(-phi^2 / 2*sigma^2)  peaks at droplet surface (phi=0),
  suppressing the penalty deep inside the viscous boundary layer where
  the PIV data is absent and gradients are unresolvable.

Gradient Aggregation
--------------------
  For an effective batch of N_eff points split into K mini-batches of
  size B = N_eff / K:

    for each mini-batch b:
        loss_b = L(b) / K          # scale so sum == full-batch loss
        loss_b.backward()          # accumulates into .grad buffers
    optimizer.step()               # single parameter update
    optimizer.zero_grad()

  This is mathematically equivalent to computing the gradient over the
  full batch, but uses only B points of memory at a time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from src.geometry.geometry import (
    OblateSpheroid,
    sample_volume,
    sample_surface,
    sdf_oblate_spheroid,
    sdf_weight,
)
from src.network.pinn_model import StreamingPINN, curl, _grad

log = logging.getLogger("trainer")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    # Physical parameters (non-dimensionalised)
    nu: float = 1.56e-5          # kinematic viscosity of air [m^2/s] (or non-dim)
    R_e: float = 1.5e-3          # equatorial semi-axis [m]
    R_p: float = 1.0e-3          # polar semi-axis [m]

    # Collocation counts (effective batch = mini_batch_size * n_mini_batches)
    n_pde_pts: int = 1000        # total PDE collocation points per epoch
    n_surf_pts: int = 200        # surface BC collocation points per epoch
    n_data_pts_per_epoch: int = 4000  # PIV points sampled per epoch (0 = all)
    mini_batch_size: int = 100   # points per gradient-aggregation mini-batch
    domain_scale: float = 5.0    # bounding box half-side = domain_scale * R_e

    # Loss weights
    lambda_data: float = 1.0
    lambda_pde: float = 1.0
    lambda_bc: float = 0.1       # surface slip-velocity BC (placeholder)

    # Network
    hidden: int = 64
    layers: int = 4

    # Optimiser
    lr: float = 1e-3
    n_epochs: int = 1000

    # Logging
    log_every: int = 100


# ---------------------------------------------------------------------------
# SDF-weighted PDE residual (extracted so gradient aggregation can call it
# on individual mini-batches)
# ---------------------------------------------------------------------------
def _pde_loss_chunk(
    model: StreamingPINN,
    xyz_chunk: torch.Tensor,   # [B, 3] float32, requires_grad=True
    phi_chunk: torch.Tensor,   # [B]    float32 SDF values
    nu: float,
    geom: OblateSpheroid,
) -> torch.Tensor:
    """
    Compute SDF-weighted Navier-Stokes momentum residual for one mini-batch.

    Returns a scalar loss (already weighted and meaned over the chunk).
    """
    psi, p = model(xyz_chunk)
    u = curl(psi, xyz_chunk)                    # [B, 3]

    ux, uy, uz = u[:, 0:1], u[:, 1:2], u[:, 2:3]

    # Pressure gradient
    dp = _grad(p, xyz_chunk)                    # [B, 3]

    # Laplacian of each velocity component: ∇²u_i = Σ_j d²u_i/dxj²
    def laplacian(s: torch.Tensor) -> torch.Tensor:
        g = _grad(s, xyz_chunk)
        return sum(_grad(g[:, j:j+1], xyz_chunk)[:, j:j+1] for j in range(3))

    lap_u = torch.cat([laplacian(ux), laplacian(uy), laplacian(uz)], dim=1)

    # Convective term (u·∇)u
    du = torch.stack([_grad(ux, xyz_chunk),
                      _grad(uy, xyz_chunk),
                      _grad(uz, xyz_chunk)], dim=1)   # [B, 3, 3]
    conv = torch.einsum('nj,nij->ni', u, du)           # [B, 3]

    # NS residual: (u·∇)u + ∇p - ν∇²u = 0
    residual = conv + dp - nu * lap_u                  # [B, 3]
    residual_sq = (residual ** 2).sum(dim=1)           # [B]

    # SDF spatial weight: w ∈ (0,1], peaks at surface
    w = sdf_weight(phi_chunk, geom=geom).float()       # [B]

    return (w * residual_sq).mean()


# ---------------------------------------------------------------------------
# Data assimilation loss (PIV MSE)
# ---------------------------------------------------------------------------
def _data_loss(
    model: StreamingPINN,
    xyz_data: torch.Tensor,    # [M, 3] float32, requires_grad=True
    u_data: torch.Tensor,      # [M, 2 or 3] float32
) -> torch.Tensor:
    psi, _ = model(xyz_data)
    u_pred = curl(psi, xyz_data)
    n_comp = u_data.shape[1]
    return ((u_pred[:, :n_comp] - u_data) ** 2).mean()


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------
class PINNTrainer:
    """
    Coordinates the full PINN training loop with gradient aggregation.

    Usage
    -----
    >>> cfg = TrainerConfig(n_epochs=5, n_pde_pts=100, mini_batch_size=50)
    >>> trainer = PINNTrainer(cfg)
    >>> trainer.train(xyz_piv, u_piv)
    """

    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info("Device: %s", self.device)

        self.geom = OblateSpheroid(R_e=cfg.R_e, R_p=cfg.R_p)
        self.model = StreamingPINN(hidden=cfg.hidden, layers=cfg.layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        if self.device.type == 'cuda':
            # TF32 gives ~2× speedup on Ampere/Ada GPUs with negligible accuracy loss.
            torch.set_float32_matmul_precision('high')

    # ------------------------------------------------------------------
    def _sample_pde_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample volume collocation points; return (xyz float32, phi float32)."""
        pts, phi = sample_volume(
            self.geom, self.cfg.n_pde_pts, self.device,
            domain_scale=self.cfg.domain_scale, exclude_interior=True,
        )
        return pts.float(), phi.float()

    # ------------------------------------------------------------------
    def _gradient_aggregation_pde(
        self,
        xyz_pde: torch.Tensor,   # [N, 3] float32
        phi_pde: torch.Tensor,   # [N]    float32
        scale: float,            # multiply loss by this before .backward()
    ) -> float:
        """
        Split xyz_pde into mini-batches, accumulate gradients.
        Returns the total (unscaled) PDE loss value for logging.
        """
        B = self.cfg.mini_batch_size
        N = xyz_pde.shape[0]
        K = max(1, (N + B - 1) // B)   # number of mini-batches (ceiling div)
        total_loss = 0.0

        for start in range(0, N, B):
            chunk_xyz = xyz_pde[start:start + B].detach().requires_grad_(True)
            chunk_phi = phi_pde[start:start + B]

            loss_chunk = _pde_loss_chunk(
                self.model, chunk_xyz, chunk_phi, self.cfg.nu, self.geom
            )
            total_loss += loss_chunk.item()

            # Scale by 1/K so that summing K chunks ≈ full-batch gradient
            (scale * self.cfg.lambda_pde * loss_chunk / K).backward()

        return total_loss / K   # mean over chunks for logging

    # ------------------------------------------------------------------
    def _gradient_aggregation_data(
        self,
        xyz_piv: torch.Tensor,   # [M, 3] float32, on device
        u_piv: torch.Tensor,     # [M, 2 or 3] float32, on device
        scale: float,
    ) -> float:
        """
        Gradient aggregation over PIV data loss.
        Identical pattern to _gradient_aggregation_pde so arbitrarily large
        PIV datasets (e.g. 21 M points) never cause VRAM OOM.
        """
        B = self.cfg.mini_batch_size
        M = xyz_piv.shape[0]
        K = max(1, (M + B - 1) // B)
        total_loss = 0.0

        for start in range(0, M, B):
            chunk_xyz = xyz_piv[start:start + B].detach().requires_grad_(True)
            chunk_u   = u_piv[start:start + B]
            loss_chunk = _data_loss(self.model, chunk_xyz, chunk_u)
            total_loss += loss_chunk.item()
            (scale * self.cfg.lambda_data * loss_chunk / K).backward()

        return total_loss / K

    # ------------------------------------------------------------------
    def train(
        self,
        xyz_piv: torch.Tensor,   # [M, 3] PIV measurement locations
        u_piv: torch.Tensor,     # [M, 2 or 3] PIV velocity observations
    ) -> list[dict]:
        """
        Run the training loop.

        Parameters
        ----------
        xyz_piv : [M, 3] float32 — non-dimensionalised PIV point locations
        u_piv   : [M, 2 or 3] float32 — PIV velocity (in-plane or 3C)

        Returns
        -------
        history : list of per-epoch loss dicts
        """
        xyz_piv = xyz_piv.to(self.device).float()
        u_piv   = u_piv.to(self.device).float()

        M_total = xyz_piv.shape[0]
        n_data  = self.cfg.n_data_pts_per_epoch
        subsample = (n_data > 0) and (n_data < M_total)

        history = []
        t_interval_start = time.perf_counter()
        t_train_start = t_interval_start

        for epoch in range(1, self.cfg.n_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            # ---- 1. Data assimilation loss — random subsample each epoch
            #         so GPU sees a fresh, compact batch rather than 21 M pts
            if subsample:
                idx = torch.randperm(M_total, device=self.device)[:n_data]
                xyz_batch = xyz_piv[idx]
                u_batch   = u_piv[idx]
            else:
                xyz_batch, u_batch = xyz_piv, u_piv

            loss_data_val = self._gradient_aggregation_data(
                xyz_batch, u_batch, scale=1.0
            )

            # ---- 2. PDE residual loss with gradient aggregation ----------
            xyz_pde, phi_pde = self._sample_pde_points()
            loss_pde_val = self._gradient_aggregation_pde(
                xyz_pde, phi_pde, scale=1.0
            )

            # ---- 3. Parameter update ------------------------------------
            self.optimizer.step()

            epoch_losses = {
                'epoch': epoch,
                'data':  loss_data_val,
                'pde':   loss_pde_val,
                'total': loss_data_val + self.cfg.lambda_pde * loss_pde_val,
            }
            history.append(epoch_losses)

            if epoch % self.cfg.log_every == 0 or epoch == 1:
                now = time.perf_counter()
                interval_s = now - t_interval_start
                elapsed_s  = now - t_train_start
                remaining  = self.cfg.n_epochs - epoch
                eta_s      = (elapsed_s / epoch) * remaining if epoch > 0 else 0.0
                log.info(
                    "Epoch %4d/%d | total=%.4e | data=%.4e | pde=%.4e"
                    " | last_%d_ep=%.1fs | ETA=%.0fs",
                    epoch, self.cfg.n_epochs,
                    epoch_losses['total'],
                    epoch_losses['data'],
                    epoch_losses['pde'],
                    self.cfg.log_every,
                    interval_s,
                    eta_s,
                )
                t_interval_start = now

        return history


# ===========================================================================
# Self-contained mock training test (CPU only)
# ===========================================================================
if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
    )

    torch.manual_seed(0)
    device = torch.device('cpu')

    # --- Mock PIV data: 50 random points in the fluid domain, 2C in-plane ---
    N_piv = 50
    xyz_piv = torch.rand(N_piv, 3) * 2e-3 + 5e-4   # rough exterior coords [m]
    u_piv   = torch.randn(N_piv, 2) * 1e-3           # mock velocities [m/s]

    cfg = TrainerConfig(
        n_epochs=5,
        n_pde_pts=100,
        mini_batch_size=50,
        hidden=32,
        layers=3,
        log_every=1,
    )

    trainer = PINNTrainer(cfg)
    print(f"\nDevice : {trainer.device}")
    print(f"Model  : {sum(p.numel() for p in trainer.model.parameters())} parameters")
    print(f"Config : {cfg.n_epochs} epochs, {cfg.n_pde_pts} PDE pts, "
          f"mini-batch={cfg.mini_batch_size}\n")

    history = trainer.train(xyz_piv, u_piv)

    print("\n--- Training history ---")
    for row in history:
        print(f"  Epoch {row['epoch']:2d} | "
              f"data={row['data']:.4e} | "
              f"pde={row['pde']:.4e} | "
              f"total={row['total']:.4e}")

    # Verify losses are finite and decreasing (or at least not NaN)
    assert all(torch.isfinite(torch.tensor(r['total'])) for r in history), \
        "NaN/Inf detected in training losses"
    print("\nAll assertions passed. Mock training loop complete.")
    sys.exit(0)
