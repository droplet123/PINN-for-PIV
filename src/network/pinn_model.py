"""
PINN model using vector potential formulation.

The network outputs Psi = (psi_x, psi_y, psi_z) and pressure p.
Velocity is derived as u = curl(Psi), which identically satisfies
div(u) = 0 by the vector calculus identity div(curl(F)) = 0.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


def _grad(
    output: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Compute gradient of scalar/vector output w.r.t. inputs via autograd.

    Args:
        create_graph: build higher-order graph (True during training for
                      Laplacian; False during inference to save VRAM).
    """
    return torch.autograd.grad(
        output, inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,   # always retain: multiple _grad calls share graph
    )[0]


def curl(
    psi: torch.Tensor,
    xyz: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Compute u = curl(Psi) = (dpsi_z/dy - dpsi_y/dz,
                              dpsi_x/dz - dpsi_z/dx,
                              dpsi_y/dx - dpsi_x/dy).

    Args:
        psi:          [N, 3] vector potential (psi_x, psi_y, psi_z)
        xyz:          [N, 3] spatial coordinates with requires_grad=True
        create_graph: if True (default), build the higher-order graph needed
                      for NS Laplacian during training.  Pass False during
                      inference to avoid building the 3rd-order graph and
                      reduce peak VRAM usage.

    Returns:
        u: [N, 3] divergence-free velocity field
    """
    psi_x, psi_y, psi_z = psi[:, 0:1], psi[:, 1:2], psi[:, 2:3]

    # Each _grad call returns [N, 3]; index columns for specific partials.
    # retain_graph=True is required because all three psi components share
    # the same xyz leaf — each call must keep the graph for the next.
    dpsi_x = _grad(psi_x, xyz, create_graph=create_graph)
    dpsi_y = _grad(psi_y, xyz, create_graph=create_graph)
    dpsi_z = _grad(psi_z, xyz, create_graph=create_graph)

    u = dpsi_z[:, 1:2] - dpsi_y[:, 2:3]   # dpsi_z/dy - dpsi_y/dz
    v = dpsi_x[:, 2:3] - dpsi_z[:, 0:1]   # dpsi_x/dz - dpsi_z/dx
    w = dpsi_y[:, 0:1] - dpsi_x[:, 1:2]   # dpsi_y/dx - dpsi_x/dy

    return torch.cat([u, v, w], dim=1)     # [N, 3]


class _MLP(nn.Module):
    """Fully-connected network with tanh activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [out_dim]
        self.net = nn.Sequential(*[
            layer
            for i in range(len(dims) - 1)
            for layer in (
                [nn.Linear(dims[i], dims[i + 1])]
                + ([nn.Tanh()] if i < len(dims) - 2 else [])
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreamingPINN(nn.Module):
    """
    Physics-Informed Neural Network for 3D acoustic streaming.

    Input:  (x*, y*, z*) — non-dimensionalised coordinates  [N, 3]
    Output: Psi = (psi_x, psi_y, psi_z) [N, 3]  +  p* [N, 1]

    Velocity is NEVER a direct network output; it is always derived
    via u = curl(Psi) so that div(u) = 0 is satisfied by construction.
    """

    def __init__(self, hidden: int = 64, layers: int = 4):
        super().__init__()
        # Shared trunk encodes spatial features
        self.trunk = _MLP(3, hidden, hidden, layers)
        # Separate heads for vector potential and pressure
        self.head_psi = nn.Linear(hidden, 3)
        self.head_p   = nn.Linear(hidden, 1)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [N, 3] with requires_grad=True

        Returns:
            psi: [N, 3] vector potential
            p:   [N, 1] kinematic pressure
        """
        features = torch.tanh(self.trunk(xyz))
        return self.head_psi(features), self.head_p(features)

    def velocity(self, xyz: torch.Tensor) -> torch.Tensor:
        """Derive divergence-free velocity u = curl(Psi)."""
        psi, _ = self(xyz)
        return curl(psi, xyz)

    def compute_losses(
        self,
        xyz: torch.Tensor,
        nu: float,
        u_data: torch.Tensor | None = None,
        xyz_data: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics and data loss components.

        Physics loss: steady incompressible Navier-Stokes momentum residual
            (u·∇)u + ∇p - ν∇²u = 0
        Note: continuity loss is omitted — curl formulation guarantees div(u)=0.

        Args:
            xyz:      [N, 3] collocation points (requires_grad=True)
            nu:       kinematic viscosity (non-dimensionalised)
            u_data:   [M, 2 or 3] PIV velocity observations (optional)
            xyz_data: [M, 3] locations of PIV observations (optional)

        Returns:
            dict with keys 'momentum', and optionally 'data'
        """
        psi, p = self(xyz)
        u = curl(psi, xyz)   # [N, 3]

        ux, uy, uz = u[:, 0:1], u[:, 1:2], u[:, 2:3]

        # --- pressure gradient ---
        dp = _grad(p, xyz)                          # [N, 3]

        # --- viscous diffusion: ∇²u = ∇(∇·u) - ∇×(∇×u); since div(u)=0,
        #     ∇²u_i = sum_j d²u_i/dxj² ---
        def laplacian(scalar: torch.Tensor) -> torch.Tensor:
            g = _grad(scalar, xyz)                  # [N, 3]
            return sum(_grad(g[:, j:j+1], xyz)[:, j:j+1] for j in range(3))

        lap_u = torch.cat([laplacian(ux), laplacian(uy), laplacian(uz)], dim=1)

        # --- convective term (u·∇)u ---
        du = torch.stack([_grad(ux, xyz), _grad(uy, xyz), _grad(uz, xyz)], dim=1)
        # du: [N, 3, 3] where du[n, i, j] = d u_i / d x_j
        conv = torch.einsum('nj,nij->ni', u, du)   # [N, 3]

        # Navier-Stokes residual: (u·∇)u + ∇p - ν∇²u = 0
        ns_residual = conv + dp - nu * lap_u        # [N, 3]
        losses = {'momentum': (ns_residual ** 2).mean()}

        # --- data assimilation loss (PIV observations) ---
        if u_data is not None and xyz_data is not None:
            # Ensure xyz_data is a leaf with grad enabled (required for curl)
            if not xyz_data.requires_grad:
                xyz_data = xyz_data.detach().requires_grad_(True)
            psi_d, _ = self(xyz_data)
            u_pred = curl(psi_d, xyz_data)
            # PIV provides in-plane components; match available columns
            n_comp = u_data.shape[1]
            losses['data'] = ((u_pred[:, :n_comp] - u_data) ** 2).mean()

        return losses


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = StreamingPINN(hidden=32, layers=3).to(device)

    xyz = torch.rand(100, 3, requires_grad=True, device=device)
    psi, p = model(xyz)
    assert psi.shape == (100, 3), f"psi shape {psi.shape}"
    assert p.shape   == (100, 1), f"p shape {p.shape}"

    u = model.velocity(xyz)
    assert u.shape == (100, 3), f"u shape {u.shape}"

    # Divergence-free proof
    ones = torch.ones(100, device=device)
    grads = [
        torch.autograd.grad(u[:, i], xyz,
                            grad_outputs=ones,
                            create_graph=False,
                            retain_graph=(i < 2))[0][:, i]
        for i in range(3)
    ]
    div_u = grads[0] + grads[1] + grads[2]
    max_div = div_u.abs().max().item()
    print(f"max |div(u)| = {max_div:.2e}")
    assert max_div < 1e-5, f"Divergence not zero: {max_div}"

    losses = model.compute_losses(xyz, nu=1e-4)
    print(f"momentum loss = {losses['momentum'].item():.4e}")
    print("All checks passed.")
