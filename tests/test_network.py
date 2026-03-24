"""
Tests for StreamingPINN — vector potential formulation.

Key proof: div(curl(Psi)) = 0 identically, so mass conservation
is satisfied by construction regardless of network weights.
"""

import pytest
import torch
from src.network.pinn_model import StreamingPINN, curl


@pytest.fixture
def model():
    torch.manual_seed(42)
    return StreamingPINN(hidden=32, layers=3)


@pytest.fixture
def xyz():
    torch.manual_seed(0)
    x = torch.rand(100, 3, requires_grad=True)
    return x


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

def test_forward_shapes(model, xyz):
    psi, p = model(xyz)
    assert psi.shape == (100, 3)
    assert p.shape   == (100, 1)


def test_velocity_shape(model, xyz):
    u = model.velocity(xyz)
    assert u.shape == (100, 3)


# ---------------------------------------------------------------------------
# THE CORE PROOF: divergence-free velocity
# ---------------------------------------------------------------------------

def test_divergence_free(model, xyz):
    """
    Prove that u = curl(Psi) satisfies div(u) = 0 to numerical precision.
    This is the mathematical identity div(curl(F)) = 0, verified via autograd.
    """
    u = model.velocity(xyz)   # [100, 3]; graph retained via create_graph=True inside curl

    ones = torch.ones(100)
    grads = [
        torch.autograd.grad(u[:, i], xyz,
                            grad_outputs=ones,
                            create_graph=False,
                            retain_graph=(i < 2))[0][:, i]
        for i in range(3)
    ]
    div_u = grads[0] + grads[1] + grads[2]

    max_div = div_u.abs().max().item()
    assert max_div < 1e-5, (
        f"Divergence not numerically zero: max|div(u)| = {max_div:.2e}\n"
        "Vector potential formulation should guarantee div(u)=0 by construction."
    )


# ---------------------------------------------------------------------------
# Curl correctness: analytical verification on a known field
# ---------------------------------------------------------------------------

def test_curl_analytical():
    """
    For Psi = (0, 0, x*y), curl(Psi) = (0, 0, y) analytically.
    Verify autograd matches.
    """
    xyz = torch.tensor([[1.0, 2.0, 3.0],
                         [2.0, 3.0, 4.0]], requires_grad=True)
    # psi_z = x*y  =>  curl = (dpsi_z/dy - 0, 0 - dpsi_z/dx, 0)
    #                       = (x, -y, 0)  ... wait:
    # u = dpsi_z/dy - dpsi_y/dz = x - 0 = x
    # v = dpsi_x/dz - dpsi_z/dx = 0 - y = -y
    # w = dpsi_y/dx - dpsi_x/dy = 0 - 0 = 0
    psi = torch.stack([
        torch.zeros(2),
        torch.zeros(2),
        xyz[:, 0] * xyz[:, 1],   # psi_z = x*y
    ], dim=1)

    u = curl(psi, xyz)
    expected_u = xyz[:, 0:1]    # x
    expected_v = -xyz[:, 1:2]   # -y
    expected_w = torch.zeros(2, 1)

    assert torch.allclose(u[:, 0:1], expected_u, atol=1e-5)
    assert torch.allclose(u[:, 1:2], expected_v, atol=1e-5)
    assert torch.allclose(u[:, 2:3], expected_w, atol=1e-5)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

def test_compute_losses_keys(model, xyz):
    losses = model.compute_losses(xyz, nu=1e-4)
    assert 'momentum' in losses
    assert 'data' not in losses   # no data provided


def test_compute_losses_with_data(model, xyz):
    torch.manual_seed(1)
    xyz_data = torch.rand(20, 3, requires_grad=True)
    u_data   = torch.rand(20, 2)   # 2C PIV (in-plane only)

    losses = model.compute_losses(xyz, nu=1e-4,
                                  u_data=u_data, xyz_data=xyz_data)
    assert 'momentum' in losses
    assert 'data' in losses
    assert losses['data'].item() >= 0.0


def test_losses_are_scalar(model, xyz):
    losses = model.compute_losses(xyz, nu=1e-4)
    for name, val in losses.items():
        assert val.shape == (), f"Loss '{name}' is not scalar: {val.shape}"


def test_losses_are_finite(model, xyz):
    losses = model.compute_losses(xyz, nu=1e-4)
    for name, val in losses.items():
        assert torch.isfinite(val), f"Loss '{name}' is not finite: {val}"


# ---------------------------------------------------------------------------
# Device agnosticism
# ---------------------------------------------------------------------------

def test_cpu_device(model, xyz):
    """Explicit CPU test — mirrors cloud GPU usage pattern."""
    model_cpu = model.to('cpu')
    xyz_cpu   = xyz.detach().requires_grad_(True)
    u = model_cpu.velocity(xyz_cpu)
    assert u.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
