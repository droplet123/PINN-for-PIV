"""
tests/test_geometry.py
======================
Formal pytest suite for src/geometry/geometry.py

Covers:
  1. OblateSpheroid — construction, derived properties, validation.
  2. sdf_oblate_spheroid — sign convention, surface accuracy, monotonicity.
  3. surface_normals — unit length, outward direction, pole/equator cases.
  4. sample_surface — point count, surface accuracy, normal consistency.
  5. sample_volume — point count, exterior constraint, SDF values.
  6. sdf_weight — peak at surface, monotone decay, boundary values.
  7. Autograd — SDF differentiable end-to-end with create_graph=True.

All tests run on CPU only; no GPU or external data required.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.geometry.geometry import (
    OblateSpheroid,
    sample_surface,
    sample_volume,
    sdf_oblate_spheroid,
    sdf_weight,
    surface_normals,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def oblate() -> OblateSpheroid:
    """Standard oblate spheroid: R_e=1.5mm, R_p=1.0mm, centred at origin."""
    return OblateSpheroid(R_e=1.5e-3, R_p=1.0e-3, centre=(0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def sphere() -> OblateSpheroid:
    """Degenerate case: perfect sphere (R_e == R_p)."""
    return OblateSpheroid(R_e=1.0e-3, R_p=1.0e-3, centre=(0.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def oblate_offset() -> OblateSpheroid:
    """Oblate spheroid with non-zero centre to test coordinate shifting."""
    return OblateSpheroid(R_e=1.5e-3, R_p=1.0e-3, centre=(5e-3, -3e-3, 2e-3))


# ---------------------------------------------------------------------------
# 1. OblateSpheroid
# ---------------------------------------------------------------------------

class TestOblateSpheroid:

    def test_aspect_ratio_oblate(self, oblate):
        assert oblate.aspect_ratio == pytest.approx(1.0e-3 / 1.5e-3, rel=1e-10)
        assert oblate.aspect_ratio < 1.0

    def test_aspect_ratio_sphere(self, sphere):
        assert sphere.aspect_ratio == pytest.approx(1.0, rel=1e-10)

    def test_volume(self, oblate):
        expected = (4.0 / 3.0) * math.pi * (1.5e-3)**2 * 1.0e-3
        assert oblate.volume == pytest.approx(expected, rel=1e-10)

    def test_invalid_axes_raises(self):
        with pytest.raises(ValueError):
            OblateSpheroid(R_e=0.0, R_p=1e-3)
        with pytest.raises(ValueError):
            OblateSpheroid(R_e=1e-3, R_p=-1e-3)

    def test_prolate_warning(self, caplog):
        """R_p > R_e is prolate — geometry.py emits a log WARNING."""
        import logging
        with caplog.at_level(logging.WARNING, logger="geometry"):
            prolate = OblateSpheroid(R_e=1e-3, R_p=2e-3)
        assert any("PROLATE" in r.message.upper() for r in caplog.records)
        assert prolate.aspect_ratio > 1.0

    def test_centre_tensor_shape(self, oblate_offset):
        ct = oblate_offset.centre_tensor(DEVICE)
        assert ct.shape == (1, 3)
        assert ct.dtype == torch.float64

    def test_centre_tensor_values(self, oblate_offset):
        ct = oblate_offset.centre_tensor(DEVICE)
        np.testing.assert_allclose(
            ct.numpy(), [[5e-3, -3e-3, 2e-3]], rtol=1e-12
        )


# ---------------------------------------------------------------------------
# 2. sdf_oblate_spheroid
# ---------------------------------------------------------------------------

class TestSDF:

    def _pts(self, coords: list) -> torch.Tensor:
        return torch.tensor(coords, dtype=torch.float64, device=DEVICE)

    def test_centre_is_inside(self, oblate):
        pts = self._pts([[0.0, 0.0, 0.0]])
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi[0].item() < 0.0

    def test_equator_on_surface(self, oblate):
        pts = self._pts([[oblate.R_e, 0.0, 0.0]])
        phi = sdf_oblate_spheroid(pts, oblate)
        assert abs(phi[0].item()) < 1e-10

    def test_pole_on_surface(self, oblate):
        pts = self._pts([[0.0, 0.0, oblate.R_p]])
        phi = sdf_oblate_spheroid(pts, oblate)
        assert abs(phi[0].item()) < 1e-10

    def test_exterior_positive(self, oblate):
        pts = self._pts([[2 * oblate.R_e, 0.0, 0.0]])
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi[0].item() > 0.0

    def test_exterior_above_pole(self, oblate):
        pts = self._pts([[0.0, 0.0, 2 * oblate.R_p]])
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi[0].item() > 0.0

    def test_sdf_increases_with_distance(self, oblate):
        """SDF must be monotonically increasing along the x-axis outside."""
        x_vals = torch.tensor(
            [[r, 0.0, 0.0] for r in [oblate.R_e, 1.5*oblate.R_e, 2.0*oblate.R_e]],
            dtype=torch.float64, device=DEVICE,
        )
        phi = sdf_oblate_spheroid(x_vals, oblate)
        assert phi[0].item() <= phi[1].item() <= phi[2].item()

    def test_sphere_sdf_equals_radial_distance(self, sphere):
        """For a sphere, SDF = |r| - R exactly."""
        r = 2.0 * sphere.R_e
        pts = self._pts([[r, 0.0, 0.0]])
        phi = sdf_oblate_spheroid(pts, sphere)
        expected = r - sphere.R_e
        assert abs(phi[0].item() - expected) < 1e-9

    def test_offset_centre(self, oblate_offset):
        """SDF must be zero at the equator of an offset spheroid."""
        cx, cy, cz = oblate_offset.centre
        pts = self._pts([[cx + oblate_offset.R_e, cy, cz]])
        phi = sdf_oblate_spheroid(pts, oblate_offset)
        assert abs(phi[0].item()) < 1e-10

    def test_output_shape(self, oblate):
        N = 50
        pts = torch.randn(N, 3, dtype=torch.float64, device=DEVICE) * oblate.R_e
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi.shape == (N,)

    def test_output_dtype(self, oblate):
        pts = torch.zeros(5, 3, dtype=torch.float64, device=DEVICE)
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi.dtype == torch.float64

    def test_batch_consistency(self, oblate):
        """Single-point and batched evaluation must agree."""
        pts_single = torch.tensor([[oblate.R_e * 1.5, 0.0, 0.0]],
                                  dtype=torch.float64, device=DEVICE)
        pts_batch  = pts_single.expand(10, -1)
        phi_s = sdf_oblate_spheroid(pts_single, oblate)
        phi_b = sdf_oblate_spheroid(pts_batch,  oblate)
        assert torch.allclose(phi_b, phi_s.expand(10), atol=1e-14)


# ---------------------------------------------------------------------------
# 3. surface_normals
# ---------------------------------------------------------------------------

class TestSurfaceNormals:

    def test_unit_length_equatorial(self, oblate):
        """Normals on the equatorial belt must have unit length."""
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        pts_np = np.stack([
            oblate.R_e * np.cos(angles),
            oblate.R_e * np.sin(angles),
            np.zeros_like(angles),
        ], axis=1)
        pts = torch.tensor(pts_np, dtype=torch.float64, device=DEVICE)
        n = surface_normals(pts, oblate)
        norms = n.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)

    def test_outward_equatorial(self, oblate):
        """Equatorial normals must point radially outward (z-component ~ 0)."""
        angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        pts_np = np.stack([
            oblate.R_e * np.cos(angles),
            oblate.R_e * np.sin(angles),
            np.zeros_like(angles),
        ], axis=1)
        pts = torch.tensor(pts_np, dtype=torch.float64, device=DEVICE)
        n = surface_normals(pts, oblate)
        # Radial dot product should be ~1
        radial = (n[:, :2] * pts[:, :2] / oblate.R_e).sum(dim=1)
        assert radial.min().item() > 0.99

    def test_pole_normal_is_z(self, oblate):
        """Normal at the north pole must be (0, 0, 1)."""
        pts = torch.tensor([[0.0, 0.0, oblate.R_p]],
                           dtype=torch.float64, device=DEVICE)
        n = surface_normals(pts, oblate)
        assert abs(n[0, 0].item()) < 1e-12
        assert abs(n[0, 1].item()) < 1e-12
        assert abs(n[0, 2].item() - 1.0) < 1e-12

    def test_south_pole_normal_is_neg_z(self, oblate):
        """Normal at the south pole must be (0, 0, -1)."""
        pts = torch.tensor([[0.0, 0.0, -oblate.R_p]],
                           dtype=torch.float64, device=DEVICE)
        n = surface_normals(pts, oblate)
        assert abs(n[0, 2].item() + 1.0) < 1e-12

    def test_output_shape(self, oblate):
        N = 30
        pts = torch.randn(N, 3, dtype=torch.float64, device=DEVICE) * oblate.R_e
        n = surface_normals(pts, oblate)
        assert n.shape == (N, 3)

    def test_sphere_normals_radial(self, sphere):
        """For a sphere, normals must equal the unit position vector."""
        angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        pts_np = np.stack([
            sphere.R_e * np.cos(angles),
            sphere.R_e * np.sin(angles),
            np.zeros_like(angles),
        ], axis=1)
        pts = torch.tensor(pts_np, dtype=torch.float64, device=DEVICE)
        n = surface_normals(pts, sphere)
        expected = F.normalize(pts.double(), p=2, dim=1)
        assert torch.allclose(n, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. sample_surface
# ---------------------------------------------------------------------------

class TestSampleSurface:

    def test_point_count(self, oblate):
        pts, norms = sample_surface(oblate, n_pts=100, device=DEVICE, seed=0)
        assert pts.shape == (100, 3)
        assert norms.shape == (100, 3)

    def test_points_on_surface(self, oblate):
        """All sampled points must satisfy |phi| < 1e-8 m."""
        pts, _ = sample_surface(oblate, n_pts=200, device=DEVICE, seed=1)
        phi = sdf_oblate_spheroid(pts, oblate)
        assert phi.abs().max().item() < 1e-8

    def test_normals_unit_length(self, oblate):
        _, norms = sample_surface(oblate, n_pts=100, device=DEVICE, seed=2)
        norms_mag = norms.norm(dim=1)
        assert torch.allclose(norms_mag, torch.ones_like(norms_mag), atol=1e-12)

    def test_normals_outward(self, oblate):
        """Normals must point away from the centre (positive dot with position)."""
        pts, norms = sample_surface(oblate, n_pts=100, device=DEVICE, seed=3)
        c = oblate.centre_tensor(DEVICE)
        dot = ((pts - c) * norms).sum(dim=1)
        assert dot.min().item() > 0.0

    def test_reproducibility(self, oblate):
        pts1, _ = sample_surface(oblate, n_pts=50, device=DEVICE, seed=42)
        pts2, _ = sample_surface(oblate, n_pts=50, device=DEVICE, seed=42)
        assert torch.allclose(pts1, pts2)

    def test_different_seeds_differ(self, oblate):
        pts1, _ = sample_surface(oblate, n_pts=50, device=DEVICE, seed=0)
        pts2, _ = sample_surface(oblate, n_pts=50, device=DEVICE, seed=1)
        assert not torch.allclose(pts1, pts2)

    def test_dtype(self, oblate):
        pts, norms = sample_surface(oblate, n_pts=10, device=DEVICE, seed=0)
        assert pts.dtype == torch.float64
        assert norms.dtype == torch.float64


# ---------------------------------------------------------------------------
# 5. sample_volume
# ---------------------------------------------------------------------------

class TestSampleVolume:

    def test_point_count(self, oblate):
        pts, phi = sample_volume(oblate, n_pts=100, device=DEVICE, seed=0)
        assert pts.shape == (100, 3)
        assert phi.shape == (100,)

    def test_exterior_only(self, oblate):
        """With exclude_interior=True, all phi must be >= 0."""
        _, phi = sample_volume(
            oblate, n_pts=200, device=DEVICE,
            exclude_interior=True, seed=0
        )
        assert phi.min().item() >= -1e-30

    def test_include_interior(self, oblate):
        """With exclude_interior=False, some phi should be negative.
        Use domain_scale=1.0 so the bounding box tightly wraps the droplet,
        guaranteeing a meaningful fraction of interior points."""
        _, phi = sample_volume(
            oblate, n_pts=500, device=DEVICE,
            domain_scale=1.0,
            exclude_interior=False, seed=0
        )
        assert phi.min().item() < 0.0

    def test_domain_scale(self, oblate):
        """All points must lie within domain_scale * R_e of the centre."""
        scale = 3.0
        pts, _ = sample_volume(
            oblate, n_pts=100, device=DEVICE,
            domain_scale=scale, seed=0
        )
        c = oblate.centre_tensor(DEVICE)
        max_dist = (pts - c).abs().max().item()
        assert max_dist <= scale * oblate.R_e + 1e-15

    def test_dtype(self, oblate):
        pts, phi = sample_volume(oblate, n_pts=10, device=DEVICE, seed=0)
        assert pts.dtype == torch.float64
        assert phi.dtype == torch.float64


# ---------------------------------------------------------------------------
# 6. sdf_weight
# ---------------------------------------------------------------------------

class TestSDFWeight:

    def test_peak_at_surface(self, oblate):
        """w(0) must equal exactly 1.0."""
        phi = torch.tensor([0.0], dtype=torch.float64)
        w = sdf_weight(phi, geom=oblate)
        assert w[0].item() == pytest.approx(1.0, abs=1e-15)

    def test_monotone_decay(self, oblate):
        """Weight must strictly decrease as |phi| increases."""
        phi = torch.tensor(
            [0.0, 0.5 * oblate.R_e, oblate.R_e, 2.0 * oblate.R_e],
            dtype=torch.float64,
        )
        w = sdf_weight(phi, geom=oblate)
        assert w[0] > w[1] > w[2] > w[3]

    def test_explicit_sigma(self, oblate):
        """Explicit sigma overrides the default R_e/4."""
        phi = torch.tensor([oblate.R_e], dtype=torch.float64)
        sigma = oblate.R_e
        w = sdf_weight(phi, sigma=sigma)
        expected = math.exp(-0.5)   # phi == sigma => exp(-1/2)
        assert w[0].item() == pytest.approx(expected, rel=1e-10)

    def test_no_geom_no_sigma_raises(self):
        phi = torch.tensor([1e-3], dtype=torch.float64)
        with pytest.raises(ValueError):
            sdf_weight(phi)

    def test_output_in_unit_interval(self, oblate):
        phi = torch.linspace(-3 * oblate.R_e, 3 * oblate.R_e, 100,
                             dtype=torch.float64)
        w = sdf_weight(phi, geom=oblate)
        assert w.min().item() >= 0.0
        assert w.max().item() <= 1.0 + 1e-15

    def test_symmetric_around_surface(self, oblate):
        """w(+phi) must equal w(-phi) — Gaussian is symmetric."""
        phi_pos = torch.tensor([oblate.R_e], dtype=torch.float64)
        phi_neg = torch.tensor([-oblate.R_e], dtype=torch.float64)
        w_pos = sdf_weight(phi_pos, geom=oblate)
        w_neg = sdf_weight(phi_neg, geom=oblate)
        assert w_pos[0].item() == pytest.approx(w_neg[0].item(), rel=1e-12)


# ---------------------------------------------------------------------------
# 7. Autograd differentiability
# ---------------------------------------------------------------------------

class TestAutograd:

    def test_sdf_gradient_nonzero(self, oblate):
        """grad(SDF) w.r.t. input coordinates must be non-zero."""
        pts = torch.randn(50, 3, dtype=torch.float64,
                          device=DEVICE, requires_grad=False) * oblate.R_e
        pts = pts.requires_grad_(True)
        phi = sdf_oblate_spheroid(pts, oblate)
        grad = torch.autograd.grad(
            outputs=phi,
            inputs=pts,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        assert grad is not None
        assert grad.norm().item() > 0.0

    def test_sdf_second_order_gradient(self, oblate):
        """Second-order gradient (needed for PINN Laplacian) must exist."""
        pts = torch.randn(10, 3, dtype=torch.float64,
                          device=DEVICE) * oblate.R_e
        pts = pts.requires_grad_(True)
        phi = sdf_oblate_spheroid(pts, oblate)
        grad1 = torch.autograd.grad(
            outputs=phi,
            inputs=pts,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        # Second derivative: grad of (sum of first grads)
        grad2 = torch.autograd.grad(
            outputs=grad1.sum(),
            inputs=pts,
            create_graph=False,
        )[0]
        assert grad2 is not None
        assert grad2.shape == pts.shape

    def test_sdf_gradient_shape(self, oblate):
        N = 30
        pts = torch.randn(N, 3, dtype=torch.float64,
                          device=DEVICE, requires_grad=True) * oblate.R_e
        phi = sdf_oblate_spheroid(pts, oblate)
        grad = torch.autograd.grad(
            outputs=phi,
            inputs=pts,
            grad_outputs=torch.ones_like(phi),
        )[0]
        assert grad.shape == (N, 3)

    def test_sdf_weight_differentiable(self, oblate):
        """sdf_weight must be differentiable (used in adaptive loss)."""
        phi = torch.linspace(0, oblate.R_e, 20,
                             dtype=torch.float64, requires_grad=True)
        w = sdf_weight(phi, geom=oblate)
        w.sum().backward()
        assert phi.grad is not None
        assert phi.grad.norm().item() > 0.0
