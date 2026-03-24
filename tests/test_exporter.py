"""
Tests for exporter.py — inference and VTK export pipeline.
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import pyvista as pv

from src.network.pinn_model import StreamingPINN
from src.geometry.geometry import OblateSpheroid
from src.postprocess.exporter import generate_3d_grid, predict_flow_field, export_vtr


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    return StreamingPINN(hidden=16, layers=2)


@pytest.fixture(scope='module')
def geom():
    return OblateSpheroid(R_e=1.5e-3, R_p=1.0e-3)


@pytest.fixture(scope='module')
def grid_10():
    """10x10x10 grid over ±5 mm domain."""
    return generate_3d_grid((-5e-3, 5e-3), (-5e-3, 5e-3), (-5e-3, 5e-3), resolution=10)


@pytest.fixture(scope='module')
def fields_10(model, grid_10):
    _, _, _, pts = grid_10
    return predict_flow_field(model, pts, batch_size=200)


# ---------------------------------------------------------------------------
# generate_3d_grid
# ---------------------------------------------------------------------------

def test_grid_shape_uniform(grid_10):
    xs, ys, zs, pts = grid_10
    assert xs.shape == (10,)
    assert ys.shape == (10,)
    assert zs.shape == (10,)
    assert pts.shape == (1000, 3)


def test_grid_shape_nonuniform():
    xs, ys, zs, pts = generate_3d_grid((0, 1), (0, 2), (0, 3), resolution=(2, 3, 4))
    assert xs.shape == (2,)
    assert ys.shape == (3,)
    assert zs.shape == (4,)
    assert pts.shape == (24, 3)


def test_grid_bounds(grid_10):
    xs, ys, zs, pts = grid_10
    assert pytest.approx(xs[0],  abs=1e-9) == -5e-3
    assert pytest.approx(xs[-1], abs=1e-9) ==  5e-3


def test_grid_dtype(grid_10):
    _, _, _, pts = grid_10
    assert pts.dtype == torch.float32


# ---------------------------------------------------------------------------
# predict_flow_field
# ---------------------------------------------------------------------------

def test_field_keys(fields_10):
    assert {'velocity', 'pressure', 'vector_potential'} <= fields_10.keys()


def test_field_shapes(fields_10):
    assert fields_10['velocity'].shape         == (1000, 3)
    assert fields_10['pressure'].shape         == (1000, 1)
    assert fields_10['vector_potential'].shape == (1000, 3)


def test_fields_finite_before_masking(fields_10):
    """Untrained model should still produce finite outputs."""
    for name, arr in fields_10.items():
        assert np.isfinite(arr).all(), f"Non-finite values in '{name}' before masking"


def test_batched_equals_full(model):
    """Batched inference must match single-pass inference."""
    torch.manual_seed(1)
    pts = torch.rand(100, 3, dtype=torch.float32) * 2e-3

    f_full   = predict_flow_field(model, pts, batch_size=100)
    f_batched = predict_flow_field(model, pts, batch_size=25)

    np.testing.assert_allclose(
        f_full['velocity'], f_batched['velocity'], atol=1e-6,
        err_msg="Batched inference differs from full-batch inference"
    )


# ---------------------------------------------------------------------------
# export_vtr
# ---------------------------------------------------------------------------

def test_vtr_file_created(model, grid_10, fields_10):
    xs, ys, zs, _ = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(xs, ys, zs, fields_10, os.path.join(d, 'out.vtr'))
        assert path.exists()
        assert path.stat().st_size > 0


def test_vtr_readable_by_pyvista(model, grid_10, fields_10):
    xs, ys, zs, _ = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(xs, ys, zs, fields_10, os.path.join(d, 'out.vtr'))
        grid = pv.read(str(path))
        assert grid.n_points == 1000
        assert 'velocity' in grid.point_data


def test_vtr_field_names(model, grid_10, fields_10):
    xs, ys, zs, _ = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(xs, ys, zs, fields_10, os.path.join(d, 'out.vtr'))
        grid = pv.read(str(path))
        for key in ('velocity', 'pressure', 'vector_potential'):
            assert key in grid.point_data


# ---------------------------------------------------------------------------
# Droplet masking
# ---------------------------------------------------------------------------

def test_interior_masked_to_nan(model, grid_10, fields_10, geom):
    xs, ys, zs, pts = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(
            xs, ys, zs, fields_10, os.path.join(d, 'masked.vtr'),
            geom=geom, mask_interior=True,
        )
        grid = pv.read(str(path))
        vel = grid.point_data['velocity']

        # At least some interior points should be NaN (droplet is inside domain)
        assert np.isnan(vel).any(), "Expected NaN values inside droplet mask"


def test_no_masking_all_finite(model, grid_10, fields_10):
    xs, ys, zs, _ = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(
            xs, ys, zs, fields_10, os.path.join(d, 'unmasked.vtr'),
            geom=None, mask_interior=False,
        )
        grid = pv.read(str(path))
        vel = grid.point_data['velocity']
        assert np.isfinite(vel).all(), "Without masking, all values should be finite"


def test_exterior_points_not_nan(model, grid_10, fields_10, geom):
    """Points clearly outside the droplet must retain finite values after masking."""
    xs, ys, zs, _ = grid_10
    with tempfile.TemporaryDirectory() as d:
        path = export_vtr(
            xs, ys, zs, fields_10, os.path.join(d, 'masked2.vtr'),
            geom=geom, mask_interior=True,
        )
        grid = pv.read(str(path))
        vel = grid.point_data['velocity']
        # Not ALL points should be NaN — the droplet is much smaller than the domain
        assert np.isfinite(vel).any(), "Some exterior points must remain finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
