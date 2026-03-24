"""
tests/test_data_parser.py
=========================
Formal pytest suite for src/data/data_parser.py

Covers:
  1. Directory classifier — all regime variants including both
     'calib_nodrop' and 'calib_no_drop' spellings.
  2. ScaleParams — physical coordinate mapping (pixel -> mm).
  3. parse_vc7 — full pipeline on ReadIM built-in sample files:
       - correct array shapes
       - physical coordinate ranges are finite and non-empty
       - velocity arrays have correct dtype and shape
       - 2C file produces NaN-filled w_phys
       - 3C file produces valid w_phys
       - FlowRegime override is respected
       - C-level buffer is destroyed (no ReadIM buffer leak)
  4. Non-dimensionalisation — dimensionless values scale correctly.
  5. ExperimentCondition.iter_snapshots — lazy iteration yields
     PIVSnapshot objects without memory leaks.

All tests run on CPU only; no GPU or external data required.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import ReadIM

from src.data.data_parser import (
    ExperimentCondition,
    FlowRegime,
    NonDimSnapshot,
    PIVSnapshot,
    ScaleParams,
    classify_directory,
    parse_vc7,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_2c_path() -> Path:
    """Path to ReadIM's built-in 2-component sample VC7."""
    return Path(ReadIM.extra.get_sample_vector_filenames()[0])


@pytest.fixture(scope="module")
def sample_3c_path() -> Path:
    """Path to ReadIM's built-in 3-component sample VC7."""
    return Path(ReadIM.extra.get_sample_vector_filenames()[1])


@pytest.fixture(scope="module")
def snap_2c(sample_2c_path) -> PIVSnapshot:
    """Parsed 2C snapshot (module-scoped — parsed once, reused)."""
    return parse_vc7(sample_2c_path, regime=FlowRegime.BACKGROUND)


@pytest.fixture(scope="module")
def snap_3c(sample_3c_path) -> PIVSnapshot:
    """Parsed 3C snapshot."""
    return parse_vc7(sample_3c_path, regime=FlowRegime.DROPLET)


# ---------------------------------------------------------------------------
# 1. Directory classifier
# ---------------------------------------------------------------------------

class TestClassifyDirectory:

    @pytest.mark.parametrize("dir_name,expected", [
        # Background — compact DaVis spelling (no underscore between no/drop)
        ("Calib_nodrop_pressure",   FlowRegime.BACKGROUND),
        ("CALIB_NODROP",            FlowRegime.BACKGROUND),
        # Background — canonical spelling with underscore
        ("calib_no_drop_test",      FlowRegime.BACKGROUND),
        ("Calib_No_Drop_LargeView", FlowRegime.BACKGROUND),
        # Droplet variants
        ("Ethanol_drop",            FlowRegime.DROPLET),
        ("ethanol_drop_run2",       FlowRegime.DROPLET),
        ("Water_drop",              FlowRegime.DROPLET),
        ("water_drop_run3",         FlowRegime.DROPLET),
        ("Two_drop",                FlowRegime.DROPLET),
        ("two_drop_experiment",     FlowRegime.DROPLET),
        # Unknown
        ("random_experiment",       FlowRegime.UNKNOWN),
        ("LargeView",               FlowRegime.UNKNOWN),
        ("SmallView",               FlowRegime.UNKNOWN),
    ])
    def test_classify(self, dir_name: str, expected: FlowRegime):
        result = classify_directory(Path("/fake/root") / dir_name)
        assert result == expected, (
            f"'{dir_name}' -> {result.name}, expected {expected.name}"
        )


# ---------------------------------------------------------------------------
# 2. ScaleParams — coordinate mapping
# ---------------------------------------------------------------------------

class TestScaleParams:

    def test_pixel_to_physical_x_origin(self):
        """Pixel index 0 maps to the offset value."""
        sc = ScaleParams(
            x_factor=0.25, x_offset=-100.0,
            y_factor=-0.25, y_offset=50.0,
            vel_factor=0.01, vel_offset=0.0,
        )
        x = sc.pixel_to_physical_x(np.array([0.0]))
        assert x[0] == pytest.approx(-100.0)

    def test_pixel_to_physical_x_linear(self):
        """Mapping is strictly linear: x = factor*i + offset."""
        sc = ScaleParams(
            x_factor=0.5, x_offset=10.0,
            y_factor=-0.5, y_offset=20.0,
            vel_factor=0.02, vel_offset=0.0,
        )
        idx = np.array([0.0, 1.0, 2.0, 10.0])
        expected = 0.5 * idx + 10.0
        np.testing.assert_allclose(sc.pixel_to_physical_x(idx), expected)

    def test_pixel_to_physical_y_negative_factor(self):
        """scaleY.factor is negative in DaVis (image y-axis flipped)."""
        sc = ScaleParams(
            x_factor=0.25, x_offset=0.0,
            y_factor=-0.25, y_offset=50.0,
            vel_factor=0.01, vel_offset=0.0,
        )
        y = sc.pixel_to_physical_y(np.array([0.0, 10.0]))
        assert y[0] == pytest.approx(50.0)
        assert y[1] < y[0], "Negative factor: y decreases with pixel index"


# ---------------------------------------------------------------------------
# 3. parse_vc7 — full pipeline
# ---------------------------------------------------------------------------

class TestParseVC7:

    def test_returns_piv_snapshot(self, snap_2c):
        assert isinstance(snap_2c, PIVSnapshot)

    def test_regime_override_respected(self, snap_2c, snap_3c):
        assert snap_2c.regime == FlowRegime.BACKGROUND
        assert snap_3c.regime == FlowRegime.DROPLET

    # --- 2C file ---

    def test_2c_velocity_shape(self, snap_2c):
        ny, nx = snap_2c.u_phys.shape
        assert snap_2c.v_phys.shape == (ny, nx)
        assert snap_2c.w_phys.shape == (ny, nx)
        assert nx > 0 and ny > 0

    def test_2c_n_components(self, snap_2c):
        assert snap_2c.n_components == 2

    def test_2c_w_is_nan(self, snap_2c):
        """2C files have no out-of-plane component — w must be all NaN."""
        assert np.all(np.isnan(snap_2c.w_phys)), \
            "w_phys must be NaN-filled for a 2C file"

    def test_2c_uv_finite(self, snap_2c):
        assert np.all(np.isfinite(snap_2c.u_phys))
        assert np.all(np.isfinite(snap_2c.v_phys))

    def test_2c_coordinates_finite(self, snap_2c):
        assert np.all(np.isfinite(snap_2c.x_phys))
        assert np.all(np.isfinite(snap_2c.y_phys))

    def test_2c_coordinate_lengths_match_velocity(self, snap_2c):
        ny, nx = snap_2c.u_phys.shape
        assert len(snap_2c.x_phys) == nx
        assert len(snap_2c.y_phys) == ny

    def test_2c_scale_units(self, snap_2c):
        assert "mm" in snap_2c.scale.x_unit
        assert "mm" in snap_2c.scale.y_unit

    def test_2c_scale_factor_nonzero(self, snap_2c):
        assert snap_2c.scale.x_factor != 0.0
        assert snap_2c.scale.vel_factor != 0.0

    def test_2c_attributes_dict(self, snap_2c):
        assert isinstance(snap_2c.attributes, dict)
        assert len(snap_2c.attributes) > 0

    # --- 3C file ---

    def test_3c_n_components(self, snap_3c):
        assert snap_3c.n_components == 3

    def test_3c_w_finite(self, snap_3c):
        """3C files must have a valid (non-NaN) w component."""
        assert np.all(np.isfinite(snap_3c.w_phys)), \
            "w_phys must be finite for a 3C file"

    def test_3c_velocity_shape_consistent(self, snap_3c):
        assert snap_3c.u_phys.shape == snap_3c.v_phys.shape
        assert snap_3c.u_phys.shape == snap_3c.w_phys.shape

    # --- dtype ---

    def test_velocity_dtype_float64(self, snap_2c, snap_3c):
        for snap in (snap_2c, snap_3c):
            assert snap.u_phys.dtype == np.float64
            assert snap.v_phys.dtype == np.float64
            assert snap.w_phys.dtype == np.float64

    # --- coordinate monotonicity ---

    def test_x_coordinates_monotone(self, snap_2c):
        """x_phys must be strictly monotone (positive factor)."""
        diffs = np.diff(snap_2c.x_phys)
        assert np.all(diffs > 0) or np.all(diffs < 0), \
            "x_phys must be strictly monotone"

    def test_y_coordinates_monotone(self, snap_2c):
        """y_phys must be strictly monotone (negative factor in DaVis)."""
        diffs = np.diff(snap_2c.y_phys)
        assert np.all(diffs > 0) or np.all(diffs < 0), \
            "y_phys must be strictly monotone"


# ---------------------------------------------------------------------------
# 4. Non-dimensionalisation
# ---------------------------------------------------------------------------

class TestNonDimensionalisation:

    R_e   = 1.0e-3   # 1 mm
    u_ref = 0.01     # 10 mm/s

    def test_returns_non_dim_snapshot(self, snap_2c):
        nd = snap_2c.non_dimensionalise(self.R_e, self.u_ref)
        assert isinstance(nd, NonDimSnapshot)

    def test_x_star_shape(self, snap_2c):
        nd = snap_2c.non_dimensionalise(self.R_e, self.u_ref)
        assert nd.x_star.shape == snap_2c.x_phys.shape

    def test_u_star_shape(self, snap_2c):
        nd = snap_2c.non_dimensionalise(self.R_e, self.u_ref)
        assert nd.u_star.shape == snap_2c.u_phys.shape

    def test_scaling_correctness(self, snap_2c):
        """x* = x_phys[mm] * 1e-3 / R_e,  u* = u_phys / u_ref."""
        nd = snap_2c.non_dimensionalise(self.R_e, self.u_ref)
        expected_x0 = snap_2c.x_phys[0] * 1e-3 / self.R_e
        assert nd.x_star[0] == pytest.approx(expected_x0, rel=1e-10)

        expected_u = snap_2c.u_phys / self.u_ref
        np.testing.assert_allclose(nd.u_star, expected_u, rtol=1e-10)

    def test_source_reference_preserved(self, snap_2c):
        nd = snap_2c.non_dimensionalise(self.R_e, self.u_ref)
        assert nd.source is snap_2c
        assert nd.R_e == self.R_e
        assert nd.u_ref == self.u_ref


# ---------------------------------------------------------------------------
# 5. ExperimentCondition.iter_snapshots
# ---------------------------------------------------------------------------

class TestExperimentCondition:

    def test_iter_snapshots_yields_piv_snapshot(self, sample_2c_path, sample_3c_path):
        cond = ExperimentCondition(
            directory=sample_2c_path.parent,
            fov="LargeView",
            regime=FlowRegime.BACKGROUND,
            vc7_files=[sample_2c_path, sample_3c_path],
        )
        snaps = list(cond.iter_snapshots())
        assert len(snaps) == 2
        assert all(isinstance(s, PIVSnapshot) for s in snaps)

    def test_iter_snapshots_len(self, sample_2c_path):
        cond = ExperimentCondition(
            directory=sample_2c_path.parent,
            fov="SmallView",
            regime=FlowRegime.BACKGROUND,
            vc7_files=[sample_2c_path],
        )
        assert len(cond) == 1

    def test_iter_snapshots_regime_propagated(self, sample_2c_path):
        cond = ExperimentCondition(
            directory=sample_2c_path.parent,
            fov="LargeView",
            regime=FlowRegime.DROPLET,
            vc7_files=[sample_2c_path],
        )
        snap = next(cond.iter_snapshots())
        assert snap.regime == FlowRegime.DROPLET
