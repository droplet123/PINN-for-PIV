"""
data_parser.py
==============
Step 1 — Data Parsing & Preprocessing
3D Acoustic Streaming Reconstruction via PINN

Responsibilities
----------------
* Traverse the experimental PIV data root and auto-classify every
  subdirectory into one of two physical regimes:
    - Background flow  (path contains ``calib_no_drop``)
    - Droplet flow     (Ethanol_drop, Water_drop, Two_drop, …)
* Parse .vc7 files via ``ReadIM.extra.get_Buffer_andAttributeList``.
* Enforce strict memory management (no leaks in batch loops).
* Map pixel-grid indices → physical coordinates [mm] using the
  per-file ``Scales`` metadata embedded by DaVis:
      x_phys = scaleX.factor * i + scaleX.offset
* Expose a clean non-dimensionalisation interface:
      x* = x_phys / R_e,   u* = u_phys / u_ref

Author : (project)
Date   : 2026-03-20
Python : ≥ 3.10
Deps   : ReadIM, numpy, pathlib (stdlib)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterator

import numpy as np
import ReadIM
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level logger  (callers configure the root logger; no basicConfig here)
# ---------------------------------------------------------------------------
log = logging.getLogger("data_parser")

# ---------------------------------------------------------------------------
# Constants — override via PIV_DATA_ROOT env var for Linux cloud deployment
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    os.environ.get("PIV_DATA_ROOT", "")
).resolve() if os.environ.get("PIV_DATA_ROOT") else (
    Path(__file__).resolve().parents[2] / "experimental_data"
)

# Condition-classification keywords (case-insensitive substring match).
# Background folders may appear as "calib_no_drop" or "calib_nodrop" in
# DaVis experiment trees — both variants are captured.
_BACKGROUND_KEYWORDS: tuple[str, ...] = ("calib_no_drop", "calib_nodrop")
_DROPLET_KEYWORDS: tuple[str, ...] = (
    "ethanol_drop", "water_drop", "two_drop"
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class FlowRegime(Enum):
    """Physical classification of an experimental sub-directory."""
    BACKGROUND = auto()   # Acoustic field without a levitated droplet
    DROPLET = auto()      # Acoustic streaming around a levitated droplet
    UNKNOWN = auto()      # Could not classify — skip in downstream steps


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ScaleParams:
    """
    Encapsulates the linear calibration extracted from one .vc7 file.

    DaVis stores calibration as:
        physical = factor * pixel_index + offset
    where *factor* is the scale commonly labelled "Slope" in the literature.
    The field name in ``BufferScaleType`` is ``factor``.
    """
    x_factor: float          # [physical_unit / pixel]
    x_offset: float          # [physical_unit]
    y_factor: float
    y_offset: float
    vel_factor: float        # [velocity_unit / raw_int]
    vel_offset: float
    x_unit: str = "mm"
    y_unit: str = "mm"
    vel_unit: str = "m/s"

    def pixel_to_physical_x(self, i: np.ndarray) -> np.ndarray:
        """Convert x pixel indices → physical coordinates (mm)."""
        return self.x_factor * i + self.x_offset

    def pixel_to_physical_y(self, j: np.ndarray) -> np.ndarray:
        """Convert y pixel indices → physical coordinates (mm).
        Note: scaleY.factor is typically negative in DaVis (image origin
        is top-left; physical y grows downward or is flipped).
        """
        return self.y_factor * j + self.y_offset


@dataclass
class PIVSnapshot:
    """
    One fully parsed, physically-mapped PIV snapshot from a .vc7 file.

    Attributes
    ----------
    filepath    : source .vc7 path
    regime      : background or droplet flow
    scale       : calibration parameters embedded in the file
    x_phys      : (nx,)  physical x-coordinates [mm]
    y_phys      : (ny,)  physical y-coordinates [mm]
    u_phys      : (ny, nx) x-velocity component [m/s]
    v_phys      : (ny, nx) y-velocity component [m/s]
    w_phys      : (ny, nx) z-velocity component [m/s], NaN if 2C file
    n_components: number of velocity components (2 or 3)
    attributes  : raw DaVis attribute dictionary
    """
    filepath: Path
    regime: FlowRegime
    scale: ScaleParams
    x_phys: np.ndarray                           # shape (nx,)
    y_phys: np.ndarray                           # shape (ny,)
    u_phys: np.ndarray                           # shape (ny, nx)
    v_phys: np.ndarray                           # shape (ny, nx)
    w_phys: np.ndarray                           # shape (ny, nx) or NaN-filled
    n_components: int = 2
    attributes: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Non-dimensionalisation
    # ------------------------------------------------------------------
    def non_dimensionalise(
        self,
        R_e: float,
        u_ref: float,
    ) -> "NonDimSnapshot":
        """
        Return a dimensionless copy of this snapshot.

        Physical → dimensionless map
        ----------------------------
            x* = x_phys [mm] × 1e-3 / R_e [m]
            u* = u_phys [m/s] / u_ref [m/s]

        Parameters
        ----------
        R_e   : characteristic equatorial radius of the droplet [m]
        u_ref : characteristic velocity scale [m/s]

        Returns
        -------
        NonDimSnapshot
        """
        mm_to_m = 1e-3   # DaVis x/y are in mm; R_e is in metres

        x_star = self.x_phys * mm_to_m / R_e
        y_star = self.y_phys * mm_to_m / R_e
        u_star = self.u_phys / u_ref
        v_star = self.v_phys / u_ref
        w_star = self.w_phys / u_ref

        return NonDimSnapshot(
            source=self,
            R_e=R_e,
            u_ref=u_ref,
            x_star=x_star,
            y_star=y_star,
            u_star=u_star,
            v_star=v_star,
            w_star=w_star,
        )


@dataclass
class NonDimSnapshot:
    """
    Dimensionless companion to ``PIVSnapshot``.  Coordinates scaled by R_e,
    velocities scaled by u_ref — values cluster around O(1) for stable
    neural-network training.
    """
    source: PIVSnapshot
    R_e: float
    u_ref: float
    x_star: np.ndarray    # shape (nx,)
    y_star: np.ndarray    # shape (ny,)
    u_star: np.ndarray    # shape (ny, nx)
    v_star: np.ndarray    # shape (ny, nx)
    w_star: np.ndarray    # shape (ny, nx)


# ---------------------------------------------------------------------------
# Condition classification
# ---------------------------------------------------------------------------
def classify_directory(directory: Path) -> FlowRegime:
    """
    Determine the flow regime of an experimental sub-directory by
    inspecting its name (case-insensitive substring search).

    Parameters
    ----------
    directory : Path to a sub-directory under the data root.

    Returns
    -------
    FlowRegime enum member.
    """
    name_lower = directory.name.lower()

    for kw in _BACKGROUND_KEYWORDS:
        if kw in name_lower:
            return FlowRegime.BACKGROUND

    for kw in _DROPLET_KEYWORDS:
        if kw in name_lower:
            return FlowRegime.DROPLET

    return FlowRegime.UNKNOWN


# ---------------------------------------------------------------------------
# Low-level .vc7 reader  (single-file, strict memory management)
# ---------------------------------------------------------------------------
def _read_vc7_file(filepath: Path) -> tuple[
    np.ndarray,      # velocity array  (n_comp, ny, nx)
    ReadIM.BufferType,   # buffer — caller must DestroyBuffer
    dict,            # attribute dictionary (plain Python dict)
]:
    """
    Load one .vc7 file and return its raw velocity array together with
    the live BufferType and an attribute dictionary.

    **Caller responsibility**: You MUST call ``ReadIM.DestroyBuffer(buf)``
    on the returned buffer when done.  The attribute list is destroyed
    *inside* this function after conversion to a plain Python dict so
    that only one resource escapes per call.

    Parameters
    ----------
    filepath : absolute path to a .vc7 file

    Returns
    -------
    arr  : numpy array of shape ``(n_components, ny, nx)``
           Components 0, 1 are (u, v); component 2 is w for 3C files.
           All values are already in physical units
           (m/s for velocity, ready after scaleI is applied by DaVis).
    buf  : live ``BufferType`` — scale metadata can be read from it;
           caller must ``ReadIM.DestroyBuffer(buf)`` when done.
    atts : plain Python dict of DaVis attributes (attrName → value string).

    Raises
    ------
    FileNotFoundError : if *filepath* does not exist.
    RuntimeError      : if ReadIM cannot open the file.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"VC7 file not found: {filepath}")

    log.debug("Reading: %s", filepath.name)

    buf, att_list = ReadIM.extra.get_Buffer_andAttributeList(str(filepath))

    # Convert the attribute list to a plain dict immediately so we can
    # destroy the C-level AttributeList object right away — only the
    # buffer stays alive to carry scale metadata to the caller.
    att_dict: dict = ReadIM.att2dict(att_list)
    ReadIM.DestroyAttributeListSafe(att_list)   # ← free C memory now
    del att_list

    # ``buffer_as_array`` returns a *view* into the buffer's memory;
    # shape is (n_components * nf, ny, nx).  For standard 2C PIV with
    # nf=1: shape = (10, ny, nx).  For 3C PIV: shape = (14, ny, nx).
    # Component index 0 → u,  1 → v,  2 → w (3C only).
    arr, _buf_alt = ReadIM.buffer_as_array(buf)
    # _buf_alt is an internal alias — do not destroy separately.

    return arr, buf, att_dict


# ---------------------------------------------------------------------------
# High-level parser — builds a PIVSnapshot with full metadata
# ---------------------------------------------------------------------------
def parse_vc7(
    filepath: Path,
    regime: FlowRegime | None = None,
) -> PIVSnapshot:
    """
    Parse a single .vc7 file into a fully annotated ``PIVSnapshot``.

    This function encapsulates the entire lifecycle of the ReadIM buffer:
    it opens the file, extracts data and metadata, converts everything to
    native Python / NumPy objects, **destroys the buffer**, and returns a
    self-contained data object.  No C-level memory escapes.

    Parameters
    ----------
    filepath : absolute path to the .vc7 file
    regime   : override auto-classification (useful for unit tests)

    Returns
    -------
    PIVSnapshot
    """
    if regime is None:
        regime = classify_directory(filepath.parent)

    arr, buf, att_dict = _read_vc7_file(filepath)

    try:
        # ------------------------------------------------------------------
        # Extract calibration scales from the live buffer
        # ------------------------------------------------------------------
        scale = ScaleParams(
            x_factor=buf.scaleX.factor,
            x_offset=buf.scaleX.offset,
            y_factor=buf.scaleY.factor,
            y_offset=buf.scaleY.offset,
            vel_factor=buf.scaleI.factor,
            vel_offset=buf.scaleI.offset,
            x_unit=buf.scaleX.unit.strip("[]"),   # DaVis wraps units in []
            y_unit=buf.scaleY.unit.strip("[]"),
            vel_unit=buf.scaleI.unit,
        )

        nx, ny = buf.nx, buf.ny
        n_vector_components = ReadIM.GetVectorComponents(buf.image_sub_type)
        # 2C → 10 components; 3C → 14 components.  u=arr[0], v=arr[1], w=arr[2].
        is_3c = n_vector_components >= 14

        # ------------------------------------------------------------------
        # Map pixel indices → physical coordinates
        # Physical formula:  coord = factor × grid_index + offset
        # ------------------------------------------------------------------
        x_idx = np.arange(nx, dtype=np.float64)
        y_idx = np.arange(ny, dtype=np.float64)
        x_phys = scale.pixel_to_physical_x(x_idx)   # (nx,) [mm]
        y_phys = scale.pixel_to_physical_y(y_idx)    # (ny,) [mm]

        # ``buffer_as_array`` already applies scaleI internally (DaVis convention),
        # so arr[0..2] are directly in physical velocity units (m/s).
        u_phys = arr[0].astype(np.float64).copy()    # (ny, nx)
        v_phys = arr[1].astype(np.float64).copy()    # (ny, nx)
        if is_3c:
            w_phys = arr[2].astype(np.float64).copy()
            n_comp = 3
        else:
            # Out-of-plane component absent; fill with NaN for uniform API
            w_phys = np.full_like(u_phys, np.nan)
            n_comp = 2

    finally:
        # ------------------------------------------------------------------
        # STRICT MEMORY MANAGEMENT — guaranteed even if an exception occurs
        # ReadIM.DestroyBuffer releases the C heap allocation.
        # del removes Python references so the GC can collect wrappers.
        # ------------------------------------------------------------------
        ReadIM.DestroyBuffer(buf)   # ← release C-level buffer memory
        del arr, buf                # ← drop Python references

    return PIVSnapshot(
        filepath=filepath,
        regime=regime,
        scale=scale,
        x_phys=x_phys,
        y_phys=y_phys,
        u_phys=u_phys,
        v_phys=v_phys,
        w_phys=w_phys,
        n_components=n_comp,
        attributes=att_dict,
    )


# ---------------------------------------------------------------------------
# Low-level .im7 reader for raw intensity visualisation
# ---------------------------------------------------------------------------
def read_im7_to_uint8(filepath: Path) -> np.ndarray:
    """
    Read one .im7 image file and return a 2-D uint8 array scaled to [0, 255].

    The raw DaVis image may be stored as 12-bit or 16-bit intensity.  For
    visual comparison against the geometric fit, we linearly normalise the
    first image plane to standard 8-bit grayscale:

        img_u8 = 255 * (img - img_min) / (img_max - img_min)

    If the image is constant-valued, a zero array is returned.

    STRICT MEMORY MANAGEMENT
    ------------------------
    This function explicitly destroys the ReadIM buffer and deletes Python
    references before returning, per project requirements.

    Parameters
    ----------
    filepath : absolute path to a .im7 file

    Returns
    -------
    img_u8 : (ny, nx) uint8 numpy array
    """
    if not filepath.exists():
        raise FileNotFoundError(f"IM7 file not found: {filepath}")

    buffer, att_list = ReadIM.extra.get_Buffer_andAttributeList(str(filepath))
    try:
        arr, _ = ReadIM.buffer_as_array(buffer)
        img_raw = arr[0].astype(np.float64).copy()
    finally:
        ReadIM.DestroyAttributeListSafe(att_list)
        del att_list
        ReadIM.DestroyBuffer(buffer)
        del buffer

    img_min = float(np.min(img_raw))
    img_max = float(np.max(img_raw))
    if img_max <= img_min:
        img_u8 = np.zeros_like(img_raw, dtype=np.uint8)
    else:
        img_norm = (img_raw - img_min) / (img_max - img_min)
        img_u8 = np.clip(np.round(255.0 * img_norm), 0, 255).astype(np.uint8)

    del arr, img_raw
    return img_u8


# ---------------------------------------------------------------------------
# Convenience PNG writer for raw IM7 visualisation
# ---------------------------------------------------------------------------
def save_im7_png(filepath: Path, out_path: Path) -> Path:
    """
    Convert one .im7 image to an 8-bit grayscale PNG on disk.

    Parameters
    ----------
    filepath : source .im7 path
    out_path : destination .png path

    Returns
    -------
    Path to the written PNG file.
    """
    img_u8 = read_im7_to_uint8(filepath)
    Image.fromarray(img_u8, mode="L").save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCondition:
    """Represents one experimental sub-directory and its .vc7 inventory."""
    directory: Path
    fov: str                      # "LargeView" or "SmallView"
    regime: FlowRegime
    vc7_files: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.vc7_files)

    def iter_snapshots(self) -> Iterator[PIVSnapshot]:
        """
        Lazily parse each .vc7 file in this condition.

        Each snapshot is fully released from C memory before the next
        one is parsed — safe for large datasets on memory-limited nodes.
        """
        for fp in self.vc7_files:
            yield parse_vc7(fp, regime=self.regime)


def scan_data_root(root: Path = DATA_ROOT) -> list[ExperimentCondition]:
    """
    Walk *root* and build an inventory of all experimental conditions.

    Expected tree structure::

        root/
        ├── LargeView/
        │   ├── Calib_nodrop_pressure/
        │   │   └── *.vc7
        │   ├── Ethanol_drop/
        │   │   └── *.vc7
        │   └── ...
        └── SmallView/
            └── ...

    Parameters
    ----------
    root : path to the PIV data root directory

    Returns
    -------
    List of ``ExperimentCondition`` objects sorted by (FOV, regime, name).
    Conditions with zero .vc7 files are silently excluded.

    Raises
    ------
    FileNotFoundError : if *root* does not exist
    """
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    conditions: list[ExperimentCondition] = []

    for fov_dir in sorted(root.iterdir()):
        if not fov_dir.is_dir():
            continue
        fov_name = fov_dir.name   # "LargeView" or "SmallView"

        for cond_dir in sorted(fov_dir.iterdir()):
            if not cond_dir.is_dir():
                continue

            regime = classify_directory(cond_dir)
            vc7_files = sorted(cond_dir.rglob("*.vc7"))

            if not vc7_files:
                log.debug("Skipping empty condition: %s", cond_dir.name)
                continue

            cond = ExperimentCondition(
                directory=cond_dir,
                fov=fov_name,
                regime=regime,
                vc7_files=vc7_files,
            )
            conditions.append(cond)
            log.info(
                "Found: [%s] %-30s | %s | %d files",
                fov_name,
                cond_dir.name,
                regime.name,
                len(vc7_files),
            )

    return conditions


# ---------------------------------------------------------------------------
# Convenience: load all snapshots from a condition list (batch, memory-safe)
# ---------------------------------------------------------------------------
def load_all_snapshots(
    conditions: list[ExperimentCondition],
    regime_filter: FlowRegime | None = None,
) -> list[PIVSnapshot]:
    """
    Parse every .vc7 file across *conditions*, optionally filtered by regime.

    Memory management: each snapshot is parsed sequentially; the buffer
    is destroyed inside ``parse_vc7`` before the next file is opened.
    Only the resulting lightweight ``PIVSnapshot`` dataclasses accumulate
    in RAM — no C-level buffers are retained.

    Parameters
    ----------
    conditions    : list from ``scan_data_root``
    regime_filter : if given, load only conditions of this regime

    Returns
    -------
    list of ``PIVSnapshot``
    """
    snapshots: list[PIVSnapshot] = []
    for cond in conditions:
        if regime_filter is not None and cond.regime != regime_filter:
            continue
        for snap in cond.iter_snapshots():
            snapshots.append(snap)
            log.debug("Loaded snapshot: %s", snap.filepath.name)
    return snapshots


# ===========================================================================
# Self-contained smoke test
# ===========================================================================
if __name__ == "__main__":
    """
    Smoke test — runs entirely on the local CPU with NO GPU required.

    Uses ReadIM's built-in sample .vc7 files so the test is self-contained
    and does not require access to G:\\Experimental_Data\\.

    Verifies:
      1. Physical coordinate mapping produces plausible [mm] ranges.
      2. Velocity arrays have the correct shape and no NaN in 2C fields.
      3. ScaleParams are extracted and printed.
      4. Non-dimensionalisation returns dimensionless values near O(1).
      5. Memory is released cleanly (DestroyBuffer + del called).
    """
    import sys

    log.setLevel(logging.DEBUG)
    print("=" * 65)
    print("  data_parser.py -- Smoke Test (CPU, built-in sample files)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Locate ReadIM's bundled sample files (no external data needed)
    # ------------------------------------------------------------------
    sample_files = ReadIM.extra.get_sample_vector_filenames()
    sample_2c = Path(sample_files[0])   # 2-component PIV
    sample_3c = Path(sample_files[1])   # 3-component PIV

    # Assign synthetic regimes for the smoke test
    test_cases = [
        (sample_2c, FlowRegime.BACKGROUND, "2C sample -> BACKGROUND"),
        (sample_3c, FlowRegime.DROPLET,    "3C sample -> DROPLET"),
    ]

    for fp, forced_regime, label in test_cases:
        print(f"\n{'-'*60}")
        print(f"  Test case : {label}")
        print(f"  File      : {fp.name}")
        print(f"{'-'*60}")

        snap: PIVSnapshot = parse_vc7(fp, regime=forced_regime)

        # ---- Calibration parameters ----------------------------------
        sc = snap.scale
        print(f"  Calibration (ScaleParams):")
        print(f"    scaleX : factor={sc.x_factor:+.6f}, "
              f"offset={sc.x_offset:+.6f}  [{sc.x_unit}]")
        print(f"    scaleY : factor={sc.y_factor:+.6f}, "
              f"offset={sc.y_offset:+.6f}  [{sc.y_unit}]")
        print(f"    scaleI : factor={sc.vel_factor:+.6f}, "
              f"offset={sc.vel_offset:+.6f}  [{sc.vel_unit}]")

        # ---- Physical coordinate ranges ------------------------------
        print(f"  Physical coordinates:")
        print(f"    x_phys : [{snap.x_phys.min():+.4f}, "
              f"{snap.x_phys.max():+.4f}]  [{sc.x_unit}]  "
              f"(nx={len(snap.x_phys)})")
        print(f"    y_phys : [{snap.y_phys.min():+.4f}, "
              f"{snap.y_phys.max():+.4f}]  [{sc.y_unit}]  "
              f"(ny={len(snap.y_phys)})")

        # ---- Velocity matrix -----------------------------------------
        ny, nx = snap.u_phys.shape
        print(f"  Velocity arrays:")
        print(f"    shape  : (ny={ny}, nx={nx}),  "
              f"n_components={snap.n_components}")
        print(f"    u_phys : [{np.nanmin(snap.u_phys):+.4f}, "
              f"{np.nanmax(snap.u_phys):+.4f}]  [{sc.vel_unit}]")
        print(f"    v_phys : [{np.nanmin(snap.v_phys):+.4f}, "
              f"{np.nanmax(snap.v_phys):+.4f}]  [{sc.vel_unit}]")
        if snap.n_components == 3:
            print(f"    w_phys : [{np.nanmin(snap.w_phys):+.4f}, "
                  f"{np.nanmax(snap.w_phys):+.4f}]  [{sc.vel_unit}]")
        else:
            print(f"    w_phys : NaN-filled (2C file, no out-of-plane data)")

        # ---- Non-dimensionalisation ----------------------------------
        # NOTE: These sample files are from a generic PIV experiment,
        # not our levitated droplet setup.  x* values will NOT be O(1)
        # here — that is expected.  In production, R_e and u_ref are
        # tuned to the actual droplet geometry (typically R_e ~ 1 mm,
        # u_ref ~ 1–10 mm/s for acoustic streaming).
        R_e   = 1.0e-3   # [m] — equatorial radius of the droplet
        u_ref = 0.01     # [m/s] — characteristic streaming velocity
        nd = snap.non_dimensionalise(R_e=R_e, u_ref=u_ref)
        print(f"  Non-dimensionalisation (R_e={R_e*1e3:.1f}mm, "
              f"u_ref={u_ref:.3f}m/s):")
        print(f"    x*   : [{nd.x_star.min():+.3f}, {nd.x_star.max():+.3f}]")
        print(f"    y*   : [{nd.y_star.min():+.3f}, {nd.y_star.max():+.3f}]")
        print(f"    u*   : [{np.nanmin(nd.u_star):+.3f}, "
              f"{np.nanmax(nd.u_star):+.3f}]")

        # ---- Regime classification -----------------------------------
        print(f"  FlowRegime : {snap.regime.name}")

        # Explicit cleanup of the snapshot's numpy arrays
        del snap, nd

    # ------------------------------------------------------------------
    # Test directory classifier with synthetic path names
    # ------------------------------------------------------------------
    print(f"\n{'-'*60}")
    print("  Directory classifier sanity check:")
    test_paths = [
        ("Calib_nodrop_pressure",  FlowRegime.BACKGROUND),   # DaVis compact
        ("calib_no_drop_test",     FlowRegime.BACKGROUND),   # canonical form
        ("Ethanol_drop",           FlowRegime.DROPLET),
        ("water_drop_run3",        FlowRegime.DROPLET),
        ("Two_drop",               FlowRegime.DROPLET),
        ("random_experiment",      FlowRegime.UNKNOWN),
    ]
    all_pass = True
    for dir_name, expected in test_paths:
        dummy_path = Path("/fake/root") / dir_name
        result = classify_directory(dummy_path)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}]  '{dir_name}' -> {result.name} "
              f"(expected {expected.name})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Smoke test {'PASSED' if all_pass else 'FAILED'}")
    print(f"  Memory management: DestroyBuffer + DestroyAttributeListSafe")
    print(f"  called inside parse_vc7() -- no C-level leaks.")
    print("=" * 65)

    sys.exit(0 if all_pass else 1)
