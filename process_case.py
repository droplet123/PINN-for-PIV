"""
process_case.py
===============
End-to-end pipeline: PIV data parsing + droplet detection + VTK export.

Usage
-----
    python process_case.py <case_directory> [--R_e MM] [--R_p MM] [--u_ref M_S]

Example
-------
    python process_case.py "./experimental_data/LargeView/Ethanol_drop/Ethanol_pressure4"

Outputs (written to <case_directory>/output/)
---------------------------------------------
    raw_image.png          — pure experimental image extracted from .im7
    droplet_fitted.png     — raw image + VC7 mask boundary (dotted) +
                             fitted oblate boundary (solid) + R_e / R_p labels
    piv_slice.vtp          — 2-D PIV point cloud with u, v, w vectors (ParaView)
    droplet_surface.vtp    — 3-D oblate spheroid surface mesh (ParaView)

Physics-based detection constraints
-------------------------------------
    1. VC7 spatial prior  — DaVis masks the droplet interior with zero/NaN
       velocity; the largest such region defines the search ROI.
    2. Morphological glare filling — large closing kernel (25×25) consolidates
       fragmented laser specular highlights into one solid blob.
    3. Oblate constraint  — R_e >= R_p enforced; prolate fits are rejected or
       axes are swapped to the physically correct orientation.

Dependencies
------------
    numpy, matplotlib, scikit-image, opencv-python, pyvista, ReadIM, torch

Author : (project)
Date   : 2026-03-21
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2  # opencv-python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyvista as pv
import ReadIM
from scipy.ndimage import binary_fill_holes
from skimage import measure

# Project modules
from src.data.data_parser import (
    FlowRegime,
    classify_directory,
    parse_vc7,
    PIVSnapshot,
    read_im7_to_uint8,
)
from src.geometry.geometry import OblateSpheroid

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("process_case")


# ===========================================================================
# Step 1a — VC7 spatial prior: extract zero/NaN mask (Constraint 1)
# ===========================================================================

def _vc7_droplet_mask(snap: PIVSnapshot) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Derive a binary mask of the droplet interior from a PIV snapshot.

    DaVis sets velocity to exactly 0.0 (or NaN) inside the droplet because
    no valid cross-correlation can be computed there.  The largest contiguous
    region of zero/NaN vectors is the droplet mask.

    Returns
    -------
    mask_vc7 : (ny_vc7, nx_vc7) bool array — True where droplet is
    roi      : (row_min, col_min, row_max, col_max) bounding box in VC7
               grid pixels, or None if no masked region found
    """
    # A vector is "masked" if both u and v are zero or NaN
    zero_u = (snap.u_phys == 0.0) | np.isnan(snap.u_phys)
    zero_v = (snap.v_phys == 0.0) | np.isnan(snap.v_phys)
    raw_mask = zero_u & zero_v   # (ny_vc7, nx_vc7)

    # Keep only the largest connected component (the droplet, not noise)
    labeled = measure.label(raw_mask)
    props = measure.regionprops(labeled)
    if not props:
        return raw_mask, None

    best = max(props, key=lambda p: p.area)
    droplet_mask = labeled == best.label

    r_min, c_min, r_max, c_max = best.bbox   # (row_min, col_min, row_max, col_max)
    log.info(
        "VC7 droplet mask: area=%d vectors, bbox rows=[%d,%d] cols=[%d,%d]",
        best.area, r_min, r_max, c_min, c_max,
    )
    return droplet_mask, (r_min, c_min, r_max, c_max)


# ===========================================================================
# Step 1b — Physics-constrained droplet detection (Constraints 1, 2, 3)
# ===========================================================================

def detect_droplet_ellipse(
    img: np.ndarray,
    vc7_roi_img: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, float, float] | None:
    """
    Detect the levitated droplet ellipse with three physics-based constraints.

    Constraint 1 — VC7 spatial prior
        If ``vc7_roi_img`` is provided (bounding box in image pixels mapped
        from the VC7 grid), candidate contours whose centroid falls outside
        this ROI are rejected.

    Constraint 2 — Morphological glare filling
        A 25×25 closing kernel consolidates fragmented laser specular
        highlights into a single solid blob before contour finding.

    Constraint 3 — Oblate spheroid law (R_e >= R_p)
        cv2.fitEllipse returns (centre, (width, height), angle).  The
        horizontal semi-axis is R_e and the vertical is R_p.  If the fit
        is prolate (height > width), the axes are swapped to enforce the
        physical constraint.  If the aspect ratio is still implausible
        (R_p/R_e > 0.99 or < 0.2) the candidate is skipped.

    Parameters
    ----------
    img         : (ny, nx) float64 or uint8 intensity array
    vc7_roi_img : (row_min, col_min, row_max, col_max) in image pixels,
                  derived from the VC7 mask bounding box scaled to the
                  image resolution.  Pass None to skip spatial filtering.

    Returns
    -------
    (row_c, col_c, R_e_px, R_p_px) — centre and semi-axes in image pixels,
    with R_e >= R_p (oblate constraint satisfied), or None on failure.
    """
    # --- Normalise to uint8 for OpenCV ---
    img_f = img.astype(np.float64)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max <= img_min:
        log.warning("detect_droplet_ellipse: constant image — aborting.")
        return None
    img_u8 = np.clip(255.0 * (img_f - img_min) / (img_max - img_min), 0, 255).astype(np.uint8)

    # Constraint 2: Gaussian blur → Otsu threshold → 25×25 morphological closing
    # The large kernel fills dark gaps between bright laser glare spots,
    # consolidating fragmented highlights into one solid blob.
    blurred = cv2.GaussianBlur(img_u8, (5, 5), sigmaX=2.0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filled = binary_fill_holes(closed > 0).astype(np.uint8) * 255

    contours_cv, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours_cv:
        log.warning("detect_droplet_ellipse: no contours found after closing.")
        return None
    contours_sorted = sorted(contours_cv, key=cv2.contourArea, reverse=True)

    for cnt in contours_sorted:
        if len(cnt) < 5:
            continue   # cv2.fitEllipse requires >= 5 points

        area = cv2.contourArea(cnt)
        if area < 25:   # reject tiny noise blobs
            continue

        # --- Constraint 1: VC7 spatial prior ---
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_cnt = M["m10"] / M["m00"]   # col (x in image)
        cy_cnt = M["m01"] / M["m00"]   # row (y in image)

        if vc7_roi_img is not None:
            r_min, c_min, r_max, c_max = vc7_roi_img
            # Add 20% margin around the VC7 ROI to tolerate scale mismatch
            margin_r = 0.2 * (r_max - r_min)
            margin_c = 0.2 * (c_max - c_min)
            if not (
                r_min - margin_r <= cy_cnt <= r_max + margin_r and
                c_min - margin_c <= cx_cnt <= c_max + margin_c
            ):
                log.debug(
                    "Contour centroid (%.1f, %.1f) outside VC7 ROI — skipping.",
                    cx_cnt, cy_cnt,
                )
                continue

        # --- Fit ellipse ---
        _, (ew, eh), angle = cv2.fitEllipse(cnt)
        # cv2 returns full diameters; convert to semi-axes
        half_w = ew / 2.0   # semi-axis along the ellipse's major axis
        half_h = eh / 2.0   # semi-axis along the ellipse's minor axis

        # Map to physical R_e (horizontal) and R_p (vertical) using the
        # ellipse orientation angle (degrees, measured from x-axis)
        angle_rad = np.deg2rad(angle)
        # Project semi-axes onto image x and y directions
        R_horiz = abs(half_w * np.cos(angle_rad)) + abs(half_h * np.sin(angle_rad))
        R_vert  = abs(half_w * np.sin(angle_rad)) + abs(half_h * np.cos(angle_rad))

        # --- Constraint 3: oblate law R_e >= R_p ---
        # R_e is the equatorial (horizontal) radius; R_p is the polar (vertical)
        R_e_px = max(R_horiz, R_vert)
        R_p_px = min(R_horiz, R_vert)

        aspect = R_p_px / R_e_px
        if aspect > 0.99 or aspect < 0.15:
            log.debug(
                "Contour aspect ratio %.3f out of physical range [0.15, 0.99] — skipping.",
                aspect,
            )
            continue

        log.info(
            "Droplet detected: centre=(%.1f, %.1f) px, "
            "R_e=%.1f px, R_p=%.1f px, aspect=%.3f",
            cx_cnt, cy_cnt, R_e_px, R_p_px, aspect,
        )
        # Return as (row_c, col_c, R_e_px, R_p_px) — row = y, col = x
        return cy_cnt, cx_cnt, R_e_px, R_p_px

    log.warning("detect_droplet_ellipse: no physically valid ellipse found.")
    return None


# ===========================================================================
# Step 1c — Visualisation: raw image + VC7 mask boundary + fitted ellipse
# ===========================================================================

def export_droplet_fitted_png(
    img: np.ndarray,
    ellipse_params: tuple | None,
    scale_mm_per_px: float,
    out_path: Path,
    vc7_mask: np.ndarray | None = None,
) -> None:
    """
    Save the fitted-droplet diagnostic figure.

    Overlays
    --------
    - VC7 mask boundary  : dotted white line — shows where DaVis masked
      the velocity field (the ground-truth droplet footprint in vector space).
    - Fitted oblate ellipse : solid lime line — the R_e / R_p fit from
      the physics-constrained detection algorithm.
    - R_e axis line (yellow) and R_p axis line (orange) with text labels.

    Parameters
    ----------
    img               : (ny, nx) raw intensity array
    ellipse_params    : (row_c, col_c, R_e_px, R_p_px) or None
    scale_mm_per_px   : [mm/pixel] for physical annotation
    out_path          : destination .png path
    vc7_mask          : (ny_vc7, nx_vc7) bool mask from _vc7_droplet_mask()
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.imshow(img, cmap="gray", origin="upper", aspect="equal")
    ax.set_title("Droplet Geometric Fit  (VC7 mask = dotted, fit = solid)", fontsize=10)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    # --- VC7 mask boundary (dotted white) ---
    if vc7_mask is not None:
        ny_img, nx_img = img.shape[:2]
        mask_resized = cv2.resize(
            vc7_mask.astype(np.uint8),
            (nx_img, ny_img),
            interpolation=cv2.INTER_NEAREST,
        )
        vc7_contours = measure.find_contours(mask_resized.astype(float), level=0.5)
        for i, c in enumerate(vc7_contours):
            ax.plot(
                c[:, 1], c[:, 0],
                color="white", linewidth=1.2, linestyle=(0, (4, 3)),
                label="VC7 mask boundary" if i == 0 else None,
            )

    # --- Fitted oblate ellipse (solid lime) ---
    if ellipse_params is not None:
        row_c, col_c, R_e_px, R_p_px = ellipse_params

        ellipse_patch = mpatches.Ellipse(
            xy=(col_c, row_c),
            width=2 * R_e_px,
            height=2 * R_p_px,
            angle=0.0,
            edgecolor="lime",
            facecolor="none",
            linewidth=2.0,
            label="Fitted oblate boundary",
        )
        ax.add_patch(ellipse_patch)
        ax.plot(col_c, row_c, "+", color="red", markersize=10, label="Centre")

        R_e_mm = R_e_px * scale_mm_per_px
        R_p_mm = R_p_px * scale_mm_per_px

        # R_e axis (horizontal, yellow)
        ax.annotate(
            "", xy=(col_c + R_e_px, row_c), xytext=(col_c, row_c),
            arrowprops=dict(arrowstyle="-", color="yellow", lw=1.5),
        )
        ax.text(col_c + R_e_px * 0.5, row_c - 6, f"$R_e$={R_e_mm:.2f}mm",
                color="yellow", fontsize=7, ha="center")

        # R_p axis (vertical, orange)
        ax.annotate(
            "", xy=(col_c, row_c - R_p_px), xytext=(col_c, row_c),
            arrowprops=dict(arrowstyle="-", color="orange", lw=1.5),
        )
        ax.text(col_c + 5, row_c - R_p_px * 0.5, f"$R_p$={R_p_mm:.2f}mm",
                color="orange", fontsize=7, va="center")

        annotation = (
            f"$R_e$ = {R_e_mm:.3f} mm\n"
            f"$R_p$ = {R_p_mm:.3f} mm\n"
            f"Aspect = {R_p_mm / R_e_mm:.3f}"
        )
        ax.text(
            0.02, 0.97, annotation,
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.legend(loc="lower right", fontsize=7)
    else:
        ax.text(
            0.5, 0.5, "No droplet detected",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="red",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ===========================================================================
# Step 2 — PIV slice → VTK point cloud (.vtp)
# ===========================================================================

def export_piv_vtp(snap: PIVSnapshot, out_path: Path) -> None:
    """
    Convert a parsed PIVSnapshot into a pyvista PolyData point cloud and
    save as a .vtp file readable by ParaView.

    Coordinate convention
    ---------------------
    The PIV plane is the x-y plane (z = 0).  Physical coordinates are
    converted from mm → m to match the geometry module's SI units.

    Velocity vectors (u, v, w) are stored as a 3-component point array
    named "velocity" so ParaView's Glyph filter renders them directly.
    """
    # Build 2-D meshgrid of physical coordinates (mm → m)
    xx, yy = np.meshgrid(snap.x_phys * 1e-3, snap.y_phys * 1e-3)  # (ny, nx)
    zz = np.zeros_like(xx)

    # Flatten to (N, 3) point array
    points = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
    ])  # (N, 3)  [m]

    # Velocity components — w is NaN for 2C; replace with zeros for VTK
    u = snap.u_phys.ravel()
    v = snap.v_phys.ravel()
    w = np.where(np.isnan(snap.w_phys.ravel()), 0.0, snap.w_phys.ravel())
    velocity = np.column_stack([u, v, w])  # (N, 3)  [m/s]

    # Mask out zero-magnitude vectors (invalid PIV vectors flagged by DaVis)
    speed = np.linalg.norm(velocity, axis=1)
    valid = speed > 0.0
    points   = points[valid]
    velocity = velocity[valid]
    speed    = speed[valid]

    cloud = pv.PolyData(points)
    cloud["velocity"] = velocity          # vector field for Glyph filter
    cloud["speed"]    = speed             # scalar for colour mapping
    cloud["u"]        = velocity[:, 0]
    cloud["v"]        = velocity[:, 1]
    cloud["w"]        = velocity[:, 2]

    cloud.save(str(out_path))
    log.info(
        "Saved PIV slice: %s  (%d valid vectors)", out_path, valid.sum()
    )


# ===========================================================================
# Step 3 — Oblate spheroid surface mesh → VTK (.vtp)
# ===========================================================================

def export_spheroid_surface_vtp(
    geom: OblateSpheroid,
    out_path: Path,
    n_theta: int = 80,
    n_phi: int = 160,
) -> None:
    """
    Generate a structured surface mesh of the oblate spheroid and save as
    a .vtp file.

    Parametric mesh
    ---------------
        x = R_e * sin(theta) * cos(phi)
        y = R_e * sin(theta) * sin(phi)
        z = R_p * cos(theta)
    with theta in [0, pi], phi in [0, 2*pi].

    Parameters
    ----------
    geom    : OblateSpheroid descriptor
    out_path: destination .vtp path
    n_theta : number of latitude divisions
    n_phi   : number of longitude divisions
    """
    theta = np.linspace(0.0, np.pi,       n_theta)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")  # (n_theta, n_phi)

    cx, cy, cz = geom.centre
    x = cx + geom.R_e * np.sin(TH) * np.cos(PH)
    y = cy + geom.R_e * np.sin(TH) * np.sin(PH)
    z = cz + geom.R_p * np.cos(TH)

    # Flatten to points and build a PolyData surface via Delaunay on the sphere
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    cloud  = pv.PolyData(points)
    surf   = cloud.delaunay_3d().extract_surface(algorithm="dataset_surface")
    surf.save(str(out_path))
    log.info("Saved spheroid surface: %s", out_path)


# ===========================================================================
# Helpers
# ===========================================================================

def _find_first_vc7(case_dir: Path) -> Path | None:
    """Return the first .vc7 file inside a PIV_MP* sub-folder of case_dir.

    Per the data layout, processed vector fields live in a sub-directory
    whose name starts with ``PIV_MP``.  Falling back to a full recursive
    search only when no such sub-directory exists.
    """
    piv_mp_dirs = sorted(d for d in case_dir.iterdir()
                         if d.is_dir() and d.name.startswith("PIV_MP"))
    for d in piv_mp_dirs:
        files = sorted(d.glob("*.vc7"))
        if files:
            return files[0]
    # Fallback: recursive search (handles flat layouts)
    files = sorted(case_dir.rglob("*.vc7"))
    return files[0] if files else None


def _find_first_im7(case_dir: Path) -> Path | None:
    """Return the first .im7 file directly inside case_dir (non-recursive).

    Raw images are stored at the case root, not in sub-directories.
    """
    files = sorted(case_dir.glob("*.im7"))
    return files[0] if files else None


def _scale_mm_per_px(snap: PIVSnapshot) -> float:
    """Estimate mm/pixel from the x-axis calibration factor."""
    return abs(snap.scale.x_factor)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(
    case_dir: Path,
    R_e_mm: float | None = None,
    R_p_mm: float | None = None,
    u_ref: float = 0.01,
) -> None:
    """
    Full end-to-end pipeline for one experimental case directory.

    Parameters
    ----------
    case_dir : path to the experimental condition folder
    R_e_mm   : equatorial radius override [mm] (auto-detected if None)
    R_p_mm   : polar radius override [mm] (auto-detected if None)
    u_ref    : reference velocity for non-dimensionalisation [m/s]
    """
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    regime = classify_directory(case_dir)
    log.info("Case: %s  |  Regime: %s", case_dir.name, regime.name)

    # Create output directory
    out_dir = case_dir / "output"
    out_dir.mkdir(exist_ok=True)
    log.info("Output directory: %s", out_dir)

    is_background = regime == FlowRegime.BACKGROUND

    # ------------------------------------------------------------------
    # A. Droplet edge detection (skipped for background cases)
    # ------------------------------------------------------------------
    detected_R_e_mm: float | None = R_e_mm
    detected_R_p_mm: float | None = R_p_mm

    if is_background:
        log.info("Background case — skipping droplet edge detection.")
    else:
        im7_path = _find_first_im7(case_dir)
        if im7_path is None:
            log.warning("No .im7 file found — skipping edge detection.")
        else:
            log.info("Loading image: %s", im7_path.name)
            raw_img_u8 = read_im7_to_uint8(im7_path)
            plt.imsave(out_dir / "raw_image.png", raw_img_u8, cmap="gray", vmin=0, vmax=255)
            log.info("Saved: %s", out_dir / "raw_image.png")

            # Constraint 1: derive VC7 spatial prior before image detection
            vc7_path = _find_first_vc7(case_dir)
            vc7_mask: np.ndarray | None = None
            vc7_roi_img: tuple | None = None
            scale_mm_px = 1.0
            if vc7_path is not None:
                snap_tmp = parse_vc7(vc7_path, regime=regime)
                scale_mm_px = _scale_mm_per_px(snap_tmp)
                vc7_mask, vc7_roi_vc7 = _vc7_droplet_mask(snap_tmp)
                # Scale VC7 grid bbox → image pixel bbox
                if vc7_roi_vc7 is not None:
                    ny_img, nx_img = raw_img_u8.shape[:2]
                    ny_vc7, nx_vc7 = snap_tmp.u_phys.shape
                    r0, c0, r1, c1 = vc7_roi_vc7
                    vc7_roi_img = (
                        r0 * ny_img / ny_vc7,
                        c0 * nx_img / nx_vc7,
                        r1 * ny_img / ny_vc7,
                        c1 * nx_img / nx_vc7,
                    )
                del snap_tmp

            ellipse = detect_droplet_ellipse(raw_img_u8, vc7_roi_img=vc7_roi_img)

            if ellipse is not None and detected_R_e_mm is None:
                _, _, R_e_px, R_p_px = ellipse
                detected_R_e_mm = R_e_px * scale_mm_px
                detected_R_p_mm = R_p_px * scale_mm_px
                log.info(
                    "Auto-detected: R_e=%.3f mm, R_p=%.3f mm",
                    detected_R_e_mm, detected_R_p_mm,
                )

            export_droplet_fitted_png(
                img=raw_img_u8,
                ellipse_params=ellipse,
                scale_mm_per_px=scale_mm_px,
                out_path=out_dir / "droplet_fitted.png",
                vc7_mask=vc7_mask,
            )
            del raw_img_u8

    # ------------------------------------------------------------------
    # B. PIV data parsing and VTK export
    # ------------------------------------------------------------------
    vc7_path = _find_first_vc7(case_dir)
    if vc7_path is None:
        log.error("No .vc7 file found in %s — aborting.", case_dir)
        return

    log.info("Parsing PIV file: %s", vc7_path.name)
    snap = parse_vc7(vc7_path, regime=regime)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(f"  PIV Snapshot Summary: {vc7_path.name}")
    print("=" * 60)
    print(f"  Regime      : {snap.regime.name}")
    print(f"  Components  : {snap.n_components}C")
    print(f"  Grid        : ny={len(snap.y_phys)}, nx={len(snap.x_phys)}")
    print(f"  x_phys [mm] : [{snap.x_phys.min():.3f}, {snap.x_phys.max():.3f}]")
    print(f"  y_phys [mm] : [{snap.y_phys.min():.3f}, {snap.y_phys.max():.3f}]")
    print(f"  u [m/s]     : [{np.nanmin(snap.u_phys):.4f}, {np.nanmax(snap.u_phys):.4f}]")
    print(f"  v [m/s]     : [{np.nanmin(snap.v_phys):.4f}, {np.nanmax(snap.v_phys):.4f}]")
    if snap.n_components == 3:
        print(f"  w [m/s]     : [{np.nanmin(snap.w_phys):.4f}, {np.nanmax(snap.w_phys):.4f}]")
    print("=" * 60)

    # Non-dimensionalise (use detected or fallback R_e)
    R_e_m = (detected_R_e_mm or 1.0) * 1e-3
    nd = snap.non_dimensionalise(R_e=R_e_m, u_ref=u_ref)
    print(f"\n  Non-dim (R_e={R_e_m*1e3:.2f} mm, u_ref={u_ref:.4f} m/s):")
    print(f"  x*  : [{nd.x_star.min():.3f}, {nd.x_star.max():.3f}]")
    print(f"  u*  : [{np.nanmin(nd.u_star):.3f}, {np.nanmax(nd.u_star):.3f}]")

    # Export PIV slice as VTP
    export_piv_vtp(snap, out_dir / "piv_slice.vtp")

    # ------------------------------------------------------------------
    # C. Droplet surface mesh (skipped for background cases)
    # ------------------------------------------------------------------
    if not is_background:
        # Use detected radii, or fall back to sensible defaults
        R_e_m  = (detected_R_e_mm or 1.5) * 1e-3
        R_p_m  = (detected_R_p_mm or 1.0) * 1e-3

        # Clamp to physically plausible range (0.1 mm – 5 mm)
        R_e_m = float(np.clip(R_e_m, 1e-4, 5e-3))
        R_p_m = float(np.clip(R_p_m, 1e-4, R_e_m))   # R_p <= R_e for oblate

        geom = OblateSpheroid(R_e=R_e_m, R_p=R_p_m, centre=(0.0, 0.0, 0.0))
        log.info(
            "Spheroid geometry: R_e=%.3f mm, R_p=%.3f mm, aspect=%.3f",
            geom.R_e * 1e3, geom.R_p * 1e3, geom.aspect_ratio,
        )
        export_spheroid_surface_vtp(geom, out_dir / "droplet_surface.vtp")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  Outputs written to: {out_dir}")
    outputs = list(out_dir.glob("*"))
    for f in sorted(outputs):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<30s}  {size_kb:7.1f} KB")
    print()


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process PIV experimental cases: detect droplet, "
                    "parse vectors, export PNG + VTP files.\n\n"
                    "Two usage modes:\n"
                    "  1) Single case:  process_case.py <case_dir>\n"
                    "  2) Batch mode:   process_case.py --data_dir DIR --case NAME",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- positional (single-case legacy mode) ---
    p.add_argument(
        "case_dir", nargs="?", type=Path, default=None,
        help="Direct path to one experimental condition directory.",
    )
    # --- batch mode ---
    p.add_argument(
        "--data_dir", type=Path, default=None,
        help="Root data directory containing sub-case folders "
             "(e.g. ./experimental_data/LargeView).",
    )
    p.add_argument(
        "--case", type=str, default=None,
        help="Sub-case folder name inside --data_dir "
             "(e.g. Ethanol_drop).  Omit to process ALL sub-folders.",
    )
    # --- physics overrides ---
    p.add_argument(
        "--R_e", type=float, default=None, metavar="MM",
        help="Equatorial radius override [mm]. Auto-detected if omitted.",
    )
    p.add_argument(
        "--R_p", type=float, default=None, metavar="MM",
        help="Polar radius override [mm]. Auto-detected if omitted.",
    )
    p.add_argument(
        "--u_ref", type=float, default=0.01, metavar="M_S",
        help="Reference velocity for non-dimensionalisation [m/s]. "
             "Default: 0.01 m/s.",
    )
    return p.parse_args()


def _resolve_case_dirs(args: argparse.Namespace) -> list[Path]:
    """Return the list of case directories to process from parsed CLI args."""
    if args.case_dir is not None:
        # Legacy positional mode — single explicit path
        return [args.case_dir.resolve()]

    if args.data_dir is None:
        raise SystemExit(
            "error: provide either a positional case_dir or --data_dir."
        )

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"error: --data_dir does not exist: {data_dir}")

    if args.case:
        # Single named sub-case
        return [(data_dir / args.case).resolve()]

    # All immediate sub-directories (each is a case)
    dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"error: no sub-directories found in {data_dir}")
    return dirs


# ===========================================================================
# __main__ — self-contained smoke test using ReadIM sample files
# ===========================================================================

if __name__ == "__main__":
    # If arguments are provided, run the real pipeline.
    # Otherwise, run a self-contained smoke test using ReadIM sample files
    # so the script can be verified without access to the experimental data.
    if len(sys.argv) > 1:
        args = _parse_args()
        for case_dir in _resolve_case_dirs(args):
            run_pipeline(
                case_dir=case_dir,
                R_e_mm=args.R_e,
                R_p_mm=args.R_p,
                u_ref=args.u_ref,
            )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Smoke test — no external data required
    # ------------------------------------------------------------------
    import tempfile
    print("=" * 65)
    print("  process_case.py — Smoke Test (CPU, ReadIM sample files)")
    print("=" * 65)

    # Use ReadIM's bundled sample vc7 as a stand-in for a real case
    sample_files = ReadIM.extra.get_sample_vector_filenames()
    sample_vc7   = Path(sample_files[0])
    sample_dir   = sample_vc7.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- B. PIV parsing + VTP export ---
        print("\n[1] Parsing sample VC7 file...")
        snap = parse_vc7(sample_vc7, regime=FlowRegime.BACKGROUND)
        print(f"    Grid: ny={len(snap.y_phys)}, nx={len(snap.x_phys)}")
        print(f"    u range: [{np.nanmin(snap.u_phys):.4f}, "
              f"{np.nanmax(snap.u_phys):.4f}] m/s")

        piv_vtp = tmp_path / "piv_slice.vtp"
        export_piv_vtp(snap, piv_vtp)
        assert piv_vtp.exists() and piv_vtp.stat().st_size > 0
        print(f"    piv_slice.vtp: {piv_vtp.stat().st_size/1024:.1f} KB  OK")

        # --- C. Spheroid surface VTP ---
        print("\n[2] Generating spheroid surface mesh...")
        geom = OblateSpheroid(R_e=1.5e-3, R_p=1.0e-3)
        surf_vtp = tmp_path / "droplet_surface.vtp"
        export_spheroid_surface_vtp(geom, surf_vtp)
        assert surf_vtp.exists() and surf_vtp.stat().st_size > 0
        print(f"    droplet_surface.vtp: {surf_vtp.stat().st_size/1024:.1f} KB  OK")

        # --- A. Detection PNG (synthetic image — no real im7 needed) ---
        print("\n[3] Testing droplet detection PNG export (synthetic image)...")
        ny, nx = len(snap.y_phys), len(snap.x_phys)
        # Synthetic image: dark background + bright elliptical droplet
        yy, xx = np.ogrid[:ny, :nx]
        cy_px, cx_px = ny // 2, nx // 2
        synthetic_img = np.where(
            (xx - cx_px)**2 / (nx * 0.2)**2 + (yy - cy_px)**2 / (ny * 0.15)**2 < 1,
            0.8, 0.1
        ).astype(np.float64)
        # Add noise
        rng = np.random.default_rng(0)
        synthetic_img += rng.normal(0, 0.05, synthetic_img.shape)

        ellipse = detect_droplet_ellipse(synthetic_img)
        png_path = tmp_path / "droplet_fitted.png"
        export_droplet_fitted_png(
            img=synthetic_img,
            ellipse_params=ellipse,
            scale_mm_per_px=abs(snap.scale.x_factor),
            out_path=png_path,
        )
        assert png_path.exists() and png_path.stat().st_size > 0
        print(f"    droplet_fitted.png: {png_path.stat().st_size/1024:.1f} KB  OK")
        if ellipse:
            _, _, sm, sn = ellipse
            print(f"    Detected semi-axes: {sm:.1f} x {sn:.1f} px")

        # --- Non-dimensionalisation check ---
        print("\n[4] Non-dimensionalisation check...")
        nd = snap.non_dimensionalise(R_e=1.5e-3, u_ref=0.01)
        print(f"    x* range: [{nd.x_star.min():.3f}, {nd.x_star.max():.3f}]")
        print(f"    u* range: [{np.nanmin(nd.u_star):.3f}, {np.nanmax(nd.u_star):.3f}]")
        del snap, nd

    print("\n" + "=" * 65)
    print("  All smoke tests PASSED")
    print("  Outputs: piv_slice.vtp, droplet_surface.vtp, droplet_fitted.png")
    print("  Run with a real case:")
    print('    python process_case.py "G:/.../.../Ethanol_drop"')
    print("=" * 65)
    sys.exit(0)
