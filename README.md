# PINN for PIV — 3D Acoustic Streaming Reconstruction

Physics-Informed Neural Network (PINN) for reconstructing the dense 3D flow field around an acoustically levitated droplet, assimilating sparse 2D PIV experimental data.

## Physics Background

Acoustic streaming generates a steady secondary flow around a levitated droplet. 2D PIV captures only planar velocity slices. This PINN reconstructs the full 3D divergence-free velocity field by:

1. Enforcing the **Navier-Stokes momentum equations** as a soft constraint
2. Assimilating **sparse 2D PIV snapshots** as data loss
3. Using a **vector potential formulation** (u = ∇×Ψ) to guarantee ∇·u = 0 by construction
4. Modelling the droplet as an **oblate spheroid** with SDF-based spatial loss weighting

## Project Structure

```
PINN_PIV/
├── src/
│   ├── data/data_parser.py        # VC7/IM7 parsing, physical calibration, non-dimensionalisation
│   ├── geometry/geometry.py       # Oblate spheroid SDF, surface normals, collocation samplers
│   ├── network/pinn_model.py      # StreamingPINN, curl operator, Navier-Stokes loss
│   ├── training/trainer.py        # PINNTrainer, gradient aggregation, SDF-weighted PDE loss
│   └── postprocess/exporter.py   # 3D grid inference, VTR export for ParaView
├── tests/                         # 121 pytest tests (all passing)
├── process_case.py                # Droplet detection + PIV → VTK pipeline
├── run_training.py                # Main training entry point
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
# PyTorch must be installed separately (use the version matching your CUDA)
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

## Quick Start

### Step 1 — Inspect a case and detect droplet geometry

```bash
python process_case.py --data_dir ./experimental_data/LargeView --case Ethanol_pressure4
```

Outputs to `<case_dir>/output/`: `raw_image.png`, `droplet_fitted.png`, `piv_slice.vtp`, `droplet_surface.vtp`.

### Step 2 — Train the PINN

```bash
python run_training.py \
  --case_dir ./experimental_data/LargeView/Ethanol_drop/Ethanol_pressure4 \
  --R_e 0.4 --R_p 0.3 \
  --u_ref 4.0 \
  --n_epochs 5000 --n_pde_pts 8000 \
  --n_data_pts 4000 --mini_batch 4000 \
  --hidden 128 --layers 6
```

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `--R_e`, `--R_p` | Droplet equatorial / polar radius [mm] |
| `--u_ref` | Reference velocity for non-dimensionalisation [m/s] — use 95th percentile of \|u\| |
| `--n_pde_pts` | PDE collocation points per epoch (increase for stronger physics constraint) |
| `--mini_batch` | Points per forward pass — increase until GPU-Util > 80% |

### Step 3 — View results in ParaView

Download `<case_dir>/output/flow_field.vtr` and open in [ParaView](https://www.paraview.org/):
- `Filters → Glyph` for velocity vectors
- `Filters → StreamTracer` for streamlines

### Batch processing

```bash
# Process all sub-cases sequentially
python process_case.py --data_dir ./experimental_data/SmallView
```

## Data Layout

```
experimental_data/
├── LargeView/
│   └── Ethanol_drop/
│       └── Ethanol_pressure4/
│           └── PIV_MP(…)/        ← .vc7 vector fields here
│               └── B00001.vc7
└── SmallView/
    ├── Ethanol_pressure3.5/
    ├── Ethanol_pressure4.5/
    └── Water_pressure6/
```

- `.vc7` — DaVis processed vector fields (inside `PIV_MP*` sub-directories)
- `.im7` — raw images (directly in case root)

## Architecture

### Vector Potential Formulation

The network predicts Ψ = (ψ_x, ψ_y, ψ_z) and pressure p. Velocity is derived as:

```
u = ∇×Ψ = (∂ψ_z/∂y − ∂ψ_y/∂z,  ∂ψ_x/∂z − ∂ψ_z/∂x,  ∂ψ_y/∂x − ∂ψ_x/∂y)
```

This guarantees ∇·u = 0 identically, without any penalty term.

### Loss Function

```
L = λ_data · MSE(u_pred(x_PIV), u_PIV)  +  λ_pde · mean(w(φ) · ‖NS_residual‖²)

w(φ) = exp(−φ² / 2σ²)   # SDF weight, peaks at droplet surface
```

### Gradient Aggregation

For large PIV datasets (> 10M points), gradients are accumulated over mini-batches before each `optimizer.step()`, making the effective batch size arbitrarily large without OOM.

## Tests

```bash
python -m pytest tests/ -v   # 121 tests, all passing
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (install separately with CUDA support)
- ReadIM (LaVision) — for `.vc7` / `.im7` parsing
- See `requirements.txt` for other dependencies

## Hardware

- **Local development**: CPU only (all tests pass)
- **Training**: CUDA GPU recommended (tested on RTX 4090, 24 GB VRAM)
