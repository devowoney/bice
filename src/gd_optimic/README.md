# gd_optimic: Gradient Descent Optimization for Initial Conditions

Production package for ML-4DVar ocean forecast optimization using frozen pretrained ocean model (glonet v1).

## Overview

This package implements **initial condition (IC) optimization** for ocean forecasting:
- **Phase 1 scope**: J_obs only (observation term)
- **Phase 1.b**: Will add J_b (background) + J_q (model error) terms
- **Frozen model**: No weight updates (scientific integrity constraint A1)
- **Object-oriented design**: Clean, modular, extensible
- **Hydra configuration**: Explicit parameter management

## Package Structure

```
src/gd_optimic/
├── __init__.py           # Package initialization
├── data.py               # Dataset loading + observation operators
├── loss.py               # Loss computation (J_obs)
├── gradient.py           # Gradient filtering + scheduled pooling
├── optimizer.py          # Main optimization loop + TensorBoard logging
├── metrics.py            # RMSE (global + basin-stratified) + PSD
├── utils.py              # Masks, normalizers, forward model
└── requirements.txt      # Dependencies

src/run_optimization.py   # Main entry script with Hydra

configs/optimize_ic.yaml  # Hydra configuration
```

## Modules

### 1. **data.py**: Dataset Loading + Observation Operators

**Classes:**
- `GlonetDataset`: Load GLORYS12 init states and observations
- `ObservationOperator`: Apply SSH/SST observation operators

**Follows R2 decisions:**
- SSH: Along-track altimetry → nearest-neighbor interpolation
- SST: Gridded L3 → direct pixel match
- QC: Light (trust CMEMS L3 pre-filtering)
- OSSE twin: GLORYS12 as truth

**Observation modes (A2-compliant):**
- `full`: GLORYS12 everywhere (idealized twin/ceiling check)
- `simulated`: GLORYS12 with realistic obs coverage (OSSE)
- `real`: Satellite SSH + SST (OSE)

### 2. **loss.py**: Loss Computation

**Classes:**
- `ObservationLoss`: MSE loss with observation masking

**Features:**
- Per-timestep MSE between forecast and observations
- Observation masking (only compute loss where obs exist)
- Dynamic or manual loss weighting across variables
- Per-variable loss tracking for TensorBoard

**Phase 1**: J_obs only  
**Phase 1.b**: Will add J_b + J_q

### 3. **gradient.py**: Gradient Filtering

**Classes:**
- `GradientFilter`: Apply spatial filtering to gradients
- `ScheduledPooling`: Multi-resolution gradient filtering (coarse-to-fine)

**Filtering strategies:**
- `none`: No filtering (raw gradients)
- `weight`: Weight by variance (emphasize high-variance regions)
- `delta`: Filter by IC perturbation magnitude
- `pooling`: Average pooling (spatial smoothing)
- `pooling+weight`: Pooling + variance weighting

**Scheduled pooling (multigrid):**
- Start with large kernel (global optimization)
- Gradually reduce to small kernel (local refinement)
- Avoids local minima early in optimization

### 4. **optimizer.py**: Main Optimization Loop

**Classes:**
- `ICOptimizer`: Main IC optimizer with TensorBoard logging

**Features:**
- Gradient descent on initial conditions
- Multi-step rollout with frozen model
- Scheduled gradient filtering
- TensorBoard logging following R8 hierarchy
- Automatic checkpointing

**TensorBoard hierarchy (R8):**
```
loss/J_obs/{total,ssh,sst,uo,vo}
metrics/rmse/{global,basin}/{var}
metrics/gradient/norm/{total,per_channel}
state/ic/{norm,update_magnitude}
```

### 5. **metrics.py**: Evaluation Metrics

**Classes:**
- `MetricsComputer`: Compute RMSE (global + basin-stratified)
- `PSDComputer`: Compute Power Spectral Density (Phase P)

**Follows R4 decisions:**
- Per-variable RMSE (SSH, T, S, U, V)
- Basin-stratified RMSE (Gulf Stream, high/low-variance)
- 2D directional spectrum by wavenumber (PSD)
- Band-energy ratio metric (mesoscale energy / total)

### 6. **utils.py**: Helper Classes

**Classes:**
- `MaskBuilder`: Create ocean/land masks and observation masks
- `ForwardModel`: Wrapper for multi-step rollout with gradient checkpointing
- `NormalizerLoader`: Load normalization/denormalization functions

**Features:**
- Ocean/land masking (zero land gradients)
- Observation masking (apply operators only where data exists)
- Memory-efficient autoregressive rollout

## Usage

### Basic Run

```bash
cd /Odyssey/private/j25lee/bice/src
python run_optimization.py
```

### Override Configuration

```bash
# Change observation mode
python run_optimization.py observations.mode=simulated

# Adjust optimization parameters
python run_optimization.py optimization.learning_rate=0.05 optimization.num_iterations=500

# Change experiment name
python run_optimization.py experiment.name=test_run
```

### View Results

```bash
# TensorBoard
tensorboard --logdir ./runs

# Outputs
ls ./outputs/<exp_id>/
```

## Configuration (Hydra)

All parameters managed in `configs/optimize_ic.yaml`:

- **Experiment**: name, description
- **Data**: file paths, sample selection
- **Model**: checkpoint paths, gradient checkpointing
- **Observations**: mode (full/simulated/real), operators
- **Loss**: weighting strategy, manual weights
- **Optimization**: learning rate, num iterations, gradient filter, scheduled pooling
- **Metrics**: RMSE (global/basin), PSD (Phase P)
- **Logging**: TensorBoard dir, log frequency, save frequency
- **Compute**: device (cuda/cpu)

## Scientific Integrity Constraints

**Enforced by code:**
- **A1**: Frozen-model invariant (no weight updates)
- **A2**: Strict obs/eval separation (observation mode enforcement)
- **A4**: Physical plausibility (land mask enforced)
- **A5**: Full reproducibility (random seeds, deterministic algorithms)

**Documented in config:**
- **A3**: Honest baselines (persistence always included)
- **A6**: No cherry-picking (all experiments logged)
- **A7**: OSE/OSSE separation (never mix real and simulated)
- **A8**: Disclosed limitations (assumptions documented)

## Development Notes

**Coding Standards (CS1 - BINDING):**
- **Visible**: Clear structure, proper spacing
- **Interpretable**: Human-readable first
- **Simple**: Straightforward logic, no clever tricks
- **Commented**: Explain WHY, not just WHAT

**Phase Progression:**
- **Phase 1**: J_obs only (current)
- **Phase 1.b**: Add J_b + J_q terms
- **Phase P**: PSD band-energy ratio, basin-stratified metrics refinement
- **Phase T**: Transfer to production, automation, maintenance

## Dependencies

See `requirements.txt` for full list.

**Key dependencies:**
- PyTorch >= 2.0
- xarray >= 2023.1
- xESMF >= 0.8 (grid regridding)
- TensorBoard >= 2.12
- Hydra >= 1.3

## Installation

```bash
cd /Odyssey/private/j25lee/bice/src/gd_optimic
pip install -r requirements.txt
```

## Contact

Project: bice (ML-4DVar for ocean forecasting)  
Phase: B (Blueprint) → R10(j) complete  
Protocol: U→B→P→T (LLMAIassistant_instruction.md)
