# Bi-Level Meta-Learning: Integration into gd_optimic

## Overview

**S(θ, x)** is a neural network that learns to predict IC updates during the main optimization loop.

Instead of:
```
x^(k+1) = x^(k) - lr·∇_x J
```

We do:
```
Main loop:
  x^(k+1) = x^(k) + S(θ, x^(k))  # Neural network predicts full update
  
Meta loop (same iteration):
  Compute: L_meta = w₁·L_perf + w₂·L_smooth + λ||θ||²
  Update: θ ← θ - α_meta·∇_θ L_meta
```

---

## Network Architecture

S(θ, x) uses a **UNet with skip connections** for multi-scale feature extraction:

- **Encoder**: Progressive downsampling (maxpool) with DoubleConv blocks
  - [C, H, W] → [32, H, W] → [64, H/2, W/2] → [128, H/4, W/4] → [256, H/8, W/8]
- **Bottleneck**: Deepest representation
  - [512, H/8, W/8]
- **Decoder**: Progressive upsampling with skip connections from encoder
  - [512, H/8, W/8] + [256, H/8, W/8] → [256, H/4, W/4]
  - → [128, H/2, W/2] → [64, H, W]
- **Output**: 1×1 conv to [C, H, W]

### Why UNet?

1. **Spatial structure preservation**: Skip connections preserve fine-grained details needed for IC updates
2. **Multi-scale learning**: Encoder learns coarse patterns; decoder refines at each scale
3. **Better gradient flow**: Skip connections improve backpropagation through meta-loss
4. **Expressive capacity**: ~3.5M parameters (vs. ~280K in simple CNN) to learn complex update rules

### Configuration

- `base_channels`: 32 (encoder depth: b → 2b → 4b → 8b)
- `num_groups`: 8 (GroupNorm groups, stable with small batch sizes)
- `output_scale`: 0.01 (soft damping via tanh prevents explosive updates)
- `temporal_steps`: 1 (single IC) or >1 (trajectory sequences)

---

### (1) Performance Loss L_perf
```
L_perf = -(J_{k-1} - J_k)
```
Encourages S to predict updates that **reduce J**. Negative improvement = bad (high loss).

### (2) Smoothness Loss L_smooth
```
L_smooth = ||∇_x J_k||²
```
Encourages S to move IC toward **flat regions** (small gradient norm).

### (3) Regularization L_reg
```
L_reg = λ·||θ||²
```
Prevents S from becoming too large; keeps updates reasonable.

---

## How It Works

### Per Iteration k:

1. **IC Update (Inner Loop)**
   ```python
   update = S(θ, x_k)          # Network predicts update
   x_{k+1} = x_k + update      # Apply (no learning rate!)
   y_pred = model(x_{k+1})     # Rollout forecast
   J_k = loss(y_pred, y_obs)   # Compute loss
   ```

2. **Meta-Learner Adaptation (Outer Loop)**
   ```python
   grad_J_k = ∇_x J_k          # Gradient at new IC
   improvement = J_{k-1} - J_k
   
   L_perf = -improvement
   L_smooth = ||grad_J_k||²
   L_reg = λ·||θ||²
   L_meta = w₁·L_perf + w₂·L_smooth + L_reg
   
   θ ← θ - α_meta·∇_θ L_meta    # Update S's parameters
   ```

3. **Repeat** for k = 1, 2, ..., max_iter

---

## Configuration

**File**: `configs/bilevel_meta_learning.yaml`

Key settings:
- `w_perf`: Weight on performance term (typically 1.0)
- `w_smooth`: Weight on smoothness term (typically 0.1)
- `lambda_reg`: Regularization weight (typically 1e-5)
- `meta_lr`: Learning rate for S's parameters (typically 1e-3)

---

## Integration with gd_optimic

### Usage Example

```python
from gd_optimic.meta_learner import NetworkS, NetworkSConfig, BiLevelICOptimizer

# Create S network (UNet with skip connections)
cfg_s = NetworkSConfig(
    input_channels=5,
    output_channels=5,
    spatial_height=330,
    spatial_width=360,
    base_channels=32,      # UNet encoder/decoder depth
    num_groups=8,          # GroupNorm groups
    temporal_steps=1,      # Single IC (not trajectory)
    output_scale=0.01,     # Prevent explosive updates
)
network_s = NetworkS(cfg_s)
print(f"S network parameters: {network_s.get_parameter_count():,}")

# Create bi-level optimizer
meta_optimizer = BiLevelICOptimizer(
    network_s,
    ocean_mask=mask,
    device="cuda",
    config=cfg.meta_learner,
)

# Main optimization loop
for k in range(max_iterations):
    # (1) Predict update using S
    update = meta_optimizer.predict_update(x_k)
    x_k_plus_1 = x_k + update
    
    # (2) Compute loss at new IC
    y_pred = model(x_k_plus_1)
    J_k = loss(y_pred, y_obs)
    
    # (3) Compute gradient at new IC
    grad_J_k = torch.autograd.grad(J_k, x_k_plus_1)[0]
    
    # (4) Update S's parameters (meta-loop)
    meta_optimizer.step(
        x_current=x_k_plus_1,
        loss_prev=J_prev,
        loss_current=J_k,
        gradient_current=grad_J_k,
    )
```

---

## Scientific Integrity (A1–A8)

- **A1**: glonet weights frozen ✅
- **A2**: Meta-loss computed only during assimilation window ✅
- **A3**: Baseline (pure gradient) paired with meta-augmented runs ✅
- **A4**: Ocean mask applied to S's updates ✅
- **A5**: Reproducible seeding for S initialization ✅
- **A6**: All runs logged, no cherry-picking ✅
- **A7**: Same initial IC, same loss function for fair comparison ✅
- **A8**: Limitations disclosed (e.g., S can fail if poorly initialized) ✅

---

## Key Differences from Pre-Training Approach

| Aspect | Pre-Training (Old ❌) | Bi-Level (New ✅) |
|---|---|---|
| When does S train? | Offline, after collecting trajectories | Online, during main loop |
| S's role | Approximate to pre-trained gradients | Learn optimization dynamics in-the-fly |
| Data needed | Thousands of trajectory files | None (learns from single IC experiment) |
| Meta-loss | Supervised MSE | Performance + smoothness + regularization |
| Flexibility | Fixed after training | Adapts to each IC problem |

---

## Expected Behavior

**Early iterations** (k=1-50):
- S learns to imitate gradient descent
- L_meta large (S learning signals)
- J decreases at similar rate to gradient descent

**Mid iterations** (k=50-200):
- S begins correcting/enhancing gradients
- L_meta stabilizes
- J might decrease faster than pure gradient (if S learns good patterns)

**Late iterations** (k=200+):
- S converges to good update rules
- L_meta ≈ 0
- J → minima (hopefully faster than baseline)

---

## Troubleshooting

**Q: S updates are NaN**
A: Check `output_scale` (too large). Reduce from 0.01 to 0.001. Also check `meta_lr` (too high).

**Q: L_meta not decreasing**
A: Meta-loss might be conflicting (performance vs. smoothness). Tune `w_perf` and `w_smooth`.

**Q: IC diverges (x → ∞)**
A: S predictions are explosive. Increase regularization `lambda_reg`, or reduce `output_scale`.

**Q: No iteration speedup vs. gradient descent**
A: S hasn't learned useful patterns yet. Check:
  - Is `meta_lr` high enough?
  - Are `w_perf` and `w_smooth` balanced?
  - Does S have enough capacity (conv_filters)?

---

**Status**: Phase B (Blueprint) → Ready for Phase P (Implementation)  
**Protocol Version**: bilevel_v1  
**Last Updated**: 2026-07-28
