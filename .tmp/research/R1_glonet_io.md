# R1 — Glonet I/O Contract (research, 2026-05-22)

## Legacy `model/glonet/`

- Tensor shape: `(B, T, C, H, W)`, T=2 input timesteps. (`modelp2.py:50`)
- Variables: thetao, so, uo, vo (10 depth levels each) + zos surface; ~40+ channels. Normalization stats: `config/stdmean/L*/{thetao,so,uo,vo}_{mean,std}.npy` (`utility.py:15-51`).
- Forward = single-step `Glonet.forward(input_st_tensors)` (`modelp2.py:50-93`).
- Architecture: skip + Fourier-Transformer (FoTF) + Encoder→TeDev(Fourier)→Decoder (`modelp2.py:40-42`).
- **CRITICAL: gradient flow is BROKEN.** Explicit `.detach()` at `modelp2.py` lines 54, 55, 67, 68, 74, 87. Also explicit `gc.collect()` + `torch.cuda.empty_cache()` at 56, 64, 78, 84, 90 — autograd through this is impossible without modifying the architecture (violates Frozen-model invariant A1).
- State persistence: pure feedforward; no RNN hidden state.

## Newer `model/glonet3/glonet2_global_e239_model_package/`

- **Grid:** 1440 × 672 (0.25° global). Source: `config/config.global.rollout10.local.yaml`.
- **Vertical:** 20 ocean depths [0.494 m … 763.333 m]. Source: `metadata/layout.json:144-165`.
- **Input channels (96 total):**
  - Ocean 3D (4 vars × 20 depths = 80): thetao, so, uo, vo.
  - Ocean surface (5): zos, ui, vi (ice vel), ice_thickness, ice_fraction.
  - **Atmospheric forcing (5 × 2 timesteps = 10):** u10, v10, t2m, mslp, sp at t-1 and t.
  - Static (1): bathymetry.
- **Output channels (80):** ocean state only — same 4 vars × 20 depths + 5 surface vars. (`layout.json:186-271`)
- **Time:** input_steps=2 (frames at t-1, t); target_offset=1 (predict t+1).
- **Max rollout (during training):** 10 steps, with feedback noise std=0.01. Trained on [2,4,7,10]-step rollouts. → autograd through rollout was INTENDED in training.
- **Checkpoint:** `epoch_0239.pt` (~964 MB). Manifest in `manifest.json`.
- **Normalization:** internal per-channel stats — `stats/train_channel_stats.json` (96 input + 80 target).
- **Differentiability:** UNKNOWN — archive has no source code, only weights + metadata. Must verify empirically.
- **State persistence:** pure feedforward; x_{t+1} = f(x_{t-1}, x_t, forcing_{t-1}, forcing_t).

## Differences (summary)

| Aspect | Legacy `glonet/` | Newer `glonet3/` (e239) |
|---|---|---|
| Vertical levels | ~10 | 20 |
| Input channels | ~40 | 96 (ocean + forcing + static) |
| Grid | inferred 1/4° | explicit 1440×672 @ 0.25° |
| Forcing | implicit | explicit (5 vars × 2 timesteps) |
| Autograd | **BROKEN** (`.detach()` in forward) | **UNKNOWN** (no source) |
| Multi-step training | not shown | up to 10 steps |
| Normalization | external `.npy` | internal JSON stats |
| Code maturity | research prototype | production checkpoint |

## Implication for ML-4DVar

- **State vector x ∈ ℝ^{B × 80 × 672 × 1440}** — ocean state only (forcing is given, not optimized).
- **Reasonable δx₀ subsets:**
  - SST + SSH only (2 channels) — barotropic / surface signal.
  - Upper-ocean T/S (0.49 – 15.8 m → 8 levels each = 16 channels) — mesoscale eddy correction.
  - Full ocean state (80) — maximum freedom, highest cost.
- **Gradient-flow risks:**
  - Legacy `glonet/` is unsuitable — `.detach()` calls in forward.
  - `glonet3/e239` MUST be verified: load checkpoint, forward small batch, `.backward()` on scalar loss, inspect `x.grad` for N=2,4,7,10 rollout steps. If `None`/zero, the checkpoint contains hidden `no_grad`/`detach` and we'd need to rebuild a differentiable forward from configs.
- **Forcing dependency (new requirement):** every assimilation/forecast call needs atmospheric forcing (u10, v10, t2m, mslp, sp). Likely ERA5. Pre-computed init_states under `/Odyssey/public/glonet/glorys12_*_init_states/` may already bundle forcing — needs confirmation.

## Files referenced

- `/Odyssey/private/j25lee/bice/model/glonet/modelp2.py:50-93`
- `/Odyssey/private/j25lee/bice/model/glonet/utility.py:15-51`
- `/Odyssey/private/j25lee/bice/model/glonet3/glonet2_global_e239_model_package/metadata/layout.json`
- `/Odyssey/private/j25lee/bice/model/glonet3/glonet2_global_e239_model_package/config/config.global.rollout10.local.yaml`
- `/Odyssey/private/j25lee/bice/model/glonet3/glonet2_global_e239_model_package/manifest.json`
