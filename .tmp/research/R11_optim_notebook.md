# R11 — Baseline `optim.ipynb` extraction (research, 2026-06-11)

Source: `/Odyssey/private/j25lee/bice/optim.ipynb`. User-supplied baseline optimization code for glonet v1. Extracted via Explore agent (read-only).

## 1. Optimization algorithm

- **Custom hand-rolled SGD** — no `torch.optim.*`. Manual update rule (cell 42):
  ```python
  x0 = x0 - CONFIG['learning_rate'] * masked_grad
  ```
- **`lr = 0.1`**, **constant**, no scheduler.
- **`num_iterations = 1000`** (hard-coded, no convergence criterion).
- **Single inner loop**; no multi-resolution outer loop in the SGD step itself.
- Gradients via `torch.autograd.grad(create_graph=False)` — first-order only, no second-order.
- **Gradient post-processing** (notable): a **`'pooling'` gradient filter** with a **scheduled kernel size** `[8, 4, 2, 1, …]` — implicit multi-resolution smoothing of the gradient field; coarse early, fine late.

## 2. Rollout (β multi-step)

- **Already implemented** in cell 25's `forward()`.
- `observation_length = 7` → 7-step autoregressive chain.
- Gradients propagate from all 7 timesteps back to δx₀ in a single `.backward()`.
- glonet weights frozen; only `x0.requires_grad = True`.

## 3. Loss components

- **J_obs ONLY.** `J_b` (background) and `J_q` (model-error) are **NOT in the notebook.**
- Functional form (cell 42):
  ```python
  J_obs = Σ_t Σ_c  MSE( ŷ ⊙ obs_mask − target ⊙ obs_mask ) / obs_var
  ```
- **`obs_var = 1.0`** scalar (identity R).
- **H operator = channel masking.** SSH (ch 0) and SST (ch 1) are observed; channels 2–4 (S, U, V) are zeroed by `obs_mask`. No spatial interpolation — obs assumed already on glonet grid.
- **Per-channel loss weighting** — two modes:
  - Manual: `[3.0, 2.0, 1.0, 1.0, 1.0]` for [SSH, T, S, U, V].
  - Dynamic: weights ∝ per-channel loss magnitude.
- Per-timestep weighting commented out; all 7 timesteps weighted equally.

## 4. Dataset / data loading

- **Warm-start x₀ source:** GLORYS12 monthly chunks at 1/4°:
  - `/Odyssey/public/glonet/glorys12_2021-01-01_to_2021-12-31_init_states/combined_input_2021-01-01_to_2021-02-15.nc`
  - `…/combined_input_2021-02-16_to_2021-04-02.nc`
- **Target sequence:** "GLORYS12 source shifted forward 7 days" (extraction wording). **AMBIGUITY** — see §6.
- **SSH obs (NEW, resolves earlier blocker):**
  - `/Odyssey/public/altimetry_traces/2010_2023/gridded/sla_l3_all_2010_2023_0.25deg_convl4.nc` — pre-gridded along-track SLA at 0.25°. **No need to populate the empty `Alongtrack_SSH_2021-...` directory.**
- **SST obs:** `ODYSSEA_SST_2021-...` (gridded L3, lat=680, lon=1440 — confirmed earlier).
- **5-channel surface order — LOCKED (different from my earlier guess):**
  - ch 0: **SSH (zos)**
  - ch 1: **THETAO** (surface T)
  - ch 2: **SO** (surface S)
  - ch 3: **UO** (surface U)
  - ch 4: **VO** (surface V)
- **Depth (40-ch sub-models 2 & 3):** loaded but **not optimized, not in loss** — frozen reference state.
- **Land/ocean mask:** derived from NaN pattern in GLORYS12.

## 5. Initial condition setup

- **Warm start from GLORYS12** at `t₀` (not zero/random):
  ```python
  x0 = torch.from_numpy(input_data[:, 0:5, :, :].copy()).float().unsqueeze(0)
  x0.requires_grad = True
  ```
- **Gradient masking over land** — yes, applied after the pooling filter.
- **No bounds / positivity / smoothness constraints** beyond ocean masking and gradient pooling.

## 6. DEVIATIONS / open questions vs. binding decisions

| Binding (in `decisions.md`) | Notebook reality | Action |
|---|---|---|
| Loss = `J_b + J_obs + J_q` (weak-constraint 4D-Var) | `J_obs` only — no J_b, no J_q | **Reconcile**: relax to "J_obs first; J_b/J_q added in Phase-1.b" OR extend code now. **User decision needed.** |
| Strict obs/eval separation: CMEMS for fit, GLORYS12 for eval only | Loss `target` is described as "GLORYS12 shifted +7d" but `obs_mask` is derived from satellite obs — unclear whether the loss-target tensor at obs locations comes from (a) the satellite file or (b) GLORYS12 sampled at the mask points | **Clarify with user**: is `target` in the loss the satellite-obs field (compliant) or GLORYS12-with-obs-mask (NON-compliant)? CRITICAL for A2. |
| Obs variance R = realistic per-product | Scalar identity (`obs_var=1`) | Acceptable as Phase-1 bootstrap; revisit when J_b/J_q come in. |
| Constant-step optimizer | ✅ Vanilla SGD, lr=0.1, no scheduler | OK |
| β rollout | ✅ Already implemented | OK |
| Surface-only (5 ch) | ✅ Depth packs frozen, not in loss | OK |
| Frozen model | ✅ glonet weights read-only | OK |

## 7. Output / artifacts

- Saves: `{CONFIG['output_dir']}/optimized_initial_conditions.nc` with attrs `best_loss`, `best_iteration`, `learning_rate`, `num_iterations`.
- Visualization: plots every `plot_frequency=100` iterations — gradients, loss curves, RMSE evolution, predictions, kernel schedule.
- Output dir: `../outputs/forecast_exp/{input_text}_idx{sample_idx}_but{idiot_idx}_{timestamp}/` — **does NOT** follow the project's `{exp_id}_{date}` convention or live under `.tmp/outputs/`.

## 8. Hyperparameter summary

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | hand-rolled SGD | `x ← x − lr·∇x` |
| Learning rate | 0.1 | constant |
| Iterations | 1000 | no convergence test |
| Forecast window | 7 days | `observation_length=7` |
| Gradient filter | `'pooling'` | scheduled kernels `[8,4,2,1,…]` |
| Loss weighting | dynamic OR manual `[3,2,1,1,1]` | per [SSH,T,S,U,V] |
| Rollout steps | 7 | autoregressive |
| Loss type | obs-only (J_obs) | no J_b, no J_q |
| Obs covariance | identity (1.0) | scalar |
| Surface channels | 5 | order: SSH, T, S, U, V |
| Init | GLORYS12 warm start | not zero/random |
| Depth state | frozen, excluded from loss | – |
