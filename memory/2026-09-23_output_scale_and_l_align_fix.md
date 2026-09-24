# Session Summary — `output_scale` Role & `l_align` Scale Fix in `meta_one_task` (2026-09-23)

**Branch/worktree:** `meta-learner-one-task` (`.vibe/worktrees/meta-learner-one-task`)
**Builds on:** `ff709b9` "Fix bug — Hybrid input for inference mode — Dimension mismatch" (adds `BiLevelICOptimizer.one_task_update()`, shared by training `step()` and inference).
**Status:** `l_align` fix applied, **uncommitted**. CPU smoke test passed; not yet validated on a real GPU run (first attempt, job 54872, hit an OOM unrelated to this change — see bottom).

---

## 1. Why `output_scale` exists

`UNetMetaGrad2D.forward()` (`src/gd_optimic/meta_learner/network_s.py:259`) ends with:

```python
out = self.output_scale * torch.tanh(out / self.output_scale)
```

So the network output `s` is **softly bounded to (−output_scale, +output_scale)**.

### History
- **`c865c4f` (2026-08-11, "Implement meta learner")** — introduced with `output_scale = 0.01`.
  At that time the network output *was* the IC update in **raw physical units**, with no destandardization:
  ```python
  ic_update = self.predict_update(x_det)
  x_new = x_det - ic_update
  ```
  A randomly initialized UNet can output arbitrary magnitudes, so the tanh cap was a safety damper
  ("soft damping via tanh prevents explosive updates" — meta_learner README). It limited the physical step to ±0.01 per grid point.
- **`54f23ab` (2026-09-07, hybrid input mode)** — raised to **`output_scale: 0.1`** in `configs/optimize_ic_hybrid.yaml` and `configs/optimize_ic.yaml`
  ("increased to allow larger updates"). The code default and `configs/batch_parallel_optim.yaml` remain `0.01`.

### Role today (meta_one_task mode)
The network now takes standardized input and its output is **destandardized with gradient statistics**:

```
z         = (g − μ) / σ                  # standardized outer gradient (unit std)
s         = network_s(input, k)          # ∈ (−output_scale, +output_scale)
ic_update = s·σ + μ                      # raw gradient units → IC step: x_new = x − ic_update
```

So `output_scale` no longer caps the physical step at a fixed value — it caps it at **±output_scale·σ of the current gradient**.
It effectively acts as the **maximum learning rate on the standardized gradient** (0.1 in our configs; for reference the plain-GD baseline uses `learning_rate = 1e-2` on the raw gradient).

**Decision (user, 2026-09-23): keep `output_scale`.** Reasons: still protects against random-init blow-ups, sets the step size relative to the gradient, and existing checkpoints were trained with it. It is now the meta-learner's step-size knob.

---

## 2. The `l_align` bug (meta_one_task mode)

### Before (STEP 3 of `BiLevelICOptimizer.step()`, present since `df0f756` "hybrid input mode doubled destandardization fixed")

```python
if self.meta_one_task:
    predicted_gradient_standardized = ic_update       # = s·σ + μ  (RAW gradient units!)
l_align = (gradient_standardized - predicted_gradient_standardized).pow(2).mean()
```

The name `predicted_gradient_standardized` is used twice:
- inside `predict_update()` it really is the standardized network output `s`;
- inside `step()` it was assigned `ic_update`, which is `s` **after destandardization** (`s·σ + μ`).

So `l_align = mean((z − (s·σ + μ))²)` compared a **z-score** with a **raw-units** quantity. Consequences:
1. With tiny gradients (σ small), `ic_update ≈ 0` relative to `z` → `l_align ≈ mean(z²) ≈ 1`, nearly constant.
2. `∂l_align/∂θ` carries a factor σ → almost no alignment signal reaches the network; the ALIGN phase taught very little.
3. Hidden in logs: for non-`meta_general_training` modes `last_l_align` was hard-coded to `0.0` and `meta_steps/L_align` was not logged.

### Alternatives considered and rejected
- **Remove the destandardization (use `s` as the IC step):** ❌ — `ic_update` is also the IC step. Using `s` directly makes the step ±output_scale in raw physical units for every variable (SSH m, T °C, S psu, U/V m/s), independent of gradient magnitude; would likely blow up and invalidates checkpoints.
- **Compare in raw units, `mean((gradient_prev − ic_update)²)`:** ❌ — algebraically equals `σ_c² · (z − s)²` per channel: tiny magnitude (swamped by `l_reg`/`l_perf`), channels weighted by their raw gradient variance, and still limited by the tanh cap.
- **Compare `z` with plain `s`:** ❌ — `s` ∈ (−0.1, 0.1) while `z` has std ≈ 1 → target unreachable; network just saturates at ±output_scale.

### After (applied)

```python
if self.meta_one_task:
    # Compare in standardized space: network output s is bounded to
    # ±output_scale by tanh, so rescale it to match the unit-std target.
    predicted_gradient_standardized = ic_update_standardized / self.network_s.output_scale
    # Ocean points only (land output is masked to 0, land target is -mean/std)
    align_mask = self._ocean_mask_like(ic_update_standardized).expand_as(ic_update_standardized)
    l_align = ((gradient_standardized - predicted_gradient_standardized).pow(2) * align_mask).sum() \
              / align_mask.sum().clamp_min(1.0)
```

- Dividing by `output_scale` is **not a second standardization** — it undoes the fixed tanh range cap so `s` is comparable with the unit-std target.
- Effective ALIGN target: **`s ≈ output_scale · z`** ⇒ `ic_update ≈ output_scale·(g − μ) + μ`, i.e. ALIGN teaches roughly one gradient step with learning rate `output_scale`; PERF then refines it to reduce forecast loss.
- Ocean-only mean: on land `s = 0` (masked) but `z = −μ/σ ≠ 0`, which would only add a constant bias.
- **IC step is unchanged:** `x_new = x_det − ic_update`, `ic_update = s·σ + μ`. Inference unaffected.
- `meta_general_training` (original IC-input mode) branch unchanged.

### Supporting changes
- `one_task_update()` now returns **`(ic_update, ic_update_standardized, gradient_standardized)`**; `optimizer.py` inference loop unpacks three values.
- New helper `_ocean_mask_like(value)` broadcasts `self.ocean_mask` to 4D/5D; replaces the duplicated mask-broadcast block in `predict_update()`.
- Logging: `L_align` is now real in one-task mode — `meta_steps/L_align` + `meta_steps/phase` in TensorBoard, `diagnostics["L_align"]` (→ `meta/L_align` per outer iteration), and in the console line together with the phase name.

### Verification
- `py_compile` OK on both files.
- CPU smoke test (small 32×64 grid, fake forward model, `output_scale=0.1`), hybrid and gradient-only:
  helper output matches a direct network call exactly; `|s| ≤ 0.1`, `s = 0` on land; `step()` runs ALIGN→TRANS→PERF with finite `L_align ≈ 1.92` at random init (was always 0 before); frozen inference loop runs.

### Expected impact / to watch
- ALIGN phase now actually trains → training behaviour differs from runs before this fix; not directly comparable.
- `l_align` now ~1–2 at start → check its size vs `l_perf` in TensorBoard; `w_align` / `w_perf` may need rebalancing.
- Old checkpoints (e.g. `meta_iter80_2026-09-10_16-40-45`) still load and run in inference; they were effectively trained mostly by PERF.

---

## 3. Job 54872 (`meta_lAlign`) OOM — NOT caused by this change

First training run with the fix crashed with `torch.OutOfMemoryError` at `optimizer.py:471`
(`torch.autograd.grad(loss, x0_current)`, iteration 0 outer gradient through GloNet) — this is **before** `meta_learner.step()` runs, so none of the modified code executed.

- GPU header at launch: `memory.free = 35103 MiB`, `utilization = 100 %` (vs. 143167 MiB free / 0 % for the working inference job 54701).
- OOM message: three other processes (3033581, 3596047, 503814) held ~35 GB each (~105 GB); this job had ~33 GB.
- Node `sl-mee-br-215` again (same contention pattern as 2026-09-19, see `memory/2026-09-19_load_model_bugfix_and_oom_diagnosis.md`). Header `GPUs:` was empty → GPU likely not reserved via `--gres`.
- Extra memory from the new `l_align` is one [1,2,5,672,1440] float32 tensor (~40 MB) — negligible.

**Action:** resubmit on a free GPU, or reserve it (`#SBATCH --gres=gpu:1` / `--exclusive`) in the user's own submit script (user manages that script).
