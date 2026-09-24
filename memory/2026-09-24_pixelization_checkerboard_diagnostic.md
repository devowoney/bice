# Session Summary — Pixelization (Checkerboard) Diagnostic for IC Updates (2026-09-23 → 2026-09-24)

**Branch/worktree:** `worktree-pixelization-diagnostic` (`.claude/worktrees/pixelization-diagnostic`), base local `main` `ff36793`
**Spec:** sub-agent task (`/Odyssey/private/j25lee/vibeSubAgent.md`)
**Status:** implemented + validated, **uncommitted** (user did not authorize commit)

## Problem

Pixelization = **checkerboard noise on the IC update** (optimized IC − initial IC). A pixelized update is not physical.
Gradient smoothing (scheduled pooling) is known to remove it. Goal of this sub-task: a **diagnostic that measures it correctly**.
(Out of scope: changing the cause — smoothing already works.)

## Why the previous metric was wrong

`_compute_finite_difference_ic_update` (removed) computed `sqrt(Σ|∇δ|² / Σδ²)` — the RMS wavenumber of the update.
It measures *any* fine-scale structure, so it cannot tell checkerboard noise from physical small-scale features:

| synthetic update | old FD score | new CF |
|---|---|---|
| physical small eddy (σ = 2 px ≈ 50 km) | 0.492 | 0.0009 |
| smooth eddy + 5 % checkerboard | 0.859 | 0.0917 |

The physical eddy scored *worse* than a 16×16 blocky field under the old metric. It also had no reference level.

## New metric — how it is computed

Function: `compute_checkerboard_fraction(field, ocean_mask)` in `src/gd_optimic/metrics.py`.
Input `field` = IC update `[C, H, W]` (last IC timestep), `ocean_mask` `[C, H, W]` (1 ocean / 0 land).

**Step 1 — checkerboard coefficient per 2×2 window.** For every overlapping window `[[a, b], [c, d]]`:

```
k = (a − b − c + d) / 2
```

`k` is the coefficient of the checkerboard pattern `[[+½, −½], [−½, +½]]`, one of the 4 orthonormal 2×2 Hadamard basis
patterns (mean, x-difference, y-difference, checkerboard). The window energy splits exactly into these 4 parts:
`a² + b² + c² + d² = mean² + xdiff² + ydiff² + k²`.
Only windows whose 4 pixels are all ocean are used (`w = 1`), so coastlines do not create fake checkerboard.

**Step 2 — amplitude** (`diag/checkerboard/amplitude/...`), per channel, in the variable's own units:

```
amplitude = sqrt( Σ w·k² / Σ w )          # RMS checkerboard coefficient over ocean windows
```

It says *how large* the checkerboard part is. It scales with the update: tiny updates → tiny amplitude
(e.g. 1e-7 SSH, 1e-4 U/V in `pixelDiagTest`). **Do not compare it with 0.038 / 0.063.**

**Step 3 — score = fraction** (`diag/checkerboard/fraction/...`): amplitude normalized by the window energy:

```
fraction = Σ w·k² / Σ w·(a² + b² + c² + d²)
         = amplitude² / E_w,     E_w = Σ w·(a²+b²+c²+d²) / Σ w   (mean energy of one ocean window)
```

This is the **share of the update's energy that is checkerboard**. Dimensionless, independent of update size.

| value | meaning |
|---|---|
| 0 | smooth update |
| ≈ 0.04 | physical (GLORYS12 day-to-day increments) |
| 0.25 | white noise (no spatial structure) |
| 1 | pure checkerboard |

**Shortcut (approximate):** each interior pixel sits in 4 windows, so `E_w ≈ 4·RMS(δ)²` and

```
fraction ≈ amplitude² / (4·RMS(δ)²)    ⇔    amplitude ≈ 2·RMS(δ)·sqrt(fraction)
```

Holds when the update is spread over open ocean (checked on `table1_refRun2/0`: SSH 0.0053 vs 0.0054, T 0.157 vs 0.159,
U 0.110 vs 0.112, V 0.099 vs 0.100). **Fails for S** (0.288 vs 0.178): the S update is concentrated at coasts, where
pixels drop out of the all-ocean windows. The exact relation `fraction = amplitude² / E_w` always holds.

**Total:** `fraction/total` = mean of the 5 channel fractions (channels have different units, so only the dimensionless
fractions are averaged). There is no amplitude total.

Checks on synthetic fields (all pass): smooth → 0.000, white noise → 0.2516, checkerboard → 1.000 (amplitude 2 for ±1),
5 % checkerboard on an eddy → 0.0917, zero update → 0 (no NaN), ×10 scaling → same fraction, GPU == CPU.

## Code changes (uncommitted)

- `src/gd_optimic/metrics.py`
  - `compute_checkerboard_fraction()` (vectorized over channels, one device→host sync).
  - Constants `CHECKERBOARD_PHYSICAL_LEVEL = 0.038`, `CHECKERBOARD_NOISE_MARGIN = 0.025`,
    `CHECKERBOARD_THRESHOLD = 0.063`.
- `src/gd_optimic/optimizer.py`
  - Removed `_compute_finite_difference_ic_update` and all `diag/finite_difference_ic_update/*` logging.
  - Per-step (`_last_ic_update`) and cumulative (`x0_current − x0_reference`) diagnostics, both kept.
  - History key `gradient_finite_difference_ic_update` → `checkerboard_fraction` (dict per variable + total).
  - Constant reference lines `ref_physical`, `ref_threshold` logged every iteration.
  - `add_custom_scalars` layout: "Pixelization (checkerboard)" → one Multiline chart per scope (total + 2 references).
- `src/gd_optimic/README.md`: TensorBoard tag list updated.

TensorBoard tags:

```
diag/checkerboard/fraction/{step,cumulative}/{SSH,T,S,U,V,total}     # score
diag/checkerboard/fraction/{step,cumulative}/ref_{physical,threshold} # 0.038 / 0.063 constants
diag/checkerboard/amplitude/{step,cumulative}/{SSH,T,S,U,V}          # size, physical units
```

The Custom Scalars chart has no data of its own — it redraws the `fraction/.../{total,ref_*}` scalars, so those must stay.

## Validation on saved multiruns (`.tmp/multiruns/*/*/states`, 141 runs)

Update = `initial_condition.nc − optimized_initial_condition.nc`, last timestep (≡ cumulative update; sign irrelevant).
`man_metaLr/1em6` skipped (no optimized IC). Total CF range per group, sorted:

| group (date) | n | weighting | smoothed | total CF |
|---|---|---|---|---|
| manFilter (08-27) | 2 | manual | yes | 0.032–0.044 |
| smooth_lossWeight (08-14) | 2 | dyn + man | yes | 0.041–0.062 |
| smooth_lossWeight2 (08-17) | 2 | dyn + man | yes | 0.049–0.054 |
| dynFilter (08-24) | 2 | dynamic | yes | 0.056–0.058 |
| dyn_lr (08-13) | 2 | dynamic | no | 0.086–0.122 |
| sLoss_typeLr (08-27) | 4 | dynamic | no | 0.086–0.096 |
| table1_refRun_lr1em2 (09-02) | 28 | manual | no | 0.088–0.179 |
| table1_refRun2 (09-09) | 22 | manual | no | 0.099–0.136 |
| table1_lr1e1 (09-06) | 8 | manual | no | 0.105–0.109 |
| table1_lr1e1_run2 (09-09) | 42 | manual | no | 0.106–0.111 |
| man_lr (08-13) | 2 | manual | no | 0.106–0.112 |
| dyn_lr (08-20) | 3 | dynamic | no | 0.106–0.164 |
| table1_3dAssim (09-21) | 17 | manual | no | 0.143–0.228 |
| man_metaLr (08-13) | 2 | manual | no | 0.154–0.194 |
| dyn_metaLr (08-14) | 3 | dynamic | no | 0.157–0.192 |

- Smoothed (n = 8): **0.032–0.062**. Unsmoothed (n = 133): **0.086–0.228**. Perfect separation, AUC = 1.000.
- Per variable AUC: S 1.000, U 1.000, V 0.992, T 0.918, **SSH 0.594** (SSH update is smooth even without smoothing).
  → read the **total**; U/V (cut ≈ 0.10) show which variable carries the artefact.
- `dyn*`/`man*` prefix = loss weighting, not smoothing. Unsmoothed dynamic-weighting runs span 0.086–0.192; the lowest
  unsmoothed runs are dynamic (`dyn_lr` lr 1e-2, `sLoss_typeLr`), consistent with the user's observation that dynamic
  weighting helps, but it does not reach the smoothed level. Meta-learner runs are the most pixelized.

## Threshold

- Physical level: GLORYS12 day-to-day increments (27 days, `table1_refRun2/0/states/ground_truth.nc`, Jan 2021),
  total CF median **0.038**, p95 0.039, max 0.041. Per var median: SSH 0.004, T 0.069, S 0.060, U 0.025, V 0.028.
- Empirical gap: smoothed max 0.062 / unsmoothed min 0.086.
- **User choice: threshold = physical 0.038 + noise margin 0.025 = 0.063** (logged as `ref_threshold`).
  Agent's alternative: 0.075 (centre of the gap, more margin; set `CHECKERBOARD_NOISE_MARGIN = 0.037`).
  With 0.063 the highest smoothed run (0.062) sits just under the line.
- Calibrated on the **cumulative** update → compare against `fraction/cumulative/total`. Step values are not calibrated.

## First live run with the new code

`.tmp/runs/pixelDiagTest_2026-09-23_21-40-04` (in the worktree; manual, lr 0.01, no smoothing, 150 it, 115 logged):
cumulative total **0.135 → 0.125** (≈ 2× threshold, inside `table1_refRun` range 0.088–0.179 → pixelized, as expected).
Cumulative per var: S 0.247 (≈ white noise), T 0.148, U 0.119, V 0.106, SSH 0.007.
Amplitudes 1e-8…1e-4 — these were first mistaken for the score; see "Step 2".

## Limitations

- Only 8 smoothed runs, all `obs.mode: full`; GLORYS reference is one month of one-day increments. Re-check threshold.
- Global scalar: a localized artefact (e.g. along satellite tracks) is diluted.
- 2×2 windows are index-space (lat-lon grid); the longitude wrap-around window is ignored (negligible).
- Full `gd_optimic.optimizer` import could not run on the login node (`xesmf`→`ESMF`, `copernicusmarine` missing;
  pre-existing, unrelated files). Verified by `py_compile`, pyflakes, standalone function tests, and the live run above.

## Side findings (not fixed, out of scope)

- `metrics.py` `set_mean_from_sequences`: uses undefined `ds` / `_np` in the `mean_field is not None` branch → always
  falls into `except` silently.
- `average_pooling` in `gradient.py` upsamples with nearest-neighbour → produces k×k blocks by design
  (not the checkerboard artefact itself, but worth knowing when reading the diagnostic).

## Reproduce (worktree `.tmp/runs/`, run with `/Odyssey/private/j25lee/bice/.venv/bin/python`)

- `checkerboard_multiruns.py` → `.tmp/outputs/checkerboard_multiruns/checkerboard_multiruns.csv` (per run + config)
- `checkerboard_threshold.py` → empirical cut + GLORYS physical level
- `verify_tb_reference.py` → TensorBoard tags / reference lines round-trip
- `verify_amp_fraction_relation.py` → amplitude ↔ fraction relation on a real update

## Next

- Commit on `worktree-pixelization-diagnostic` when authorized.
- Re-calibrate threshold after new smoothed runs (other obs modes, seasons).
