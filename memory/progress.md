# PROGRESS

## 2026-05-22 — Init
- Scaffolded per `LLMAIassistant_instruction.md`:
  - Created `CLAUDE.md` (project constitution + state).
  - Created `memory/` with `task_plan.md`, `findings.md`, `progress.md`, `decisions.md`.
  - Created `.tmp/` ephemeral workbench.
- Recorded repository snapshot in `findings.md`.
- **Phase:** U — Understand (M-O-R) — awaiting user input on Motivation.

## 2026-05-22 — Phase U / Motivation captured
- Recorded in `findings.md` (Phase U — Motivation section).
- Headline: probe ML ocean emulator's sensitivity to ICs; use the surrogate's gradient to estimate the optimal initial perturbation that explains observations.

## 2026-05-22 — Phase U / Objective captured
- Recorded in `findings.md` (Phase U — Objective section).
- Headline: ML-4DVar-style problem — find IC via gradient descent through a **frozen** pretrained glonet that (a) fits observations in the assimilation window and (b) gives best forecast in the forecast window, with physical consistency on both IC and trajectory.

## 2026-05-22 — Phase U / Result captured · Phase U COMPLETE
- Recorded in `findings.md` (Phase U — Result section + consolidated U-Summary).
- Headline: forecast improvement + **high-frequency consistency** to ground truth → diagnose & remediate ML emulator artifacts.
- **Phase transition:** U → **B (Blueprint)**. Next prompt: North Star (Q1 of 5).

## 2026-05-22 — Phase B / Q1 North Star captured
- Recorded in `findings.md` (Phase B — North Star section).
- Headline: global 1/4° ocean forecast, GLORYS12 ground truth, RMSE + PSD metrics, 7-day assim + 21-day forecast, obs-mismatch loss (path toward weak-constraint 4D-Var).
- Open: baseline definition, variable list, thresholds, eval period — carried to Phase B Q2–Q5.
- Next prompt: Q2 — Integrations.

## 2026-05-22 — Phase B / Q2 Integrations captured
- Recorded in `findings.md` (Phase B — Integrations section).
- Headline: IMTA SLURM server, PyTorch + TensorBoard, Jupyter for dev; **CMEMS observations** = assim target, **GLORYS12** = eval-only (strict separation).
- Next prompt: Q3 — Source of Truth.

## 2026-05-22 — Phase B / Q3 Source of Truth captured
- Recorded in `findings.md` (Phase B — Source of Truth section).
- Headline: data root `/Odyssey/public/glonet/`; NetCDF now, Zarr-ready; 2-step download/interpolate under 500 GB/node; OSE+OSSE; artifact promotion `.tmp/` → `src/gd_optimic/` after sign-off; naming `{exp_id}_{date}`.
- Filesystem actions: created `.tmp/runs/` and `.tmp/outputs/`.
- Next prompt: Q4 — Delivery Payload.

## 2026-05-22 — Phase B / Q4 Delivery Payload captured
- Recorded in `findings.md` (Phase B — Delivery Payload section).
- Headline: per-exp bundle = artifacts + Jupyter HTML report + MD summary; **+ observations_ose.nc / observations_osse.nc**; group-level promotion adds **superposed plots + mean/spread + PDF**; TensorBoard is live-inspection only, not a deliverable.
- Data-schema seeds drafted (state vector, obs vector, obs-mismatch loss, future 4D-Var weak-constraint extension).
- Next prompt: Q5 — Behavioral Rules.

## 2026-05-22 — Disk-full incident (resolved)
- `/Odyssey` hit 100% during Q4 capture; `findings.md` and `decisions.md` truncated to 0 bytes by failed fsync.
- User freed space (541 GB free post-cleanup).
- Both files rebuilt losslessly from conversation context; sizes verified (12.3 KB and 3.7 KB).
- No phase regression — Q4 content preserved.

## 2026-05-22 — Phase B / Q5 Behavioral Rules captured · Phase B Q1–Q5 COMPLETE
- Recorded in `findings.md` (Phase B — Behavioral Rules section + Blueprint Summary).
- (a) Scientific Integrity A1–A8 confirmed verbatim by user; (b)–(e) accepted.
- Binding implications logged in `decisions.md`.
- **Next step:** Phase B — **Research** (R1–R8 queued in `findings.md`).

## 2026-05-22 — Phase B / Research R1+R5+R7 complete
- Three read-only sub-agents executed in parallel; full reports archived under `.tmp/research/R1_glonet_io.md`, `.tmp/research/R5_glorys12_status.md`, `.tmp/research/R7_prior_art.md`.
- Summary + critical findings + reuse candidates + external references appended to `findings.md`.

## 2026-05-22 — Phase split (user decision)
- **Phase 1 (now):** legacy `model/glonet/` (v1) with user-supplied OVERRIDE that strips the `.detach()` / `gc.collect()` / `empty_cache()` calls. Ocean-only state vector; no atmospheric forcing; 2021 data already on disk.
- **Phase 2 (deferred):** `model/glonet3/glonet2_global_e239` (atmospheric forcing + sea-ice + Zarr dataset). User has source code; dataset download on IMTA still in flight.
- Decision recorded in `decisions.md`. Gate-item list in `findings.md` updated.

## 2026-05-22 — G4 confirmed; forecast vs. reanalysis window distinction
- G4 — eval period 2021 confirmed by user (glonet v1 unseen + SSH/SST 2021 obs companions on disk). Logged in `decisions.md`.
- New scheme distinction logged: forecast (short window, Phase-1 North Star) vs. reanalysis (long-term, out of scope).

## 2026-06-11 — Loss policy + obs-modes + A2 resolved
- **Loss policy refined:** forecast scheme (Phase 1) uses **`J_obs` only** (sufficient — the scientific question is whether ML emulator alone recovers a best IC); reanalysis scheme (Phase ≥ 2) gets the **full weak-4D-Var** (`J_b + J_obs + J_q` with B/R/Q matrices).
- **Three observation modes locked:** `cfg.obs.mode ∈ {'full', 'simulated', 'real'}` — `full`/`simulated` use GLORYS12 as target by design (twin / OSSE); `real` uses satellite SSH+SST and is the headline configuration. Replaces the earlier OSE/OSSE binary.
- **A2 refined to mode-aware:** GLORYS12-in-loss is permitted explicitly for `full` and `simulated` (labelled twin configurations); only `real`-mode results may carry headline forecast-skill claims.
- **File-management policy:** `optim.ipynb` is a REFERENCE file; production code lives in `src/gd_optimic/` per project rules.
- All bindings recorded in `decisions.md` (2026-06-11).

## 2026-06-11 — `optim.ipynb` ingested + analyzed
- Full extraction archived at `.tmp/research/R11_optim_notebook.md`; synthesis added to `findings.md`.
- **Confirmed already implemented:** β rollout (7 steps), constant-step custom SGD (lr=0.1, 1000 iters), surface-only (5 ch), frozen glonet, land-mask gradients, GLORYS12 warm-start IC.
- **Surface channel order LOCKED:** `[SSH, T, S, U, V]` (ch 0–4) — corrects my earlier ordering guess.
- **SSH obs path resolved:** `/Odyssey/public/altimetry_traces/2010_2023/gridded/sla_l3_all_2010_2023_0.25deg_convl4.nc` (106 GB, pre-gridded). The empty `Alongtrack_SSH_2021-...` directory was a red herring.
- **New detail noted:** multi-resolution gradient-pooling filter with kernel schedule `[8,4,2,1,…]`.
- **Two flags raised in `decisions.md`:**
  - (1) Loss currently = J_obs only — deviation from "weak-4D-Var" binding; user must pick Path A (J_obs-only Phase-1.a, J_b+J_q later) or Path B (extend now).
  - (2) **CRITICAL A2 ambiguity** — the loss `target` may be GLORYS12-shifted (A2-violating) or the satellite-obs tensor (A2-compliant). Must clarify before any run.

## 2026-06-11 — Phase-1 design decisions
- R10(c) rollout = **β** (explicit multi-step 4D-Var). 7 model calls per `.backward()` for the 7-day assim window.
- Loss = manually authored **weak-constraint 4D-Var** (J_b + J_obs + J_q) — not a torch built-in.
- Optimizer = constant-step (fixed LR, no scheduler) for Phase-1 start.
- "Patching" = region cropping (glonet uses FNO; no CNN/FNO-internal patch concept).
- User will provide a baseline optimization code file; R10(d/e) finalized after drop-in.
- R10(e) `y_k` semantics locked: CMEMS observations within each rollout step's time slice (NOT GLORYS12). → R2 is the next concrete blocker.

## 2026-05-22 — Import fix + checkpoint copy
- Edited `model/glonet/glonetLit_grdckpt.py` line 16: `sys.path.append(str(Path(__file__).parent.parent.parent / "src/glonet"))` → `sys.path.append(str(Path(__file__).parent))`. `from modelp2 import Glonet` now resolves to repo-local file.
- Copied checkpoint set B → `model/glonet/weights/`:
  - `glonet_part1.pth` (1,133,664,558 B)
  - `glonet_part2.pth` (1,244,463,598 B)
  - `glonet_part3.pth` (1,244,463,598 B)
  - sizes byte-exact vs source; 2.1 TB free post-copy.
- Decisions logged: set-B chosen; import fix applied.

## 2026-05-22 — Locator pass on Phase-1 assets
- External `Glonet` source path in override (`/Odyssey/private/j25lee/src/glonet/modelp2.py`) does NOT exist. Only on-disk copy is `bice/model/glonet/modelp2.py`. Fix plan: change L16 to `sys.path.append(str(Path(__file__).parent))`.
- Two candidate checkpoint sets located under `/Odyssey/public/glonet/TrainedWeights/`:
  - `glonet_p{1,2,3}.pt` (Jul 2025, ~1.67 GB total).
  - `glonet_part{1,2,3}.pth` (Nov 2025, ~3.62 GB total).
- Normalization stats `TrainedWeights/L0/` confirm 5 surface vars: thetao, so, uo, vo, zos. Phase-1 scope perfectly aligned.
- **Awaiting user choice** on checkpoint set before copying to `model/glonet/weights/`.

## 2026-05-22 — Override v1 dropped in + scope tightened
- User added `model/glonet/glonetLit_grdckpt.py`. Inspected; key facts written to `findings.md` (Phase 1 override section).
- **Scope tightened by user:** Phase 1 optimizes the **5-channel surface IC only** (model_1). Depth packs (model_2/3) are not part of δx₀. → decision logged.
- Architecture confirmed differentiable via `torch.utils.checkpoint` (memory-light + autograd-safe). Land masking already wired. Lightning module saves optimized IC as NetCDF to `cfg.model.output_path` — slots into our artifact-bundle convention.
- Pre-flight items surfaced (R10): (a) `sys.path` import in L16 resolves outside repo; (b) `cfg.model.checkpoint_paths.part_1` location must be pinned; (c) multi-step rollout for the 7-day assim window is not in the override — design choice pending.
- **Pending user input:** eval-period confirmation (2021), architectural-skeleton choice, R10 items, then R9 smoke test + R2/R3/R4/R8.

## 2026-06-12 — Copilot CLI handoff + 3 Phase-B ambiguities resolved

- Agent transitioned from Claude Code to **GitHub Copilot CLI (Claude Sonnet 4.6)**.
- Re-read full repository: confirmed protocol = `LLMAIassistant_instruction.md` (phases U→B→P→T).
- **A2 ambiguity closed:** `target` tensor is mode-dependent (GLORYS12 for `full`/`simulated`; satellite obs for `real`). All modes A2-compliant within their labelled scope.
- **Loss scope locked (Path A):** Phase-1.a = `J_obs` only; `J_b + J_q` deferred to Phase-1.b.
- **Baseline locked (R3):** forecast scheme baseline = **persistence** (initial state held constant for 21 days).
- Decisions logged in `decisions.md` (2026-06-12 entries).

## 2026-06-12 — R2 (Observation Handling) finalized

- SSH operator: along-track altimetry (native) → grid via nearest-neighbor
- SST operator: direct pixel match (1:1 colocation)
- QC: Light QC (trust CMEMS L3 pre-filtering)
- OSSE twin: GLORYS12 as truth
- Status: Ready for dataset class design in `src/gd_optimic/`

## 2026-06-12 — R8 (Experiment-Tracking Conventions) finalized

- TensorBoard hierarchy: hierarchical `loss/J_obs/{total,ssh,sst,...}` + placeholders for J_background, J_modelerror
- Metrics: RMSE (global, per-variable, per-basin) + PSD band-energy + gradient norms
- Logging: every iteration; histograms/embeddings every 50 iters
- Exp ID: `{config_summary}_{timestamp}` (e.g., `refIC_fullobs_2026-06-12_15-30-45`)
- Status: Ready for src/gd_optimic/ logging module

## 2026-06-12 — R9 (Differentiability Sanity Check) — IMPLEMENTED

- **Script:** `.tmp/runs/r9_sanity_check.py` (human-readable, commented)
- **What it tests:**
  1. Gradients flow through forward pass
  2. Gradient magnitudes are finite (no NaN/Inf)
  3. Per-channel gradient statistics (min, max, mean, std)
  4. **Land pixels are masked:** gradients over land = 0
  5. Memory usage reported
  
- **Usage:** `python .tmp/runs/r9_sanity_check.py`
- **Output:** `.tmp/outputs/R9_sanity_check/r9_results.json`
- **Status:** Ready to run. User can execute after confirming checkpoint paths in CONFIG.

## Pending / blockers
- R10(j) — Author `src/gd_optimic/` package: ready to proceed (all research items locked).

## 2026-06-12 — R10(j) (Production Optimization Package) — IMPLEMENTED

- **Package:** `src/gd_optimic/` (~2700 lines, 6 OOP modules + Hydra config)
- **Modules:**
  1. **data.py**: `GlonetDataset` + `ObservationOperator` (R2-compliant SSH/SST operators)
  2. **loss.py**: `ObservationLoss` (J_obs with observation masking, dynamic/manual weighting)
  3. **gradient.py**: `GradientFilter` + `ScheduledPooling` (multigrid optimization)
  4. **optimizer.py**: `ICOptimizer` (main loop with TensorBoard logging per R8 hierarchy)
  5. **metrics.py**: `MetricsComputer` (R4 RMSE) + `PSDComputer` (Phase P)
  6. **utils.py**: `MaskBuilder` + `ForwardModel` + normalizers

- **Configuration:** Hydra YAML (`configs/optimize_ic.yaml`) for explicit parameter management
- **Entry Point:** `src/run_optimization.py` (Hydra CLI)
- **Usage:**
  ```bash
  python run_optimization.py                           # Default config
  python run_optimization.py observations.mode=simulated  # Override params
  ```

- **Phase 1 Scope:** J_obs only. J_b (background) + J_q (model error) deferred to Phase 1.b.
- **CS1 Compliance:** Human-readable, extensively commented, simple structure
- **Scientific Integrity:** Enforces A1 (frozen model), A2 (obs/eval separation), A5 (reproducibility)
- **Status:** R10(j) complete ✅ | **Phase B research 10/10 complete** → ready for Phase P (Professor)

## 2026-07-01 — Short-line edits (code hygiene)

- Edited small blocks in core modules to comply with CS1 and `ruler = 132`:
  - `src/gd_optimic/output_handler.py`: wrapped long imshow() calls and attributes dictionaries across multiple lines to improve readability.
  - `src/gd_optimic/optimizer.py`: shortened logger messages and used temporary vars for clarity when adjusting ocean masks.
- Validation: `python -m py_compile` succeeded for modified files.
- Status: Applied in-tree.

## 2026-07-16 — Diagnostic anomaly views added
- Added anomaly versions of the main diagnostic PNG/GIF in `src/gd_optimic/output_handler.py` using the shared mean field when available.
- `save_outputs()` now receives `mean_field` from the metrics computer so raw and anomaly plots stay consistent with the anomaly diagnostics.
- Saved outputs now include `diagnostics/ic_correction_anomaly_comparison.png` and `states/forecast_comparison_anomaly.gif` alongside the raw figures.

## 2026-07-17 — TensorBoard IC-update fix
- Fixed TensorBoard IC-update rendering in `src/gd_optimic/optimizer.py`: use `fig.canvas.buffer_rgba()` for matplotlib capture, and log per-channel fallback images separately under `state/ic/update_image/{var}`.
- This resolved the `FigureCanvasAgg.tostring_rgb()` crash and the fallback `TypeError` from trying to log a 5-channel tensor as one image.
- Kept the north-up lat/lon visualization and the single figure legend/colorbar for the main path.

## 2026-07-17 — Gradient-norm scalar metric added
- Added per-iteration gradient-norm scalars in `src/gd_optimic/optimizer.py` under `metrics/gradient/norm/total` and `metrics/gradient/norm/per_channel/{SSH,T,S,U,V}`.
- The logged norm is computed from the masked last IC timestep, so it can be used as a scalar proxy for pixelization/artifacts during optimization.

## 2026-07-18 — Laplacian roughness metric added
- Added Laplacian-norm scalars in `src/gd_optimic/optimizer.py` under `metrics/gradient/norm/laplacian_total` and `metrics/gradient/norm/laplacian_per_channel/{SSH,T,S,U,V}`.
- The Laplacian is applied to the masked last IC timestep, making it a better scalar proxy for pixelization / local roughness than plain gradient norm.

## 2026-07-27 — Gradient-change metrics removed
- Removed `temporal_jump` and `neighbor_smoothness` from `src/gd_optimic/optimizer.py`.
- TensorBoard now focuses on loss, RMSE, IC RMSE, and update visualizations only.

## 2026-07-27 — IC spatial-gradient metric added
- Removed every previous pixelization diagnostic (wrong metrics).
- Added a finite-difference IC update norm for the correction field in `src/gd_optimic/optimizer.py`.
- TensorBoard logs `metrics/gradient/finite_difference_ic_update/{total,per_channel}` from the last IC update field.

## 2026-08-17 — Feature 1: Meta-learner Checkpointing & Mode Selection — IMPLEMENTED ✅

- **New Module:** `src/gd_optimic/meta_learner/checkpoint_manager.py`
  - `MetaLearnerCheckpointManager` class for save/load meta-learner weights
  - Checkpoint versioning and automatic discovery
  - Freeze/unfreeze utilities for inference mode

- **Three Operating Modes:**
  1. **Training** (`mode: "training"`): Train from scratch (random init, all params trainable)
  2. **Fine-tuning** (`mode: "fine_tune"`): Load pre-trained checkpoint and continue training (unfrozen params)
  3. **Inference** (`mode: "inference"`): Load pre-trained checkpoint, freeze all meta-learner params (read-only)

- **Configuration:**
  - `mode`: Execution mode (training/fine_tune/inference)
  - `load_checkpoint`: Checkpoint path (null for scratch, explicit path for loading)
  - Path format: `.tmp/runs/{exp_id}_{timestamp}/checkpoints/meta_learner/meta_learner_iterX.pt`

- **Files Modified:**
  - `src/gd_optimic/meta_learner/__init__.py`: Export MetaLearnerCheckpointManager
  - `src/gd_optimic/optimizer.py`: Instantiate checkpoint manager, handle mode selection, save meta-learner checkpoints
  - `configs/optimize_ic.yaml`: Add mode and load_checkpoint parameters

- **Status:** Feature 1 complete ✅ | Ready for testing and git commit

## 2026-08-18 — Feature 2: Structure Consistency Loss — IMPLEMENTED ✅

- **New Module:** `src/gd_optimic/structural_loss.py`
  - `StructuralOperator`: Base class for spatial derivative operators
  - `GradientOperator`: First-order gradients (∇) — central difference stencils
  - `LaplacianOperator`: Second-order Laplacian (∇²) — 4-neighbor curvature operator
  - `StructureConsistencyLoss`: Main loss class combining operator + MSE with observation masking

- **Loss Function Integration:**
  - Total loss: `L = J_obs + λ_struct * J_struct`
  - ObservationLoss now supports optional structural consistency term
  - Combined loss computed in single forward pass

- **Configuration Parameters:**
  - `loss.use_structural_loss`: Boolean enable/disable (default: false)
  - `loss.structural_operator`: "gradient" or "laplacian" (default: "gradient")
  - `loss.structural_loss_weight`: Scaling factor (default: 0.01, typical range: 0.001–0.1)

- **Files Modified:**
  - `src/gd_optimic/loss.py`: Added structural loss integration to ObservationLoss class
  - `src/gd_optimic/__init__.py`: Export all structural loss classes
  - `configs/optimize_ic.yaml`: Add loss configuration parameters
  - `src/run_optimization.py`: Pass structural loss params to ObservationLoss instantiation

- **Implementation Details:**
  - Gradient operator: Central difference on interior, replicate padding on boundaries
  - Laplacian operator: 5-point Von Neumann stencil with replicate padding
  - Both operators support [B, T, C, H, W] tensors with automatic batch processing
  - Observation masking applied to structural loss same as observation loss

- **Usage Examples:**
  - `python run_optimization.py loss.use_structural_loss=true loss.structural_operator=gradient loss.structural_loss_weight=0.01`
  - `python run_optimization.py loss.use_structural_loss=true loss.structural_operator=laplacian loss.structural_loss_weight=0.02`

- **Documentation:** Created `FEATURE_2_DOCUMENTATION.md` and `FEATURE_2_SUMMARY.md`

- **Status:** Feature 2 core implementation complete ✅ | Awaiting TensorBoard logging enhancements

## 2026-08-19 — Feature 2: TensorBoard Logging Enhancements — COMPLETE ✅

- **Gradient Channel Combination Algorithm:**
  - Gradient operator outputs [B, T, 2*C, H, W] (∂f/∂x and ∂f/∂y concatenated)
  - Implemented RMS combination to recover per-variable losses: `mag(∇f) = √(∂f/∂x)² + (∂f/∂y)²`
  - Returns struct_mse_per_var [B, C] with proper channel mapping back to original variables

- **Per-Variable Structural Loss Logging:**
  - New TensorBoard section: `loss/S_loss/{total,SSH,SST,SSS,UO,VO}`
  - Mirrors J_obs logging structure for consistency
  - Only appears when `use_structural_loss=true`
  - Weighted by `structural_loss_weight` parameter

- **Files Modified:**
  - `src/gd_optimic/structural_loss.py`: Updated compute_structure_mse() to return per-variable losses
  - `src/gd_optimic/loss.py`: Added struct_mse_per_var_weighted processing and details dict
  - `src/gd_optimic/optimizer.py`: Added TensorBoard logging for S_loss per-variable metrics

- **Documentation:** Created `FEATURE_2_LOGGING_ENHANCEMENTS.md`

- **Status:** Feature 2 complete ✅ | All files compile successfully | Ready for integration testing


## 2026-08-19 — Feature 2: Variable Weighting Applied to Structural Loss — COMPLETE ✅

- **Problem Solved:**
  - J_obs was weighted (manual or dynamic) but J_struct was unweighted
  - Inconsistent variable importance across loss terms
  
- **Solution Implemented:**
  - Apply same weighting scheme to structural loss: `J_struct_weighted = Σ_c w_c * struct_mse[c]`
  - Both J_obs and J_struct now respect variable priorities
  - Dynamic/manual weights flow automatically from loss configuration

- **Implementation:**
  - Updated `src/gd_optimic/loss.py` __call__() method
  - Extract weights from compute_weighted_loss()
  - Apply weights to struct_mse_per_var: `weighted_by_var = struct_mse_per_var * weights`
  - Use weighted values for both loss computation and TensorBoard logging

- **Total Loss Formula (Updated):**
  ```
  L_total = J_obs + λ_struct * J_struct
  where both J_obs and J_struct use same variable weighting [w_SSH, w_SST, w_SSS, w_UO, w_VO]
  ```

- **Benefits:**
  - Consistent variable balancing across both loss terms
  - Dynamic weighting: emphasizes structural consistency where observations are reliable
  - Manual weighting: encodes domain knowledge about variable importance
  - Better-balanced optimization gradients

- **Configuration:**
  - No new parameters needed
  - Automatic application via existing `loss.weighting` and `loss.manual_weights`
  
- **Documentation:** Created `FEATURE_2_WEIGHTING_ENHANCEMENT.md`

- **Status:** Feature 2 fully complete ✅ | All files compile successfully | Ready for integration testing

## 2026-08-19 — Feature 2: Separate Dynamic Structural Weighting — COMPLETE ✅

- Manual weighting reuses the configured normalized channel weights directly for both `J_obs` and `J_struct`.
    - Dynamic weighting is recalculated independently for `J_struct` from `struct_mse_per_var` channel magnitudes using the same inverse-magnitude logic as `J_obs`.
- TensorBoard `loss/S_loss/*` values now use the structural weights actually applied to the structural term.
- Validation: `python3 -m py_compile src/gd_optimic/loss.py src/gd_optimic/structural_loss.py` passed.

## 2026-08-24 — Feature 3: Downsampling Method Selection — IMPLEMENTED ✅

- **New Capability:** Choose between average-pooling and Gaussian low-pass filter for gradient smoothing
- **Problem Addressed:** User requested option to replace averaging-pooling with low-pass filter (better structure preservation)
- **Note:** Multi-resolution coarse-grid optimization deferred (model is fixed resolution) — only downsampling operators implemented

- **New Class:** `GaussianLowPassFilter` in `src/gd_optimic/gradient.py`
  - Creates 2D Gaussian kernels dynamically
  - Applies depthwise convolution for per-channel filtering
  - Preserves spatial structure better than non-overlapping pooling
  
- **Updated GradientFilter:** Enhanced to support multiple downsampling methods
  - `downsampling_method` parameter: 'average_pooling' or 'gaussian_lowpass'
  - Backward compatible (defaults to 'average_pooling')

- **Configuration Parameters (NEW):**
  - `optimization.downsampling_method`: "average_pooling" or "gaussian_lowpass"

- **Files Modified:**
  - `src/gd_optimic/gradient.py`: Added GaussianLowPassFilter class, updated GradientFilter
  - `src/gd_optimic/optimizer.py`: Added downsampling_method parameter and usage
  - `configs/optimize_ic.yaml`: Added downsampling_method configuration
  - `src/run_optimization.py`: Pass downsampling_method to GradientFilter and ICOptimizer

- **Validation:** All Python files compile successfully ✅

- **Status:** Feature 3 implementation complete ✅ | Ready for integration testing

