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
