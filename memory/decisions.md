# DECISIONS

## 2026-05-22 — Initialization
- **Decision:** Adopt `LLMAIassistant_instruction.md` as governing protocol; create scaffolding (CLAUDE.md, memory/, .tmp/).
- **Rationale:** User invoked agent init referencing this spec; the spec mandates this file layout.
- **Status:** In effect.

## 2026-06-11 — R10(c): rollout = β (explicit multi-step 4D-Var)
- **Decision:** Phase-1 optimization uses **β explicit rollout** — wrap `gradcheckp_model_1.forward` in a Python loop over `N_window` steps; δx₀ is the single tensor at t=0; observations from each step contribute to a single accumulated loss; one `.backward()` per epoch.
- **Rationale (user):** Strict 4D-Var semantics required — the trajectory matters, not a bag of independent single-step pairs. PSD/physical-consistency claims are about the rolled-forward trajectory.
- **Memory:** gradient checkpointing in the override (`torch.utils.checkpoint(use_reentrant=False)`) is what makes this feasible across N steps.
- **Status:** Binding.

## 2026-06-11 — Loss = manual weak-constraint 4D-Var (`J_b + J_obs + J_q`)
- **Decision:** `cfg.training.loss` is NOT a Hydra-instantiated `torch.nn.*Loss`. The Phase-1 loss is **manually authored**: `L = J_b(δx₀) + J_obs(H(x_k), y_k; R) + J_q(model_error_residuals; Q)`.
- **Rationale (user):** Phase-1 already targets the weak-constraint 4D-Var formulation, not a vanilla MSE warm-up.
- **Status:** Binding. Hydra config will reference this loss as a project-local class (e.g., `bice.losses.WeakConstraint4DVar`), not a torch built-in.

## 2026-06-11 — Patching = REGION CROPPING (not FNO/CNN patching)
- **Decision:** In `cfg.data.computing.enable_patching` / `cfg.model.patch_size`, "patching" means **spatial cropping to a region of interest** (e.g., Gulf Stream box, North Atlantic), NOT patch-based attention or CNN tiling. glonet's FNO operates on full fields by construction; this flag only restricts the spatial domain on which δx₀ is defined and optimized.
- **Rationale (user):** clarified terminology; prevents confusion in code/config.
- **Status:** Binding.

## 2026-06-11 — Optimizer = constant-step (Phase-1 start)
- **Decision:** Phase-1 starts with a **constant-step** optimizer (fixed learning rate, no scheduler). Concrete choice (vanilla SGD vs. Adam without scheduler) to be set in the user-supplied baseline code.
- **Rationale (user):** project bootstrap — keep optimization machinery simple while the loss + dataset are stabilized.
- **Status:** Binding for Phase-1 start; revisit once the loss landscape is understood.

## 2026-06-11 — Loss policy: refined to scheme-specific (REFINES the 2026-06-11 weak-4D-Var binding)
- **Decision:**
  - **Forecast scheme (Phase 1):** loss = **`J_obs` only** is sufficient and scientifically aligned. The question we are answering is *"can a frozen ML emulator alone recover a best IC from past observations, without a numerical ocean model?"* — obs-mismatch is the right scoreboard.
  - **Reanalysis scheme (Phase ≥ 2):** loss = **full weak-constraint 4D-Var** = `J_b + J_obs + J_q` with explicit **B, R, Q covariance matrices**. Out of Phase-1 scope but the Phase-1 loss is designed as "a target to develop" — extensible.
- **Rationale (user):** the two schemes ask different scientific questions; matching the loss to the scheme is more honest than over-engineering the forecast-scheme loss.
- **Status:** Binding. Supersedes the unconditional "weak-4D-Var binding" of earlier today.
- **Closes:** "Path A vs Path B" reconciliation in favor of Path A.

## 2026-06-11 — Observation modes — THREE-WAY taxonomy
- **Decision:** `cfg.obs.mode` ∈ {`'full'`, `'simulated'`, `'real'`} — explicit, mutually exclusive, declared per run.

  | Mode | Target tensor | Mask | Use |
  |---|---|---|---|
  | `'full'` | GLORYS12 reference state | none (every grid point) | idealized twin / theoretical ceiling — feasibility check on the gradient pipeline |
  | `'simulated'` | GLORYS12 reference state | sampled from real-obs patterns (SSH track + SST gridded availability) | classic **OSSE** — twin experiment with realistic obs coverage |
  | `'real'` | satellite SSH (altimetry) + satellite SST (ODYSSEA) | observation availability (NaN handling) | **OSE** — real-world assimilation, the headline configuration |
- **Replaces:** earlier OSE/OSSE binary.
- **Status:** Binding. Every artifact bundle's `config.yaml` MUST declare `obs.mode` (extends Behavioral Rule A5 reproducibility).

## 2026-06-11 — A2 (Strict obs/eval separation) — REFINED for the three modes
- **Refined statement:**
  - In `obs.mode='full'` and `obs.mode='simulated'`, GLORYS12 IS the target by design — these are **labelled twin / OSSE configurations**, NOT leakage. Their results MAY NOT be reported as headline forecast-skill numbers; they are diagnostic / ceiling probes.
  - In `obs.mode='real'`, GLORYS12 does NOT enter the loss. The optimized IC's downstream forecast is then evaluated against GLORYS12 — **this is the only mode whose results may carry headline forecast-skill claims** (the North Star).
- **Status:** Binding. Supersedes the original 2026-05-22 "Strict obs/eval separation" decision; the principle (no leakage between fit and headline-evaluation) is preserved, but the mechanism is now mode-aware. Original decision stays referenceable for audit.

## 2026-06-11 — File management: notebook is REFERENCE, not the project code
- **Decision:** `optim.ipynb` is a **reference file**. The Phase-1 production code is to be authored in **`src/gd_optimic/`** following project rules (`{exp_id}_{date}` naming for runs, artifacts in `.tmp/outputs/{exp_id}_{date}/`, promotion to `src/gd_optimic/<group_id>/members/` only after sign-off).
- **Status:** Binding. The notebook stays in-place as documentation; do NOT promote it as code.

## 2026-06-11 — Baseline `optim.ipynb` ingested (full extract in `.tmp/research/R11_optim_notebook.md`)
- **Decision:** Adopt the user's `optim.ipynb` as the algorithmic basis for `src/gd_optimic/`. The β multi-step rollout, custom SGD, gradient-pooling schedule, and channel-masking H operator are accepted as the Phase-1 baseline.
- **Channel order LOCKED:** `[SSH (zos), THETAO_surf, SO_surf, UO_surf, VO_surf]` (ch 0–4).
- **SSH obs path resolved:** `/Odyssey/public/altimetry_traces/2010_2023/gridded/sla_l3_all_2010_2023_0.25deg_convl4.nc`. Empty `Alongtrack_SSH_2021-...` directory deprecated (do not use).
- **Status:** Binding for Phase-1 implementation.

## 2026-06-11 — DEVIATION: notebook loss = J_obs only (binding said weak-4D-Var)
- **Issue:** `optim.ipynb` currently implements **only J_obs** — no J_b (background) and no J_q (model-error) terms. Binding decision in this file (2026-06-11 — "Loss = manual weak-constraint 4D-Var") said full `J_b + J_obs + J_q`.
- **Status:** Open — user decision pending. Two reconciliation paths:
  - **Path A (recommended):** Run Phase-1.a with J_obs-only first; add J_b + J_q as Phase-1.b before any publication-grade claim.
  - **Path B:** Extend the notebook now to include J_b + J_q before any optimization.

## 2026-06-11 — A2 AMBIGUITY: loss `target` source unclear
- **Issue:** In the notebook, the dataset loads "GLORYS12 shifted +7 d" as a `target` sequence AND also loads satellite SSH + SST. The loss `MSE(ŷ ⊙ obs_mask − target ⊙ obs_mask)` is computed against `target`. **It is not clear from the extraction whether `target` at the loss site is the satellite-obs tensor (A2-compliant) or GLORYS12 sampled at the obs locations (A2-violating).**
- **Status:** CRITICAL — must be clarified before any optimization is run, since A2 (Strict obs/eval separation) is a non-negotiable scientific-integrity rule.

## 2026-06-11 — Baseline optimization code to be provided by user
- **Decision:** User will supply a **baseline optimization code file** that we integrate with rather than designing the optimizer + training loop from scratch. R10(d) Hydra config and R10(e) dataset stub will be finalized once that file is dropped in.
- **Status:** Awaiting drop-in.

## 2026-05-22 — Phase 1 checkpoints: set B (`.pth`, Nov 17 2025) into `model/glonet/weights/`
- **Decision:** Use **Set B** = `glonet_part{1,2,3}.pth` (~3.62 GB total) from `/Odyssey/public/glonet/TrainedWeights/` as the Phase-1 weights. Copied into `model/glonet/weights/`.
- **Rationale (user):** Newer (Nov 2025) re-trained checkpoints; preferred over older `.pt` set.
- **Status:** Binding for Phase 1. `cfg.model.checkpoint_paths.part_{1,2,3}` must point at `model/glonet/weights/glonet_part{1,2,3}.pth`.

## 2026-05-22 — `glonetLit_grdckpt.py` L16 import fix
- **Decision:** Replaced the broken external `sys.path.append(... / "src/glonet")` with `sys.path.append(str(Path(__file__).parent))` so `from modelp2 import Glonet` resolves to the repo-local `model/glonet/modelp2.py`.
- **Rationale:** External path `/Odyssey/private/j25lee/src/glonet/modelp2.py` doesn't exist on disk; the override needs the repo's own `Glonet` class. Fix is purely a path correction — no architectural / weight change → A1 Frozen-model invariant preserved.
- **Status:** Applied 2026-05-22.

## 2026-05-22 — G4: held-out eval period = 2021 (CONFIRMED)
- **Decision:** Held-out evaluation period is **calendar year 2021** (rolling assim/forecast windows inside this year).
- **Rationale (user):** (i) glonet v1 was never trained on 2021 (clean held-out); (ii) obs companions `Alongtrack_SSH_2021-01-01_to_2021-12-31/` (SSH) and `ODYSSEA_SST_2021-01-01_to_2021-12-31/` (SST) both exist on disk; (iii) raw 1/12° GLORYS12 for 2021 is fully cached at `/Odyssey/public/glonet/raw/glorys12/`; (iv) pre-computed 1/4° init_states already chunked for the year.
- **Status:** Binding.

## 2026-05-22 — Forecast scheme vs. reanalysis scheme (window-length distinction)
- **Decision:** Two operating schemes are explicitly distinguished:
  - **Forecast scheme** — short window = `assim_window + forecast_window` (Phase-1 baseline: 7 d + 21 d = 28 d). This is what the North Star measures.
  - **Reanalysis scheme** — long-term window (multi-month / multi-year), continuous assimilation. **Out of scope for Phase 1**, candidate Phase 3 application.
- **Rationale (user):** ML-4DVar through glonet is targeted at the short-window forecast use first; reanalysis is a separate downstream application built on the same gradient machinery once the forecast loop is proven.
- **Status:** Binding. Reanalysis-window design choices (cycling, drift correction, B-matrix evolution) are not Phase-1 concerns.

## 2026-05-22 — Phase 1 scope = SURFACE-ONLY (5 channels via model_1)
- **Decision:** Phase 1 optimizes only the **surface-state IC** of glonet v1 — **5 channels** = `{thetao_surf, so_surf, uo_surf, vo_surf, zos}`. Depth state (model_2 / model_3, 40 channels each = 4 vars × 10 depths) is **not optimized in Phase 1**.
- **Rationale (user):** glonet v1 is structured as 3 independent Glonet models (surface + 2 depth packs). Investigating depth IC is "not worth it" at this stage; surface delivers the headline science (SSH + SST PSD) and matches the available 2021 obs companions directly. Depth variables come in Phase 2 (glonet2/e239 with Zarr forcing dataset).
- **Implications:**
  - δx₀ = `init_input1` only, shape `(B, T=2, 5, 672, 1440)`. `init_input2` / `init_input3` either left at GLORYS12 defaults (frozen baseline state) or removed from the loss entirely — TBD with user.
  - Observation operator H must map the 5-channel surface state to {SSH along-track, SST L3}. Direct: zos↔SSH-obs, surface thetao↔SST-obs.
  - State vector dimensionality cut from ~80 channels (full e239) to **5** — much cheaper optimization.
- **Status:** Binding for Phase 1.

## 2026-05-22 — Phase 1 = glonet v1 (override) · Phase 2 = glonet2 (e239)
- **Decision:** Build the ML-4DVar Best-IC prototype on **legacy `model/glonet/` (v1)** using a **user-supplied override** of the forward pass with `.detach()` calls removed (preserves weights and architecture intent; only clears the autograd-blocking calls). Glonet2 / `glonet2_global_e239` (with full atmospheric forcing + sea-ice + Zarr-format dataset requirements) is deferred to **Phase 2** of the project.
- **Rationale:**
  - User holds the override version of glonet v1 — it bypasses the gradient-blocking ops while keeping the same trained weights and architecture intent (no retraining; Frozen-model invariant A1 still satisfied if the override changes ONLY autograd-blocking calls).
  - User has glonet2 source code but the dependent dataset (Zarr w/ atmospheric forcing, sea-ice, etc.) is still being downloaded on IMTA. Blocking on it delays everything.
  - Phase 1 needs only ocean-state data, which is already on disk for 2021 (raw + 1/4° init_states + SSH/SST obs companions).
- **Status:** Binding. Phase 2 (glonet2 upgrade) is a separate scope.
- **Implication for `findings.md` / R1:** the differentiability sanity check is now on the **override**, not on e239. State vector for Phase 1 is the legacy layout: ~41 channels (4 vars × 10 depths + zos) × `(B, T=2, H, W)`, no atmospheric forcing.
- **Override scope (must respect):** the override may ONLY remove `.detach()` / `gc.collect()` / `torch.cuda.empty_cache()` calls in the forward pass. Any further architectural change re-opens the Frozen-model invariant question and requires a new decision.

## 2026-05-22 — Behavioral Rules (Phase B Q5)
- **Decision:** Adopt rules (a) Scientific Integrity A1–A8, (b) Reproducibility, (c) Operational/Safety, (d) Communication/Tone, (e) Project-specific (placeholders) as recorded in `findings.md` → "Phase B — Behavioral Rules (Q5)".
- **Rationale:** User confirmed (a) verbatim and accepted (b)–(e); these become the governing rulebook for all subsequent phases.
- **Status:** Binding. Notably implies:
  - Agent NEVER fine-tunes / modifies model weights or architecture.
  - Agent NEVER writes/deletes outside the allowed dirs without explicit OK.
  - Agent NEVER submits SLURM jobs, commits, or pushes without sign-off.
  - Every run produces a self-describing artifact bundle (seed, hash, variant, OSE/OSSE, windows, optimizer, slice, git SHA).
  - PSD/spectral checks are mandatory; A RMSE-only improvement is not a success.

## 2026-05-22 — Delivery: per-exp vs. group deliverables
- **Decision:** Per-experiment runs produce Jupyter HTML report + Markdown summary in their bundle. **PDF is group-level only** (concluding a set of related runs, e.g., uncertain-IC ensemble).
- **Promotion (group):** must include superposed plots + mean/spread of headline metrics + one group PDF.
- **TensorBoard is not a deliverable** — live interactive surface only.
- **`observations_{ose,osse}.nc` is part of the bundle** so trajectories and the obs they were fit against stay co-located.
- **Status:** Binding for Phase P deliverables and Phase T transfer.

## 2026-05-22 — Two-step download/interpolation (500 GB/node limit)
- **Decision:** Any new data acquisition follows: (a) **download original-resolution** to `/Odyssey/public/glonet/raw/`; (b) a **separate script** interpolates to 1/4° in its target directory. Never a single combined job.
- **Rationale:** IMTA SLURM nodes cap at 500 GB; combined download+interpolate OOMs.
- **Status:** Binding for all download/preprocessing code under this project.

## 2026-05-22 — Artifact promotion lifecycle (`.tmp/` → `src/gd_optimic/`)
- **Decision:** All optimization outputs land in `.tmp/` first (`.tmp/runs/` for scripts, `.tmp/outputs/` for results/viz). They are only **promoted** into `src/gd_optimic/` after explicit user validation. Promotion includes code, scripts, results, data, and visualizations.
- **Naming convention:** `{exp_id}_{date}` (e.g., `gd001_2026-05-22`) — applied to run directories at both stages.
- **Rationale:** Protects `src/gd_optimic/` from in-flight drafts; gives a clear "validated" boundary; user retains sign-off control per the protocol's Feedback step (Phase P).
- **Status:** Binding.

## 2026-05-22 — Format policy: NetCDF now, Zarr-ready
- **Decision:** All read/write code paths should be written to allow either NetCDF or Zarr (e.g., via `xarray.open_dataset` engine abstraction) even though today's primary store is NetCDF.
- **Rationale:** Repository must remain liftable to cloud-deployable Zarr without an internal-API rewrite.
- **Status:** Soft-binding (won't block PRs but is the default when introducing new I/O).

## 2026-05-22 — Strict obs/eval separation
- **Decision:** Assimilation loss is computed against **CMEMS observations**. **GLORYS12 is reserved for evaluation only.**
- **Rationale:** Prevents leakage — we must not "fit to the thing we score on." High-frequency / artifact diagnostics depend on GLORYS12 being an independent reference.
- **Status:** In effect for all experiments unless explicitly overridden (any override must be flagged loudly).

## 2026-05-22 — Phase U closed
- **Decision:** Project framed as ML-4DVar — gradient-based optimal-IC search through a frozen pretrained glonet, judged on forecast skill **and** high-frequency consistency.
- **Rationale:** Driven by the scientific gap (ML emulator artifacts); user confirmed architecture stays fixed and that high-frequency fidelity is the impact-bearing criterion.
- **Status:** In effect; governs all Phase B scoping.

## 2026-06-12 — A2 ambiguity resolved: `target` tensor is mode-dependent

- **Decision:** The loss `target` tensor in `optim.ipynb` is **mode-dependent**:
  - `obs.mode='full'` or `'simulated'` → GLORYS12 sampled at obs locations (labelled twin; A2-compliant within OSSE scope).
  - `obs.mode='real'` → satellite obs tensor (SSH altimetry + SST ODYSSEA); A2-compliant for headline claims.
- **Status:** Binding. Closes the 2026-06-11 critical A2 ambiguity flag.

## 2026-06-12 — Loss scope: Path A selected (J_obs only for Phase-1.a)

- **Decision:** Phase-1.a proceeds with **`J_obs` only**. Full 4D-Var (`J_b + J_q`) is Phase-1.b, deferred to after first optimization runs.
- **Rationale (user):** Unlock runs immediately; extend loss once pipeline is validated.
- **Status:** Binding for Phase-1.a. Supersedes earlier "Loss = manual weak-constraint 4D-Var" binding for Phase-1 start.

## 2026-06-12 — Baseline definition (R3): Persistence

- **Decision:**
  - **Forecast scheme (Phase 1):** Baseline = **persistence forecast** from the same starting IC (initial ocean state held constant for 21 days). Compared against: glonet forward from the 4D-Var optimized IC.
  - **Reanalysis scheme (Phase ≥ 2):** Benchmarking against physical ocean state persistence TBD — will require dedicated benchmark design before results can be reported.
- **Rationale (user):** Persistence is the natural zero-skill reference for the forecast scheme. Reanalysis physical persistence benchmarking is a separate, more complex task.
- **Status:** Binding for Phase 1. Reanalysis baseline is open.

## 2026-06-12 — R2 (Observation Handling) — LOCKED

- **Decision:**
  - **SSH operator H:** Use along-track altimetry (native resolution) → interpolate to 1/4° grid via **nearest neighbor** (fastest, deterministic).
  - **SST operator H:** Use pre-gridded L3 (ODYSSEA) → direct pixel match (1:1 colocation to model grid).
  - **QC masking:** For Phase-1.a, trust all valid pixels (no strict QC filtering); assume CMEMS L3 products are pre-QC'd.
  - **OSSE twin generation:** Use GLORYS12 as truth trajectory to generate synthetic observations for `obs.mode='simulated'` and `obs.mode='full'`.
  
- **Rationale (user):** Simplicity-first for Phase 1; along-track SSH more realistic than gridded; direct pixel match for SST mirrors typical assimilation practice; GLORYS12 twin is the baseline for idealized experiments.

- **Status:** Binding. Closes R2 blocker for dataset class design.

## 2026-06-12 — Coding Standards (CS1: Human-Readable Code)

- **Decision:** All code written by agent must be:
  - **Visible** — clear structure, proper spacing, logical flow
  - **Interpretable** — readable by humans first, machines second
  - **Simple** — straightforward logic; no clever tricks or one-liners
  - **Straightforward** — direct path from input to output
  - **Commented** — explanations for WHY, not just WHAT

- **Rationale (user):** "Strict rule of thumb when write a new code, you have to make the code visible, and interpretable by human. The code should be simple, straight forward and commented with explaination."

- **Examples:**
  ```python
  # ❌ BAD (clever but opaque)
  result = [f(x) for x in data if g(x)]
  
  # ✅ GOOD (clear intent)
  # Filter data items that pass validation criteria
  filtered_items = []
  for item in data:
      if meets_quality_threshold(item):  # Quality check
          filtered_items.append(process_item(item))
  ```

- **Status:** Binding for all Phase B → P → T code generation.

## 2026-06-12 — R4 (Metrics & Thresholds) — LOCKED

- **Decision:**
  - **RMSE:** Per-variable (SSH, T, S, U, V) across assimilation + forecast window; basin-stratified (Gulf Stream, high-variance region, low-variance region, coastal points).
  - **PSD:** 2D directional spectrum (kx, ky) represented by wavenumber bands (mesoscale ~50–500 km priority).
  - **PSD match metric:** Band-energy ratio (energy in mesoscale / total energy) — higher ratio = better high-frequency consistency.
  - **Success threshold (RMSE):** Adaptive — user decides case-by-case after first runs (no fixed % improvement yet).
  - **Physical floor:** Report full wavenumber range; highlight energetic scales (100 km–1000 km); confirm scales ≥ 4 grid cells resolvable.

- **Rationale (user):** Per-variable + basin breakdown critical for dynamics diagnosis; 2D spectrum + band-energy reveals whether ML emulator preserves mesoscale energy (key scientific question); threshold flexible for Phase 1 exploration.

- **Status:** Binding. Closes R4 blocker for verification gate (Phase P).

## 2026-06-12 — R8 (Experiment-Tracking Conventions) — LOCKED

- **Decision:**
  - **TensorBoard hierarchy (Phase-1.a ready for Phase-1.b extension):**
    ```
    loss/
      ├── J_obs/
      │   ├── total
      │   ├── ssh
      │   ├── sst
      │   ├── uo
      │   └── vo
      ├── J_background/     (placeholder for Phase-1.b)
      └── J_modelerror/     (placeholder for Phase-1.b)
    
    metrics/
      ├── rmse/
      │   ├── global/       (per variable: ssh, sst, uo, vo, ...)
      │   ├── gulf_stream/  (per variable, per basin)
      │   ├── high_var/
      │   ├── low_var/
      │   └── coastal/
      ├── psd/
      │   ├── band_energy/global
      │   ├── band_energy/[basin]
      │   └── spectral_slope
      └── gradient/
          ├── norm/total
          └── norm/per_channel
    
    state/
      ├── ic/norm
      └── ic/update_magnitude
    ```
  
  - **Logging frequency:** Every iteration (gradient step)
  - **Histogram + embeddings:** Every N iterations (e.g., 50) to reduce I/O; log IC field correction to diagnose pixelization
  - **Experiment ID:** `{config_summary}_{timestamp}` (e.g., `refIC_fullobs_2026-06-12_15-30-45`, `idiotIC_realObs_2026-06-12_...`, `outIC_onlySSHObs_...`)
  
- **Rationale (user):** Hierarchical ready for Phase-1.b extensions; sampled histograms reduce I/O; config-based ID makes experiments immediately interpretable.

- **Status:** Binding. Closes R8 blocker for logging infrastructure.

## 2026-06-12 — R9 (Differentiability Sanity Check) — REDEFINED & LOCKED

- **Decision:** R9 is NOT a backward-pass test. Instead, R9 checks:
  - **Gradient memory usage:** Monitor peak memory consumption during 7-step multi-step rollout with `torch.utils.checkpoint`. Confirm memory-efficiency goal (gradient checkpointing should keep peak memory ~constant vs linear growth without checkpointing).
  - **Land-pixel masking:** Verify that gradients on land pixels (ocean_mask==0) are explicitly zeroed BEFORE optimizer step, preventing spurious IC updates over land.
  - **Gradient statistics:** Per-channel gradient norms (min, max, mean, std) to confirm reasonable magnitudes (not NaN, not exploding).

- **Implementation:** Quick probe script in `src/gd_optimic/` (or `.tmp/runs/`) that:
  1. Loads checkpoint set B + glonetLit_grdckpt.py
  2. Creates synthetic IC + observations
  3. Runs 7-step forward + backward
  4. Reports: peak memory, gradient norms per channel, masked/unmasked gradient comparison
  5. Logs results to `.tmp/outputs/R9_sanity_check/`

- **Success criteria:**
  - Peak memory < [user threshold] MB (TBD after first run)
  - All gradient norms finite and reasonable (1e-6 to 1e+2 typical)
  - Land gradients = 0 after masking

- **Status:** Ready for implementation. Blocker for Phase 1 runs: user confirms memory threshold after first probe.

## Open items
- R4 — Metrics & thresholds (RMSE/PSD success criteria, physical floor): pending.
- R8 — Experiment-tracking conventions (TensorBoard tag scheme): pending.
- R9 — Differentiability sanity check: pending user OK + checkpoint path confirmation.
- R10(j) — Author `src/gd_optimic/` package: pending R4 finalization.

## 2026-06-12 — R10(j): Production Optimization Package — IMPLEMENTED

**Decision**: Implemented `src/gd_optimic/` as object-oriented package with Hydra configuration.

**Components**:
1. **data.py**: `GlonetDataset` + `ObservationOperator` (R2-compliant SSH/SST operators)
2. **loss.py**: `ObservationLoss` (J_obs with observation masking, dynamic/manual weighting)
3. **gradient.py**: `GradientFilter` + `ScheduledPooling` (multigrid optimization)
4. **optimizer.py**: `ICOptimizer` (main loop with TensorBoard logging per R8 hierarchy)
5. **metrics.py**: `MetricsComputer` (R4 RMSE) + `PSDComputer` (Phase P)
6. **utils.py**: `MaskBuilder` + `ForwardModel` + normalizers

**Configuration**: Hydra YAML (`configs/optimize_ic.yaml`) manages all parameters explicitly:
- Experiment metadata (name, description)
- Data paths (GLORYS12, SSH obs, SST obs)
- Model checkpoints (glonet v1)
- Observation mode (full/simulated/real - A2 compliant)
- Loss weighting (dynamic/manual)
- Optimization (learning rate, num iterations, gradient filter, scheduled pooling)
- Metrics (RMSE global/basin, PSD deferred to Phase P)
- Logging (TensorBoard hierarchy per R8, save frequency)

**Entry Point**: `src/run_optimization.py` uses Hydra for CLI:
```bash
python run_optimization.py                           # Default config
python run_optimization.py observations.mode=simulated  # Override params
```

**Phase 1 Scope**: J_obs only (observation term). J_b (background) + J_q (model error) deferred to Phase 1.b.

**Scientific Integrity**:
- A1 enforced: model.parameters().requires_grad = False
- A2 enforced: observation mode determines truth source
- A5 enforced: random seeds, deterministic algorithms

**CS1 Compliance**: All code human-readable, extensively commented, simple structure.

**Status**: R10(j) complete ✅ | Phase B research 10/10 complete → ready for Phase P
