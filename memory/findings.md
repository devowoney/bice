# FINDINGS

## Phase B — Behavioral Rules (Q5, captured 2026-05-22)

### (a) Scientific Integrity
- **A1. Frozen-model invariant.** `glonet` is a fixed forward operator — no fine-tuning, no architecture changes, no weight updates anywhere. Model hash + variant tag (legacy vs. `e239`) logged in every `config.yaml`.
- **A2. Independence of evaluation (no leakage).** CMEMS = fit; GLORYS12 = score. No quantity computed from GLORYS12 may enter the loss. Restates the binding decision in `decisions.md` as non-negotiable.
- **A3. Honest baselines.** Every improvement claim names an explicit baseline using the same window/model/eval set; baseline IC construction declared in `config.yaml`.
- **A4. Physical plausibility is part of "correct."** A run that reduces RMSE but produces NaN / exploding gradients / sub-grid noise / broken spectra / energy non-conservation is NOT a success — it is a candidate artifact for diagnosis. PSD/spectral checks are mandatory verification artifacts in every report.
- **A5. Full reproducibility per run.** Artifact bundle MUST include: seed, model hash + variant tag, obs config (OSE/OSSE — never inferred), assimilation/forecast windows, optimizer settings (lr, schedule, # iters, control vars in δx₀), dataset slice (dates/region), git SHA.
- **A6. No cherry-picking.** Group reports include all members (failures included). Removed members flagged with reason in `decisions.md`.
- **A7. OSE/OSSE never silently mixed.** Group PDF states which obs config produced which curve/number/map.
- **A8. Disclosed limitations.** Per-exp `REPORT.md` carries a Limitations section: scope (region/season/period), known artifacts, seed sensitivity, deviations from `decisions.md`.

### (b) Reproducibility
- Every run starts from a `config.yaml`; no hidden hyperparameters.
- Random seed pinned and logged.
- `pyproject.toml` pinned for releases; updates only via PR.

### (c) Operational / Safety (must-not-dos)
- No combined download+interpolate jobs (500 GB/node limit).
- No writes outside `.tmp/`, `src/gd_optimic/`, `model/`, `memory/`, `CLAUDE.md` without explicit user OK.
- No deletion of anything in `/Odyssey/public/glonet/` without explicit user OK.
- No long-running jobs launched by the agent (incl. SLURM submissions) without user sign-off.
- No commits / pushes without user sign-off.

### (d) Communication / Tone
- Be terse. Show data, not narration.
- Phase transitions must be announced and recorded in `progress.md`.
- On any blocker (disk full, OOM, OOM-killed SLURM job, NaN loss), pause and report root cause before retry (Phase-T Self-Annealing loop, applied universally).

### (e) Project-specific (placeholders — to firm up in Research)
- Physical-consistency floor (mass conservation? geostrophic balance? non-negative thickness? spectral-slope bound?) — _TBD_.
- CMEMS license / acknowledgement rules for published outputs — _TBD_.
- Publishing / disclosure timing (preprint vs internal report) — _TBD_.

---

## Phase B — Blueprint Summary (consolidated 2026-05-22)
- **North Star:** global 1/4° forecast at 21 days from gradient-optimized IC (frozen `glonet`, 7-day assim, obs-mismatch loss → weak-constraint 4D-Var) beats baseline on RMSE-vs-GLORYS12 AND matches GLORYS12 PSD better than baseline.
- **Integrations:** IMTA SLURM; PyTorch + TensorBoard; Jupyter for dev; CMEMS obs (assim); GLORYS12 (eval-only).
- **Source of Truth:** `/Odyssey/public/glonet/` (NetCDF now, Zarr-ready); 2-step download/interpolate (500 GB/node); OSE + OSSE through same loss interface; artifact promotion `.tmp/` → `src/gd_optimic/` on sign-off; naming `{exp_id}_{date}`.
- **Delivery:** per-exp bundle (incl. `observations_{ose,osse}.nc`, Jupyter HTML, REPORT.md); group-level promotion adds superposed plots, mean/spread, group PDF; TensorBoard is live-inspect only.
- **Behavioral Rules:** A1–A8 integrity; reproducibility via `config.yaml` + seed + pin; no unilateral writes/deletes/jobs/commits; pause-and-diagnose on blockers.

→ Phase B Q1–Q5 closed. **Research step pending.**

## Phase B — Research (queued 2026-05-22)
**Items carried over from Phase B Q1–Q5 — to be addressed in the Research sub-step before any code change.**

**R1. Glonet I/O contract**
- Enumerate input/output variables, depth levels, grid, tensor shapes (legacy `glonet` AND `glonet2_global_e239`).
- Identify the differentiable forward call (entry point in `model/glonet/` / `model/glonet3/`).
- Pin the set of control variables in δx₀ (subset of state vars on which gradient is taken).

**R2. Observation handling (OSE + OSSE)**
- For each CMEMS product (`SST_GLO_PHY_L3S_MY_010_039`, `SEALEVEL_GLO_PHY_L3_NRT_008_044`): native format, time/space sampling, QC flags, error model.
- Observation operator H: along-track for SSH, gridded L3 for SST — interpolation choice (nearest / bilinear / super-obs binning) and time alignment.
- OSSE twin pipeline: which trajectory to use as truth, noise model, sampling pattern.

**R3. Baseline definition (closes North Star)**
- Choose one of: (i) raw-IC forecast from `/Odyssey/public/glonet/glorys12_*_init_states/`, (ii) persistence, (iii) climatology from `1993-06-01_climatology/`, (iv) un-optimized δx₀=0 case. Record rationale.

**R4. Metrics & thresholds**
- RMSE: per-variable, per-day, global + basin-stratified. Threshold for "won."
- PSD: 1D/2D wavenumber spectra, frequency bands of interest (mesoscale ~50–500 km matters most for SSH; SST has its own bands). Choose match metric (Wasserstein on log-PSD? band-energy ratio?).
- Physical-consistency floor (closes Behavioral Rule (e) part 1).

**R5. GLORYS12 status**
- Confirm whether `/Odyssey/public/glonet/raw/` already holds raw GLORYS12 for the experiment period, or if the in-flight `down_glorsy12_37633.log` is the source.
- Confirm 1/4° re-interpolation script exists (else implement as the "step (b)" of the 2-step pattern).
- Pin the held-out evaluation period (start/end).

**R6. Older `glonet` weight relocation**
- Locate weights currently outside the repo; move under `model/glonet/weights/` (or similar) and document the path mapping. User-task.

**R7. Prior art / reuse**
- Search the protocol's intended sources (literature, GitHub) for ML-4DVar / differentiable-emulator DA prior work. Note relevant repos + papers in this file.
- Specifically: any prior work using `glonet` for DA; differentiable-DA frameworks (`4DVarNet`, etc.); CMEMS-side DA codebases (likely Mercator-internal).

**R8. Experiment-tracking conventions**
- TensorBoard tag scheme (e.g., `loss/total`, `loss/sst`, `loss/ssh`, `ic/norm`, `grad/norm`, per-day RMSE during val).
- Group-id assignment rule (UTC date + slug? counter?).

## Phase B — Delivery Payload (Q4, captured 2026-05-22)

### A) Per-experiment artifact bundle (under `.tmp/outputs/{exp_id}_{date}/`)
```
.tmp/outputs/{exp_id}_{date}/
├── config.yaml                 # windows, lr, loss weights, model rev, obs cfg, OSE/OSSE flag
├── ic_optimized.nc             # final optimized IC (all variables)
├── ic_initial.nc               # starting IC for comparison
├── observations_ose.nc         # OSE obs (real CMEMS, post-QC, collocated)
├── observations_osse.nc        # OSSE synthetic obs (twin) — present when applicable
├── forecast/
│   ├── trajectory.nc           # 28-day rollout from optimized IC
│   └── baseline_trajectory.nc  # 28-day rollout from un-optimized IC
├── metrics/
│   ├── rmse_vs_glorys12.json
│   ├── psd_vs_glorys12.json
│   └── loss_history.json
├── tb/                         # TensorBoard event files
├── figures/                    # per-experiment plots
├── report.ipynb / report.html  # Headline B-1
└── REPORT.md                   # Headline B-3 (embedded PNGs)
```
**Note:** `observations_{ose,osse}.nc` added per user request — obs co-located with trajectories for audit.

### B) Headline deliverable
- (B-1) Jupyter notebook report — per-experiment, auto-executed → HTML.
- (B-3) Markdown summary — per-experiment, embedded PNGs.
- (B-2) PDF — **group-level only** (e.g., uncertain-IC ensemble conclusions).
- TensorBoard — **not** a deliverable; live interactive inspection only (gradient field, IC correction).

### C) Promotion (.tmp/outputs/ → `src/gd_optimic/`)
- Artifact-bundle promotion is sufficient for this phase.
- **Group-level augmentation at promotion (required):** superposed plots, mean/spread of headline metrics, group PDF.
- Proposed group layout:
  ```
  src/gd_optimic/<group_id>/
  ├── members/{exp_id}_{date}/   # promoted bundles
  └── summary/
      ├── superposed/            # overlay plots
      ├── aggregate_metrics.json
      └── group_report.pdf
  ```

### Implied data-schema seeds (firm up in Research)
- **State vector x**: tensor over glonet I/O variables × depth × 1/4° lat × lon (global).
- **Observation vector y**: per-product irregular records + per-product H operator (state → obs locations/times).
- **Phase-1 loss (obs-mismatch):** L = Σ_p (H_p(x_t) − y_p,t)ᵀ R_p⁻¹ (H_p(x_t) − y_p,t), summed over assim window.
- **Phase-2 loss (weak-constraint 4D-Var):** + J_b (background) + J_q (model-error).

## Phase B — Source of Truth (Q3, captured 2026-05-22)

**1) Models (pretrained, frozen)**
- **Two model versions** in use: `glonet` and an **improved newer version** (`glonet2_global_e239`, in `model/glonet3/`).
- Weights for the original `glonet` currently live **outside the repo**; user will relocate them in-tree later. → **Open task:** consolidate `glonet` weights under `model/glonet/weights/` (or equivalent) once moved.
- Newer weights bundle: `model/glonet3/glonet2_global_e239_model_package(.tar.gz)`.

**2) Primary data root**
- **`/Odyssey/public/glonet/`** is the canonical data root on IMTA.
- Observed sub-tree (snapshot 2026-05-22):
  - `1993-06-01_climatology/` — climatology fields.
  - `2022-06-01_init_states/`, `glorys12_1993-01-01_to_1993-06-30_init_states/`, `glorys12_2020-01-01_to_2020-12-31_init_states/`, `glorys12_2021-01-01_to_2021-12-31_init_states/` — pre-computed initial-state inputs for glonet.
  - `Alongtrack_SSH_2021-01-01_to_2021-12-31/` — along-track SSH (likely companion to `SEALEVEL_GLO_PHY_L3_NRT_008_044`).
  - `ODYSSEA_SST_2021-01-01_to_2021-12-31/` — SST L3 (likely companion to `SST_GLO_PHY_L3S_MY_010_039`).
  - `raw/` — pre-interpolation downloads.
  - `statistics/` — normalization stats / climatologies.
  - `TrainedWeights/` — pretrained model weights store.
  - `tmp.ipynb` — scratch notebook (ignore).
- **Format:** primarily **NetCDF** today; **Zarr** support is a target so the codebase can be lifted to cloud.

**3) Download / interpolation pipeline — HARD CONSTRAINT**
- **Node memory limit: 500 GB.** Combined download-and-interpolate is infeasible.
- **Two-step pattern (mandatory):**
  1. Download original-resolution file to `/Odyssey/public/glonet/raw/`.
  2. A **separate** interpolation script reads from `raw/` and writes the 1/4° product to its dated `*_init_states/` (or equivalent) directory.
- Existing `down_glorsy12_*.log` jobs follow (or should follow) this pattern.

**4) Observations (assimilation target)**
- Configurations supported: **OSE** (real obs) **and** **OSSE** (synthetic obs from a twin experiment) — both must work through the same loss interface.
- Confirmed CMEMS product IDs:
  - `SST_GLO_PHY_L3S_MY_010_039` — multi-sensor SST L3S, multi-year (→ companion: `ODYSSEA_SST_2021-01-01_to_2021-12-31/`).
  - `SEALEVEL_GLO_PHY_L3_NRT_008_044` — along-track sea-level L3 NRT (→ companion: `Alongtrack_SSH_2021-01-01_to_2021-12-31/`).
- Detailed obs configuration (collocation operator H, QC, gridding/binning, error model) → deferred to a Research task.

**5) Optimization artifacts — lifecycle**
- **Stage 1 (dev, ephemeral):** everything lands in `.tmp/`.
  - `.tmp/runs/`    → run scripts / configs (created 2026-05-22).
  - `.tmp/outputs/` → results, plots, exported diagnostics (created 2026-05-22).
  - TensorBoard event files also under `.tmp/` until promoted.
- **Stage 2 (validated):** after explicit user sign-off, artifacts (code, scripts, results, data, viz) are **promoted** into `src/gd_optimic/`.
- **Naming convention:** `{exp_id}_{date}` (e.g., `gd001_2026-05-22`).

## Phase B — Integrations (Q2, captured 2026-05-22)

**Compute**
- **IMTA SLURM server** (remote) — single-host, no multi-site federation.
- All data + model artifacts live on that server (no cloud bucket / off-host I/O during runs).

**Data sources**
- **Copernicus Marine (CMEMS)** — used for **observations** (the assimilation-loss target).
  - Pulled via `copernicusmarine` toolbox (CLI/API).
- **GLORYS12** reanalysis (re-interpolated to 1/4°) — used **only as evaluation ground-truth**, NOT as assimilation source.

**ML stack**
- **PyTorch** (assumed — required for autograd through frozen glonet).
- **TensorBoard** — optimization tracking (loss curves, IC norms, per-iteration diagnostics).

**Pretrained model**
- Canonical: `model/glonet/` (source) + `model/glonet3/glonet2_global_e239_model_package` (weights).

**Dev/UX**
- **Jupyter notebook** during development for interactive exploration; production/batch runs go through SLURM scripts later.

**Critical architectural note:**
- **Assimilation target = real CMEMS observations** (not pseudo-obs from GLORYS12).
- **Evaluation reference = GLORYS12.**
- Strict separation between "what we fit" and "what we evaluate on".

## Phase B — North Star (Q1, captured 2026-05-22)

**Statement (working):**
> *On a held-out evaluation period, the IC produced by gradient descent through a frozen pretrained `glonet` — minimizing observation-mismatch loss over a 7-day assimilation window — yields a 21-day global forecast at 1/4° whose **RMSE against GLORYS12 (re-interpolated to 1/4°)** beats the baseline, and whose **PSD** matches GLORYS12 better than the baseline.*

**Pinned dimensions:**
- **Domain:** global ocean. **Resolution:** 1/4°. **Ground truth:** GLORYS12 (re-interpolated to 1/4°).
- **Skill metric:** RMSE vs GLORYS12. **High-frequency criterion:** PSD analysis (forecast vs ground-truth).
- **Windows (baseline experiment):** 7-day assimilation + 21-day forecast = 28-day total.
- **Loss (Phase 1):** observation-mismatch. **Loss (Phase 2 target):** weak-constraint 4D-Var.

**Open / to-pin items:**
- Baseline definition (vs raw analysis IC? persistence? un-optimized GLORYS12 IC into glonet?).
- Variables in state vector / loss (depends on glonet's I/O).
- Quantitative thresholds: "by ≥ X%" for RMSE; tolerance for PSD match; frequency bands.
- Eval period and geographic stratification.

## Phase U — Summary (M-O-R consolidated 2026-05-22)
- **M (Motivation):** ML ocean emulators have a credibility gap — they miss high-frequency features and instabilities; we lack a method to test whether they actually encode the true dynamics.
- **O (Objective):** Using a **frozen** pretrained glonet, find the IC by gradient descent through the model that fits observations in the assimilation window AND produces the best forecast in the forecast window — with physical consistency on both IC and trajectory. (Structurally: ML-4DVar.)
- **R (Result):** Forecast skill improvement from the optimized IC, **with high-frequency consistency to ground truth**, demonstrating that the method can both diagnose and correct artifacts of the ML emulator.

→ Phase U complete. Next: **Phase B — Blueprint** (North Star → Integrations → SoT → Delivery → Behavioral Rules → Research).

## Phase U — Result (captured 2026-05-22)
- **Primary criterion (necessary):** Forecast initialized from the gradient-optimized IC must be **improved** versus the baseline.
- **Scientific-impact criterion (sufficient for the project's claim):** The improvement must extend to **high-frequency consistency** with ground truth — not only mean-state / low-frequency error reduction.
- **Motivation for that criterion:** ML emulators routinely exhibit **artifacts** (over-smoothing, spurious modes, distorted spectra). A method that diagnoses AND remediates these artifacts is the high-impact contribution.
- **Implied probes (verification time):** power-spectra (temporal/spatial), high-frequency-band error, artifact-localization map, baseline vs optimized side-by-side.

## Phase U — Objective (captured 2026-05-22)
- **Model:** pretrained `glonet` (or newer same-family). **Architecture is fixed — no retraining, no architectural changes.**
- **Hypothesis:** Given the frozen pretrained surrogate, there exists a best initial condition — recoverable by gradient descent through the model — that (a) explains observations inside an assimilation window, and (b) yields the best forecast in the subsequent forecast window.
- **Physical-consistency requirement:** Both the optimized IC AND the model trajectory must be physically plausible (not just loss-minimizing).
- **Target question:** *Can we find a single IC that simultaneously explains observations in the assimilation window AND maximizes forecast skill in the forecast window, using only the pretrained model's gradient?*
- **Framing note:** This is **variational data assimilation (4D-Var)** with a learned forward operator (ML-4DVar / differentiable-emulator DA).

## Phase U — Motivation (captured 2026-05-22)
- **Gap:** Methodological gap between ML-based ocean emulators and the physical criteria required to produce trustworthy data.
- **Symptom:** ML/data-driven ocean models show limitations in representing high-frequency features and instabilities.
- **Approach hypothesis:** Investigate the **sensitivity of the ML model with respect to initial conditions**. If the surrogate truly encodes the dynamics, a physically consistent IC exists that explains the observations.
- **Concrete idea:** Estimate the **optimal initial perturbation** using the **ML model's gradient** (gradient-based optimization through the surrogate). → likely role of `src/gd_optimic/`.
- **Components in play:** `model/glonet/` (surrogate code) + `model/glonet3/` (`glonet2_global_e239` pretrained); `src/gd_optimic/` will host the gradient-based optimal-IC machinery.

## Repository snapshot (2026-05-22, at init)
- `main.py` — stub entry (`print("Hello from bice!")`).
- `pyproject.toml` — name `bice`, Python ≥3.12, no dependencies declared yet.
- `model/glonet/` — model source: `dblock.py`, `fcu.py`, `modelp2.py`, `modules1.py`, `NN.py`, `NO.py`, `sblock.py`, `utility.py`, `utils.py`.
- `model/glonet3/` — `glonet2_global_e239_model_package` (extracted + `.tar.gz`, ~885 MB).
- `src/gd_optimic/` — present but empty.
- `utils/` — empty.
- `README.md` — empty.
- `.gitignore` — Python defaults + `tmp*`, `sb_*`, `.claude/`.

_(Note: the section below was the pre-Research placeholder; superseded by the Research sections above. Kept for audit only.)_

## External references (populated 2026-05-22 via Research R7)
- **GLONET paper (definitive):** El Aouni et al., JGR-MLC 2025 — arXiv 2412.05454.
- **OceanBench (NeurIPS 2025):** github.com/mercator-ocean/oceanbench — likely the right eval harness; user has internal access.
- **FengWu-4DVar (closest architectural analog):** Xiao et al., ICML 2024 — arXiv 2312.12455; github.com/OpenEarthLab/FengWu-4DVar.
- **AI-VarDA modular DA scaffold:** github.com/xiaoyi018/AI-VarDA.
- **4DVarNet (ocean):** github.com/CIA-Oceanix/ocean4dvarnet — modular cost-function design template.
- **Differentiable Veros toy IC-recovery:** arXiv 2511.17427; github.com/team-ocean/veros — MWE template.
- **Weak-constraint 4D-Var w/ ML surrogate:** arXiv 2503.02665.
- **Adjoint via NNs (Hatfield et al. JAMES 2021):** 10.1029/2021MS002521.
- **J_q via NN (Farchi et al. JAMES 2023):** 10.1029/2022MS003474.
- **Spectral diagnostic refs:** FourCastNet 3 (arXiv 2507.12144); Pangu mesoscale KE artifact (PMC12049474); spherical-harmonic loss (arXiv 2501.19374); ocean spectral slopes (Capet/Klein JPO 2013; Storer 2023 Sci Adv).
- **Autodiff stability tips:** adaptive checkpoint adjoint (arXiv 2006.02493); Jacobian regularization (arXiv 2602.04608, 2603.05538).
- **Full bibliography & per-item relevance:** `.tmp/research/R7_prior_art.md`.

## Prior art / reuse candidates (populated 2026-05-22 via Research R7)
1. **FengWu-4DVar / AI-VarDA** — adopt as the architectural skeleton (frozen-emulator autodiff = 4D-Var). Wrap glonet as the forecast plugin.
2. **OceanBench** — adopt as the evaluation harness end-to-end.
3. **4DVarNet modular cost** — mirror their J_obs / J_b / J_q factoring.
4. **Veros differentiable demo** — borrow as a small-region unit-test for our gradient pipeline before scaling to global.
5. **FourCastNet 3 spectral-aware loss** — candidate auxiliary loss for PSD-match.
6. **Adaptive checkpointing + Jacobian regularization** — fallbacks for double-backward instability over 7-day rollouts.

## Research Findings Summary (2026-05-22)

### R1 — Glonet I/O (full report: `.tmp/research/R1_glonet_io.md`)
- **Legacy `model/glonet/` is UNSUITABLE for ML-4DVar.** `modelp2.py` has `.detach()` at lines 54/55/67/68/74/87 plus `gc.collect()` + `torch.cuda.empty_cache()` calls. Gradient flow back to the IC is broken. Using it would violate Behavioral Rule A1 (Frozen-model invariant — cannot modify architecture to "patch" the detach calls).
- **Newer `model/glonet3/glonet2_global_e239` is the candidate.** 96 input channels (80 ocean state + 10 atm forcing + bathymetry); 80 output channels (ocean only); 1440×672 @ 0.25°; 20 depths; 2-step input (t-1, t); trained on up to 10-step rollouts with feedback noise → autograd through rollout was intended.
- **CAVEAT:** package ships weights + metadata only, no source. **Differentiability MUST be verified** with a sanity check (`x.requires_grad=True` → forward N steps → `.backward()` → inspect `x.grad`). This is the gate.
- **New requirement surfaced:** the model consumes **atmospheric forcing** (u10, v10, t2m, mslp, sp at t and t-1). Likely ERA5. Pre-computed `glorys12_*_init_states/` files may bundle this — needs check.
- **State vector x ∈ ℝ^{B × 80 × 672 × 1440}** (ocean only). Reasonable δx₀ subsets: (i) SST + SSH (2 ch), (ii) upper-ocean T/S (16 ch), (iii) full state (80 ch).

### R5 — GLORYS12 status (full report: `.tmp/research/R5_glorys12_status.md`)
- **Raw GLORYS12:** `/Odyssey/public/glonet/raw/glorys12/` — 510 GB, **2021 only**, native 1/12°.
- **Pre-computed 1/4° init_states:** 1993-H1, 2020 full year, 2021 full year (~224 GB), partial 2022-06.
- **Re-interpolation:** xESMF logic exists inside `model/glonet/utility.py` (lines 211–326) with pre-computed weights `xe_weights14/L*.nc`. **No standalone batch script** — needs one to honor the 2-step download/interp decision (non-blocking for 2021).
- **Download bug:** `down_glorsy12_37633.log` failed with an h5py concurrency error (also 37634–37653, 37677–37682). Affects 2020+ but **2021 data is already cached** so not blocking.
- **Eval-period verdict: 2021** — unique year with raw + 1/4° init_states + SSH (Alongtrack) + SST (ODYSSEA) companions.

### R7 — Prior art (full report: `.tmp/research/R7_prior_art.md`)
- 6 reuse candidates ranked; FengWu-4DVar / AI-VarDA + OceanBench are the highest-leverage adoptions.
- Spectral-diagnostic literature gives concrete slope targets (k^-3 QG, k^-5/3 SQG) and a candidate auxiliary loss (spherical-harmonic-decomposed).

## Phase 1 — Baseline `optim.ipynb` ingested (2026-06-11)

**Full extraction:** `.tmp/research/R11_optim_notebook.md`. User-supplied baseline; we will integrate against this rather than rebuild.

### What's already done in the notebook (good news)
- **β rollout implemented.** `forward()` (cell 25) autoregressively chains 7 model_1 calls; gradients flow back to δx₀ in one `.backward()`. R10(c) effectively closed in code.
- **Surface-only scope respected.** Depth packs (model_2/3) loaded but **not in loss, not optimized**.
- **Frozen-model invariant (A1) respected.** glonet weights read-only; only `x0.requires_grad = True`.
- **Constant-step optimizer** as agreed: hand-rolled SGD, `lr=0.1`, 1000 iters, no scheduler. `x ← x − lr·masked_grad`.
- **Warm-start IC from GLORYS12** at `t₀` (not zero/random).
- **Land-mask gradient projection** applied after gradient filtering.

### Surface-channel order — LOCKED (correction to earlier guess)
```
ch 0 → SSH (zos)
ch 1 → THETAO (surface T)
ch 2 → SO (surface S)
ch 3 → UO (surface U)
ch 4 → VO (surface V)
```

### Loss as currently implemented (`J_obs` only)
```
J = Σ_t Σ_c  MSE( ŷ ⊙ obs_mask − target ⊙ obs_mask ) / obs_var
```
- `obs_var = 1.0` scalar (identity R).
- H operator = **channel masking**: SSH (ch 0) and SST (ch 1) observed; ch 2–4 zeroed by mask. No spatial interpolation — obs assumed already on glonet grid.
- Channel weighting `[3.0, 2.0, 1.0, 1.0, 1.0]` (manual) or dynamic per-channel rebalancing.

### Unexpected detail — multi-resolution gradient pooling
- `'pooling'` gradient filter with kernel schedule `[8, 4, 2, 1, …]` applied to ∇x₀ before the SGD step. Implicit multi-resolution smoothing: coarse early, fine late. Worth keeping in mind when we describe the algorithm in any report.

### Data sources (paths verified on disk)
- **Warm-start IC** (GLORYS12 1/4°): `/Odyssey/public/glonet/glorys12_2021-*_init_states/combined_input_*.nc`.
- **SSH obs** ✅ found at `/Odyssey/public/altimetry_traces/2010_2023/gridded/sla_l3_all_2010_2023_0.25deg_convl4.nc` (~106 GB, pre-gridded along-track SLA at 0.25°). **The empty `Alongtrack_SSH_2021-...` directory was a red herring** — actual data is here.
- **SST obs**: `/Odyssey/public/glonet/ODYSSEA_SST_2021-.../combined_obs_*.nc` (gridded L3, lat=680, lon=1440).

### Deviations from earlier binding decisions — RESOLVED 2026-06-11
1. ~~Loss = J_obs only conflicts with weak-4D-Var binding~~ → **RESOLVED.** Loss policy refined: forecast scheme (Phase 1) = `J_obs`; reanalysis scheme = full weak-4D-Var (Phase ≥ 2). See `decisions.md` 2026-06-11 — Loss policy.
2. ~~A2 ambiguity on `target`~~ → **RESOLVED.** Target tensor is mode-dependent via `cfg.obs.mode` ∈ {`full`, `simulated`, `real`}. `full`/`simulated` use GLORYS12 by design (twin / OSSE); only `real` is the headline configuration. See `decisions.md` 2026-06-11 — Observation modes + A2 refinement.
3. **R = identity scalar.** Still acceptable for Phase-1 bootstrap; revisit when reanalysis scheme arrives.
4. **Output dir convention.** Notebook writes to `../outputs/forecast_exp/.../{timestamp}/` — when porting to `src/gd_optimic/`, redirect to `.tmp/outputs/{exp_id}_{date}/`.

### Implications for `src/gd_optimic/` (Phase-1 implementation target)
- The notebook is a **reference**, not the production code. Phase-1 code lives in `src/gd_optimic/`.
- Required modules (sketch):
  - `data/` — dataset that yields per-step `(x_warm_start, target, obs_mask)` tuples; switches on `cfg.obs.mode`.
  - `loss/` — `J_obs` (Phase-1) with channel weighting and gradient pooling; pluggable interface for future J_b / J_q.
  - `optim/` — custom-SGD update loop with the pooling-kernel schedule (mirroring the notebook).
  - `model_wrap/` — instantiation of `GlonetGradientInitialCondition` (already in `model/glonet/`).
  - `runner/` — Hydra-config-driven entrypoint that writes the artifact bundle to `.tmp/outputs/{exp_id}_{date}/`.

### Implications for R10(d) / (e) / (f)
- `cfg.training.loss.J_obs_weight = 1`, `J_b_weight = 0`, `J_q_weight = 0` for Phase-1.a (matches current code). Flip these on for Phase-1.b.
- `cfg.training.optimizer` stays "constant-step custom SGD"; consider exposing the gradient-pooling schedule via config.
- Dataset class (R10(e)) needs to expose: warm-start IC tensor, satellite obs tensors, obs_mask, ocean_mask, current_coords.
- Output directory needs to be redirected to `.tmp/outputs/{exp_id}_{date}/`.

## Phase 1 design — locked decisions (2026-06-11)

| Item | Decision |
|---|---|
| Rollout | **β** — explicit multi-step (Python loop over N_window calls to `gradcheckp_model_1.forward`); 4D-Var semantics |
| Loss | **Manually authored weak-constraint 4D-Var**: `J_b + J_obs + J_q`. Not a torch built-in loss. |
| Optimizer | **Constant-step** (fixed LR, no scheduler) for Phase-1 start |
| Patching | **Region cropping** (spatial domain restriction), NOT FNO/CNN patching |
| Baseline code | **User-supplied file** will drive the optimizer + training loop integration. |

### Implications for R10(d) Hydra config

| Field | Meaning | Phase-1 value |
|---|---|---|
| `cfg.model.checkpoint_paths.part_{1,2,3}` | repo-local `.pth` paths | `model/glonet/weights/glonet_part{1,2,3}.pth` |
| `cfg.model.patch_size` | **region crop**, only used if `enable_patching=True` | full grid `(672, 1440)` initially; region box later (e.g., GS, NATL) |
| `cfg.data.computing.enable_patching` | region-crop on/off | `False` to start (global) |
| `cfg.model.output_path` | optimized-IC NetCDF dir | `.tmp/outputs/{exp_id}_{date}/` |
| `cfg.training.loss` | weak-4D-Var Python class (project-local) | `_target_: bice.losses.WeakConstraint4DVar` (or equivalent) with sub-fields `J_b_weight`, `J_obs_weight`, `J_q_weight`, `R_diag`, `Q_diag` |
| `cfg.training.optimizer` | fixed-LR optimizer | `_target_: torch.optim.SGD`, `lr: <tbd>`, `momentum: 0` (or Adam with no scheduler — set in baseline code) |
| `cfg.training.scheduler` | OMIT in Phase-1 | — |
| `cfg.training.rollout_steps` (new) | β window length, in model steps | `7` (matches 7-day assim window at 1 day/step) |
| `cfg.exp_id`, `cfg.seed`, `cfg.data.period` (new) | required by Behavioral Rule A5 | per-run |

### Implications for R10(e) dataset class

- **`y_k` semantics is now bound** by the manual `J_obs` term: `y` represents **CMEMS observations** within each rollout step's time slice (NOT GLORYS12). Concretely:
  - Per rollout step `k`, the dataset must provide the **set of obs** within the time slice `[t_k - Δ/2, t_k + Δ/2]` and the **H operator** to map model state → obs locations.
  - SSH obs are along-track (irregular x,y,t); SST obs are gridded L3.
  - This is exactly what R2 must settle.
- **`ocean_mask_{1,2,3}`** shapes still as before: `(5,672,1440)`, `(40,672,1440)`, `(40,672,1440)`.
- **`current_coords`** still `{time, lat, lon}` — used for NetCDF coord building on save.
- **The 2nd / 3rd sub-models** (depth packs) under β: either (i) keep their inputs as frozen GLORYS12 reference and DROP `init_input2/3` from the loss, or (ii) drive them with the surface-pack output (cross-pack coupling — depends on how glonet v1's 3 sub-models communicate). Needs inspection of `modelp2.py` interactions OR confirmation from the user.

## Phase 1 override — `model/glonet/glonetLit_grdckpt.py` (dropped in 2026-05-22)

User-supplied override + Lightning wrapper. Key facts read from the file:

### Architecture
- **3 independent Glonet sub-models** (`gradcheckp_model_{1,2,3}`), each loaded from its own checkpoint:
  - `model_1`: `shape_in=(2, 5, 672, 1440)` — **5 surface channels** (4 ocean vars + zos).
  - `model_2`: `shape_in=(2, 40, 672, 1440)` — 40 depth channels (4 vars × 10 levels).
  - `model_3`: `shape_in=(2, 40, 672, 1440)` — 40 depth channels (likely staggered window).
- Override class `GlonetGradientCheckpointing(Glonet)` replaces the legacy `.detach()`-laden `forward` with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` blocks — memory-light AND fully differentiable.
- All sub-model parameters frozen: `param.requires_grad = False` (A1 preserved).
- Single-step forward returns `(y1_hat, y2_hat, y3_hat)`. **No multi-step rollout loop in this file** — open question whether 7-day assim window is implemented elsewhere or needs to be added.

### Lightning module: `GlonetGradientInitialCondition`
- Initializes `init_input{1,2,3}` as `nn.Parameter(..., requires_grad=True)` from the **first batch** (training_step lazy init).
- Loss = `loss_fn(y1_hat,y1) + loss_fn(y2_hat,y2) + loss_fn(y3_hat,y3)` (Hydra-instantiated).
- Land mask applied in `on_after_backward`: `init_input{1,2,3}.grad *= dataset.ocean_mask_{1,2,3}` — gradients zeroed over land. Need a dataset that exposes `.ocean_mask_*` and `.current_coords`.
- Optimizer + scheduler via Hydra config (`cfg.training.optimizer`, `cfg.training.scheduler`).
- Logs `train_loss`, `learning_rate`, and per-input `grad_norm/input{1,2,3}` (good TensorBoard tags for R8).
- `on_train_end` saves `optimal_input{1,2,3}.nc` to `cfg.model.output_path` with metadata (description, creation_date, model config, training_epochs) — matches our artifact-bundle convention.

### Import / portability issue (locator results 2026-05-22)
- Line 16: `sys.path.append(__file__.parent.parent.parent / "src/glonet"); from modelp2 import Glonet`.
- Resolves to `/Odyssey/private/j25lee/src/glonet/modelp2.py` — **path does NOT exist on disk** (verified via `ls`).
- `moiai/glonet/` exists but contains only deployment/orchestration code (Docker, cron, web); no `modelp2.py` anywhere under `moiai/`.
- **Only `Glonet` source on disk:** `bice/model/glonet/modelp2.py` (the repo's own copy).
- **Fix proposal:** replace L16 with `sys.path.append(str(Path(__file__).parent)); from modelp2 import Glonet` — imports from the same directory the override lives in. Simple, no relative-import gotchas, keeps the file script-runnable.

### Checkpoint paths (located 2026-05-22 — `/Odyssey/public/glonet/TrainedWeights/`)
Two candidate sets — user must pick:
- **Set A — `glonet_p{1,2,3}.pt`** (Jul 10 2025): 533 MB / 570 MB / 570 MB → ~1.67 GB total. Older.
- **Set B — `glonet_part{1,2,3}.pth`** (Nov 17 2025): 1134 MB / 1244 MB / 1244 MB → ~3.62 GB total. Newer, larger.
- Normalization stats for the surface block are at `TrainedWeights/L0/` and confirm the 5 surface vars: `{thetao, so, uo, vo, zos}` × `{_mean.npy, _std.npy}` — matches Phase-1 scope exactly.
- Cannot introspect state-dict shapes here (no torch in this shell); checkpoint compatibility with the override's `shape_in=(2,5,672,1440)` is best confirmed by the user.

### Updated Phase-1 I/O contract (supersedes R1's legacy section)
- **State vector for δx₀ (Phase 1):** `(B, 2, 5, 672, 1440)` — **5 surface channels** ordered to be confirmed (likely `thetao_surf, so_surf, uo_surf, vo_surf, zos`).
- **No atmospheric forcing.** No sea-ice. No depth state in the optimized δx₀.
- **Output:** model_1 produces a 5-channel surface prediction at t+1 from inputs at t-1, t.
- **Gradients flow through `torch.utils.checkpoint` (use_reentrant=False)** → double-backward-safe, memory-efficient. This is the cleanest possible setup for ML-4DVar.

## CRITICAL FINDINGS — gate items before any further work (updated 2026-05-22)
1. **Project split into two phases:**
   - **Phase 1 (now):** use **legacy `model/glonet/` (v1) with a user-supplied override** that removes the `.detach()` / `gc.collect()` / `empty_cache()` calls from the forward pass. State vector: ~41 ocean channels × T=2 × H × W. **No atmospheric forcing needed.** Eval period: 2021.
   - **Phase 2 (deferred):** `model/glonet3/glonet2_global_e239` (full atmospheric forcing + sea-ice + Zarr dataset). Source code held by user. Blocked on Zarr-format dataset download (in-flight on IMTA).
   - Decision recorded in `decisions.md` (2026-05-22 — Phase 1 = glonet v1 (override) · Phase 2 = glonet2).
2. **Differentiability sanity check shifts target.** Now run on the **override-version of glonet v1** (once user drops it into `model/glonet/`), not on e239. Smoke test: `x.requires_grad=True` → forward 1 step → `.backward()` → confirm `x.grad` is finite and non-zero.
3. **Forcing question → Phase 2.** Not relevant for Phase 1 since legacy glonet v1 is pure-ocean.
4. **Held-out eval window = 2021.** Pending explicit confirmation from user.
5. **Architectural skeleton** — adopt FengWu-4DVar / AI-VarDA + OceanBench, vs. minimal in-house? Still open.

## External references
