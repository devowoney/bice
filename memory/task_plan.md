# TASK PLAN

## Active Phase
**B — Blueprint** — to begin with North Star (Phase U complete 2026-05-22).

## Checklist

### Phase U — Understand (M-O-R)
- [x] Motivation — captured 2026-05-22 (sensitivity of ML ocean emulator to IC; gradient-based optimal initial perturbation)
- [x] Objective — captured 2026-05-22 (ML-4DVar-style: best IC via gradient through frozen pretrained model; must satisfy physics in IC and forecast; covers both assimilation and forecast windows)
- [x] Result — captured 2026-05-22 (forecast improvement WITH high-frequency consistency; diagnose+remediate ML artifacts)

### Phase B — Blueprint
- [x] (a) North Star — captured 2026-05-22 (global 1/4°; GLORYS12 truth; RMSE + PSD; 7d assim + 21d forecast; obs-mismatch loss → weak-constraint 4D-Var)
- [x] (b) Integrations — captured 2026-05-22 (IMTA SLURM; CMEMS obs for assim; GLORYS12 for eval ONLY; PyTorch + TensorBoard; Jupyter for dev)
- [x] (c) Source of Truth — captured 2026-05-22 (`/Odyssey/public/glonet/` root, NetCDF→Zarr-ready, 2-step download/interpolate under 500 GB/node, OSE+OSSE, artifact promotion lifecycle)
- [x] (d) Delivery Payload — captured 2026-05-22 (per-exp bundle in `.tmp/outputs/{exp_id}_{date}/` with obs.nc, Jupyter+MD per-exp report; group PDF + superposed plots + mean/spread at promotion)
- [x] (e) Behavioral Rules — captured 2026-05-22 (Scientific Integrity A1–A8 + reproducibility + safety + tone; (e) project-specific physical floor + license + disclosure placeholders carry into Research)
- [~] Research — R1 ✅, R5 ✅, R7 ✅ done 2026-05-22 (reports in `.tmp/research/`, summary in `findings.md`); R2/R3/R4/R8 pending user input.
  - [x] R1 — Glonet I/O contract (legacy glonet v1 = Phase 1 target; e239 = Phase 2)
  - [ ] R2 — Observation handling (SSH along-track via NN, SST direct pixel, GLORYS12 twin, light QC) ✅ 2026-06-12
  - [ ] R3 — Baseline definition
  - [ ] R4 — Metrics & thresholds
  - [x] R5 — GLORYS12 status (2021 ready)
  - [x] R6 — Drop-in of override `glonet` v1 → `model/glonet/glonetLit_grdckpt.py` (DONE 2026-05-22)
  - [x] R7 — Prior art
  - [ ] R8 — Tracking conventions (override already gives us: `train_loss`, `learning_rate`, `grad_norm/input1` — refine)
  - [ ] R9 — Differentiability sanity check on `GlonetGradientCheckpointing` (model_1 only) — agent task, awaiting user OK + checkpoint paths
  - [~] R10 — Phase-1 prerequisites:
    - [x] (a) fix `sys.path` import in glonetLit_grdckpt.py L16 → repo-local `modelp2.py` (2026-05-22)
    - [x] (b) checkpoints: set-B `glonet_part{1,2,3}.pth` copied to `model/glonet/weights/` (verified byte-exact 2026-05-22)
    - [x] (c) rollout = β (explicit multi-step 4D-Var) — locked 2026-06-11; already implemented in `optim.ipynb` cell 25
    - [~] (d) Hydra config — drafting depends on resolving A2 ambiguity + J_b/J_q reconciliation (see decisions 2026-06-11)
    - [~] (e) dataset class — semantics now clear; needs: warm-start IC, SSH satellite tensor, SST satellite tensor, obs_mask, ocean_mask, current_coords
    - [x] (f) baseline optimization code ingested 2026-06-11 — `R11_optim_notebook.md`
    - [x] (g) loss policy resolved 2026-06-11 (Path A: J_obs for forecast scheme, full 4D-Var for reanalysis)
    - [x] (h) A2 ambiguity resolved 2026-06-12 — target is mode-dependent; all modes A2-compliant within labelled scope
    - [x] (g) loss policy resolved 2026-06-12 (Path A: J_obs for Phase-1.a; J_b+J_q deferred to Phase-1.b)
    - [x] R3 — Baseline definition resolved 2026-06-12 (forecast scheme: persistence; reanalysis: TBD)
    - [ ] (i) redirect notebook output dir to `.tmp/outputs/{exp_id}_{date}/` convention (will happen as part of port to `src/gd_optimic/`)
    - [ ] (j) author `src/gd_optimic/` package structure from notebook reference (Phase-1 implementation)

### Phase P — Professor
- [ ] Verification artifact (test / screenshot / one-liner)
- [ ] Explicit sign-off captured → `decisions.md`
- [ ] Visualization presented

### Phase T — Trigger
- [ ] Transfer to production parity
- [ ] Automation/firing mechanism documented
- [ ] Maintenance Log finalized in `CLAUDE.md`
- [ ] Self-Annealing repair loop ready

## Upcoming
- Define data schema (input/output) once Phase B answers are in.
- **Open Phase B research items (carry into Research step):**
  - Enumerate exact glonet I/O variables & tensor shapes (to define the IC vector that gradient descent will optimize).
  - Detail observation handling: H operator, QC/masking, error model, binning for OSE; twin-generation pipeline for OSSE.
  - Confirm GLORYS12 native → 1/4° re-interpolation script status.
  - Locate `glonet` (older) weights → move into `model/glonet/weights/` (user task).
