# R7 — Prior Art (research, 2026-05-22)

## Bucket A — Differentiable / ML-based 4D-Var frameworks

- **4DVarNet (core + ocean4dvarnet)** — learned variational scheme: NN prior + learned solver unrolling 4D-Var gradient descent. Origin: SSH/altimetry interpolation; now generalized.
  - Fablet, Beauchamp et al. 2021–2023. GMD 2023: https://gmd.copernicus.org/articles/16/2119/2023/
  - Repos: https://github.com/CIA-Oceanix/4dvarnet-core, https://github.com/CIA-Oceanix/ocean4dvarnet
  - **Relevance:** most mature open-source ocean variational-ML codebase; modular cost-function design (J_obs + learned prior) is a direct template for our J_obs ⊕ J_b ⊕ J_q.
- **FengWu-4DVar** — couples a frozen pretrained AI weather model with classical 4D-Var via autodiff (no hand-coded adjoint); >1 year of stable cycling.
  - Xiao et al. ICML 2024. arXiv: https://arxiv.org/abs/2312.12455. Repo: https://github.com/OpenEarthLab/FengWu-4DVar
  - **Relevance:** closest architectural analog to `bice` — frozen pretrained emulator + autodiff = 4D-Var. Their gradient-equivalence proof (autodiff = adjoint) is reusable justification.
- **AI-VarDA** — modular DA framework unifying FengWu-4DVar, VAE-Var, LoRA-EnVar with pluggable forecast/B-matrix/obs-operator interfaces.
  - Repo: https://github.com/xiaoyi018/AI-VarDA
  - **Relevance:** ready scaffold; wrap glonet as the "forecast model" plugin and inherit cycling.
- **DiffDA** — diffusion model conditioning on sparse obs at inference (GraphCast backbone). Not 4D-Var.
  - Huang et al. ICML 2024. arXiv: https://arxiv.org/abs/2401.05932
  - **Relevance:** contrasting paradigm; baseline/foil.
- **Variational DA with Learned Inverse Observation Operator** — reformulates the loss in physics-space via a learned obs-inverse.
  - Frerix, Kochkov, Smith, Cremers, Brenner, Hoyer. ICML 2021. arXiv: https://arxiv.org/abs/2102.11192. Code: https://github.com/googleinterns/invobs-data-assimilation
  - **Relevance:** can warm-start the IC if our sparse SSH/SST/Argo H operator is ill-posed.
- **Building TL/Adjoint Models with NNs** — Hatfield, Chantry, Dueben, Lopez, Geer, Palmer. JAMES 2021. https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002521
  - **Relevance:** principled background on NN-as-adjoint; underwrites autodiff approach.
- **Reparameterizing 4DVAR with Neural Fields** — represents the IC as a neural field; spectral-bias as implicit regularizer.
  - arXiv 2025: https://arxiv.org/abs/2509.21751
  - **Relevance:** drop-in replacement for control variable if direct IC gradients are pathological.

## Bucket B — Glonet & Mercator-Ocean lineage

- **GLONET paper (JGR-MLC 2025)** — hierarchical transformer + neural-operator backbone trained on GLORYS12; 1/4° global, 10-day forecasts.
  - El Aouni et al. 2025. arXiv: https://arxiv.org/abs/2412.05454. Journal: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000686
  - **Relevance:** definitive reference for the frozen forward operator.
- **OceanBench (NeurIPS 2025)** — Mercator's open benchmark; tracks vs reanalysis / analysis / Class-4 obs; includes GLO12, GLONET, XiHe, Wenhai baselines.
  - Repo: https://github.com/mercator-ocean/oceanbench. Poster: https://neurips.cc/virtual/2025/poster/121394
  - **Relevance:** direct evaluation harness; Class-4 / IV-TT protocol fits our RMSE + PSD goal.
- **OceanForecastBench** — companion/alternative ML ocean benchmark. arXiv: https://arxiv.org/pdf/2511.18732
- **Eddy-Resolving Global Ocean Forecasting with Multi-Scale GNNs** — adjacent ML emulator with explicit PSD at 60-day rollout. arXiv: https://arxiv.org/abs/2601.12775
- **Mercator press release on GLONET/OceanBench (Dec 2025)** — https://www.mercator-ocean.eu/press-release/mercator-ocean-international-makes-ai-ocean-forecasting-operational-with-glonet-validated-by-oceanbench-at-neurips-2025/
- **Not verified:** no public DA-using-glonet paper found. Likely internal Mercator artifact — user is the source of truth.

## Bucket C — Optimization-through-learned-physics for ICs

- **Differentiable Veros (JAX)** — autodiff ocean model with toy IC-correction inverse-problem demo.
  - arXiv 2025: https://arxiv.org/abs/2511.17427. Repo: https://github.com/team-ocean/veros
  - **Relevance:** minimal MWE template for "recover IC by gradient descent through ocean dynamics."
- **Online Model Error Correction with NNs in Incremental 4D-Var** — Farchi, Laloyaux, Bonavita, Bocquet. JAMES 2023. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003474
  - **Relevance:** textbook reference for the J_q model-error term.
- **Weakly-Constrained 4D-Var for Downscaling with Surrogates** — arXiv: https://arxiv.org/abs/2503.02665
  - **Relevance:** closest published recipe for J_obs + J_b + J_q with an ML surrogate.
- **Adjoint-Matching NN Surrogates for Fast 4D-Var** — arXiv: https://arxiv.org/abs/2111.08626
  - **Relevance:** backstop if direct autodiff through glonet is too heavy.
- **Convergence of ML and DA in Earth system science (npj AI 2026)** — https://www.nature.com/articles/s44387-026-00107-0
  - **Relevance:** orientation review.
- **Practical autodiff stability tips:**
  - Adaptive Checkpoint Adjoint — arXiv: https://arxiv.org/abs/2006.02493
  - Jacobian regularization for long NeuralODE rollouts — arXiv: https://arxiv.org/abs/2602.04608
  - JAWS spatially-adaptive Jacobian regularization for neural operators — arXiv: https://arxiv.org/abs/2603.05538
  - **Relevance:** remedies if double-backward through 7-day glonet rollout is unstable / OOM.

## Bucket D — Spectral / high-frequency diagnostics for ML emulators

- **FourCastNet 3** — angular PSD vs ERA5; CRPS loss + geometric design retains spectra to 60-day lead. arXiv: https://arxiv.org/html/2507.12144v1
- **Pangu vs MPAS mesoscale KE spectra** — Pangu fails to reproduce -5/3 slope; underestimates KE <1000 km. PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12049474/
  - **Relevance:** canonical example of the artifact `bice` aims to remediate.
- **JAWS / Jacobian regularization for spectral stability** — arXiv: https://arxiv.org/abs/2603.05538
- **Thermalizer** — stable autoregressive emulation for chaotic spatiotemporal systems. arXiv: https://arxiv.org/abs/2503.18731
- **Capet/Klein et al. ocean mesoscale/submesoscale spectral references** — k^-3 (QG), k^-5/3 (SQG), k^-2 (IGW): JPO 2013 https://journals.ametsoc.org/view/journals/phoc/43/11/jpo-d-13-063.1.xml ; Drake Passage https://pordlabs.ucsd.edu/sgille/pub_dir/jpo-d-15-0087.pdf
  - **Relevance:** slope targets for `bice`'s PSD diagnostic at 50–500 km.
- **Storer et al. 2023 — Global ocean KE cascade** — Sci Adv: https://www.science.org/doi/10.1126/sciadv.adi7420
- **"Fixing the Double Penalty" via spherical-harmonic loss** — arXiv: https://arxiv.org/abs/2501.19374
  - **Relevance:** candidate auxiliary loss to push glonet output toward correct PSD.

## Reuse recommendations (ranked by leverage)

1. **Adopt FengWu-4DVar / AI-VarDA as the skeleton.** They already solve "autodiff through frozen pretrained model = 4D-Var gradient." Swap forecast plugin → glonet.
2. **Use OceanBench as the evaluation harness.** Three tracks (reanalysis / analysis / Class-4) and process diagnostics align with `bice`'s RMSE + PSD goals; user has internal access.
3. **Mirror 4DVarNet's modular cost-function** for our J_obs ⊕ J_b ⊕ J_q split.
4. **Borrow the Veros differentiable-IC-recovery toy** as a unit-test template — validates the gradient implementation on a small region before scaling.
5. **Add a spectral auxiliary loss** (FourCastNet 3 / spherical-harmonic decomposed) — directly serves the PSD-match North Star.
6. **Apply Jacobian regularization / adaptive checkpointing** as fallbacks when double-backward over 7-day rollouts becomes unstable.

## Caveats

- No public repo/paper for `bice`-style DA on glonet — flag for user (likely internal).
- Some surveyed papers are arXiv-only (2509.21751, 2503.02665, 2511.17427, 2601.12775).
- The Wiley JGR-MLC GLONET paper may need institutional access (user has it).
