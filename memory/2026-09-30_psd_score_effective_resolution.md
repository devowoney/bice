# PSD score and effective resolution — method, results, limits

*2026-09-30 · branch `worktree-paper-visualization` (uncommitted) · code: `src/gd_optimic/spectral_scores.py`*

This note explains the spectral skill diagnostic that replaced the local-patch EBCR in `OutputHandler.save_outputs()`: what it measures, why it is computed on segments, how it was validated, what it shows on the `table1_3dAssim_2026-09-24_08-48-56` runs, and what is still open.

---

## 1. Why the previous spectral diagnostics were replaced

| Diagnostic | What it did | Problem |
|---|---|---|
| Global radial PSD (`psd_analysis.csv`, still produced) | One 2D FFT over the whole globe, land set to 0, one Hann window over the map | Land = 0 creates steps at every coastline → broadband spurious small-scale energy. Wavenumber in cycles/degree mixes latitudes whose km-per-degree differ by cos(lat). |
| Local-patch EBCR (removed) | `(PSD_opt − PSD_truth) / (PSD_ref − PSD_truth)` on 200 km patches (8×8 px) | Compares spectral **amplitudes only**: a forecast with the right eddy energy in the wrong place scores perfectly. An 8×8 patch has ~4 radial bins and cannot resolve > 200 km, yet was reported on 65–500 km. |

Both are known ways of producing spectra that look informative but are not interpretable as scale-dependent skill.

## 2. What the PSD score measures

Following the ocean-data-challenges mapping studies (`2023a_SSH_mapping_OSE`, `src/mod_spectral.py`):

$$
\text{Score}(k) = 1 - \frac{\mathrm{PSD}\big(\text{forecast} - \text{truth}\big)(k)}{\mathrm{PSD}\big(\text{truth}\big)(k)}
$$

- The numerator is the spectrum of the **error field**. Misplaced features produce error even when their amplitude is right, so the score penalises **phase (position) and amplitude** errors together.
- Score = 1: perfect at that scale. Score = 0: error energy equals signal energy (no skill). Score < 0: the forecast adds energy absent from the truth.
- **Effective resolution λ_eff** = wavelength where the score crosses 0.5 (error variance = half the signal variance). Scales larger than λ_eff are resolved. The crossing is the first one scanning from large to small scales, interpolated linearly in log-wavenumber (identical to `compute_crossing` in `mod_spectral.py`).
- Truth = GLORYS12. λ_eff is therefore *relative to GLORYS12*, not to the real ocean.

## 3. Why segments (pseudo-tracks)

The OSE challenge computes spectra on 1000 km pieces of altimeter tracks. Here the truth is a **full gridded state** — equivalent to perfect, noise-free observations everywhere — so the same pieces are cut directly out of the grid ("pseudo-tracks"). This solves several problems at once:

1. **No land in any spectrum.** A segment is kept only if it is 100 % ocean, so coastline leakage is removed by construction rather than corrected afterwards.
2. **A true km axis.** Meridional spacing is a constant 0.25° = 27.83 km. Zonal spacing is 0.25° · cos(lat), constant along each latitude row; each row uses its own point count for 1000 km and its spectrum is interpolated (log-log) onto the common grid k = m / 1000 km. Fields are never resampled, so no aliasing at high latitude.
3. **Finite, windowable records.** The FFT assumes periodicity. A 1000 km, Hann-windowed, mean-removed segment satisfies it; a global map with land and a latitude-dependent grid does not.
4. **Local statistics.** Western boundary currents, the equator and the gyres have different dynamics. Segments can be pooled by region or by sliding box instead of blending the globe into one curve.
5. **Comparability.** Segment length, overlap, window, detrending and threshold follow the challenge, so λ_eff is defined like published effective resolutions (e.g. Ballarotta et al., 2019, for DUACS).

### Settings

| Setting | Value | Reason |
|---|---|---|
| Segment length | 1000 km (36 points meridionally) | As the challenge. Resolvable range 55.7 km (Nyquist, 2 × 27.83 km) to 1001.9 km |
| Overlap | 75 % (stride = npt / 4) | As `segment_overlapping=0.25` |
| Spectrum | `scipy.signal.welch(fs=1/dx, nperseg=npt, noverlap=0, window='hann', detrend='constant', scaling='density')` | As `mod_spectral` |
| Directions | Meridional + zonal, pooled | More segments, closer to isotropic |
| Fields | truth, reference, optimized, ref − truth, opt − truth | Error spectra per forecast |
| Lead days | 1, 3, 5, 7, 14, 21, 28 (`DEFAULT_TIMESTEPS`) | Errors saturate within ~1 week, early days carry the information |
| Regions | global, gulf_stream, high_var, low_var (from `MaskBuilder.build_regional_masks`) | A segment belongs to a region if its centre does |
| Maps | 10° × 10° boxes slid on a 1° step, ≥ 3 segments per box | As `mod_spectral` (`vlat/vlon = arange(…, 1)`, box = centre ± 5°). Smooth, but true map resolution ≈ 10° |
| Map land mask | Box map put on the native 0.25° grid: each ocean pixel takes its nearest box value, land = NaN (`boxes_to_native`); land patches ≤ 9 px (3 × 3, ~80 km) are drawn as ocean (`MAP_MAX_ISLAND_PIXELS`) | Boxes centred on land still contain ocean segments and would blur coastlines by up to ~5°; values are unchanged, only the coastline becomes the model's own. The ~210 one-pixel model islands would otherwise speckle the open ocean (display only; segments still exclude them) |
| Ocean mask (backfill) | Land = NaN or 0 at **every** loaded lead day of the ground truth | A single-day `!= 0` test marked ocean pixels that are exactly 0.0 that day as land (~1 500 px for U/V), producing spurious one-pixel holes and breaking segments |
| Out of range | `unresolved` (score < 0.5 even at 1000 km) → stored as 1001.9 km; `resolved_all` (score ≥ 0.5 down to 56 km) → stored as 55.7 km; flagged in `status` | Keeps every scored ocean box on the map; the flag keeps censored values distinguishable |

## 4. Validation

### 4.1 Method checks against known answers

| Test | Expected | Obtained |
|---|---|---|
| Forecast = truth with all scales < 200 km removed (separable red-noise field) | λ_eff ≈ 200 km | **192 km** (≈ 4 % Hann-window leakage bias) |
| Same with a land block inserted | Unchanged | **192 km** (17 020 vs 20 000 segments) — land does not contaminate |
| Perfect forecast | Fully resolved | `resolved_all` |
| Crossing interpolation on an analytic ratio | 200 km | **200.4 km** |
| Vectorised map crossing vs scalar `compute_crossing` (5000 random curves) | Identical | **0 mismatches** |

### 4.2 Physical plausibility on real runs

- **Truth spectrum shape.** The global SSH truth PSD falls as ≈ k^-2.9 from 1000 to 100 km, a normal slope for a global average of energetic and quiet regions.
- **Monotonic error growth.** λ_eff grows steadily with lead time (Table 1), as expected from forecast error cascading to larger scales.
- **Consistency with RMSE.** U and V become unresolved at every scale by day 3–5, matching their RMSE curves saturating near 0.2 m s⁻¹.
- **Link to the known pixelization artefact.** Below ≈ 100 km the forecast spectrum flattens into a noise floor (≈ 7× GLORYS12 energy at 67 km), driving the score strongly negative. This is the same small-scale emulator artefact as `memory/2026-09-24_pixelization_checkerboard_diagnostic.md` — the score detects an independently known problem.
- **Spatial pattern.** Maps are coarse in the equatorial band and fine at mid/high latitudes, the same latitude dependence reported for DUACS altimetry maps (Ballarotta et al., 2019).

## 5. Results (`table1_3dAssim_2026-09-24_08-48-56`)

**Table 1 — Global SSH effective resolution, run 0 (reference → optimized)**

| Lead day | 1 | 3 | 5 | 7 | 14 | 21 | 28 |
|---|---|---|---|---|---|---|---|
| λ_eff (km) | 86 → 86 | 243 → 241 | 433 → 432 | 684 → 686 | > 1000 | > 1000 | > 1000 |

**Table 2 — Global λ_eff at day 7, run 0 (reference → optimized)**

| Variable | SSH | T | S | U | V |
|---|---|---|---|---|---|
| λ_eff (km) | 684 → 686 | 646 → 646 | 462 → 461 | > 1000 | > 1000 |

**Table 3 — Share of ocean map boxes that are unresolved (> 1000 km), SSH, run 0 (46 939 scored boxes)**

| Lead day | 1 | 3 | 5 | 7 | 14 | 21 | 28 |
|---|---|---|---|---|---|---|---|
| Unresolved | 0.2 % | 4 % | 12 % | 31 % | 58 % | 68 % | 74 % |

Over the 3 runs checked so far, the optimized IC is **≈ 5–20 km finer than the reference on days 1–5** (all variables) and indistinguishable afterwards. The 28-run aggregate (`aggregate_all_v3.py`, figures `effective_resolution_{region}` and `psd_score_{region}_t{day}`) is pending the backfill re-run.

Headline: the IC optimization barely changes the scale-dependent skill. Skill is lost at all scales below ~700 km within one week for SSH, and within 3–5 days for velocities. The dominant small-scale deficiency is the emulator's noise floor below ~100 km, which the IC cannot fix.

## 6. Limits and open points

1. **Significance of the optimized gain.** The 5–20 km gain on days 1–5 needs a paired test across the 28 runs (per-run differences, not overlapping ± std bands).
2. **Single-snapshot estimates.** Each lead day is one daily field; the challenge pools a year. Per-box map values are noisier, especially in small boxes.
3. **Upper censoring at 1000 km.** From ~day 7 most values sit at the bound. Longer segments (e.g. 2000 km) would extend the range at the cost of coastal and small-basin coverage.
4. **Isotropy assumption.** Meridional and zonal segments are pooled. Scoring them separately would show whether this matters, especially near the equator, where zonal and meridional dynamics differ.
5. **Truth is not the ocean.** GLORYS12 has its own effective resolution; λ_eff is relative to it.
6. **Sensitivity not yet tested.** Segment length, linear vs constant detrend, window choice.
7. **Map resolution ≈ 10°.** The 1° sliding step makes maps smooth, not sharper. The native land mask gives exact coastlines, but a coastal pixel still carries the value of a 10° box that mixes open-ocean segments; ~2 % of ocean pixels (small enclosed seas, grid edges) have no box with ≥ 3 segments and stay NaN.
8. **Hann bias.** A sharp cutoff is recovered ~4 % too fine (192 vs 200 km); acceptable, but a known systematic.
9. **Coherence not computed.** The challenge also computes spectral coherence, which separates phase from amplitude error — a natural next diagnostic.

## 7. Outputs and how to run

Per run (written by `save_outputs` for new runs, or by the backfill for finished ones):

| File | Content |
|---|---|
| `metrics/effective_resolution.csv` | Per region × lead day × variable: `lambda_eff_{reference,optimized}_km`, `n_segments`, `status_*` |
| `metrics/psd_segment_scores.csv` | Per wavelength (56–1000 km): PSD of truth, reference, optimized, both errors; `score_*` |
| `metrics/effective_resolution_maps.nc` | 1° box maps (timestep, variable, lat, lon): λ_eff, `n_segments`, `status_*` (−1 no data, 0 ok, 1 multiple, 2 resolved_all, 3 unresolved); zlib, ≈ 13 MB |
| `metrics/effective_resolution_maps_native.nc` | Same λ_eff maps on the native 0.25° grid with the model land mask (land = NaN); zlib, ≈ 17 MB |
| `metrics/effective_resolution.json` | Method settings + λ_eff table |
| `diagnostics/effective_resolution_map_{var}.png` | Quick-look maps from the native-grid product, rows = lead days, columns = reference / optimized |

Backfill finished runs from their `states/*.nc`:

```bash
cd /Odyssey/private/j25lee/bice/.tmp/multiruns
/Odyssey/private/j25lee/miniforge3/envs/oceanai/bin/python \
  /Odyssey/private/j25lee/bice/.claude/worktrees/paper-visualization/.tmp/multiruns/backfill_effective_resolution.py \
  table1_3dAssim_2026-09-24_08-48-56 [--timesteps 1 3 5 7] [--skip-existing] [--out-root DIR]
```

Aggregate paper figures across runs: `aggregate_all_v3.py <multirun> -o <out>` → `aggregated_diagnostics/effective_resolution_{region}.{pdf,png}`, `psd_score_{region}_t{day}.{pdf,png}`, `effective_resolution_aggregated.csv`.

## References

- ocean-data-challenges, *2023a_SSH_mapping_OSE*, `src/mod_spectral.py` and `nb_diags_global/` — https://github.com/ocean-data-challenges/2023a_SSH_mapping_OSE
- Ballarotta, M. et al. (2019). On the resolutions of ocean altimetry maps. *Ocean Science*, 15, 1091–1109.
