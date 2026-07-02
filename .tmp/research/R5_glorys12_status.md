# R5 — GLORYS12 Status (research, 2026-05-22)

## On-disk inventory

### Raw 1/12° GLORYS12 (`/Odyssey/public/glonet/raw/glorys12/`, ~510 GB)

- `surface/` — 1 file (2021 only): `glorys_surface_2021-01-01_to_2021-12-31.nc` (~30 GB).
- `depths/` — 20 files (2021), one per depth level [0.49 m … 763.33 m].
- `static/` — `glorys_static_bathymetry.nc`.
- **Total: 22 NetCDF files, 2021 only.** Years 2014–2020 were attempted but write FAILED in recent jobs.

### Pre-computed 1/4° init_states (`/Odyssey/public/glonet/glorys12_*_init_states/`)

| Period | Layout | Total size |
|---|---|---|
| 1993-01-01 → 1993-06-30 | 1 large `combined_input.nc` (111 GB) + 3 input components (HDF5) | ~112 GB |
| 2020-01-01 → 2020-12-31 | 13 monthly-window files `combined_input_YYYY-MM-DD_to_*.nc` (~19 GB each) + stats subdir | ~225 GB |
| 2021-01-01 → 2021-12-31 | 8 chunked period files (~27–29 GB each) + stats subdir (628 MB `mean.nc`) | ~224 GB |
| 2022-06-01 | inaccessible (permission) — likely similar mid-2022 | unknown |

These init_states are already at 1/4° (per `utility.py:212` step = 1/4 = 0.25°).

## Download job (log 37633)

- **Status:** FAILED on 2026-05-22 at 16:02 UTC (after 7m56s).
- **Product:** `GLOBAL_MULTIYEAR_PHY_001_030`, version 202311 (inferred).
- **Variables:** `zos, uo, vo, so, thetao` across multiple depths.
- **Error:** `RuntimeError: Can't synchronously determine if attribute exists by name (invalid identifier type to function)` — h5py concurrency / file-locking issue inside copernicusmarine toolbox during xarray-to-NetCDF write. Not data corruption.
- **Script:** invoked via `/Odyssey/private/j25lee/moiai/glonet2/.venv/bin/download_glorys`; source at `/Odyssey/private/j25lee/moiai/glonet2/src/glorys_download.py` (and `ocean_download.py`).
- **Pattern:** multiple recent jobs (37633, 37634–37636, 37645–37653, 37677–37682) share the same h5py failure.

## Re-interpolation pipeline (1/12° → 1/4°)

- **Already implemented in legacy code:** `model/glonet/utility.py` lines 211–226, 264–280, 310–326.
- **Tooling:** xESMF (`xe.Regridder(..., 'bilinear', weights_fn, reuse_weights=True)`).
- **Pre-computed weights:** `xe_weights14/L*.nc`.
- **Mechanism:** per-depth regrid, concatenated; called on-the-fly during model input loading.
- **MISSING:** a standalone batch preprocessing script. The regridding is currently fused into model input loading — fine for inference, but not what the project's "2-step download/interpolate" decision intends. For batch reference-data generation we still need a dedicated 1/12° → 1/4° preprocessing script that writes intermediate `.nc`/`.zarr` files.

## Eval-period options

| Year | Raw GLORYS12 | 1/4° init_states | Obs companions (SSH, SST) | Verdict |
|---|---|---|---|---|
| **2021** | ✅ full year | ✅ 8 chunked files | ✅ `Alongtrack_SSH_2021-...`, `ODYSSEA_SST_2021-...` | **READY — recommended** |
| 2020 | failed download | ✅ 13 monthly files | ❌ no SSH/SST 2020 dirs | partial — no obs |
| 1993 H1 | ❌ absent | ✅ 1 large file | ❌ none | climatology only |

## Implication

- **2021 is the unambiguous held-out eval period.** Raw, 1/4°, and obs companions all exist.
- **xESMF regridding is operational** in legacy code; reuse the regridder logic, but factor it into a standalone preprocessing script per the 2-step rule.
- **Action items before any run:**
  1. Verify `xe_weights14/L*.nc` exist and are readable.
  2. Build a standalone batch regridder (eventually — not blocking 2021 work since init_states already exist).
  3. Pin held-out eval window inside 2021 (suggestion: keep some 2021 dates untouched until publication; develop on a different 2021 subset).
  4. Don't block on the download bug — 2021 is sufficient; surface the h5py issue to whoever maintains `moiai/glonet2/src/glorys_download.py`.
