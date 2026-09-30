"""
Effective spatial resolution from pseudo-track wavenumber spectra.

Gridded adaptation of the ocean-data-challenges 2023a_SSH_mapping_OSE spectral
diagnostic (src/mod_spectral.py):

- The OSE challenge cuts 1000 km along-track segments out of altimeter tracks.
  Here the truth (GLORYS12) is a full gridded state, so segments are cut out of
  the grid itself along meridional and zonal pseudo-tracks. Segments must be
  100% ocean, so no land-zeroing / coastline leakage enters the spectrum.
- Each segment spectrum uses scipy.signal.welch(fs=1/dx, nperseg=npt,
  noverlap=0, scaling='density') (Hann window, constant detrend), as in the
  challenge. Segments overlap by 75 %.
- Meridional spacing is uniform (0.25 deg = 27.83 km). Zonal spacing is
  0.25 deg * cos(lat), so each latitude row uses npt = round(L / dx) at its
  native spacing and its spectrum is interpolated (log-log) onto the common
  meridional wavenumber grid k = m / L. No resampling of the fields is needed.
- Score(k) = 1 - PSD(forecast - truth) / PSD(truth). The effective resolution
  is the wavelength where PSD(err)/PSD(truth) crosses 0.5, interpolated
  linearly in log-wavenumber (compute_crossing in mod_spectral).

Input for one lead time: truth/reference/optimized fields [C, H, W], ocean mask
[C, H, W] (bool), lat [H], lon [W] (degrees, regular global grid).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import welch

KM_PER_DEG = 111.32
SEGMENT_LENGTH_KM = 1000.0
SEGMENT_STRIDE_FRACTION = 0.25  # 75 % overlap, as segment_overlapping=0.25 in the challenge
SCORE_THRESHOLD = 0.5
LAMBDA_MIN_KM = 65.0
LAMBDA_MAX_KM = 500.0
MIN_SEGMENTS_PER_BOX = 3
# Map: 10x10 deg boxes slid on a 1 deg step, as mod_spectral (vlat/vlon = arange(..., 1), box = centre +- 5 deg).
# Neighbouring boxes overlap by 9 deg, so the map is smooth but its true resolution is ~10 deg.
BOX_DEG = 10.0
BOX_STEP_DEG = 1.0
# Map display: land patches up to this many native pixels (3x3 = ~80 km) are drawn as ocean
MAP_MAX_ISLAND_PIXELS = 9
# Map status codes; censored boxes hold the bound (Nyquist / segment length) as lambda
MAP_STATUS = {"no_data": -1, "ok": 0, "multiple": 1, "resolved_all": 2, "unresolved": 3, "nan": -1}
# Forecast errors saturate within ~1 week (lambda_eff > segment length), so the
# early lead days carry most of the information.
DEFAULT_TIMESTEPS = (1, 3, 5, 7, 14, 21, 28)
FIELDS = ("truth", "reference", "optimized", "err_reference", "err_optimized")


def _valid_window_starts(valid: np.ndarray, npt: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """(row, start) of windows of length npt that are fully valid along axis 1."""
    csum = np.concatenate(
        [np.zeros((valid.shape[0], 1), dtype=np.int32), np.cumsum(valid, axis=1, dtype=np.int32)],
        axis=1,
    )
    full = (csum[:, npt:] - csum[:, :-npt]) == npt
    full[:, np.arange(full.shape[1]) % stride != 0] = False
    return np.nonzero(full)


def _segment_psds(
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: np.ndarray,
    starts: np.ndarray,
    npt: int,
    dx_km: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Welch PSD per segment for truth, reference, optimized and both errors."""
    idx = starts[:, None] + np.arange(npt)[None, :]
    truth, ref, opt = (a[rows[:, None], idx].astype(np.float64) for a in arrays)
    segments = {
        "truth": truth,
        "reference": ref,
        "optimized": opt,
        "err_reference": ref - truth,
        "err_optimized": opt - truth,
    }
    psds = {}
    k = None
    for name, seg in segments.items():
        k, p = welch(seg, fs=1.0 / dx_km, nperseg=npt, noverlap=0,
                     window="hann", detrend="constant", scaling="density", axis=-1)
        psds[name] = p[:, 1:]  # drop k = 0
    return k[1:], psds


def _to_common_grid(k_src: np.ndarray, p_src: np.ndarray, k_common: np.ndarray) -> np.ndarray:
    """Log-log interpolation of segment spectra [n, nk_src] onto k_common."""
    if k_src.size == k_common.size and np.allclose(k_src, k_common):
        return p_src
    logp = np.log(np.clip(p_src, 1e-300, None))
    out = np.empty((p_src.shape[0], k_common.size))
    lk_src, lk_dst = np.log(k_src), np.log(k_common)
    for i in range(p_src.shape[0]):
        out[i] = np.interp(lk_dst, lk_src, logp[i])
    return np.exp(out)


def _crossing_many(ratio: np.ndarray, k: np.ndarray, threshold: float = SCORE_THRESHOLD):
    """Vectorised compute_crossing over rows of ratio [n, nk] (all finite); returns (lambda, status code)."""
    s = np.sign(ratio - threshold)
    change = s[:, 1:] != s[:, :-1]
    n_cross = change.sum(axis=1)
    i = np.argmax(change, axis=1)
    rows = np.arange(ratio.shape[0])
    a1, a2 = ratio[rows, i] - threshold, ratio[rows, i + 1] - threshold
    l1, l2 = np.log(k[i]), np.log(k[i + 1])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = 1.0 / np.exp(l1 - a1 * (l1 - l2) / (a1 - a2))
    status = np.where(n_cross > 1, MAP_STATUS["multiple"], MAP_STATUS["ok"]).astype(np.int8)
    none = n_cross == 0
    status[none] = np.where(ratio[none, 0] < threshold, MAP_STATUS["resolved_all"], MAP_STATUS["unresolved"])
    lam[none] = np.nan
    return lam, status


def compute_crossing(ratio: np.ndarray, k: np.ndarray, threshold: float = SCORE_THRESHOLD) -> Tuple[float, str]:
    """
    Wavelength (km) where PSD(err)/PSD(truth) first crosses `threshold`, scanning
    from large to small scales; linear interpolation in log(k).

    status: 'ok', 'multiple' (ok, but more crossings further on),
    'resolved_all' (ratio < threshold at every k: finer than the grid Nyquist),
    'unresolved' (ratio >= threshold at every k: coarser than the segment length),
    'nan' (no valid spectrum).
    """
    valid = np.isfinite(ratio)
    if valid.sum() < 2:
        return float("nan"), "nan"
    r, kk = ratio[valid], k[valid]
    crossings = np.nonzero(np.diff(np.sign(r - threshold)))[0]
    if crossings.size == 0:
        return float("nan"), ("resolved_all" if r[0] < threshold else "unresolved")
    i = crossings[0]
    a1, a2 = r[i] - threshold, r[i + 1] - threshold
    l1, l2 = np.log(kk[i]), np.log(kk[i + 1])
    log_k = l1 - a1 * (l1 - l2) / (a1 - a2)
    return float(1.0 / np.exp(log_k)), ("multiple" if crossings.size > 1 else "ok")


class _Accumulator:
    """Running sums of segment PSDs per region and per map cell (box_step_deg)."""

    def __init__(self, nk: int, region_names, n_cell_lat: int, n_cell_lon: int):
        self.regions = {r: {"n": 0, **{f: np.zeros(nk) for f in FIELDS}} for r in region_names}
        self.cell_n = np.zeros((n_cell_lat, n_cell_lon))
        self.cell = {f: np.zeros((n_cell_lat, n_cell_lon, nk)) for f in FIELDS}

    def add(self, psds, center_lat_idx, center_lon_idx, cell_i, cell_j, region_masks_2d):
        for name, mask2d in region_masks_2d.items():
            sel = mask2d[center_lat_idx, center_lon_idx]
            if np.any(sel):
                acc = self.regions[name]
                acc["n"] += int(sel.sum())
                for f in FIELDS:
                    acc[f] += psds[f][sel].sum(axis=0)
        np.add.at(self.cell_n, (cell_i, cell_j), 1)
        for f in FIELDS:
            np.add.at(self.cell[f], (cell_i, cell_j), psds[f])


def _box_sums(cell: np.ndarray, cells_per_box: int) -> np.ndarray:
    """Sum map cells into (cells_per_box x cells_per_box) boxes on a one-cell step; lon wraps."""
    out = np.zeros_like(cell)
    for di in range(cells_per_box):
        for dj in range(cells_per_box):
            shifted = np.roll(cell, -dj, axis=1)
            if di:
                shifted = np.concatenate([shifted[di:], np.zeros_like(shifted[:di])], axis=0)
            out += shifted
    return out


def compute_spectral_scores(
    truth: np.ndarray,
    reference: np.ndarray,
    optimized: np.ndarray,
    ocean_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    segment_length_km: float = SEGMENT_LENGTH_KM,
    box_deg: float = BOX_DEG,
    box_step_deg: float = BOX_STEP_DEG,
) -> Dict:
    """
    Pseudo-track PSDs and effective resolution for one lead time.

    Args:
        truth, reference, optimized: [C, H, W] fields (land values are ignored)
        ocean_mask: [C, H, W] bool, True = ocean
        lat, lon: 1D regular grid coordinates in degrees (lon spans the globe)
        region_masks: {name: [C, H, W]} (>0 = in region); a segment belongs to a
            region if its centre point does. Defaults to {'global': ocean_mask}.

    Returns:
        {'wavenumber_cpkm': [nk],
         'regions': {region: {channel: {'n_segments', 'psd': {field: [nk]},
                                        'lambda_eff_km': {'reference', 'optimized'},
                                        'status': {'reference', 'optimized'}}}},
         'maps': {channel: {'lat', 'lon', 'n_segments', 'lambda_eff_reference_km',
                            'lambda_eff_optimized_km'}}}
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ocean_mask = np.asarray(ocean_mask) > 0
    if region_masks is None:
        region_masks = {"global": ocean_mask}
    region_masks = {k: np.asarray(v) > 0 for k, v in region_masks.items()}

    dy_km = float(np.abs(lat[1] - lat[0])) * KM_PER_DEG
    dlon = float(np.abs(lon[1] - lon[0]))
    npt_mer = int(round(segment_length_km / dy_km))
    k_common = np.arange(1, npt_mer // 2 + 1) / (npt_mer * dy_km)
    nk = k_common.size

    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    n_cell_lat, n_cell_lon = int(round(180 / box_step_deg)), int(round(360 / box_step_deg))
    cell_i_of_lat = np.clip(((lat + 90.0) // box_step_deg).astype(int), 0, n_cell_lat - 1)
    cell_j_of_lon = np.clip(((lon180 + 180.0) // box_step_deg).astype(int), 0, n_cell_lon - 1)
    cells_per_box = int(round(box_deg / box_step_deg))

    result = {"wavenumber_cpkm": k_common, "regions": {r: {} for r in region_masks}, "maps": {}}
    n_channels, n_lat, n_lon = truth.shape

    for c in range(n_channels):
        fields = (truth[c], reference[c], optimized[c])
        valid = ocean_mask[c] & np.isfinite(truth[c]) & np.isfinite(reference[c]) & np.isfinite(optimized[c])
        fields = tuple(np.where(valid, f, 0.0) for f in fields)
        regions_2d = {r: m[c] for r, m in region_masks.items()}
        acc = _Accumulator(nk, region_masks, n_cell_lat, n_cell_lon)

        # Meridional pseudo-tracks: columns, uniform spacing dy.
        stride = max(1, int(npt_mer * SEGMENT_STRIDE_FRACTION))
        cols, starts = _valid_window_starts(valid.T, npt_mer, stride)
        if cols.size:
            _, psds = _segment_psds(tuple(f.T for f in fields), cols, starts, npt_mer, dy_km)
            ci = starts + npt_mer // 2
            acc.add(psds, ci, cols, cell_i_of_lat[ci], cell_j_of_lon[cols], regions_2d)

        # Zonal pseudo-tracks: rows at native spacing dlon*cos(lat); lon wraps.
        for i in range(n_lat):
            dx_km = dlon * KM_PER_DEG * np.cos(np.deg2rad(lat[i]))
            if dx_km <= 0:
                continue
            npt = int(round(segment_length_km / dx_km))
            if npt < npt_mer or npt >= n_lon:
                continue  # need Nyquist >= common grid, segment shorter than the circle
            row_valid = np.concatenate([valid[i], valid[i, : npt - 1]])[None, :]
            _, starts_z = _valid_window_starts(row_valid, npt, max(1, int(npt * SEGMENT_STRIDE_FRACTION)))
            starts_z = starts_z[starts_z < n_lon]
            if starts_z.size == 0:
                continue
            row_fields = tuple(np.concatenate([f[i], f[i, : npt - 1]])[None, :] for f in fields)
            k_row, psds = _segment_psds(row_fields, np.zeros_like(starts_z), starts_z, npt, dx_km)
            psds = {f: _to_common_grid(k_row, p, k_common) for f, p in psds.items()}
            cj = (starts_z + npt // 2) % n_lon
            ci = np.full_like(cj, i)
            acc.add(psds, ci, cj, cell_i_of_lat[ci], cell_j_of_lon[cj], regions_2d)

        for r, sums in acc.regions.items():
            n = sums["n"]
            psd = {f: (sums[f] / n if n else np.full(nk, np.nan)) for f in FIELDS}
            lam, status = {}, {}
            for key in ("reference", "optimized"):
                lam[key], status[key] = compute_crossing(psd[f"err_{key}"] / psd["truth"], k_common)
            result["regions"][r][c] = {"n_segments": n, "psd": psd, "lambda_eff_km": lam, "status": status}

        box_n = _box_sums(acc.cell_n, cells_per_box)
        box = {f: _box_sums(acc.cell[f], cells_per_box) for f in ("truth", "err_reference", "err_optimized")}
        # No crossing -> censor at the resolvable bound so every scored box gets a value
        scored = box_n >= MIN_SEGMENTS_PER_BOX
        lam_maps, status_maps = {}, {}
        for key in ("reference", "optimized"):
            m = np.full(box_n.shape, np.nan)
            s = np.full(box_n.shape, MAP_STATUS["no_data"], dtype=np.int8)
            lam, status = _crossing_many(box[f"err_{key}"][scored] / box["truth"][scored], k_common)
            lam[status == MAP_STATUS["resolved_all"]] = 1.0 / k_common[-1]
            lam[status == MAP_STATUS["unresolved"]] = 1.0 / k_common[0]
            m[scored], s[scored] = lam, status
            lam_maps[key], status_maps[key] = m, s
        # Box (i, j) covers cells i..i+cells_per_box-1, so its centre is at its upper-right cell corner
        result["maps"][c] = {
            "lat": -90.0 + (np.arange(n_cell_lat) + cells_per_box / 2.0) * box_step_deg,
            "lon": -180.0 + (np.arange(n_cell_lon) + cells_per_box / 2.0) * box_step_deg,
            "n_segments": box_n,
            "lambda_eff_reference_km": lam_maps["reference"],
            "lambda_eff_optimized_km": lam_maps["optimized"],
            "status_reference": status_maps["reference"],
            "status_optimized": status_maps["optimized"],
        }
    return result


def boxes_to_native(box_map: np.ndarray, box_lat: np.ndarray, box_lon: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """
    Put a box map [..., C, ny, nx] on the native grid [..., C, H, W]: each ocean
    pixel takes the value of the nearest box centre (lon wraps), land is NaN.
    Values are unchanged; only the coastline becomes the model's native one.
    """
    step_lat = float(box_lat[1] - box_lat[0])
    step_lon = float(box_lon[1] - box_lon[0])
    lon180 = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    i = np.clip(np.round((np.asarray(lat) - box_lat[0]) / step_lat).astype(int), 0, box_lat.size - 1)
    j = np.round((lon180 - box_lon[0]) / step_lon).astype(int) % box_lon.size
    native = box_map[..., i[:, None], j[None, :]].astype(np.float32)
    return np.where(_fill_small_islands(np.asarray(ocean_mask) > 0), native, np.float32(np.nan))


def _fill_small_islands(ocean_mask: np.ndarray, max_pixels: int = MAP_MAX_ISLAND_PIXELS) -> np.ndarray:
    """
    Map display only: treat land patches of <= max_pixels (8-connected) as ocean,
    so single-pixel model islands do not speckle the open ocean. Segments are
    unaffected (they still exclude every land pixel).
    """
    from scipy.ndimage import label

    out = ocean_mask.copy()
    structure = np.ones((3, 3), dtype=bool)
    for c in range(out.shape[0]):
        labels, n = label(~ocean_mask[c], structure=structure)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        small = sizes <= max_pixels
        small[0] = False  # background (ocean)
        out[c] |= small[labels]
    return out


def save_effective_resolution(
    metrics_dir,
    diagnostics_dir,
    run_id: str,
    reference: np.ndarray,
    optimized: np.ndarray,
    truth: np.ndarray,
    ocean_mask: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    var_names,
    region_masks: Optional[Dict[str, np.ndarray]] = None,
    timesteps=DEFAULT_TIMESTEPS,
) -> Optional[Dict]:
    """
    Compute pseudo-track spectral scores at the given lead days (those within
    the forecast horizon) and write:
      metrics/psd_segment_scores.csv        per region/timestep/variable/wavenumber PSDs + scores
      metrics/effective_resolution.csv      per region/timestep/variable lambda_eff (ref, opt)
      metrics/effective_resolution_maps.nc  10-degree box maps on a 1-degree sliding step
      metrics/effective_resolution.json     method metadata + summary
      diagnostics/effective_resolution_map_{var}.png

    Forecast arrays are [T, C, H, W] with physical lead day t stored at index t - 1.
    """
    import csv
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import xarray as xr

    metrics_dir, diagnostics_dir = Path(metrics_dir), Path(diagnostics_dir)
    horizon = min(reference.shape[0], optimized.shape[0], truth.shape[0])
    timesteps = [int(t) for t in timesteps if 1 <= t <= horizon]
    if not timesteps:
        return None

    score_rows, lambda_rows = [], []
    map_ref, map_opt, map_n, map_sref, map_sopt = [], [], [], [], []
    map_lat = map_lon = None
    for t in timesteps:
        res = compute_spectral_scores(
            truth[t - 1], reference[t - 1], optimized[t - 1], ocean_mask, lat, lon, region_masks
        )
        k = res["wavenumber_cpkm"]
        for region, per_channel in res["regions"].items():
            for c, r in per_channel.items():
                psd = r["psd"]
                lambda_rows.append({
                    "run_id": run_id, "region": region, "timestep": t, "variable": var_names[c],
                    "n_segments": r["n_segments"],
                    "lambda_eff_reference_km": r["lambda_eff_km"]["reference"],
                    "lambda_eff_optimized_km": r["lambda_eff_km"]["optimized"],
                    "status_reference": r["status"]["reference"],
                    "status_optimized": r["status"]["optimized"],
                })
                for i, kv in enumerate(k):
                    row = {"run_id": run_id, "region": region, "timestep": t,
                           "variable": var_names[c], "wavenumber_cpkm": kv,
                           "wavelength_km": 1.0 / kv, "n_segments": r["n_segments"]}
                    row.update({f"psd_{f}": psd[f][i] for f in FIELDS})
                    row["score_reference"] = 1.0 - psd["err_reference"][i] / psd["truth"][i]
                    row["score_optimized"] = 1.0 - psd["err_optimized"][i] / psd["truth"][i]
                    score_rows.append(row)
        chans = sorted(res["maps"])
        map_ref.append([res["maps"][c]["lambda_eff_reference_km"] for c in chans])
        map_opt.append([res["maps"][c]["lambda_eff_optimized_km"] for c in chans])
        map_n.append([res["maps"][c]["n_segments"] for c in chans])
        map_sref.append([res["maps"][c]["status_reference"] for c in chans])
        map_sopt.append([res["maps"][c]["status_optimized"] for c in chans])
        map_lat, map_lon = res["maps"][chans[0]]["lat"], res["maps"][chans[0]]["lon"]

    for name, rows in (("psd_segment_scores.csv", score_rows), ("effective_resolution.csv", lambda_rows)):
        with open(metrics_dir / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    variables = list(var_names[: len(map_ref[0])])
    dims = ("timestep", "variable", "lat", "lon")
    xr.Dataset(
        {
            "lambda_eff_reference_km": (dims, np.array(map_ref)),
            "lambda_eff_optimized_km": (dims, np.array(map_opt)),
            "n_segments": (dims, np.array(map_n)),
            "status_reference": (dims, np.array(map_sref)),
            "status_optimized": (dims, np.array(map_sopt)),
        },
        coords={"timestep": timesteps, "variable": variables, "lat": map_lat, "lon": map_lon},
        attrs={"description": "Effective resolution in 10x10 degree boxes on a sliding 1 degree step (as mod_spectral); "
                              "boxes without a 0.5 crossing hold the resolvable bound (see status)",
               "status_codes": ", ".join(f"{v}={k}" for k, v in MAP_STATUS.items() if k != "nan"),
               "run_id": run_id},
    ).to_netcdf(
        metrics_dir / "effective_resolution_maps.nc",
        encoding={v: {"zlib": True, "complevel": 4} for v in (
            "lambda_eff_reference_km", "lambda_eff_optimized_km", "n_segments",
            "status_reference", "status_optimized")},
    )

    # Same maps on the native grid with the model land mask (exact coastline)
    ocean_c = np.asarray(ocean_mask)[: len(variables)] > 0
    native = {key: boxes_to_native(np.array(m), map_lat, map_lon, lat, lon, ocean_c)
              for key, m in (("reference", map_ref), ("optimized", map_opt))}
    native_dims = ("timestep", "variable", "lat", "lon")
    xr.Dataset(
        {f"lambda_eff_{key}_km": (native_dims, v) for key, v in native.items()},
        coords={"timestep": timesteps, "variable": variables, "lat": np.asarray(lat), "lon": np.asarray(lon)},
        attrs={"description": "effective_resolution_maps.nc on the native grid: each ocean pixel takes "
                              "its nearest 1-degree box value, land = NaN (status codes: see the 1-degree file)",
               "run_id": run_id},
    ).to_netcdf(
        metrics_dir / "effective_resolution_maps_native.nc",
        encoding={f"lambda_eff_{key}_km": {"zlib": True, "complevel": 4} for key in native},
    )

    metadata = {
        "run_id": run_id,
        "timesteps": timesteps,
        "method": {
            "reference": "ocean-data-challenges 2023a_SSH_mapping_OSE src/mod_spectral.py, gridded adaptation",
            "pseudo_tracks": "meridional (dy = 0.25 deg) and zonal (dx = 0.25 deg * cos(lat)) segments, "
                             "100% ocean, pooled on a common wavenumber grid",
            "segment_length_km": SEGMENT_LENGTH_KM,
            "segment_overlap": 1.0 - SEGMENT_STRIDE_FRACTION,
            "psd": "scipy.signal.welch(nperseg=npt, noverlap=0, window='hann', "
                   "detrend='constant', scaling='density')",
            "score": "1 - PSD(forecast - truth) / PSD(truth)",
            "effective_resolution": f"wavelength where PSD(err)/PSD(truth) crosses {SCORE_THRESHOLD} "
                                    "(first crossing from large scales, log-k interpolation)",
            "km_per_degree": KM_PER_DEG,
            "maps": f"10x10 deg boxes, sliding {BOX_STEP_DEG:g} deg step (mod_spectral), >= {MIN_SEGMENTS_PER_BOX} segments",
        },
        "effective_resolution": lambda_rows,
    }
    with open(metrics_dir / "effective_resolution.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    # Per-run quick-look maps: rows = lead days, columns = reference / optimized.
    # Values outside [LAMBDA_MIN_KM, LAMBDA_MAX_KM] saturate to the end colours
    # (> max, incl. unresolved boxes = worst colour); grey = land (native mask) / too few segments.
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#d9d9d9")
    cmap.set_over(cmap(1.0))
    cmap.set_under(cmap(0.0))
    lon180 = np.where(np.asarray(lon) > 180.0, np.asarray(lon) - 360.0, np.asarray(lon))
    order = np.argsort(lon180)
    for c, var in enumerate(variables):
        fig, axes = plt.subplots(len(timesteps), 2, figsize=(12, 2.6 * len(timesteps)),
                                 sharex=True, sharey=True, squeeze=False, constrained_layout=True)
        for ti, t in enumerate(timesteps):
            for col, (label, key) in enumerate((("Reference", "reference"), ("Optimized", "optimized"))):
                ax = axes[ti, col]
                im = ax.pcolormesh(lon180[order], lat, np.ma.masked_invalid(native[key][ti, c][:, order]),
                                   cmap=cmap, vmin=LAMBDA_MIN_KM, vmax=LAMBDA_MAX_KM,
                                   shading="nearest", rasterized=True)
                ax.set_title(f"{label} | {var} | day {t}")
        fig.colorbar(im, ax=axes, extend="both", shrink=0.6,
                     label=f"Effective resolution (km); > {LAMBDA_MAX_KM:.0f} incl. unresolved = worst")
        fig.savefig(diagnostics_dir / f"effective_resolution_map_{var}.png", dpi=150)
        plt.close(fig)
    return metadata
