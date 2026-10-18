#!/usr/bin/env python3
"""
Backfill regional zooms of the IC correction figure on finished runs from their saved states
(no re-optimization). diagnostics/ic_correction_anomaly_comparison.png is global, where
grid-scale pixelization is invisible; this re-draws it over western boundary current boxes
and the equatorial Pacific.

Columns: initial anomaly | optimized anomaly | correction (optimized - initial).
Anomaly = field - time mean of states/ground_truth.nc (same mean_field as the live run).
The correction is mean-free by construction, so its anomaly equals the raw correction.

Usage:
  python backfill_ic_correction_regional.py table1_3dAssim_2026-09-24_08-48-56          # multirun dir
  python backfill_ic_correction_regional.py .tmp/runs/pixel_mergeTest_2026-09-24_13-56-41 # single run dir
  python backfill_ic_correction_regional.py table1_* --regions gulf_stream kuroshio
  python backfill_ic_correction_regional.py table1_* --region med=30,46,-6,36  # extra box
  python backfill_ic_correction_regional.py table1_* --out-root /tmp/test --skip-existing

Reads per run:   states/{initial_condition,optimized_initial_condition,ground_truth}.nc
Writes per run:  diagnostics/ic_correction_regional_<region>.png
(or under --out-root/<multirun>/<run>/diagnostics/ when given).
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

VAR_META = [("SSH", "Sea Surface Height", "m"), ("T", "Temperature", "°C"), ("S", "Salinity", "PSU"),
            ("U", "Zonal Velocity", "m/s"), ("V", "Meridional Velocity", "m/s")]
VAR_NAMES = [v for v, _, _ in VAR_META]
# (lat_min, lat_max, lon_min, lon_max); WBC boxes as in aggregate_all_v3.py / statistics_biceOutput_v2.py
REGIONS = {"gulf_stream": (30, 45, -85, -30), "kuroshio": (25, 45, 130, 180), "agulhas": (-35, -20, 20, 60),
           "equatorial": (-10, 10, -160, -110)}  # equatorial Pacific: f -> 0, strongest checkerboard
REGION_TITLES = {"gulf_stream": "Gulf Stream", "kuroshio": "Kuroshio", "agulhas": "Agulhas",
                 "equatorial": "Equatorial Pacific"}


def crop(ds, box):
    lat0, lat1, lon0, lon1 = box
    lat_slice = slice(lat0, lat1) if ds.lat[0] < ds.lat[-1] else slice(lat1, lat0)
    return ds.sel(lat=lat_slice, lon=slice(lon0, lon1))


def load_region(states, box, t_idx):
    """Cropped [C, H, W] initial / optimized IC at t_idx, ground-truth time mean, lat, lon, attrs."""
    with xr.open_dataset(states / "initial_condition.nc") as ds:
        sub = crop(ds, box).isel(time=t_idx)
        init = np.stack([sub[v].values for v in VAR_NAMES]).astype(np.float64)
        lat, lon = sub.lat.values, sub.lon.values
        ic_time = str(sub.time.values)[:10]
    with xr.open_dataset(states / "optimized_initial_condition.nc") as ds:
        sub = crop(ds, box).isel(time=t_idx)
        opt = np.stack([sub[v].values for v in VAR_NAMES]).astype(np.float64)
        attrs = dict(ds.attrs)
    mean = None
    gt_path = states / "ground_truth.nc"
    if gt_path.exists():
        with xr.open_dataset(gt_path) as ds:
            sub = crop(ds, box)
            mean = np.stack([np.nan_to_num(sub[v].values).mean(axis=0) for v in VAR_NAMES])
    return init, opt, mean, lat, lon, ic_time, attrs


def robust_vmax(*fields):
    vals = np.concatenate([np.abs(f[np.isfinite(f)]).ravel() for f in fields])
    if not vals.size:
        return 1e-12
    vmax = float(np.percentile(vals, 99))
    return max(vmax if vmax > 0 else float(vals.max()), 1e-12)  # sparse fields: 99th pct can be 0


def plot_region(out_path, name, init, opt, mean, lat, lon, title_info):
    land = init == 0  # land is stored as 0 in states/*.nc
    if mean is not None:
        init_a, opt_a = init - mean, opt - mean
    else:
        init_a, opt_a = init, opt
    delta = opt - init
    for arr in (init_a, opt_a, delta):
        arr[land] = np.nan

    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    extent = (lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2)
    origin = "lower" if lat[0] < lat[-1] else "upper"
    aspect = 1.0 / np.cos(np.deg2rad(np.mean(lat)))  # locally equal-area pixels
    width = lon.max() - lon.min()
    height = (lat.max() - lat.min()) * aspect
    panel_w = 5.0
    panel_h = max(panel_w * height / width, 1.2)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")
    state_label = "anomaly" if mean is not None else "state"
    col_titles = [f"Initial {state_label}", f"Optimized {state_label}", "Correction (optimized - initial)"]

    fig, axes = plt.subplots(5, 3, figsize=(3 * panel_w + 2, 5 * panel_h + 1.2), constrained_layout=True)
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")
    for ch, (var, long_name, units) in enumerate(VAR_META):
        vmax_state = robust_vmax(init_a[ch], opt_a[ch])
        vmax_delta = robust_vmax(delta[ch])
        for col, (field, vmax) in enumerate(((init_a[ch], vmax_state), (opt_a[ch], vmax_state),
                                             (delta[ch], vmax_delta))):
            ax = axes[ch, col]
            im = ax.imshow(field, cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent, origin=origin,
                           aspect=aspect, interpolation="none")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=units if col == 2 else None)
            ax.tick_params(labelsize=7)
            if ch < 4:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])
        axes[ch, 0].set_ylabel(f"{var}\n{long_name}", fontsize=9)
        d = delta[ch][np.isfinite(delta[ch])]
        if not d.size or not np.any(d):
            axes[ch, 2].text(0.5, 0.5, "no correction", transform=axes[ch, 2].transAxes,
                             ha="center", va="center", fontsize=10, color="0.3")
        else:
            axes[ch, 2].text(0.01, 0.02, f"rms {np.sqrt(np.mean(d ** 2)):.2e}", transform=axes[ch, 2].transAxes,
                             fontsize=7, bbox=dict(facecolor="white", alpha=0.7, lw=0))

    fig.suptitle(f"IC correction | {REGION_TITLES.get(name, name)} {title_info}", fontsize=13)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_run(run_dir, regions, t_idx, out_root=None, skip_existing=False):
    states = run_dir / "states"
    if not (states / "initial_condition.nc").exists() or not (states / "optimized_initial_condition.nc").exists():
        print(f"  - {run_dir.name}: missing states/{{initial,optimized_initial}}_condition.nc, skipped")
        return
    if out_root is None:
        diag_dir = run_dir / "diagnostics"
    else:
        diag_dir = Path(out_root) / run_dir.parent.name / run_dir.name / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, box in regions.items():
        out_path = diag_dir / f"ic_correction_regional_{name}.png"
        if skip_existing and out_path.exists():
            continue
        init, opt, mean, lat, lon, ic_time, attrs = load_region(states, box, t_idx)
        if init.shape[1] < 2 or init.shape[2] < 2:
            print(f"  ! {run_dir.name}: region {name} {box} is empty on this grid, skipped")
            continue
        info = (f"({box[0]}..{box[1]}°N, {box[2]}..{box[3]}°E) | IC {ic_time} | "
                f"best iter {attrs.get('best_iteration', '?')}\n{run_dir.parent.name}/{run_dir.name}")
        if mean is None:
            info += " | no ground_truth.nc: raw states"
        plot_region(out_path, name, init, opt, mean, lat, lon, info)
        written.append(name)
    print(f"  ✓ {run_dir.name}: {', '.join(written) if written else 'nothing new'} -> {diag_dir}")


def parse_box(spec):
    name, _, vals = spec.partition("=")
    box = tuple(float(v) for v in vals.split(","))
    if not name or len(box) != 4:
        raise argparse.ArgumentTypeError(f"expected name=lat_min,lat_max,lon_min,lon_max, got {spec!r}")
    return name, box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="multirun or single run directories (globs allowed)")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS),
                    help="predefined regions to draw (default: all)")
    ap.add_argument("--region", action="append", type=parse_box, default=[],
                    help="extra box name=lat_min,lat_max,lon_min,lon_max (repeatable)")
    ap.add_argument("--t-idx", type=int, default=-1, help="IC timestep to draw (default: latest, as the live figure)")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of inside each run")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    regions = {r: REGIONS[r] for r in args.regions}
    regions.update(dict(args.region))

    parents = sorted({Path(p).resolve() for pat in args.dirs for p in (glob(pat) or [pat]) if Path(p).is_dir()})
    if not parents:
        sys.exit("no run directories found")
    for parent in parents:
        if (parent / "states").is_dir():  # single run
            runs = [parent]
        else:
            runs = sorted((d for d in parent.iterdir() if d.is_dir() and d.name.isnumeric()), key=lambda d: int(d.name))
        print(f"{parent.name} ({len(runs)} runs)")
        for run_dir in runs:
            try:
                process_run(run_dir, regions, args.t_idx, args.out_root, args.skip_existing)
            except Exception as e:  # keep going over the other runs
                print(f"  ✗ {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
