#!/usr/bin/env python3
"""
Backfill the pseudo-track effective-resolution score on finished runs from
their saved states/*.nc (no re-optimization).

Usage:
  python backfill_effective_resolution.py table1_3dAssim_2026-09-24_08-48-56
  python backfill_effective_resolution.py table1_* --out-root /tmp/test   # write elsewhere
  python backfill_effective_resolution.py table1_* --skip-existing

Reads per run:   states/{reference_forecast,optimized_forecast,ground_truth}.nc, config.yaml
Writes per run:  metrics/{psd_segment_scores.csv, effective_resolution.csv,
                 effective_resolution_maps.nc, effective_resolution.json}
                 diagnostics/effective_resolution_map_{var}.png
(or under --out-root/<multirun>/<run>/ when given).
"""

import argparse
import sys
from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from gd_optimic.spectral_scores import DEFAULT_TIMESTEPS, save_effective_resolution  # noqa: E402

VAR_NAMES = ["SSH", "T", "S", "U", "V"]


def load_stack(path, indices):
    """[T_total, C, H, W] float32 with only `indices` filled (others stay zero)."""
    with xr.open_dataset(path) as ds:
        n_time = ds.sizes["time"]
        out = np.zeros((n_time, len(VAR_NAMES), ds.sizes["lat"], ds.sizes["lon"]), dtype=np.float32)
        sub = ds.isel(time=indices)
        for c, v in enumerate(VAR_NAMES):
            out[indices, c] = sub[v].values
        return out, ds.lat.values, ds.lon.values


def build_masks(truth_loaded, lat, lon, cfg):
    """
    Ocean mask from the loaded ground-truth days [n_days, C, H, W] + the run's regional masks.

    Land = NaN or 0 at *every* loaded day. A single-day `!= 0` test marks ocean
    pixels whose value is exactly 0.0 that day as land (~1500 px for U/V).
    """
    import torch
    from gd_optimic.utils import MaskBuilder

    ocean = np.any(np.isfinite(truth_loaded) & (truth_loaded != 0.0), axis=0)
    regions = MaskBuilder(device="cpu").build_regional_masks(
        lat, lon, torch.from_numpy(ocean.astype(np.float32)),
        variance_ssh_path=cfg["data"]["variance_ssh_path"],
        high_var_threshold=cfg["metrics"]["high_var_threshold"],
    )
    return ocean, regions


def process_run(run_dir, out_root=None, skip_existing=False, timesteps=DEFAULT_TIMESTEPS):
    states = run_dir / "states"
    paths = {k: states / f"{k}.nc" for k in ("reference_forecast", "optimized_forecast", "ground_truth")}
    if not all(p.exists() for p in paths.values()):
        print(f"  - {run_dir.name}: missing states/*.nc, skipped")
        return
    if out_root is None:
        metrics_dir, diag_dir = run_dir / "metrics", run_dir / "diagnostics"
    else:
        base = Path(out_root) / run_dir.parent.name / run_dir.name
        metrics_dir, diag_dir = base / "metrics", base / "diagnostics"
    if skip_existing and (metrics_dir / "effective_resolution.csv").exists():
        print(f"  - {run_dir.name}: already done, skipped")
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    with xr.open_dataset(paths["ground_truth"]) as ds:
        n_time = ds.sizes["time"]
    timesteps = [t for t in timesteps if 1 <= t <= n_time]
    indices = [t - 1 for t in timesteps]
    ref, lat, lon = load_stack(paths["reference_forecast"], indices)
    opt, _, _ = load_stack(paths["optimized_forecast"], indices)
    truth, _, _ = load_stack(paths["ground_truth"], indices)
    ocean, regions = build_masks(truth[indices], lat, lon, cfg)

    meta = save_effective_resolution(
        metrics_dir, diag_dir, run_id=f"{run_dir.parent.name}/{run_dir.name}",
        reference=ref, optimized=opt, truth=truth, ocean_mask=ocean, lat=lat, lon=lon,
        var_names=VAR_NAMES, region_masks=regions, timesteps=timesteps,
    )
    g = [r for r in meta["effective_resolution"] if r["region"] == "global" and r["variable"] == "SSH"]
    summary = ", ".join(f"d{r['timestep']}: {r['lambda_eff_reference_km']:.0f}->{r['lambda_eff_optimized_km']:.0f} km"
                        for r in g)
    print(f"  ✓ {run_dir.name}: global SSH lambda_eff (ref->opt) {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("multiruns", nargs="+", help="multirun directories (globs allowed)")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of inside each run")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS),
                    help="lead days to score (default: %(default)s)")
    args = ap.parse_args()

    parents = sorted({Path(p).resolve() for pat in args.multiruns for p in (glob(pat) or [pat]) if Path(p).is_dir()})
    for parent in parents:
        runs = sorted((d for d in parent.iterdir() if d.is_dir() and d.name.isnumeric()), key=lambda d: int(d.name))
        print(f"{parent.name} ({len(runs)} runs)")
        for run_dir in runs:
            try:
                process_run(run_dir, args.out_root, args.skip_existing, args.timesteps)
            except Exception as e:  # keep going over the other runs
                print(f"  ✗ {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
