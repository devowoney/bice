#!/usr/bin/env python3
"""
Backfill regional forecast animations on finished runs.
states/gulf_stream_forecast_comparison.gif only covers the Gulf Stream and has no ground truth
or anomaly view; this draws every region of backfill_ic_correction_regional.py.

Columns: ground truth | reference forecast | optimized forecast, one frame per forecast day.
Two GIFs per region:
  raw      fields, viridis, per-variable range (1st..99th percentile over all panels and frames)
  anomaly  field - time mean of the ground truth (same mean_field as the live run), RdBu_r
Colour ranges are fixed over the animation so frames are comparable.

Fields are taken from the first source available in each run:
  ground truth         states/ground_truth.nc -> GLORYS dataset of the run config (data.init_state_files,
                       time sample_idx + sequence_length, forecast_horizon steps); a missing file
                       (e.g. node-local /SCRATCH copy) is looked up by name under data.root_path
  reference/optimized  states/{reference,optimized}_forecast.nc -> best_solution.pt
                       (reference_forecast / full_forecast) -> forward model re-run from the ICs
  ICs (model re-run)   states/{initial,optimized_initial}_condition.nc -> dataset input
                       (+ initial_condition.noise as in the run) / best_solution.pt x0
--rerun-model skips best_solution.pt forecasts; --save-states writes rebuilt fields back as
states/*.nc (~540 MB each, same format as the live run).

Usage:
  python backfill_forecast_regional_gif.py table1_3dAssim_2026-09-24_08-48-56          # multirun dir
  python backfill_forecast_regional_gif.py .tmp/runs/pixel_mergeTest_2026-09-24_13-56-41 # single run dir
  python backfill_forecast_regional_gif.py .tmp/runs/parallel_table2_smoothGrad/2026-10-11_11-08-53  # parallel
  python backfill_forecast_regional_gif.py table1_* --regions gulf_stream equatorial
  python backfill_forecast_regional_gif.py table1_* --region med=30,46,-6,36  # extra box
  python backfill_forecast_regional_gif.py table1_* --out-root /tmp/test --skip-existing
  python backfill_forecast_regional_gif.py table1_* --rerun-model --device cuda --save-states

Reads per run:   states/*.nc, best_solution.pt, config.yaml (+ dataset / model weights when needed)
                 parallel runs (gpu<rank>_sample<idx>/): ../.hydra/config.yaml with data.sample_idx = <idx>
Writes per run:  states/forecast_regional_<region>.gif, states/forecast_regional_<region>_anomaly.gif
(or under --out-root/<multirun>/<run>/states/ when given).
"""

import argparse
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xarray as xr  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_ic_correction_regional import REGION_TITLES, REGIONS, VAR_META, VAR_NAMES, crop, parse_box  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
SOURCES = [("ground_truth", "Ground truth"), ("reference_forecast", "Reference forecast"),
           ("optimized_forecast", "Optimized forecast")]
STATE_TITLES = {"ground_truth": "Ground Truth (GLORYS12)", "reference_forecast": "Reference Forecast (from initial IC)",
                "optimized_forecast": "Optimized Forecast"}
_DATASET_CACHE = {}
_MODEL_CACHE = {}


def load_run_config(run_dir):
    """
    Run config: <run>/config.yaml (single / multirun), else the shared <parent>/.hydra/config.yaml of
    run_optimization_slurm_parallel.py, whose gpu<rank>_sample<idx> dirs carry the real data.sample_idx.
    Returns (cfg, label of the source).
    """
    if (run_dir / "config.yaml").exists():
        with open(run_dir / "config.yaml") as f:
            return yaml.safe_load(f), "config.yaml"
    shared = run_dir.parent / ".hydra" / "config.yaml"
    m = re.fullmatch(r"gpu\d+_sample(\d+)", run_dir.name)
    if m and shared.exists():
        with open(shared) as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["sample_idx"] = int(m.group(1))
        return cfg, f"../.hydra/config.yaml, sample_idx={m.group(1)}"
    raise FileNotFoundError(f"no {run_dir / 'config.yaml'} (nor ../.hydra/config.yaml for a gpu*_sample* dir)")


def find_runs(parent):
    """Run dirs directly under a multirun / parallel parent (any child with states/), natural sort."""
    def key(d):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d.name)]
    return sorted((d for d in parent.iterdir() if (d / "states").is_dir()), key=key)


def resolve_data_file(root, fname):
    """root/fname as in GlonetDataset; if missing (node-local scratch copy), the same file name under root."""
    path = Path(root) / fname
    if path.exists():
        return path
    matches = sorted(Path(root).glob(path.name)) + sorted(Path(root).glob(f"*/{path.name}"))
    if not matches:
        raise FileNotFoundError(f"{path} not found, and no {path.name} under {root}")
    print(f"    · {path} missing, using {matches[0]}")
    return matches[0]


def open_glorys(cfg):
    """GLORYS init-state dataset of the run (same concat as GlonetDataset._load_dataset), cached across runs."""
    data = cfg["data"]
    key = (data["root_path"], tuple(data["init_state_files"]))
    if key not in _DATASET_CACHE:
        files = [resolve_data_file(data["root_path"], f) for f in data["init_state_files"]]
        # Dask-backed (as GlonetDataset lazy_load): a plain concat would read both ~30 GB files into memory
        _DATASET_CACHE[key] = xr.concat([xr.open_dataset(f, chunks={"time": 1, "ch": 5}) for f in files], dim="time",
                                        combine_attrs="override").sortby("time")
    return _DATASET_CACHE[key]


def to_state_dataset(arr, time, lat, lon, title, attrs):
    """[T, C, H, W] array -> Dataset with one variable per channel (format of states/*.nc)."""
    ds = xr.Dataset({var: xr.DataArray(arr[:, ch], dims=["time", "lat", "lon"],
                                       coords={"time": time, "lat": lat, "lon": lon},
                                       attrs={"units": units, "long_name": long_name})
                     for ch, (var, long_name, units) in enumerate(VAR_META)})
    ds.attrs.update(title=title, **attrs)
    return ds


def load_best_solution(run_dir):
    path = run_dir / "best_solution.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be mmapped
        return torch.load(path, map_location="cpu", weights_only=False)


def load_ics(run_dir, cfg, best):
    """Initial and optimized IC [1, T=2, C, H, W] float32 tensors, plus where each came from."""
    states = run_dir / "states"
    ics, origin = [], []
    for name in ("initial_condition", "optimized_initial_condition"):
        path = states / f"{name}.nc"
        if path.exists():
            with xr.open_dataset(path) as ds:
                ics.append(torch.from_numpy(np.stack([ds[v].values for v in VAR_NAMES], axis=1)).float()[None])
            origin.append(f"states/{name}.nc")
        elif name == "optimized_initial_condition" and best is not None:
            ics.append(best["x0"].float().clone())
            origin.append("best_solution.pt x0")
        elif name == "initial_condition":
            # Same input as run_optimization.py: dataset window at sample_idx (+ IC noise when enabled)
            data, ic_cfg = cfg["data"], cfg.get("initial_condition") or {}
            seq = open_glorys(cfg).isel(time=slice(data["sample_idx"], data["sample_idx"] + data["sequence_length"]))
            if ic_cfg.get("noise", False):
                sys.path.insert(0, str(SRC))
                from gd_optimic import ObservationOperator

                seq = seq.isel(ch=slice(0, 5)).load()
                mean_field = xr.open_dataset(data["stats_file"])["data"]
                seq = ObservationOperator(device="cpu").add_gaussian_noise(
                    seq, mean_field, ic_cfg.get("noise_scale", 1.0),
                    seed=[ic_cfg.get("seed", 0) + data["sample_idx"], 1])
            arr = np.nan_to_num(seq["data"].isel(ch=slice(0, 5)).values)
            ics.append(torch.from_numpy(arr).float()[None])
            origin.append("dataset input" + (" + IC noise" if ic_cfg.get("noise", False) else ""))
        else:
            raise FileNotFoundError("no optimized IC (states/optimized_initial_condition.nc or best_solution.pt)")
    return ics, origin


def get_forward_model(cfg, device):
    """ForwardModel as built by run_optimization.py (ocean mask from the dataset at sample_idx), cached."""
    sys.path.insert(0, str(SRC))
    from gd_optimic import ForwardModel, MaskBuilder

    model_path = str(Path(cfg["model"]["location"]) / cfg["model"]["checkpoint_files"]["part1"])
    sample = open_glorys(cfg).isel(time=cfg["data"]["sample_idx"])["data"].values
    ocean_mask = MaskBuilder(device=device).build_ocean_mask(sample).to(device)
    key = (model_path, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()  # one glonet on the device at a time
        _MODEL_CACHE[key] = ForwardModel(model_path=model_path, normalizer_path=cfg["model"]["location"],
                                         device=device, use_gradient_checkpointing=False, ocean_mask=ocean_mask)
    model = _MODEL_CACHE[key]
    model.ocean_mask = ocean_mask  # masks may differ between runs (sample_idx)
    return model


def build_sources(run_dir, cfg, rerun_model=False, device="cuda"):
    """
    {source: Dataset} for ground truth / reference / optimized (states/*.nc format, land = 0), plus
    {source: origin} and the Datasets rebuilt in memory (for --save-states).
    """
    states = run_dir / "states"
    data = cfg["data"]
    horizon = int(data["forecast_horizon"])
    srcs, origin, rebuilt = {}, {}, {}
    for name, _ in SOURCES:
        if (states / f"{name}.nc").exists():
            srcs[name], origin[name] = xr.open_dataset(states / f"{name}.nc"), f"states/{name}.nc"

    if "ground_truth" not in srcs:
        start = data["sample_idx"] + data["sequence_length"]
        gt = open_glorys(cfg).isel(time=slice(start, start + horizon), ch=slice(0, 5))
        arr = np.nan_to_num(gt["data"].values).astype(np.float32)  # land NaN -> 0, as in the run
        srcs["ground_truth"] = rebuilt["ground_truth"] = to_state_dataset(
            arr, gt.time.values, gt.lat.values, gt.lon.values, STATE_TITLES["ground_truth"], {})
        origin["ground_truth"] = "dataset (run config)"
    truth = srcs["ground_truth"]

    missing = [n for n in ("reference_forecast", "optimized_forecast") if n not in srcs]
    if missing:
        best = load_best_solution(run_dir)
        attrs = {"best_iteration": int(best["iteration"]), "best_loss": float(best["loss"])} if best is not None else {}
        forecasts = {}
        if best is not None and not rerun_model:
            for name, key in (("reference_forecast", "reference_forecast"), ("optimized_forecast", "full_forecast")):
                if name in missing and key in best:
                    forecasts[name] = best[key][0].float().numpy()
                    origin[name] = f"best_solution.pt {key}"
        todo = [n for n in missing if n not in forecasts]
        if todo:
            (x0_init, x0_opt), ic_origin = load_ics(run_dir, cfg, best)
            model = get_forward_model(cfg, device)
            for name, x0, ic_from in (("reference_forecast", x0_init, ic_origin[0]),
                                      ("optimized_forecast", x0_opt, ic_origin[1])):
                if name in todo:
                    print(f"    · forward model: {name} from {ic_from} ({horizon} steps, {device})")
                    with torch.no_grad():
                        _, y = model.forward(x0.to(device), horizon)
                    forecasts[name] = y[0].float().cpu().numpy()
                    origin[name] = f"model re-run from {ic_from}"
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        for name, arr in forecasts.items():
            arr = arr[: truth.sizes["time"]]
            srcs[name] = rebuilt[name] = to_state_dataset(
                arr, truth.time.values[: arr.shape[0]], truth.lat.values, truth.lon.values, STATE_TITLES[name], attrs)
    return srcs, origin, rebuilt


def load_region(srcs, box):
    """Cropped [S=3, T, C, H, W] fields (truth, reference, optimized), lat, lon, dates."""
    fields = []
    for name, _ in SOURCES:
        sub = crop(srcs[name], box)
        fields.append(np.nan_to_num(np.stack([sub[v].values for v in VAR_NAMES], axis=1)).astype(np.float32))
        lat, lon = sub.lat.values, sub.lon.values
        if name == "ground_truth":
            dates = [str(t)[:10] for t in sub.time.values]
    horizon = min(f.shape[0] for f in fields)
    return np.stack([f[:horizon] for f in fields]), lat, lon, dates[:horizon]


def color_range(field, symmetric):
    vals = field[np.isfinite(field)]
    if not vals.size:
        return -1.0, 1.0
    if symmetric:
        vmax = float(np.percentile(np.abs(vals), 99))
        vmax = max(vmax if vmax > 0 else float(np.abs(vals).max()), 1e-12)
        return -vmax, vmax
    vmin, vmax = (float(v) for v in np.percentile(vals, [1, 99]))
    return (vmin, vmax) if vmax > vmin else (vmin - 1e-12, vmax + 1e-12)


def save_gif(out_path, data, lat, lon, dates, title, anomaly):
    """data [S=3, T, C, H, W] with NaN on land."""
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    extent = (lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2)
    origin = "lower" if lat[0] < lat[-1] else "upper"
    aspect = 1.0 / np.cos(np.deg2rad(np.mean(lat)))  # locally equal-area pixels
    panel_w = 4.5
    panel_h = max(panel_w * (lat.max() - lat.min()) * aspect / (lon.max() - lon.min()), 1.2)

    cmap = plt.get_cmap("RdBu_r" if anomaly else "viridis").copy()
    cmap.set_bad("0.75")
    fig, axes = plt.subplots(5, 3, figsize=(3 * panel_w + 2, 5 * panel_h + 1.2), constrained_layout=True)
    for col, (_, label) in enumerate(SOURCES):
        axes[0, col].set_title(f"{label}{' anomaly' if anomaly else ''}", fontsize=11, fontweight="bold")

    images = []
    for ch, (var, long_name, units) in enumerate(VAR_META):
        vmin, vmax = color_range(data[:, :, ch], symmetric=anomaly)
        row = []
        for col in range(3):
            ax = axes[ch, col]
            im = ax.imshow(data[col, 0, ch], cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, origin=origin,
                           aspect=aspect, interpolation="none", animated=True)
            ax.tick_params(labelsize=7)
            if ch < 4:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])
            row.append(im)
        fig.colorbar(row[-1], ax=list(axes[ch]), fraction=0.03, pad=0.01, label=units)
        axes[ch, 0].set_ylabel(f"{var}\n{long_name}", fontsize=9)
        images.append(row)

    suptitle = fig.suptitle(f"{title} | day 1/{len(dates)} ({dates[0]})", fontsize=13)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # freeze the constrained layout: re-solving it every frame dominates runtime

    def update(t):
        suptitle.set_text(f"{title} | day {t + 1}/{len(dates)} ({dates[t]})")
        for ch, row in enumerate(images):
            for col, im in enumerate(row):
                im.set_data(data[col, t, ch])
        return [suptitle] + [im for row in images for im in row]

    anim = animation.FuncAnimation(fig, update, frames=len(dates), interval=500, blit=False, repeat=True)
    try:
        anim.save(out_path, writer=animation.PillowWriter(fps=2), dpi=100)
    finally:
        plt.close(fig)


def process_run(run_dir, regions, out_root=None, skip_existing=False, rerun_model=False, device="cuda",
                save_states=False):
    out_dir = run_dir / "states" if out_root is None else Path(out_root) / run_dir.parent.name / run_dir.name / "states"
    paths = {name: (out_dir / f"forecast_regional_{name}.gif", out_dir / f"forecast_regional_{name}_anomaly.gif")
             for name in regions}
    if skip_existing:
        paths = {name: p for name, p in paths.items() if not (p[0].exists() and p[1].exists())}
        if not paths:
            print(f"  - {run_dir.name}: already done, skipped")
            return
    cfg, cfg_from = load_run_config(run_dir)
    print(f"    config: {cfg_from}")
    srcs, origin, rebuilt = build_sources(run_dir, cfg, rerun_model, device)
    print(f"    sources: " + ", ".join(f"{n}: {origin[n]}" for n, _ in SOURCES))
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_states:
        for name, ds in rebuilt.items():
            ds.to_netcdf(out_dir / f"{name}.nc")
            print(f"    · saved {out_dir / f'{name}.nc'}")
    attrs = srcs["optimized_forecast"].attrs

    written = []
    for name, (raw_path, anom_path) in paths.items():
        box = regions[name]
        data, lat, lon, dates = load_region(srcs, box)
        if data.shape[-2] < 2 or data.shape[-1] < 2:
            print(f"  ! {run_dir.name}: region {name} {box} is empty on this grid, skipped")
            continue
        land = data[0, :1] == 0  # land is stored as 0 in states/*.nc; [1, C, H, W]
        mean = data[0].mean(axis=0)  # ground-truth time mean [C, H, W]
        raw = np.where(land[None], np.nan, data)
        anom = np.where(land[None], np.nan, data - mean[None, None])

        title = (f"{REGION_TITLES.get(name, name)} ({box[0]}..{box[1]}°N, {box[2]}..{box[3]}°E) | "
                 f"best iter {attrs.get('best_iteration', '?')} | {run_dir.parent.name}/{run_dir.name}")
        save_gif(raw_path, raw, lat, lon, dates, title, anomaly=False)
        save_gif(anom_path, anom, lat, lon, dates, title, anomaly=True)
        written.append(name)
    for ds in srcs.values():
        ds.close()
    print(f"  ✓ {run_dir.name}: {', '.join(written) if written else 'nothing new'} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="multirun or single run directories (globs allowed)")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS),
                    help="predefined regions to draw (default: all)")
    ap.add_argument("--region", action="append", type=parse_box, default=[],
                    help="extra box name=lat_min,lat_max,lon_min,lon_max (repeatable)")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of inside each run")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rerun-model", action="store_true",
                    help="re-run the forward model for missing forecast .nc even if best_solution.pt has them")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for the forward model re-run (default: cuda if available)")
    ap.add_argument("--save-states", action="store_true",
                    help="write rebuilt ground truth / forecasts as states/*.nc (~540 MB each)")
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
            runs = find_runs(parent)
        print(f"{parent.name} ({len(runs)} runs)")
        for run_dir in runs:
            print(f"  {run_dir.name}")
            try:
                process_run(run_dir, regions, args.out_root, args.skip_existing, args.rerun_model, args.device,
                            args.save_states)
            except Exception as e:  # keep going over the other runs
                print(f"  ✗ {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
