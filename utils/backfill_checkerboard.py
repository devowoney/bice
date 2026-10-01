#!/usr/bin/env python3
"""
Backfill the pixelization (checkerboard fraction) score on finished runs from their saved
artifacts (no re-optimization). Older runs logged the finite-difference norm instead, which
is not comparable with the current compute_checkerboard_fraction() score.

Only the *cumulative* IC update (x0 - x0_reference, last IC timestep) can be rebuilt: the
*step* update needs consecutive iterations, and checkpoints are save_frequency apart.

Usage:
  python backfill_checkerboard.py table1_3dAssim_2026-09-24_08-48-56          # multirun dir
  python backfill_checkerboard.py .tmp/runs/pixel_mergeTest_2026-09-24_13-56-41 # single run dir
  python backfill_checkerboard.py table1_* --out-root /tmp/test   # write elsewhere
  python backfill_checkerboard.py table1_* --skip-existing --tensorboard

Reads per run:   states/{initial_condition,optimized_initial_condition}.nc,
                 checkpoints/checkpoint_iter*.pt, config.yaml
Writes per run:  metrics/{checkerboard_backfill.csv, checkerboard_backfill.json}
                 tb/events.*  (with --tensorboard: diag/checkerboard/{fraction,amplitude}/cumulative/*,
                               same tags and Custom Scalars chart as the live optimizer)
(or under --out-root/<multirun>/<run>/ when given).
"""

import argparse
import csv
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from gd_optimic.metrics import (  # noqa: E402
    CHECKERBOARD_PHYSICAL_LEVEL,
    CHECKERBOARD_THRESHOLD,
    compute_checkerboard_fraction,
)

VAR_NAMES = ["SSH", "T", "S", "U", "V"]
_MASK_CACHE = {}


def load_last_ic(path):
    """Last IC timestep [C, H, W] float32 from a states/*initial_condition.nc file, plus its attrs."""
    with xr.open_dataset(path) as ds:
        last = ds.isel(time=-1)
        return torch.from_numpy(np.stack([last[v].values for v in VAR_NAMES]).astype(np.float32)), dict(ds.attrs)


def build_ocean_mask(cfg, x0_ref):
    """
    Same mask as the live run: NaNs of the raw init-state data at data.sample_idx
    (run_optimization.py -> MaskBuilder.build_ocean_mask). Cached across runs sharing the data.

    Fallback when the raw data is unreachable: nonzero pixels of the saved reference IC
    (land is stored as 0 in states/*.nc). Returns (mask [C, H, W], source label).
    """
    data = cfg["data"]
    key = (data["root_path"], tuple(data["init_state_files"]), int(data["sample_idx"]))
    if key not in _MASK_CACHE:
        try:
            from gd_optimic.data import GlonetDataset
            from gd_optimic.utils import MaskBuilder

            dataset = GlonetDataset(data_path=key[0], data_files=list(key[1]), lazy_load=True)
            sample = dataset.dataset.isel(time=key[2])["data"].values
            _MASK_CACHE[key] = (MaskBuilder(device="cpu").build_ocean_mask(sample).float().cpu(), "init_state_nan")
        except Exception as e:
            print(f"    ! raw init-state mask unavailable ({e}); using nonzero pixels of initial_condition.nc")
            return (x0_ref != 0).float(), "initial_condition_nonzero"
    mask, source = _MASK_CACHE[key]
    if mask.shape != x0_ref.shape:
        print(f"    ! mask shape {tuple(mask.shape)} != IC {tuple(x0_ref.shape)}; using nonzero pixels of IC")
        return (x0_ref != 0).float(), "initial_condition_nonzero"
    return mask, source


def iter_checkpoints(ckpt_dir):
    """(iteration, path) sorted by iteration."""
    found = []
    for p in ckpt_dir.glob("checkpoint_iter*.pt"):
        m = re.fullmatch(r"checkpoint_iter(\d+)\.pt", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def load_checkpoint_ic(path):
    """Last IC timestep [C, H, W] of the checkpoint's x0 [1, T, C, H, W] (mmap: skips predictions)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:  # legacy (non-zip) serialization cannot be mmapped
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["x0"][0, -1].float().clone()


def score(update, mask):
    res = compute_checkerboard_fraction(update, mask, tuple(VAR_NAMES))
    return res["fraction"], res["amplitude"]


def write_tensorboard(tb_dir, rows):
    """Cumulative checkerboard scalars with the live optimizer's tags + reference lines + chart layout."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tb_dir), filename_suffix=".checkerboard_backfill")
    writer.add_custom_scalars({
        "Pixelization (checkerboard)": {
            "cumulative total vs reference": [
                "Multiline",
                [r"diag/checkerboard/fraction/cumulative/(total|ref_physical|ref_threshold)$"],
            ]
        }
    })
    for r in rows:
        if r["source"] != "checkpoint":
            continue  # best IC duplicates a checkpoint iteration
        it = r["iteration"]
        for quantity in ("fraction", "amplitude"):
            for var, value in r[quantity].items():
                writer.add_scalar(f"diag/checkerboard/{quantity}/cumulative/{var}", value, it)
        writer.add_scalar("diag/checkerboard/fraction/cumulative/ref_physical", CHECKERBOARD_PHYSICAL_LEVEL, it)
        writer.add_scalar("diag/checkerboard/fraction/cumulative/ref_threshold", CHECKERBOARD_THRESHOLD, it)
    writer.close()


def process_run(run_dir, out_root=None, skip_existing=False, tensorboard=False):
    states = run_dir / "states"
    ref_path, best_path = states / "initial_condition.nc", states / "optimized_initial_condition.nc"
    if not ref_path.exists():
        print(f"  - {run_dir.name}: missing states/initial_condition.nc, skipped")
        return
    if out_root is None:
        metrics_dir, tb_dir = run_dir / "metrics", run_dir / "tb"
    else:
        base = Path(out_root) / run_dir.parent.name / run_dir.name
        metrics_dir, tb_dir = base / "metrics", base / "tb"
    if skip_existing and (metrics_dir / "checkerboard_backfill.csv").exists():
        print(f"  - {run_dir.name}: already done, skipped")
        return
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    x0_ref, _ = load_last_ic(ref_path)
    mask, mask_source = build_ocean_mask(cfg, x0_ref)

    rows = []
    for it, path in iter_checkpoints(run_dir / "checkpoints"):
        fraction, amplitude = score(load_checkpoint_ic(path) - x0_ref, mask)
        rows.append({"source": "checkpoint", "iteration": it, "fraction": fraction, "amplitude": amplitude})
    if best_path.exists():
        x0_best, attrs = load_last_ic(best_path)
        fraction, amplitude = score(x0_best - x0_ref, mask)
        rows.append({"source": "best", "iteration": int(attrs.get("best_iteration", -1)),
                     "fraction": fraction, "amplitude": amplitude})
    if not rows:
        print(f"  - {run_dir.name}: no checkpoints or optimized IC, skipped")
        return

    with open(metrics_dir / "checkerboard_backfill.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "iteration", "scope", "variable", "fraction", "amplitude", "above_threshold"])
        for r in rows:
            for var in VAR_NAMES + ["total"]:
                frac = r["fraction"][var]
                w.writerow([r["source"], r["iteration"], "cumulative", var, frac,
                            r["amplitude"].get(var, ""), frac > CHECKERBOARD_THRESHOLD])
    with open(metrics_dir / "checkerboard_backfill.json", "w") as f:
        json.dump({
            "run_id": f"{run_dir.parent.name}/{run_dir.name}",
            "scope": "cumulative",
            "field": "x0[0, -1] - x0_reference[0, -1]",
            "ocean_mask_source": mask_source,
            "physical_level": CHECKERBOARD_PHYSICAL_LEVEL,
            "threshold": CHECKERBOARD_THRESHOLD,
            "rows": rows,
        }, f, indent=2)
    if tensorboard:
        write_tensorboard(tb_dir, rows)

    last = rows[-1]
    flag = "PIXELIZED" if last["fraction"]["total"] > CHECKERBOARD_THRESHOLD else "smooth"
    print(f"  ✓ {run_dir.name}: {len(rows)} ICs, {last['source']} iter {last['iteration']} "
          f"total CF {last['fraction']['total']:.3f} ({flag}, threshold {CHECKERBOARD_THRESHOLD:.3f}) "
          f"[mask: {mask_source}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="multirun or single run directories (globs allowed)")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of inside each run")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tensorboard", action="store_true",
                    help="also write the cumulative scalars into each run's tb/ (same tags as live runs)")
    args = ap.parse_args()

    parents = sorted({Path(p).resolve() for pat in args.dirs for p in (glob(pat) or [pat]) if Path(p).is_dir()})
    for parent in parents:
        if (parent / "states").is_dir():  # single run
            runs = [parent]
        else:
            runs = sorted((d for d in parent.iterdir() if d.is_dir() and d.name.isnumeric()), key=lambda d: int(d.name))
        print(f"{parent.name} ({len(runs)} runs)")
        for run_dir in runs:
            try:
                process_run(run_dir, args.out_root, args.skip_existing, args.tensorboard)
            except Exception as e:  # keep going over the other runs
                print(f"  ✗ {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
