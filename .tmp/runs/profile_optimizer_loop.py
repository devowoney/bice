#!/usr/bin/env python3
"""
Instrumented run of the REAL ICOptimizer.optimize() loop (src/gd_optimic/optimizer.py)
to find exactly where the ~2h gap goes: production runs 550 iterations in ~7h, but the
pure model forward+backward cost curve (profile_step_cost.py) only accounts for
~32.16s/iteration * 550 ~= 4.9h. This script does NOT change any production code — it
monkeypatches timing wrappers around the loop's sub-steps (forward, backward, loss,
gradient filter, metrics, TensorBoard logging, checkpointing) for the duration of this
process only, runs a short REAL optimization (small num_iterations, real data/model/obs),
and reports where the per-iteration wall time actually goes.

This mirrors src/run_optimization_slurm.py's setup (steps 1-9) exactly, so the measured
loop is the real production code path, not a synthetic approximation.

Usage:
    python .tmp/runs/profile_optimizer_loop.py [--iterations 15] [--save-frequency 10]

Output (each run gets its own isolated directory, timestamped):
    .tmp/runs/profile_optimizer_loop_<timestamp>/                  (ICOptimizer's own outputs:
                                                                      tb/, checkpoints/, metrics/,
                                                                      config.yaml, best_solution.pt)
    .tmp/runs/profile_optimizer_loop_<timestamp>/profiling_results/timing_breakdown.json
    .tmp/runs/profile_optimizer_loop_<timestamp>/profiling_results/timing_breakdown.png
"""

import argparse
import functools
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gd_optimic import (  # noqa: E402
    GlonetDataset,
    ObservationOperator,
    ObservationLoss,
    GradientFilter,
    ICOptimizer,
    MetricsComputer,
    MaskBuilder,
    ForwardModel,
)

# ---------------------------------------------------------------------------
# Timing instrumentation (process-local only — no source files are modified)
# ---------------------------------------------------------------------------
TIMES = defaultdict(float)          # name -> cumulative seconds
COUNTS = defaultdict(int)           # name -> call count
PER_ITER = defaultdict(list)        # name -> [seconds per call] (chronological)


def timed(name):
    """Wrap a callable so every call is timed (with CUDA sync for accurate GPU cost)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            TIMES[name] += dt
            COUNTS[name] += 1
            PER_ITER[name].append(dt)
            return result
        return wrapper
    return deco


def instrument(optimizer: ICOptimizer):
    """Patch the real objects the loop calls, in place, for this process only."""
    # --- one-time setup / teardown (previously folded into "unaccounted") ---
    optimizer.setup_directories = timed("setup.setup_directories")(optimizer.setup_directories)
    optimizer._save_final_results = timed("teardown._save_final_results")(optimizer._save_final_results)
    # SummaryWriter doesn't exist yet (created inside setup_directories), so patch the
    # CLASS's close method — applies to whichever instance gets created later.
    from torch.utils.tensorboard import SummaryWriter
    SummaryWriter.close = timed("teardown.writer_close")(SummaryWriter.close)

    # Model forward (bound-method override on the instance)
    optimizer.forward_model.forward = timed("forward_model.forward")(optimizer.forward_model.forward)

    # loss_fn / gradient_filter are called as `obj(...)`, which resolves `__call__` on
    # the CLASS (dunder methods bypass instance-attribute lookup), so we patch the class.
    type(optimizer.loss_fn).__call__ = timed("loss_fn.__call__")(type(optimizer.loss_fn).__call__)
    type(optimizer.gradient_filter).__call__ = timed("gradient_filter.__call__")(
        type(optimizer.gradient_filter).__call__
    )

    # Metrics + logging + checkpointing are normal instance methods on ICOptimizer.
    optimizer.metrics_computer.compute_all_metrics = timed("metrics_computer.compute_all_metrics")(
        optimizer.metrics_computer.compute_all_metrics
    )
    optimizer._log_to_tensorboard = timed("optimizer._log_to_tensorboard")(optimizer._log_to_tensorboard)
    optimizer._log_histograms_to_tensorboard = timed("optimizer._log_histograms_to_tensorboard")(
        optimizer._log_histograms_to_tensorboard
    )
    optimizer._save_checkpoint = timed("optimizer._save_checkpoint")(optimizer._save_checkpoint)
    optimizer._compute_finite_difference_ic_update = timed(
        "optimizer._compute_finite_difference_ic_update"
    )(optimizer._compute_finite_difference_ic_update)
    optimizer._print_progress = timed("optimizer._print_progress")(optimizer._print_progress)

    # torch.autograd.grad is a free function called inline in optimize() — patch globally
    # for the lifetime of this process (restored at the end via _uninstrument).
    global _orig_autograd_grad
    _orig_autograd_grad = torch.autograd.grad
    torch.autograd.grad = timed("backward (torch.autograd.grad)")(_orig_autograd_grad)

    # Iteration-boundary tick: compute_all_metrics runs once per iteration, ~85% through
    # the loop body. Timestamping each call (already recorded via PER_ITER durations +
    # the wrapper's own perf_counter reads) lets us reconstruct true per-iteration wall
    # time as the delta between consecutive calls — this captures EVERYTHING in the loop
    # body (including any code we didn't individually wrap), not just the sum of our
    # named sub-timers.
    global TICKS
    TICKS = []
    _orig_metrics = optimizer.metrics_computer.compute_all_metrics

    @functools.wraps(_orig_metrics)
    def tick_wrapper(*args, **kwargs):
        TICKS.append(time.perf_counter())
        return _orig_metrics(*args, **kwargs)

    optimizer.metrics_computer.compute_all_metrics = tick_wrapper


def uninstrument():
    torch.autograd.grad = _orig_autograd_grad


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=20, help="num_iterations override (kept small)")
    p.add_argument("--save-frequency", type=int, default=10, help="checkpoint save frequency override")
    p.add_argument("--config-name", type=str, default="optimize_ic")
    p.add_argument(
        "--latent-off", action="store_true",
        help="Test-only: set model.inner_checkpoint_blocks.latent=false (verifies the new "
             "ForwardModel/GlonetGradientCheckpointing inner_checkpoint_blocks passthrough).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cfg = OmegaConf.load(REPO_ROOT / "configs" / f"{args.config_name}.yaml")
    cfg.optimization.num_iterations = args.iterations
    cfg.logging.save_frequency = args.save_frequency
    if args.latent_off:
        cfg.model.inner_checkpoint_blocks = {
            "spatial": True, "latent": False, "temporal": True, "predictions": True,
        }
        print("Test override: model.inner_checkpoint_blocks.latent = False")
    # Keep production defaults for log_frequency / histogram_frequency (this is exactly
    # what we're trying to measure the cost of).

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.compute.device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_iterations={args.iterations}  save_frequency={args.save_frequency}  "
          f"log_frequency={cfg.logging.log_frequency}  histogram_frequency={cfg.logging.histogram_frequency}")

    # ---- Steps 1-9: identical setup to src/run_optimization_slurm.py ----
    print("\n[1/9] Loading dataset...")
    dataset = GlonetDataset(
        data_path=cfg.data.root_path,
        data_files=cfg.data.init_state_files,
        ssh_obs_files=cfg.data.ssh_obs_files if cfg.observations.mode != "full" else None,
        sst_obs_files=cfg.data.sst_obs_files if cfg.observations.mode != "full" else None,
        lazy_load=True,
    )

    input_sequence = dataset.get_sequence(start_idx=cfg.data.sample_idx, length=cfg.data.sequence_length)
    target_start_idx = cfg.data.sample_idx + cfg.data.sequence_length
    target_sequence = dataset.get_sequence(start_idx=target_start_idx, length=cfg.data.observation_length)
    ground_truth_sequence = dataset.get_sequence(start_idx=target_start_idx, length=cfg.data.forecast_horizon)

    input_sequence = dataset.align_grid(input_sequence, input_sequence)
    target_sequence = dataset.align_grid(target_sequence, input_sequence)
    ground_truth_sequence = dataset.align_grid(ground_truth_sequence, input_sequence)

    print("[2/9] Applying observation operators...")
    obs_operator = ObservationOperator(device=device)
    stats_field = None
    if cfg.data.get("stats_file", None):
        import xarray as _xr
        stats_ds = _xr.open_dataset(Path(cfg.data.stats_file))
        stats_field = stats_ds["data"] if "data" in stats_ds.data_vars else stats_ds[list(stats_ds.data_vars)[0]]

    ssh_mask = sst_mask = None
    if cfg.observations.mode != "full":
        ssh_obs = dataset.get_ssh_obs(target_start_idx, cfg.data.observation_length)
        target_sequence, ssh_mask = obs_operator.apply_ssh_operator(
            target_sequence, ssh_obs, cfg.observations.mode, mdt=stats_field
        )
        sst_obs = dataset.get_sst_obs(target_start_idx, cfg.data.observation_length)
        target_sequence, sst_mask = obs_operator.apply_sst_operator(target_sequence, sst_obs, cfg.observations.mode)

    print("[3/9] Creating masks...")
    mask_builder = MaskBuilder(device=device)
    sample_data = dataset.dataset.isel(time=cfg.data.sample_idx)["data"].values
    ocean_mask = mask_builder.build_ocean_mask(sample_data)
    regional_masks = mask_builder.build_regional_masks(
        input_sequence.coords["lat"].values,
        input_sequence.coords["lon"].values,
        ocean_mask,
        variance_ssh_path=cfg.data.variance_ssh_path,
        high_var_threshold=cfg.metrics.high_var_threshold,
    )
    obs_mask = mask_builder.build_obs_mask(
        ocean_mask, cfg.data.observation_length,
        ssh_nanmask=ssh_mask, sst_nanmask=sst_mask, obs_mode=cfg.observations.mode,
    )

    print("[4/9] Preparing tensors...")
    input_data = np.nan_to_num(input_sequence["data"].values, nan=0.0)
    x0_init = torch.from_numpy(input_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
    target_data = np.nan_to_num(target_sequence["data"].values, nan=0.0)
    target_tensor = torch.from_numpy(target_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)

    print("[5/9] Initializing forward model...")
    inner_checkpoint_blocks = cfg.model.get("inner_checkpoint_blocks", None)
    if inner_checkpoint_blocks is not None:
        inner_checkpoint_blocks = dict(inner_checkpoint_blocks)
    forward_model = ForwardModel(
        model_path=str(Path(cfg.model.location) / cfg.model.checkpoint_files.part1),
        normalizer_path=cfg.model.location,
        device=device,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        ocean_mask=ocean_mask.to(device) if ocean_mask is not None else None,
        inner_checkpoint_blocks=inner_checkpoint_blocks,
    )

    print("[6/9] Initializing loss function...")
    loss_fn = ObservationLoss(
        obs_mask=obs_mask,
        loss_weighting=cfg.loss.weighting,
        manual_weights=cfg.loss.manual_weights if cfg.loss.weighting == "manual" else None,
        use_structural_loss=cfg.loss.use_structural_loss,
        structural_operator=cfg.loss.structural_operator,
        structural_loss_weight=cfg.loss.structural_loss_weight,
        device=device,
    )

    print("[7/9] Initializing gradient filter...")
    gradient_filter = GradientFilter(
        enable=cfg.optimization.use_gradient_smoothing,
        downsampling_method=cfg.optimization.downsampling_method,
        device=device,
    )

    print("[8/9] Initializing metrics computer...")
    metrics_computer = MetricsComputer(ocean_mask=ocean_mask, device=device, stats_field=stats_field)
    metrics_computer.set_mean_from_sequences(
        input_sequence_xr=input_sequence, ground_truth_sequence_xr=ground_truth_sequence
    )

    print("[9/9] Initializing optimizer...")
    optimizer = ICOptimizer(
        forward_model=forward_model,
        loss_fn=loss_fn,
        gradient_filter=gradient_filter,
        metrics_computer=metrics_computer,
        learning_rate=cfg.optimization.learning_rate,
        num_iterations=cfg.optimization.num_iterations,
        device=device,
        output_dir=cfg.logging.output_dir,
        tensorboard_subdir=cfg.logging.tensorboard_subdir,
        checkpoints_subdir=cfg.logging.checkpoints_subdir,
        metrics_subdir=cfg.logging.metrics_subdir,
        save_frequency=cfg.logging.save_frequency,
        log_frequency=cfg.logging.log_frequency,
        histogram_frequency=cfg.logging.histogram_frequency,
        forecast_horizon=cfg.data.forecast_horizon,
        use_meta_learner=False,
    )

    # ---- Give this run its own isolated directory under .tmp/runs/ ----
    # ICOptimizer.setup_directories() (src/gd_optimic/optimizer.py) hardcodes
    # `self.exp_dir = Path(".")` and writes tb/, checkpoints/, metrics/,
    # config.yaml, best_solution.pt relative to the CURRENT WORKING DIRECTORY.
    # In production this is masked by Hydra's `hydra.job.chdir: true`, which
    # cd's into `.tmp/runs/{exp_name}_{timestamp}/` before main() ever runs.
    # This script doesn't go through Hydra, so without this chdir its outputs
    # would land directly in the repo/worktree root -- and collide with any
    # other run using the same relative paths (this is what crashed job 53774:
    # two profiling runs both wrote to ./tb/ at once). Do the same thing Hydra
    # would do, so outputs are isolated under .tmp/runs/ like every other run.
    exp_id = f"profile_optimizer_loop_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = REPO_ROOT / ".tmp" / "runs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    print(f"\nRun directory (isolated, matches Hydra's chdir behavior): {run_dir}")

    # ---- Instrument, then run the REAL loop for a few iterations ----
    instrument(optimizer)

    print(f"\nRunning {args.iterations} REAL optimization iterations (instrumented)...\n")
    t_total0 = time.perf_counter()
    try:
        optimizer.optimize(
            x0_init=x0_init,
            target_sequence=target_tensor,
            ocean_mask=ocean_mask,
            exp_id=exp_id,
            regional_masks=regional_masks,
            input_sequence_xr=input_sequence,
            target_sequence_xr=target_sequence,
            ground_truth_sequence_xr=ground_truth_sequence,
        )
    finally:
        uninstrument()
    total_wall = time.perf_counter() - t_total0

    report(total_wall, args.iterations, run_dir)


def report(total_wall, num_iterations, run_dir):
    # Timing-breakdown artifacts live alongside this run's own optimizer outputs
    # (tb/, checkpoints/, metrics/, best_solution.pt), under the same isolated
    # .tmp/runs/<exp_id>/ directory rather than a shared fixed path.
    out_dir = run_dir / "profiling_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("TIMING BREAKDOWN (real ICOptimizer.optimize() loop, instrumented)")
    print("=" * 78)
    print(f"Total wall time for {num_iterations} iterations: {total_wall:.2f}s "
          f"({total_wall/num_iterations:.2f}s/iteration avg, includes one-time setup inside optimize())")

    summary = {}
    print(f"\n{'component':<45} {'calls':>6} {'total_s':>10} {'avg_s/call':>12} {'% of wall':>10}")
    print("-" * 90)
    for name, total in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        n = COUNTS[name]
        avg = total / n if n else 0.0
        pct = 100.0 * total / total_wall if total_wall else 0.0
        print(f"{name:<45} {n:>6} {total:>10.2f} {avg:>12.4f} {pct:>9.1f}%")
        summary[name] = {
            "calls": n, "total_s": total, "avg_s_per_call": avg, "pct_of_wall": pct,
            "per_call_s": PER_ITER[name],
        }

    accounted = sum(TIMES.values())
    unaccounted = total_wall - accounted
    print("-" * 90)
    print(f"{'SUM (accounted)':<45} {'':>6} {accounted:>10.2f} {'':>12} {100*accounted/total_wall:>9.1f}%")
    print(f"{'UNACCOUNTED (residual)':<45} {'':>6} {unaccounted:>10.2f} "
          f"{'':>12} {100*unaccounted/total_wall:>9.1f}%")

    # ---- True per-iteration wall time from consecutive tick timestamps ----
    # TICKS[i] = time.perf_counter() taken right as iteration i's compute_all_metrics
    # call starts. The delta TICKS[i]-TICKS[i-1] is the FULL wall time of everything
    # that happened in between: the rest of iteration i-1 (logging/checkpoint/cleanup)
    # plus the start of iteration i (forward/backward/filter/update) up to its metrics
    # call. This captures ALL loop-body code, not just our named sub-timers, so it is
    # the authoritative per-iteration cost -- independent of what we did or didn't wrap.
    tick_deltas = [TICKS[i] - TICKS[i - 1] for i in range(1, len(TICKS))]
    print(f"\nTrue per-iteration wall time (consecutive metrics-call tick deltas, "
          f"n={len(tick_deltas)}):")
    for i, d in enumerate(tick_deltas, start=2):
        print(f"  iteration {i:2d}: {d:.2f}s")

    setup_s = TIMES.get("setup.setup_directories", 0.0)
    teardown_s = TIMES.get("teardown._save_final_results", 0.0) + TIMES.get("teardown.writer_close", 0.0)
    recurring_s = None
    if tick_deltas:
        steady_ticks = tick_deltas[1:] if len(tick_deltas) > 1 else tick_deltas  # drop iter-2 cold start too
        recurring_s = sum(steady_ticks) / len(steady_ticks)
        print(f"\nSetup (setup_directories, one-time):     {setup_s:8.2f}s")
        print(f"Teardown (_save_final_results + writer.close, one-time): {teardown_s:8.2f}s")
        print(f"Steady-state per-iteration (tick-based, excludes iter 1-2): {recurring_s:8.2f}s/iteration")
        print(f"  -> extrapolated to 550 iterations (loop only): {recurring_s*550/3600:.2f}h")
        print(f"  -> + one-time setup/teardown (negligible at scale): "
              f"{(recurring_s*550 + setup_s + teardown_s)/3600:.2f}h")

    with open(out_dir / "timing_breakdown.json", "w") as f:
        json.dump({
            "num_iterations": num_iterations,
            "total_wall_s": total_wall,
            "components": summary,
            "unaccounted_s": unaccounted,
            "tick_deltas_s": tick_deltas,
            "setup_s": setup_s,
            "teardown_s": teardown_s,
            "steady_state_per_iter_s_tick_based": recurring_s,
        }, f, indent=2)

    plot_breakdown(summary, unaccounted, out_dir / "timing_breakdown.png")
    print(f"\nSaved: {out_dir}")


def plot_breakdown(summary, unaccounted, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    labels = list(summary.keys()) + ["UNACCOUNTED"]
    totals = [v["total_s"] for v in summary.values()] + [max(unaccounted, 0.0)]
    order = sorted(range(len(labels)), key=lambda i: -totals[i])
    labels = [labels[i] for i in order]
    totals = [totals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, totals, color="tab:blue")
    ax.set_xlabel("Total time (s)")
    ax.set_title("Per-iteration loop: where the time goes")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
