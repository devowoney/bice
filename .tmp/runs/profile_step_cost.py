#!/usr/bin/env python3
"""
Profile forward and forward+backward cost of the glonet surface rollout
as a function of the number of autoregressive steps.

Motivation
----------
Production optimization (src/gd_optimic/optimizer.py) runs `num_iterations`
outer gradient-descent steps; each outer step does exactly ONE call to
`ForwardModel.forward(x0, num_forecast_steps)` (a multi-step rollout, e.g.
observation_length=7 steps) followed by one `torch.autograd.grad` backward.
550 outer iterations currently take ~7h on an H200 — not operational.

Before changing anything, we need a cost curve: how does forward-only time,
and forward+backward time, scale with the number of rollout steps? This
tells us whether the per-iteration cost is linear in steps (expected) or
super-linear (would point at the nested gradient-checkpointing structure:
the outer rollout is checkpointed as ONE block in ForwardModel.forward,
while each inner glonet call ALSO checkpoints its own sub-blocks).

This script uses the exact production `ForwardModel` wrapper
(src/gd_optimic/utils.py) with a real glonet checkpoint, but with synthetic
(random) initial conditions and an all-ocean mask, so no dataset I/O is
needed to isolate pure model compute cost.

Usage
-----
    .venv/bin/python .tmp/runs/profile_step_cost.py

Output
------
    .tmp/runs/profile_step_cost_results/results.json
    .tmp/runs/profile_step_cost_results/results.csv
    .tmp/runs/profile_step_cost_results/cost_curve.png
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

# -----------------------------------------------------------------------
# Make the production package importable (src/gd_optimic/utils.py)
# -----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# `gd_optimic/__init__.py` eagerly imports the whole package (dataset,
# optimizer, meta-learner, ...), which drags in heavy geospatial/CMEMS
# client deps (xesmf/esmpy, copernicusmarine) that ForwardModel itself
# never touches. Those may be absent on this interactive node even though
# they ARE present on the SLURM/H200 production environment. To keep this
# profiling script importable everywhere without editing production code,
# stub the optional deps AND register a bare 'gd_optimic' package object
# (with the real __path__) so `from gd_optimic.utils import ForwardModel`
# loads only utils.py + its direct deps, skipping __init__.py entirely.
import types

for _optional_dep in ("xesmf", "copernicusmarine"):
    try:
        __import__(_optional_dep)
    except ImportError:
        sys.modules[_optional_dep] = types.ModuleType(_optional_dep)

_gd_optimic_pkg = types.ModuleType("gd_optimic")
_gd_optimic_pkg.__path__ = [str(REPO_ROOT / "src" / "gd_optimic")]
sys.modules["gd_optimic"] = _gd_optimic_pkg

from gd_optimic.utils import ForwardModel  # noqa: E402


CONFIG = {
    "model_path": str(REPO_ROOT / "model" / "glonet" / "weights" / "glonet_part1.pth"),
    "normalizer_path": "/Odyssey/public/glonet/TrainedWeights",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "grid_shape": (672, 1440),   # 1/4 degree global
    "n_channels": 5,             # SSH, T, S, U, V
    "batch_size": 1,
    "time_steps": 2,             # glonet requires T=2 input
    # Rollout lengths to profile. 7 = the current assim-window length used
    # in production (configs/optimize_ic.yaml: data.observation_length).
    "step_counts": [1, 2, 3, 4, 5, 6, 7],
    "repeats": 3,      # timed repeats per (step_count, mode) after warmup
    "warmup": 1,       # untimed warmup repeats (excludes cuDNN autotune / allocator effects)
    # NOTE: kept under .tmp/runs/ (not .tmp/outputs/) so results stay
    # alongside the script and are not excluded by .gitignore's
    # ".tmp/* except .tmp/runs/" rule.
    "output_dir": REPO_ROOT / ".tmp" / "runs" / "profile_step_cost_results",
}


def make_ocean_mask():
    """All-ocean mask: isolates model compute cost from dataset land-mask I/O."""
    return torch.ones(CONFIG["n_channels"], *CONFIG["grid_shape"], device=CONFIG["device"])


def make_x0(requires_grad: bool):
    shape = (CONFIG["batch_size"], CONFIG["time_steps"], CONFIG["n_channels"], *CONFIG["grid_shape"])
    x0 = torch.randn(shape, device=CONFIG["device"]) * 0.01
    x0.requires_grad_(requires_grad)
    return x0


def sync():
    if CONFIG["device"] == "cuda":
        torch.cuda.synchronize()


def time_forward_only(forward_model, num_steps):
    """Mirrors the no_grad initial-prediction call in optimizer.optimize()."""
    x0 = make_x0(requires_grad=False)
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        forward_model.forward(x0, num_steps)
    sync()
    return time.perf_counter() - t0


def time_forward_backward(forward_model, num_steps):
    """Mirrors the production per-iteration step: forward + torch.autograd.grad."""
    x0 = make_x0(requires_grad=True)
    sync()
    t0 = time.perf_counter()
    _, y_hat = forward_model.forward(x0, num_steps)
    loss = y_hat.pow(2).mean()
    torch.autograd.grad(loss, x0)
    sync()
    return time.perf_counter() - t0


def peak_mem_gb():
    if CONFIG["device"] != "cuda":
        return None
    return torch.cuda.max_memory_allocated(CONFIG["device"]) / 1e9


def profile_mode(forward_model, num_steps, timing_fn):
    # Warmup (not timed) — first call pays cuDNN algo search / allocator growth.
    for _ in range(CONFIG["warmup"]):
        timing_fn(forward_model, num_steps)

    if CONFIG["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats(CONFIG["device"])

    times = [timing_fn(forward_model, num_steps) for _ in range(CONFIG["repeats"])]
    return {
        "times_s": times,
        "mean_s": sum(times) / len(times),
        "min_s": min(times),
        "max_s": max(times),
        "peak_mem_gb": peak_mem_gb(),
    }


def force_correct_model_class(forward_model):
    """Work around a production landmine in ForwardModel._load_model (src/gd_optimic/utils.py):

    When use_gradient_checkpointing=False, _load_model falls back to the RAW
    `Glonet` class (modelp2.py) instead of the `GlonetGradientCheckpointing`
    override. Per findings.md (R1), raw `Glonet.forward()` has hardcoded
    `.detach()` calls that break autograd back to the IC — it was never meant
    to be used for gradient-based optimization; GlonetGradientCheckpointing
    was written specifically to fix that. So the ForwardModel-level
    `use_gradient_checkpointing` flag conflates two unrelated things: (a)
    whether the OUTER rollout is wrapped in torch.utils.checkpoint, and (b)
    which model class gets loaded. For a fair "outer checkpoint on vs off"
    comparison we always want the differentiable class; only (a) should vary.

    This patches forward_model.model IN PLACE after construction — it does
    NOT modify src/gd_optimic/utils.py.
    """
    sys.path.insert(0, str(REPO_ROOT / "model" / "glonet"))
    from optimIC_GD_glonetLit import GlonetGradientCheckpointing

    checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    model = GlonetGradientCheckpointing(shape_in=(2, CONFIG["n_channels"], *CONFIG["grid_shape"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(CONFIG["device"])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    forward_model.model = model
    return forward_model


def run_sweep(use_gradient_checkpointing: bool):
    """Profile forward-only and forward+backward cost across step_counts.

    use_gradient_checkpointing mirrors ForwardModel's own flag (production
    default: True). Comparing True vs False isolates the cost contributed
    by the outer rollout-level checkpoint recomputation on backward. The
    model class is pinned to GlonetGradientCheckpointing in both cases (see
    force_correct_model_class) so this is an apples-to-apples comparison of
    the OUTER wrap only, not a comparison of two different model classes.
    """
    label = "checkpointed" if use_gradient_checkpointing else "no_checkpoint"
    print(f"\n{'='*70}\nSweep: {label}\n{'='*70}")

    ocean_mask = make_ocean_mask()
    forward_model = ForwardModel(
        model_path=CONFIG["model_path"],
        normalizer_path=CONFIG["normalizer_path"],
        device=CONFIG["device"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        ocean_mask=ocean_mask,
    )
    forward_model = force_correct_model_class(forward_model)

    rows = []
    for num_steps in CONFIG["step_counts"]:
        try:
            fwd = profile_mode(forward_model, num_steps, time_forward_only)
            fwd_bwd = profile_mode(forward_model, num_steps, time_forward_backward)
        except torch.cuda.OutOfMemoryError as e:  # noqa: F821 (py>=3.12 has this on torch)
            print(f"  steps={num_steps}: OOM ({e}) — stopping sweep here")
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()
            break

        backward_only_s = fwd_bwd["mean_s"] - fwd["mean_s"]
        row = {
            "config": label,
            "num_steps": num_steps,
            "forward_mean_s": fwd["mean_s"],
            "forward_backward_mean_s": fwd_bwd["mean_s"],
            "backward_only_mean_s": backward_only_s,
            "forward_peak_mem_gb": fwd["peak_mem_gb"],
            "forward_backward_peak_mem_gb": fwd_bwd["peak_mem_gb"],
        }
        rows.append(row)
        peak_mem_str = (
            f"{fwd_bwd['peak_mem_gb']:.2f}GB" if fwd_bwd["peak_mem_gb"] is not None else "n/a"
        )
        print(
            f"  steps={num_steps:2d}  "
            f"forward={fwd['mean_s']:6.3f}s  "
            f"fwd+bwd={fwd_bwd['mean_s']:6.3f}s  "
            f"backward~={backward_only_s:6.3f}s  "
            f"peak_mem(fwd+bwd)={peak_mem_str}"
        )

    # Free model before returning (each sweep loads its own copy).
    del forward_model
    if CONFIG["device"] == "cuda":
        torch.cuda.empty_cache()

    return rows


def report_scaling(rows, key):
    """Print per-step marginal cost to reveal linear vs super-linear scaling."""
    print(f"\nMarginal {key} cost per added rollout step:")
    prev = None
    for r in rows:
        marginal = None if prev is None else r[key] - prev
        marginal_str = f"{marginal:6.3f}s" if marginal is not None else "   n/a"
        print(f"  steps={r['num_steps']:2d}  total={r[key]:6.3f}s  marginal={marginal_str}")
        prev = r[key]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Override CONFIG['step_counts'], e.g. --steps 1 2 4 7",
    )
    parser.add_argument("--repeats", type=int, default=None, help="Override timed repeats per point")
    parser.add_argument("--warmup", type=int, default=None, help="Override untimed warmup repeats")
    parser.add_argument(
        "--skip-no-checkpoint", action="store_true",
        help="Only profile the production (checkpointed) config; skip the comparison sweep.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps is not None:
        CONFIG["step_counts"] = args.steps
    if args.repeats is not None:
        CONFIG["repeats"] = args.repeats
    if args.warmup is not None:
        CONFIG["warmup"] = args.warmup

    print("=" * 70)
    print("FORWARD / FORWARD+BACKWARD COST CURVE PROFILE")
    print("=" * 70)
    print(f"Device: {CONFIG['device']}")
    if CONFIG["device"] == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Step counts: {CONFIG['step_counts']}")
    print(f"Repeats per point: {CONFIG['repeats']} (+{CONFIG['warmup']} warmup)")

    out_dir = CONFIG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each sweep loads its own model copy and can fail independently (e.g.
    # OOM, or an autograd break). Never let one sweep's failure discard the
    # other's already-collected, potentially expensive-to-recompute rows.
    all_rows = []
    try:
        all_rows += run_sweep(use_gradient_checkpointing=True)
    except Exception:
        import traceback
        print("\nSweep 'checkpointed' FAILED (results collected so far are still kept):")
        traceback.print_exc()

    if not args.skip_no_checkpoint:
        try:
            all_rows += run_sweep(use_gradient_checkpointing=False)
        except Exception:
            import traceback
            print("\nSweep 'no_checkpoint' FAILED (results collected so far are still kept):")
            traceback.print_exc()

    checkpointed_rows = [r for r in all_rows if r["config"] == "checkpointed"]
    if checkpointed_rows:
        report_scaling(checkpointed_rows, "forward_mean_s")
        report_scaling(checkpointed_rows, "forward_backward_mean_s")

        # Sanity-check extrapolation against the reported 550-iteration / 7h run.
        seven_step_row = next((r for r in checkpointed_rows if r["num_steps"] == 7), None)
        if seven_step_row is not None:
            per_iter = seven_step_row["forward_backward_mean_s"]
            est_550 = per_iter * 550
            print(
                f"\nExtrapolation (checkpointed, 7 steps): "
                f"{per_iter:.3f}s/iteration -> {est_550/3600:.2f}h for 550 iterations "
                f"(on THIS GPU: {torch.cuda.get_device_name(0) if CONFIG['device']=='cuda' else 'cpu'}, "
                f"not necessarily H200)."
            )

    # ---- Save results ----
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": CONFIG_SERIALIZABLE(), "rows": all_rows}, f, indent=2)

    if all_rows:
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    plot_cost_curve(all_rows, out_dir / "cost_curve.png")

    print(f"\nResults saved to: {out_dir}")


def CONFIG_SERIALIZABLE():
    cfg = dict(CONFIG)
    cfg["output_dir"] = str(cfg["output_dir"])
    return cfg


def plot_cost_curve(rows, out_path):
    """Render forward-only and forward+backward time vs. rollout steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot, JSON/CSV still saved.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "checkpointed": {"linestyle": "-"},
        "no_checkpoint": {"linestyle": "--"},
    }
    for config_label, style in styles.items():
        subset = [r for r in rows if r["config"] == config_label]
        if not subset:
            continue
        steps = [r["num_steps"] for r in subset]
        ax.plot(
            steps, [r["forward_mean_s"] for r in subset],
            marker="o", color="tab:blue", label=f"forward only ({config_label})", **style,
        )
        ax.plot(
            steps, [r["forward_backward_mean_s"] for r in subset],
            marker="s", color="tab:red", label=f"forward+backward ({config_label})", **style,
        )

    ax.set_xlabel("Rollout steps (autoregressive days)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("glonet surface rollout: cost vs. number of steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
