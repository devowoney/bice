# Session Summary — Compute Cost Profiling (2026-09-16)

**Branch/worktree:** `compute-cost-profiling` (`.claude/worktrees/compute-cost-profiling`)
**Goal:** 550-iteration constant-step IC optimization takes ~7h on an H200 — not operational. Understand where the time goes and reduce it, without changing scientific behavior.

## Diagnostic tools built (all in `.tmp/runs/`, tracked in git)

All three write outputs into their own isolated `.tmp/runs/<name>_<timestamp>/` (or a dedicated results subfolder) — never the repo root — so concurrent runs never collide.

- **`profile_step_cost.py`** + `sb_profile_step_cost.sh` — forward-only / forward+backward wall-time and peak-memory cost curve vs. number of rollout steps, using the real `ForwardModel` (synthetic IC, all-ocean mask, no dataset I/O). Supports comparing the outer `ForwardModel`-level checkpoint on vs. off (`--skip-no-checkpoint` to disable).
- **`profile_optimizer_loop.py`** + `sb_profile_optimizer_loop.sh` — runs the *real* `ICOptimizer.optimize()` loop (same setup as `src/run_optimization_slurm.py`) for a small number of iterations, with in-process timing wrappers (no source edits) around every sub-step (forward, backward, loss, gradient filter, metrics, TensorBoard logging, checkpointing, setup, teardown) plus a tick-based true per-iteration wall-time series.
- **`profile_inner_checkpoint.py`** + `sb_profile_inner_checkpoint.sh` — sweeps which of the 4 internal checkpointed blocks inside a single glonet forward call (`spatial`, `latent`, `temporal`, `predictions`) are worth checkpointing, via a test-only subclass of `Glonet` (does not modify `model/glonet/optimIC_GD_glonetLit.py`).

## Findings

1. **Forward/backward cost is linear in rollout steps** (no runaway blow-up from the nested checkpointing structure by itself). At 7 steps (production assim window) on H200 NVL: forward ≈9.3s, backward ≈22.8s, fwd+bwd ≈32.1s, peak mem ≈67GB (with both outer and all 4 inner checkpoints on — today's production default).

2. **Real per-iteration loop cost breakdown** (instrumented, real data/model/obs):
   - ~87% model forward+backward compute (matches the isolated cost curve almost exactly — cross-validated).
   - ~9-12% TensorBoard image logging (`_log_to_tensorboard`, 2 full matplotlib figures rendered *every* iteration because `logging.log_frequency: 1` — not yet addressed).
   - Steady-state per-iteration: ~37-38s (not the naive ~32s from pure compute alone).
   - `_save_final_results` (end-of-run: 2 extra 28-step forecasts + NetCDF/PNG/GIF diagnostics) costs ~9 minutes, one-time, not iteration-scaling.
   - **Periodic performance bursts**: clusters of ~5 iterations run 30-50% slower than steady-state, recurring a few times per 80-iteration test. Not explained by checkpoint saves or histogram logging; likely external (node/allocator contention). Not resolved — would need cluster-level (`nvidia-smi`) monitoring to pin down, not just Python timers.
   - Extrapolated 550-iteration total (pre-fix): ~5.8-6.0h vs. observed ~7h — most of the gap explained; residual ~1h plausibly from the burst pattern compounding over a longer run.

3. **Nested/double gradient checkpointing** — `ForwardModel.forward()` (`src/gd_optimic/utils.py`) wraps the *entire* multi-step rollout in one **outer** `torch.utils.checkpoint`, while `GlonetGradientCheckpointing.forward()` (`model/glonet/optimIC_GD_glonetLit.py`) *also* wraps its own 4 internal blocks in **inner** checkpoints per model call. This forces a redundant third recompute pass over the 28 (7 steps × 4 blocks) inner-checkpointed regions during backward.
   - **Measured: removing the outer checkpoint costs zero memory** (peak identical: 66.97GB either way) while cutting backward time **41%** and total forward+backward **29%**.
   - I made this edit once; **the user reverted it** — they want `use_gradient_checkpointing` kept as a hyperparameter for low-memory GPUs rather than removed. No production code change is currently applied from this finding.

4. **Inner-block sweep results** (outer checkpoint off, 7 steps, H200 ~140GB budget):

   | block turned off | fwd+bwd | peak mem | vs. all-on | verdict |
   |---|---|---|---|---|
   | (none — all_on, current production) | 22.70s | 66.97GB | baseline | — |
   | `spatial` (jump+space, full-res) | — | **OOM** | — | must stay checkpointed |
   | `temporal` (dynamics/TeDev) | — | **OOM** | — | must stay checkpointed (surprising — assumed cheap since downsampled; wrong) |
   | `latent` (maps/Encoder) | 21.75s | 86.29GB | **−4.2%** | safe, worth doing (~54GB headroom left) |
   | `predictions` (mapsback/Decoder) | 22.14s | 126.42GB | −2.5% | risky — only ~14GB headroom for a small gain |
   | all 4 off | — | **OOM** | — | — |

   Combined potential (outer off + inner `latent` off) vs. today's actual default (everything checkpointed): 32.09s → 21.75s ≈ **−32%** forward+backward time, using 86GB of ~140GB.

5. **Two pre-existing production landmines found (not yet fixed, not requested to fix):**
   - `ForwardModel._load_model()` (`src/gd_optimic/utils.py`): when `use_gradient_checkpointing=False`, it loads the **raw** `Glonet` class instead of `GlonetGradientCheckpointing` — the raw class has hardcoded `.detach()` calls (documented in `findings.md` R1) that break autograd back to the IC. Anyone setting that flag to `False` today would silently get broken gradients. Discovered because it crashed the `no_checkpoint` comparison arm of `profile_step_cost.py` before being worked around test-side (`force_correct_model_class`).
   - `ICOptimizer.setup_directories()` (`src/gd_optimic/optimizer.py`) hardcodes `self.exp_dir = Path(".")`, ignoring `output_dir`. Masked in production by Hydra's `hydra.job.chdir: true`. My diagnostic scripts don't go through Hydra, so they replicate the chdir manually now (`profile_optimizer_loop.py` creates and `os.chdir()`s into its own `.tmp/runs/<exp_id>/` before calling `optimize()`) — this also fixed a real crash (two profiling runs collided writing to the same relative `tb/` path).
   - `_compute_finite_difference_ic_update` is called 3x per iteration on identical inputs (duplicate work, currently negligible cost — not fixed).

## Decisions / open items

- User wants `use_gradient_checkpointing` (outer wrap) kept as-is / configurable, not removed — my edit was reverted. Not applying the outer-checkpoint-off change to production for now.
- Next planned step (in progress when this summary was written): expose which of the 4 *inner* blocks get checkpointed as a configurable option (defaulting to today's safe all-on behavior), so `latent` can be turned off for high-memory GPUs like H200 without a code change. Not yet implemented.
- Do not turn off `spatial` or `temporal` inner checkpointing under any circumstance on a 140GB-class GPU — confirmed OOM.
- `predictions_off` is a marginal, risky option (2.5% gain, ~14GB headroom) — not recommended unless further memory margin is confirmed safe across other run configurations (different batch/step counts).
- `latent_off + predictions_off` combined was not tested (likely OOM — would exceed ~140GB); skip unless specifically requested.
- TensorBoard image logging every iteration (`log_frequency: 1`) is a known, still-unaddressed ~9-12%/iteration cost — throttling image logging specifically (not scalar logging) to a lower cadence (e.g. `histogram_frequency`) was proposed but not implemented.
- The periodic per-iteration slowdown bursts remain unexplained; not pursued further this session.
