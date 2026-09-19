# Session Summary — `_load_model` Bug Fix, OOM Diagnosis (2026-09-19)

**Branch/worktree:** `compute-cost-profiling` (`.claude/worktrees/compute-cost-profiling`)
**Continues from:** `memory/2026-09-18_inner_checkpoint_feature_and_cleanup.md`

## Trigger

User ran a real 550-iteration production job (`compFullTest`, via their own `sb_run_optimization.sh`) with `configs/optimize_ic.yaml` set to `use_gradient_checkpointing: false` + `inner_checkpoint_blocks: {spatial: true, latent: false, temporal: true, predictions: true}`. It crashed immediately after computing the initial loss:

```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

## Root cause (confirmed, fixed)

`ForwardModel._load_model()` (`src/gd_optimic/utils.py`) used `use_gradient_checkpointing` to decide **which model class to load**:
- `True` → `GlonetGradientCheckpointing` (the differentiable override, and the only class that respects `inner_checkpoint_blocks`)
- `False` → raw `Glonet` (`model/glonet/modelp2.py`), which has hardcoded `.detach()` calls (documented in `findings.md` R1) that permanently sever the autograd graph back to the IC.

This is the exact landmine flagged (but left unfixed) in the 2026-09-16 and 2026-09-18 session notes — it had only been worked around inside diagnostic scripts (`force_correct_model_class` in `profile_step_cost.py`) until now. Setting `use_gradient_checkpointing: false` in a real config — which the user reasonably expected to only disable the *outer* rollout-level checkpoint — silently swapped in the broken model class instead, and separately made `inner_checkpoint_blocks` a no-op (only `GlonetGradientCheckpointing` reads it).

**Fix applied** (`src/gd_optimic/utils.py`, `_load_model`): removed the `if use_gradient_checkpointing / else` branch entirely. `GlonetGradientCheckpointing` is now **always** loaded, regardless of `use_gradient_checkpointing`. That flag now correctly controls *only* the outer wrap in `ForwardModel.forward()`, which was always its sole legitimate purpose. `inner_checkpoint_blocks` is always honored.

**Verified**: resubmitting the exact same `compFullTest` config (job 54206) got past the point of the original crash — it now reaches deep into the real forward/backward rollout (traceback shows it failing inside `mapsback` → `group_norm`, i.e. well past the point where autograd used to break) before hitting an unrelated CUDA OOM (see below). No `grad_fn` error this time — the fix works.

## Separate issue: CUDA OOM on job 54206 — NOT a code bug

That resubmitted run then failed with:
```
torch.OutOfMemoryError: ... Process 2268909 has 35.29 GiB ... Process 3016939 has 34.69 GiB ...
Process 3033581 has 34.78 GiB ... this process has 34.06 GiB ...
```
Three *other* processes were already resident on the GPU (~105GB combined) before this job even started — the job's own GPU-status header showed only 35879 MiB free out of 143771 MiB total at launch. The run landed on **`sl-mee-br-215`**, which had an unrelated `interactive` session (user `s26ny`) running on it at the time — confirmed via `sinfo`/`squeue` cluster-wide. Their chosen config (`use_gradient_checkpointing: false` + `latent: false`) needs ~86GB alone (per the 2026-09-16 profiling), which comfortably fits an *exclusive* H200 (~140GB) but not a node with only ~35GB free.

**Cluster check at time of writing:** `sl-mee-br-214` (2× H200) was fully `idle` with zero jobs queued/running on it; `sl-mee-br-215` and `sl-mee-br-216` both had other users' jobs running. The user's own `sb_run_optimization.sh` (their personal SLURM script, not one authored this session) targets `sl-mee-br-215` (line 7) — an edit to point it at `sl-mee-br-214` instead was proposed but **rejected by the user**; that script is theirs to manage, not something to modify without being asked.

## Status

- `src/gd_optimic/utils.py`: `_load_model` fix applied, uncommitted, on `compute-cost-profiling`.
- The `use_gradient_checkpointing=False` + `inner_checkpoint_blocks` combination in `configs/optimize_ic.yaml` (as currently set on disk) should now work correctly memory/gradient-wise; remaining risk for the next attempt is purely node contention, not code.
- No changes made to `sb_run_optimization.sh` (user's own script) — user manages node selection themselves.
- My own verification job for this fix (job 54202, 5-iteration smoke test) was cancelled by the user before completing (`sacct`: `CANCELLED`, 11m elapsed) — the confirmation that the fix works comes from the user's own `compFullTest_54206.log`, not from a clean run on my side.

## Open items carried forward (unchanged)

- TensorBoard image logging every iteration (`log_frequency: 1`) still costs ~9-12%/iteration — not throttled.
- Periodic per-iteration performance bursts remain unexplained (out of scope this session).
- Outer-checkpoint removal (29% faster, zero memory cost, measured 2026-09-16) still not applied to production — kept as the existing `use_gradient_checkpointing` hyperparameter per user preference.
