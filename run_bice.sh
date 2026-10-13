#!/bin/bash
# ============================================================================
# BICE (ML-4DVar) parallel IC optimization launcher -- standalone 4x H100 server (no SLURM)
# ============================================================================
# Usage:  edit the PARAMETERS section below, then run
#
#     ./run_bice.sh
#
# Launches one process per GPU. Each process gets the SLURM_* variables that
# src/run_optimization_slurm_parallel.py reads (NTASKS / PROCID / LOCALID), so the
# samples are split across GPUs exactly as under srun.
#
# Artifacts : $ARTIFACTS_ROOT/runs/parallel_<name>/<timestamp>/    single run
#             $ARTIFACTS_ROOT/multiruns/<name>_<timestamp>/<job>/  hydra sweep (MULTIRUN=true)
#             one gpu<rank>_sample<idx>/ folder per sample (~5 GB each), rank<k>.log per GPU
# ============================================================================

# ============================================================================
# PARAMETERS
# ============================================================================

# ---- Run ----
EXP_NAME="table4_noiseIC_1em1"   # experiment.name
GPUS="0,1,2,3"                 # GPU ids, one process each
BATCH_SIZE=50                  # number of samples (data.batch_size)
START_IDX=0                    # first sample (data.sample_idx): runs START_IDX .. START_IDX+BATCH_SIZE-1
                               # (sample 0 = IC on 2021-01-01; 365 days available)
CONFIG="batch_parallel_optim"  # hydra config in configs/

RUN_IN_BACKGROUND=true         # true: detach with nohup (survives logout), follow the printed tail command
DRY_RUN=false                  # true: only print the resolved command
MULTIRUN=false                 # true: hydra sweep (-m); give comma separated values below, e.g.
                               #       "observations.mode=10,20,50" -> one job per value, run in sequence

# ---- Paths ----
DATA_DIR="/mnt/shared/bice"                       # input states, stats, SSH variance
MODEL_DIR="/mnt/shared/bice/TrainedWeights"       # glonet_part1.pth + L0/*.npy normalizers
ARTIFACTS_ROOT="/mnt/shared/bice/tmp"             # runs/ and multiruns/ (root disk is too small)
CONDA_ENV="/home/sysadmin/jlee/miniforge3/envs/oceanai"

# ---- Hydra overrides (any key of configs/$CONFIG.yaml) ----
HYDRA_OVERRIDES=(
    # Observations
    "observations.mode=full"            # full | simulated | real | <0-100> blocks observed
    "observations.noise=false"
    "observations.noise_scale=1.0"
    "initial_condition.noise=true"
    "initial_condition.noise_scale=0.1"

    # Optimizer (tab4 reference)
    "optimization.num_iterations=550"
    "optimization.learning_rate=10"
    "optimization.ic_optimizer.name=adagrad"
    "optimization.use_gradient_smoothing=true"
    "optimization.use_scheduled_pooling=true"
    "optimization.pooling_schedule.schedule_steps=[50,100,150,200,250,300,350,400,450,500]"
    "optimization.pooling_schedule.kernel_sizes=[24,16,8,4,2,1,4,2,1,2,1]"

    # Model memory -- H100 80GB: all inner blocks checkpointed (latent=false OOMs at >79GB,
    # all true peaks at ~69GB). The outer rollout checkpoint adds ~40% time and saves nothing.
    "model.use_gradient_checkpointing=false"
    "model.inner_checkpoint_blocks.spatial=true"
    "model.inner_checkpoint_blocks.latent=true"
    "model.inner_checkpoint_blocks.temporal=true"
    "model.inner_checkpoint_blocks.predictions=true"

    # Logging
    "logging.save_frequency=50"
)

# ============================================================================
# END OF PARAMETERS
# ============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INIT_FILE="$DATA_DIR/glorys12_2021-01-01_to_2021-12-31_init_states/combined_input_2021-01-01_to_2021-12-31.nc"
STATS_FILE="$DATA_DIR/glorys12_2021-01-01_to_2021-12-31_init_states/stats/mean.nc"
VAR_FILE="$DATA_DIR/variance_cmems_dt_allsat.nc"

for f in "$INIT_FILE" "$STATS_FILE" "$VAR_FILE" "$MODEL_DIR/glonet_part1.pth" "$MODEL_DIR/L0/zos_mean.npy"; do
    [[ -e "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done
if [[ $(stat -c %s "$MODEL_DIR/glonet_part1.pth") -lt 1000000 ]]; then
    echo "ERROR: $MODEL_DIR/glonet_part1.pth is a git-lfs pointer, not the checkpoint"; exit 1
fi

# One timestamp shared by all ranks (each rank would otherwise resolve ${now:} separately), so
# they all write to the same hydra run/sweep dir. A detached child reuses the parent's.
TIMESTAMP="${BICE_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
if [[ $MULTIRUN == true ]]; then
    RUN_DIR="$ARTIFACTS_ROOT/multiruns/${EXP_NAME}_${TIMESTAMP}"
    MODE_FLAG=(--multirun)
else
    RUN_DIR="$ARTIFACTS_ROOT/runs/parallel_${EXP_NAME}/${TIMESTAMP}"
    MODE_FLAG=()
fi

# Plumbing overrides (paths, sample range, run dir) appended after the user ones
OVERRIDES=(
    "${HYDRA_OVERRIDES[@]}"
    "experiment.name=$EXP_NAME"
    "experiment.timestamp=$TIMESTAMP"
    "artifacts_root=$ARTIFACTS_ROOT"
    "data.batch_size=$BATCH_SIZE"
    "data.sample_idx=$START_IDX"
    "data.root_path=$DATA_DIR"
    "data.init_state_files=[$INIT_FILE]"
    "data.stats_file=$STATS_FILE"
    "data.variance_ssh_path=$VAR_FILE"
    "model.location=$MODEL_DIR"
)

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC=${#GPU_LIST[@]}

if [[ -z "${BICE_CHILD:-}" ]]; then
    echo "========================================================================"
    echo "BICE parallel optimization"
    echo "========================================================================"
    echo "Experiment : $EXP_NAME$([[ $MULTIRUN == true ]] && echo '  (hydra multirun)')"
    echo "GPUs       : $GPUS ($NPROC processes)"
    echo "Samples    : $START_IDX .. $((START_IDX + BATCH_SIZE - 1))"
    echo "Config     : $CONFIG"
    echo "Run dir    : $RUN_DIR"
    echo "Overrides  :"; printf '    %s\n' "${OVERRIDES[@]}"
    echo "========================================================================"
fi

if [[ $DRY_RUN == true ]]; then exit 0; fi

# ~5 GB per sample (best_solution.pt + states/ + checkpoints every 50 iterations).
# A full disk does not crash the ranks: output_handler logs it and the outputs are lost.
mkdir -p "$RUN_DIR"
NEED_GB=$(( BATCH_SIZE * 5 ))
FREE_GB=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc '0-9')
if (( FREE_GB < NEED_GB )); then
    echo "ERROR: $ARTIFACTS_ROOT has ${FREE_GB} GB free, run needs ~${NEED_GB} GB"; exit 1
fi

if [[ $RUN_IN_BACKGROUND == true && -z "${BICE_CHILD:-}" ]]; then
    BICE_CHILD=1 BICE_TIMESTAMP="$TIMESTAMP" nohup "$0" > "$RUN_DIR/launcher.log" 2>&1 &
    echo "Started in background (PID $!). Follow with:"
    echo "  tail -f $RUN_DIR/launcher.log $RUN_DIR/rank*.log"
    echo "Stop with: kill $!"
    exit 0
fi

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
PYTHON="$CONDA_ENV/bin/python"
# torch pulls in the system libstdc++ first, which is too old for the env's netCDF4/pnetcdf
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# Keep the cores shared fairly between the ranks
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) / NPROC / 2 ))}"

cd "$PROJECT_DIR"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

PIDS=()
cleanup() { echo "Stopping ranks..."; kill "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup INT TERM

for rank in "${!GPU_LIST[@]}"; do
    # Each rank sees only its GPU, so it is cuda:0 inside the process
    CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" \
    SLURM_JOB_ID="bice_${TIMESTAMP}" SLURM_JOB_NAME="$EXP_NAME" \
    SLURM_NTASKS="$NPROC" SLURM_PROCID="$rank" SLURM_LOCALID=0 \
    SLURM_CPUS_PER_TASK="$OMP_NUM_THREADS" \
        "$PYTHON" src/run_optimization_slurm_parallel.py --config-name="$CONFIG" "${MODE_FLAG[@]}" "${OVERRIDES[@]}" \
        > "$RUN_DIR/rank${rank}.log" 2>&1 &
    PIDS+=($!)
    echo "rank $rank -> GPU ${GPU_LIST[$rank]} (PID ${PIDS[-1]}), log: $RUN_DIR/rank${rank}.log"
done

FAILED=0
for rank in "${!PIDS[@]}"; do
    if wait "${PIDS[$rank]}"; then
        echo "rank $rank finished OK"
    else
        echo "rank $rank FAILED (exit $?), see $RUN_DIR/rank${rank}.log"; FAILED=1
    fi
done

# output_handler logs save failures (e.g. disk full) but the rank still exits 0
if grep -l "Error saving outputs" "$RUN_DIR"/rank*.log 2>/dev/null; then
    echo "Output saving FAILED in the logs above (grep 'Error saving outputs')"; FAILED=1
fi

echo "========================================================================"
[[ $FAILED -eq 0 ]] && echo "All ranks completed. Results: $RUN_DIR" || echo "Some ranks failed. Results: $RUN_DIR"
echo "TensorBoard: tensorboard --logdir $RUN_DIR"
echo "========================================================================"
exit $FAILED
