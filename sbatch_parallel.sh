#!/bin/bash
#SBATCH --partition=Odyssey_GPU         # Partition name
#SBATCH --account=odyssey               # Account name
#SBATCH --qos=low

#SBATCH --nodes=1
#SBATCH --nodelist=sl-mee-br-210        # node request
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2              # 4 GPUs per node for parallel processing
#SBATCH --mem-per-gpu=160G            # Memory per GPU
#SBATCH --cpus-per-task=32            # CPUs per task

#SBATCH --job-name=batPar_mergeTest      # Job name for parallel optimization
#SBATCH --output=/Odyssey/private/j25lee/.bin/log/%x_%j.log     # Standard output and error log
# SBATCH --time=12:00:00

# ============================================================================
# BICE (ML-4DVar) Parallel IC Optimization SLURM Submission Script
# ============================================================================
# Usage:
#   sbatch sbatch_script.sh                           # Default parallel run
#   sbatch --job-name=test_parallel sbatch_script.sh  # Custom job name
#   sbatch --ntasks=8 sbatch_script.sh               # Use 8 GPUs
#   sbatch --time=24:00:00 sbatch_script.sh          # 24-hour limit
#
# Environment variables (optional):
#   BATCH_SIZE: Number of optimization samples to run (default: 50)
#   START_IDX: Index of the first sample (default: 0); runs START_IDX .. START_IDX+BATCH_SIZE-1
#   OPT_PARAMS: Additional Hydra parameters for the parallel script
#   LOG_LEVEL: Logging level (default: INFO)
#
# Example with parameters:
#   export BATCH_SIZE=100
#   export OPT_PARAMS="observations.mode=simulated optimization.num_iterations=500"
#   sbatch --ntasks=8 sbatch_script.sh
#
# ============================================================================


set -e  # Exit on error

# Print job info
echo "========================================================================"
echo "SLURM Job Information"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Time limit: $SLURM_TIMELIMIT"
echo "========================================================================"
echo ""

# Set working directory
export HOME=/Odyssey/private/$USER
source $HOME/.bashrc
mamba activate oceanai

# Disable Hydra's automatic directory creation to prevent nesting
# This overrides hydra.run.dir from the config
export HYDRA_JOB_CHDIR=false
export HYDRA_RUN_DIR=$(pwd)
export HYDRA_OUTPUT_SUBDIR=null

srun sleep 5

PROJECT_DIR="/Odyssey/private/j25lee/bice/"
cd "$PROJECT_DIR"

# # Create log directory
# mkdir -p logs

# # Setup uv environment
# echo "Setting up Python environment with uv..."
# export UV_PROJECT_DIRECTORY="$PROJECT_DIR"

# # Use pre-synced virtual environment (avoids uv key auth issues in sbatch)
# if [ ! -d .venv ]; then
#     echo "ERROR: .venv not found. Run 'uv sync --all-extras' locally first."
#     exit 1
# fi

# # Activate virtual environment
# source .venv/bin/activate

# Print Python info
echo ""
echo "========================================================================"
echo "Python Environment"
echo "========================================================================"
python --version
which python
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# SETUP DISTRIBUTED TRAINING ENVIRONMENT VARIABLES
# ============================================================================
# These are required for PyTorch distributed training
if [ -n "$SLURM_JOB_ID" ]; then
    # Get the first node from the node list for MASTER_ADDR
    MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_ADDR=$MASTER_NODE
    export MASTER_PORT=12345
    export WORLD_SIZE=$SLURM_NTASKS
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    
    echo "Distributed Training Environment:"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  RANK: $RANK"
    echo "  LOCAL_RANK: $LOCAL_RANK"
    echo ""
fi

# Verify GPU access
echo "========================================================================"
echo "GPU Status"
echo "========================================================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
        --format=csv,noheader || echo "⚠ nvidia-smi failed"
else
    echo "⚠ nvidia-smi not found"
fi
echo ""

# ============================================================================
# PARALLEL OPTIMIZATION PARAMETERS
# ============================================================================

# Set default config if not provided
OPT_CONFIG="${OPT_CONFIG:-optimize_ic}"

# Set default batch size (number of optimization samples)
BATCH_SIZE="${BATCH_SIZE:-4}"  # Default: 50 samples

# Set first sample index (samples START_IDX .. START_IDX+BATCH_SIZE-1 are processed)
START_IDX="${START_IDX:-0}"

# Set default parameters for parallel optimization
PARAM1="experiment.name=parallel_optim_test_outputFix"
PARAM2="data.batch_size=$BATCH_SIZE data.sample_idx=$START_IDX"  # Total number of samples, from START_IDX
PARAM3="optimization.num_iterations=20 \
        logging.save_frequency=10 \
        loss.weighting=manual"


# Combine all parameters
OPT_PARAMS="$PARAM1 $PARAM2 $PARAM3"
# ============================================================================
# CONSTRUCT AND RUN PARALLEL OPTIMIZATION COMMAND
# ============================================================================

# Command for parallel optimization
CMD="srun python src/run_optimization_slurm_parallel.py"

# Add config override if specified
if [ ! -z "$OPT_CONFIG" ]; then
    CMD="$CMD --config-name=$OPT_CONFIG"
fi

# Add any additional parameters
if [ ! -z "$OPT_PARAMS" ]; then
    CMD="$CMD $OPT_PARAMS"
fi

# Print command
echo "========================================================================"
echo "Running Parallel Optimization"
echo "========================================================================"
echo "Command: $CMD"
echo "Batch Size: $BATCH_SIZE samples (from index $START_IDX)"
echo "Number of Tasks: $SLURM_NTASKS processes"
echo "========================================================================"
echo ""

# Run parallel optimization
$CMD

# Print completion
echo ""
echo "========================================================================"
echo "Parallel Job Completed"
echo "========================================================================"
echo "Results saved to: .tmp/runs/"
echo "TensorBoard logs: .tmp/runs/"
echo ""
echo "To view results:"
echo "  tensorboard --logdir runs/"
echo "========================================================================"
