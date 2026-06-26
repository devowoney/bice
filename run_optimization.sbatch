#!/bin/bash
#SBATCH --partition=Odyssey_GPU         # Partition name
#SBATCH --account=odyssey               # Account name
#SBATCH --qos=low

#SBATCH --nodes=1
#SBATCH --nodelist=sl-mee-br-211                        # node request
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1                       # 2 GPUs per node (4 total across 2 nodes)
#SBATCH --mem-per-gpu=200G                      # Memory per GPU
#SBATCH --cpus-per-task=32                      # CPUs per GPU

#SBATCH --job-name=bice_output
#SBATCH --output=./logs/%x_%j.log             # Standard output and error log (%x for j_name, %j for j_id)
# SBATCH --time=12:00:00

# ============================================================================
# BICE (ML-4DVar) IC Optimization SLURM Submission Script
# ============================================================================
# Usage:
#   sbatch run_optimization.sbatch                           # Default (1x A100)
#   sbatch --job-name=test run_optimization.sbatch           # Custom job name
#   sbatch --gpus-per-node=h100:1 run_optimization.sbatch    # Use H100
#   sbatch --time=24:00:00 run_optimization.sbatch           # 24-hour limit
#
# Environment variables (optional):
#   OPT_CONFIG: Override config file (default: optimize_ic)
#   OPT_PARAMS: Additional Hydra parameters
#   LOG_LEVEL: Logging level (default: INFO)
#
# Example with parameters:
#   export OPT_PARAMS="observations.mode=simulated optimization.num_iterations=500"
#   sbatch --gpus-per-node=a100:2 run_optimization.sbatch
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

srun sleep 5

PROJECT_DIR="/Odyssey/private/j25lee/bice"
cd "$PROJECT_DIR"

# Create log directory
mkdir -p logs

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

# Set default config if not provided
OPT_CONFIG="${OPT_CONFIG:-optimize_ic}"
# OPT_PARAMS="${OPT_PARAMS:-}"
OPT_PARAMS="optimization.num_iterations=50 logging.save_frequency=10 logging.histogram_frequency=20"

# Construct command
CMD="python src/run_optimization_slurm.py"

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
echo "Running Optimization"
echo "========================================================================"
echo "Command: $CMD"
echo "========================================================================"
echo ""

# Run optimization
$CMD

# Print completion
echo ""
echo "========================================================================"
echo "Job Completed"
echo "========================================================================"
echo "Results saved to: .tmp/runs/"
echo "TensorBoard logs: .tmp/runs/"
echo ""
echo "To view results:"
echo "  tensorboard --logdir runs/"
echo "========================================================================"
