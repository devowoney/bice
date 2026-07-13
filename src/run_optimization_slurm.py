#!/usr/bin/env python3
"""
SLURM-aware IC optimization runner.

Detects SLURM environment and configures resources automatically.
Compatible with submitit for job submission.

Usage:
    python run_optimization_slurm.py                              # Local or SLURM
    python run_optimization_slurm.py observations.mode=simulated  # Override params
    python -m submitit.launch_job run_optimization_slurm.py       # Submitit launcher
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gd_optimic import (
    GlonetDataset,
    ObservationOperator,
    ObservationLoss,
    GradientFilter,
    ICOptimizer,
    MetricsComputer,
    MaskBuilder,
    ForwardModel,
    ScheduledPooling,
)

def setup_slurm_environment(cfg: DictConfig) -> DictConfig:
    """
    Detect and configure SLURM environment.

    Automatically sets:
    - GPU device(s) from SLURM_GPUS
    - num_workers based on available CPUs
    - output paths with job ID

    Args:
        cfg: Hydra configuration

    Returns:
        Updated configuration with SLURM settings
    """
    # Check if running under SLURM
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ntasks = os.environ.get("SLURM_NTASKS", "1")
    gpus = os.environ.get("SLURM_GPUS", "0")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    job_name = os.environ.get("SLURM_JOB_NAME", "optimization")

    # Log SLURM info
    logger.info("\n" + "=" * 60)
    logger.info("SLURM Environment Detection")
    logger.info("=" * 60)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Job Name: {job_name}")
    logger.info(f"N Tasks: {ntasks}")
    logger.info(f"GPUs: {gpus}")
    logger.info(f"CPUs per task: {cpus_per_task}")

    # Verify GPU availability
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}")
    else:
        logger.warning("No GPU detected - will use CPU (slow!)")

    logger.info("=" * 60 + "\n")

    # Update config with SLURM settings
    cfg.compute.num_workers = min(int(cpus_per_task), 4)

    # # Append job ID to output directory
    # if job_id != "local":
    #     job_suffix = f"_job{job_id}"
    #     cfg.logging.output_dir = cfg.logging.output_dir + job_suffix

    return cfg


def setup_logging(cfg: DictConfig):
    """
    Configure Python logging.

    Args:
        cfg: Hydra configuration
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging level: {log_level}")


def setup_reproducibility(seed: int = 42):
    """
    Set up reproducibility (A5 - scientific integrity).

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="../configs", config_name="optimize_ic")
def main(cfg: DictConfig):
    """
    Main optimization function with SLURM integration.

    Args:
        cfg: Hydra configuration
    """
    # Setup SLURM environment
    cfg = setup_slurm_environment(cfg)

    # Setup logging
    setup_logging(cfg)

    # Setup reproducibility (A5)
    setup_reproducibility(seed=42)

    logger.info("\n" + "=" * 60)
    logger.info("IC Optimization with SLURM Integration")
    logger.info("=" * 60)
    logger.debug("Configuration:\n" + OmegaConf.to_yaml(cfg))
    logger.info("=" * 60 + "\n")

    device = cfg.compute.device if torch.cuda.is_available() else "cpu"
    cfg.compute.device = device
    logger.info(f"Using device: {device}")

    # Generate experiment ID
    if cfg.logging.auto_generate_exp_id:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_id = f"{cfg.experiment.name}_{timestamp}"
    else:
        exp_id = cfg.experiment.name

    logger.info(f"Experiment ID: {exp_id}\n")
    # # Create experiment directory under the configured base output_dir (e.g., runs/{exp_id})
    # base_output = Path(cfg.logging.output_dir)
    # exp_output_dir = base_output / exp_id
    exp_output_dir = Path(".")
    # exp_output_dir.mkdir(parents=True, exist_ok=True)
    # Save a copy of the active config to the experiment folder for reproducibility
    cfg_path = exp_output_dir / "config.yaml"
    with open(cfg_path, "w") as cf:
        cf.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Experiment directory created: {exp_output_dir}")

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    logger.info("Loading dataset...")
    dataset = GlonetDataset(
        data_path=cfg.data.root_path,
        data_files=cfg.data.init_state_files,
        ssh_obs_files=cfg.data.ssh_obs_files if cfg.observations.mode != "full" else None,
        sst_obs_files=cfg.data.sst_obs_files if cfg.observations.mode != "full" else None,
        lazy_load=True,
    )

    # Extract sequences
    input_sequence = dataset.get_sequence(
        start_idx=cfg.data.sample_idx, length=cfg.data.sequence_length
    )

    target_start_idx = cfg.data.sample_idx + cfg.data.sequence_length
    target_end_idx = target_start_idx + cfg.data.observation_length
    target_sequence = dataset.get_sequence(
        start_idx=target_start_idx, length=cfg.data.observation_length
    )
    
    # Load full ground truth for RMSE diagnostics (T=0 to T=forecast_horizon)
    ground_truth_end_idx = target_start_idx + cfg.data.forecast_horizon
    ground_truth_sequence = dataset.get_sequence(
        start_idx=target_start_idx, length=cfg.data.forecast_horizon
    )

    # Align all loaded sequences to the same grid before applying observation operators.
    input_sequence = dataset.align_grid(input_sequence, input_sequence)
    target_sequence = dataset.align_grid(target_sequence, input_sequence)
    ground_truth_sequence = dataset.align_grid(ground_truth_sequence, input_sequence)

    logger.info(f"Loaded dataset")
    logger.debug(f"  Input sequence: {input_sequence['data'].shape}")
    logger.debug(f"  Target sequence (assimilation): {target_sequence['data'].shape}")
    logger.debug(f"  Ground truth sequence (full forecast): {ground_truth_sequence['data'].shape}")
    # -------------------------------------------------------------------------
    # 2. Apply Observation Operators
    # -------------------------------------------------------------------------
    logger.info("Applying observation operators...")
    obs_operator = ObservationOperator(device=device)

    ssh_mask = None
    sst_mask = None

    if cfg.observations.mode != "full":
        ssh_obs = dataset.get_ssh_obs(target_start_idx, cfg.data.observation_length)
        target_sequence, ssh_mask = obs_operator.apply_ssh_operator(
            target_sequence, ssh_obs, cfg.observations.mode
        )

        sst_obs = dataset.get_sst_obs(target_start_idx, cfg.data.observation_length)
        target_sequence, sst_mask = obs_operator.apply_sst_operator(
            target_sequence, sst_obs, cfg.observations.mode
        )

    logger.info(f"Applied observation operators (mode: {cfg.observations.mode})")
    # -------------------------------------------------------------------------
    # 3. Create Masks
    # -------------------------------------------------------------------------
    logger.info("Creating masks...")
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
        ocean_mask,
        cfg.data.observation_length,
        ssh_nanmask=ssh_mask,
        sst_nanmask=sst_mask,
        obs_mode=cfg.observations.mode,
    )

    logger.info("Created masks")
    # -------------------------------------------------------------------------
    # 4. Prepare Data Tensors
    # -------------------------------------------------------------------------
    logger.info("Preparing data tensors...")

    input_data = input_sequence["data"].values
    input_data = np.nan_to_num(input_data, nan=0.0)
    x0_init = torch.from_numpy(input_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)

    target_data = target_sequence["data"].values
    target_data = np.nan_to_num(target_data, nan=0.0)
    target_tensor = (
        torch.from_numpy(target_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
    )
     
    # Convert full ground truth for RMSE diagnostics
    ground_truth_data = ground_truth_sequence["data"].values
    ground_truth_data = np.nan_to_num(ground_truth_data, nan=0.0)
    ground_truth_tensor = (
        torch.from_numpy(ground_truth_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
    )

    logger.info("Prepared tensors")
    # -------------------------------------------------------------------------
    # 5. Initialize Forward Model
    # -------------------------------------------------------------------------
    logger.info("Initializing forward model...")
    forward_model = ForwardModel(
        model_path=str(Path(cfg.model.location) / cfg.model.checkpoint_files.part1),
        normalizer_path=cfg.model.location,
        device=device,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
        ocean_mask=ocean_mask,
    )
    logger.info("Initialized forward model")

    # -------------------------------------------------------------------------
    # 6. Initialize Loss Function
    # -------------------------------------------------------------------------
    logger.info("Initializing loss function...")
    loss_fn = ObservationLoss(
        obs_mask=obs_mask,
        loss_weighting=cfg.loss.weighting,
        manual_weights=cfg.loss.manual_weights if cfg.loss.weighting == "manual" else None,
        device=device,
    )
    logger.info("Initialized loss function")
    # -------------------------------------------------------------------------
    # 7. Initialize Gradient Filter
    # -------------------------------------------------------------------------
    logger.info("Initializing gradient filter...")
    gradient_filter = GradientFilter(filter_type=cfg.optimization.gradient_filter, device=device)

    scheduled_pooling = None
    if cfg.optimization.use_scheduled_pooling:
        scheduled_pooling = ScheduledPooling(
            schedule_type=cfg.optimization.pooling_schedule.type,
            num_iterations=cfg.optimization.num_iterations,
            initial_kernel=cfg.optimization.pooling_schedule.initial_kernel,
            final_kernel=cfg.optimization.pooling_schedule.final_kernel,
            schedule_steps=cfg.optimization.pooling_schedule.schedule_steps,
            kernel_sizes=cfg.optimization.pooling_schedule.kernel_sizes,
        )
    logger.info("Initialized gradient filter")

    # -------------------------------------------------------------------------
    # 8. Initialize Metrics Computer
    # -------------------------------------------------------------------------
    logger.info("Initializing metrics computer...")
    metrics_computer = MetricsComputer(ocean_mask=ocean_mask, device=device)
    logger.info("Initialized metrics computer")
    # -------------------------------------------------------------------------
    # 9. Initialize Optimizer
    # -------------------------------------------------------------------------
    logger.info("Initializing optimizer...")

    # Create output directories following project structure
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
        scheduled_pooling=scheduled_pooling,
        forecast_horizon=cfg.data.forecast_horizon,
    )
    logger.info("Initialized optimizer")

    # -------------------------------------------------------------------------
    # 10. Run Optimization
    # -------------------------------------------------------------------------
    logger.info("Starting optimization...")

    best_x0, results = optimizer.optimize(
        x0_init=x0_init,
        target_sequence=target_tensor,
        ocean_mask=ocean_mask,
        exp_id=exp_id,
        regional_masks=regional_masks,
        input_sequence_xr=input_sequence,
        target_sequence_xr=target_sequence,
        ground_truth_sequence_xr=ground_truth_sequence
    )

    logger.info("\n" + "=" * 60)
    logger.info("Optimization Complete!")
    logger.info("=" * 60)
    logger.info(f"Best loss: {results['best_loss']:.6f} (iteration {results['best_iteration']})")
    logger.info(f"Initial loss: {results['initial_loss']:.6f}")
    logger.info(f"Final loss: {results['final_loss']:.6f}")
    improvement = (1 - results["final_loss"] / results["initial_loss"]) * 100
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info("=" * 60 + "\n")

    exp_output_dir = Path(cfg.logging.output_dir) / exp_id
    logger.info(f"Results saved to: {exp_output_dir}")


if __name__ == "__main__":
    main()
