#!/usr/bin/env python3
"""
SLURM-aware Parallel IC optimization runner.

Detects SLURM distributed environment and runs parallel optimizations
across available GPUs. Each GPU handles a minibatch of optimization problems.

Usage:
    python run_optimization_slurm_parallel.py                              # Local or SLURM
    python run_optimization_slurm_parallel.py observations.mode=simulated  # Override params
    srun python run_optimization_slurm_parallel.py       # SLURM distributed launch
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





def setup_slurm_process_info():
    """
    Get process information from SLURM environment variables.
    Simple SLURM-based process detection without PyTorch distributed training.
    
    Returns:
        Tuple of (world_size, global_rank, local_rank, device)
    """
    # Check if running under SLURM
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Get process information from SLURM
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    global_rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    
    # For non-SLURM environments, use single process
    if job_id == "local":
        world_size = 1
        global_rank = 0
        local_rank = 0
    
    # Set device based on local rank
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"
    
    return world_size, global_rank, local_rank, device

def setup_slurm_environment(cfg: DictConfig, local_rank: int = 0) -> DictConfig:
    """
    Detect and configure SLURM environment.

    Automatically sets:
    - GPU device(s) from SLURM_GPUS
    - num_workers based on available CPUs
    - output paths with job ID

    Args:
        cfg: Hydra configuration
        local_rank: Local rank for GPU assignment

    Returns:
        Updated configuration with SLURM settings
    """
    # Check if running under SLURM
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ntasks = os.environ.get("SLURM_NTASKS", "1")
    gpus = os.environ.get("SLURM_GPUS", "0")
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    job_name = os.environ.get("SLURM_JOB_NAME", "optimization")

    # Log SLURM info (only from rank 0 to avoid clutter)
    if local_rank == 0:
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

    return cfg


def setup_logging(cfg: DictConfig, global_rank: int = 0):
    """
    Configure Python logging for distributed environment.

    Args:
        cfg: Hydra configuration
        global_rank: Global rank for logging setup
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO")

    # Only configure logging for rank 0 to avoid duplicate logs
    if global_rank == 0:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        # Suppress logging for non-rank-0 processes
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    logger = logging.getLogger(__name__)
    if global_rank == 0:
        logger.info(f"Logging level: {log_level}")


def setup_reproducibility(seed: int = 42, global_rank: int = 0):
    """
    Set up reproducibility for distributed training.

    Args:
        seed: Random seed
        global_rank: Global rank for seed adjustment
    """
    # Use different seeds for different processes to avoid correlation
    process_seed = seed + global_rank
    
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(process_seed)
        torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_batch_indices(total_batch_size: int, world_size: int, global_rank: int, first_idx: int = 0):
    """
    Calculate start and end indices for this process's batch.
    
    Args:
        total_batch_size: Total number of optimization problems
        world_size: Total number of processes
        global_rank: Current process rank
        first_idx: Dataset index of the first sample (offset applied to all ranks)
        
    Returns:
        Tuple of (start_idx, end_idx) for this process's batch
    """
    # Calculate batch size per process
    batch_size_per_process = total_batch_size // world_size
    remainder = total_batch_size % world_size
    
    # Distribute remainder across first few processes
    if global_rank < remainder:
        start_idx = global_rank * (batch_size_per_process + 1)
        end_idx = start_idx + (batch_size_per_process + 1)
    else:
        start_idx = global_rank * batch_size_per_process + remainder
        end_idx = start_idx + batch_size_per_process
    
    return first_idx + start_idx, first_idx + end_idx


def run_optimization_on_batch(
    cfg: DictConfig,
    dataset,
    obs_operator: ObservationOperator,
    mask_builder: MaskBuilder,
    forward_model: ForwardModel,
    loss_fn: ObservationLoss,
    gradient_filter: GradientFilter,
    scheduled_pooling,
    metrics_computer: MetricsComputer,
    device: str,
    batch_start_idx: int,
    batch_end_idx: int,
    global_rank: int,
    local_rank: int,
    exp_id: str
):
    """
    Run optimization on a batch of samples SEQUENTIALLY.
    
    IMPORTANT: This function processes samples one-by-one in a for loop,
    NOT in parallel. This ensures that each GPU only runs one optimization
    at a time, preventing OOM errors. The parallelism comes from different
    GPUs handling different samples simultaneously.
    
    Args:
        cfg: Hydra configuration
        dataset: Dataset object
        obs_operator: Observation operator
        mask_builder: Mask builder
        forward_model: Forward model (shared across samples)
        loss_fn: Loss function (updated per sample)
        gradient_filter: Gradient filter (shared across samples)
        scheduled_pooling: Scheduled pooling object
        metrics_computer: Metrics computer (updated per sample)
        device: Device to use
        batch_start_idx: Start index for this batch
        batch_end_idx: End index for this batch
        global_rank: Global rank
        local_rank: Local rank
        exp_id: Experiment ID
        
    Returns:
        Dictionary with optimization results
    """
    results = {}
    
    for sample_idx in range(batch_start_idx, batch_end_idx):
        if global_rank == 0:
            logger.info(f"Processing sample {sample_idx} on rank {global_rank}")
        
        # Extract sequences for this sample
        input_sequence = dataset.get_sequence(
            start_idx=sample_idx, length=cfg.data.sequence_length
        )

        target_start_idx = sample_idx + cfg.data.sequence_length
        target_end_idx = target_start_idx + cfg.data.observation_length
        target_sequence = dataset.get_sequence(
            start_idx=target_start_idx, length=cfg.data.observation_length
        )
        
        # Load full ground truth for RMSE diagnostics
        ground_truth_end_idx = target_start_idx + cfg.data.forecast_horizon
        ground_truth_sequence = dataset.get_sequence(
            start_idx=target_start_idx, length=cfg.data.forecast_horizon
        )

        # Align all loaded sequences to the same grid
        input_sequence = dataset.align_grid(input_sequence, input_sequence)
        target_sequence = dataset.align_grid(target_sequence, input_sequence)
        ground_truth_sequence = dataset.align_grid(ground_truth_sequence, input_sequence)

        # Apply observation operators
        stats_field = None
        if cfg.data.get('stats_file', None):
            try:
                import xarray as _xr
                from pathlib import Path as _Path
                stats_ds = _xr.open_dataset(_Path(cfg.data.stats_file))
                if 'data' in stats_ds.data_vars:
                    stats_field = stats_ds['data']
                else:
                    first_var = list(stats_ds.data_vars)[0]
                    stats_field = stats_ds[first_var]
            except Exception as e:
                if global_rank == 0:
                    logger.warning(f"Failed to load stats file {cfg.data.stats_file}: {e}")

        ssh_mask = None
        sst_mask = None

        if cfg.observations.mode != "full":
            ssh_obs = dataset.get_ssh_obs(target_start_idx, cfg.data.observation_length)
            target_sequence, ssh_mask = obs_operator.apply_ssh_operator(
                target_sequence, ssh_obs, cfg.observations.mode, mdt=stats_field
            )

            sst_obs = dataset.get_sst_obs(target_start_idx, cfg.data.observation_length)
            target_sequence, sst_mask = obs_operator.apply_sst_operator(
                target_sequence, sst_obs, cfg.observations.mode
            )

        # Create masks
        sample_data = dataset.dataset.isel(time=sample_idx)["data"].values
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

        # Prepare data tensors
        input_data = input_sequence["data"].values
        input_data = np.nan_to_num(input_data, nan=0.0)
        x0_init = torch.from_numpy(input_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)

        target_data = target_sequence["data"].values
        target_data = np.nan_to_num(target_data, nan=0.0)
        target_tensor = (
            torch.from_numpy(target_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
        )
        
        ground_truth_data = ground_truth_sequence["data"].values
        ground_truth_data = np.nan_to_num(ground_truth_data, nan=0.0)
        ground_truth_tensor = (
            torch.from_numpy(ground_truth_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
        )



        # Create metrics computer for this sample with the sliced ocean mask (5 channels)
        # Slice ocean_mask to first 5 channels to match model output
        ocean_mask_5ch = ocean_mask[0:5, :, :]
        
        sample_metrics_computer = MetricsComputer(
            ocean_mask=ocean_mask_5ch,
            device=device,
            stats_field=None
        )
        
        # Update metrics computer with new sequences (slice to 5 channels to match model output)
        # Create sliced sequences with only the first 5 channels
        input_sequence_5ch = input_sequence["data"][:, 0:5, :, :]
        ground_truth_sequence_5ch = ground_truth_sequence["data"][:, 0:5, :, :]
        
        # Create temporary xarray datasets with the sliced data
        import xarray as xr
        input_seq_sliced = xr.Dataset({
            "data": (("time", "ch", "lat", "lon"), input_sequence_5ch.values)
        }, coords={
            "time": input_sequence.coords["time"],
            "ch": input_sequence.coords["ch"][:5],
            "lat": input_sequence.coords["lat"],
            "lon": input_sequence.coords["lon"]
        })
        
        ground_truth_seq_sliced = xr.Dataset({
            "data": (("time", "ch", "lat", "lon"), ground_truth_sequence_5ch.values)
        }, coords={
            "time": ground_truth_sequence.coords["time"],
            "ch": ground_truth_sequence.coords["ch"][:5],
            "lat": ground_truth_sequence.coords["lat"],
            "lon": ground_truth_sequence.coords["lon"]
        })
        
        sample_metrics_computer.set_mean_from_sequences(
            input_sequence_xr=input_seq_sliced,
            ground_truth_sequence_xr=ground_truth_seq_sliced,
        )

        # Create loss function for this sample with the computed obs_mask
        sample_loss_fn = ObservationLoss(
            obs_mask=obs_mask,
            loss_weighting=cfg.loss.weighting,
            manual_weights=cfg.loss.manual_weights if cfg.loss.weighting == "manual" else None,
            use_structural_loss=cfg.loss.use_structural_loss,
            structural_operator=cfg.loss.structural_operator,
            structural_loss_weight=cfg.loss.structural_loss_weight,
            device=device,
        )

        # Create unique subdirectory for this GPU rank and sample
        rank_sample_dir = f"gpu{global_rank}_sample{sample_idx}"
        
        # Create sample-specific output directories
        sample_output_dir =  rank_sample_dir
        
        sample_tensorboard_dir = None
        if cfg.logging.tensorboard_subdir:
            sample_tensorboard_dir = str(Path(cfg.logging.tensorboard_subdir) / rank_sample_dir)
            
        sample_checkpoints_dir = None
        if cfg.logging.checkpoints_subdir:
            sample_checkpoints_dir = str(Path(cfg.logging.checkpoints_subdir) / rank_sample_dir)
            
        sample_metrics_dir = None
        if cfg.logging.metrics_subdir:
            sample_metrics_dir = str(Path(cfg.logging.metrics_subdir) / rank_sample_dir)

        # Create optimizer for this sample
        # sample_exp_id = f"{exp_id}_sample{sample_idx}_rank{global_rank}"
        sample_exp_id = ""
        
        optimizer = ICOptimizer(
            forward_model=forward_model,
            loss_fn=sample_loss_fn,
            gradient_filter=gradient_filter,
            metrics_computer=sample_metrics_computer,
            learning_rate=cfg.optimization.learning_rate,
            num_iterations=cfg.optimization.num_iterations,
            device=device,
            output_dir=sample_output_dir,
            tensorboard_subdir=sample_tensorboard_dir,
            checkpoints_subdir=sample_checkpoints_dir,
            metrics_subdir=sample_metrics_dir,
            save_frequency=cfg.logging.save_frequency,
            log_frequency=cfg.logging.log_frequency,
            histogram_frequency=cfg.logging.histogram_frequency,
            scheduled_pooling=scheduled_pooling,
            forecast_horizon=cfg.data.forecast_horizon,
            use_meta_learner=cfg.optimization.get("use_meta_learner", False),
            meta_learner_config=dict(cfg.optimization.meta_learner) if cfg.optimization.get("use_meta_learner", False) else None,
        )

        # Run optimization
        best_x0, sample_results = optimizer.optimize(
            x0_init=x0_init,
            target_sequence=target_tensor,
            ocean_mask=ocean_mask,
            exp_id=sample_exp_id,
            regional_masks=regional_masks,
            input_sequence_xr=input_sequence,
            target_sequence_xr=target_sequence,
            ground_truth_sequence_xr=ground_truth_sequence
        )
        
        # Store results
        results[f"sample_{sample_idx}"] = sample_results
        
        if global_rank == 0:
            logger.info(f"Completed sample {sample_idx}: loss={sample_results['final_loss']:.6f}")
        
        # =========================================================================
        # MEMORY CLEANUP - Critical for sequential processing on same GPU
        # =========================================================================
        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicit garbage collection
        import gc
        gc.collect()
        
        # Clean up tensors for this sample
        del x0_init, target_tensor, ground_truth_tensor
        del input_sequence, target_sequence, ground_truth_sequence
        del ocean_mask, regional_masks, obs_mask
        if ssh_mask is not None:
            del ssh_mask
        if sst_mask is not None:
            del sst_mask
        
        # Force garbage collection again after deletion
        gc.collect()
    
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="optimize_ic")
def main(cfg: DictConfig):
    """
    Main parallel optimization function with SLURM integration.

    Args:
        cfg: Hydra configuration
    """
    # Setup distributed environment
    world_size, global_rank, local_rank, device = setup_slurm_process_info()
    
    try:
        # Setup SLURM environment
        cfg = setup_slurm_environment(cfg, local_rank)

        # Setup logging
        setup_logging(cfg, global_rank)

        # Setup reproducibility
        setup_reproducibility(seed=42, global_rank=global_rank)

        if global_rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("Parallel IC Optimization with SLURM Integration")
            logger.info("=" * 60)
            logger.info(f"World size: {world_size}")
            logger.info(f"Global rank: {global_rank}")
            logger.info(f"Local rank: {local_rank}")
            logger.info(f"Device: {device}")
            logger.debug("Configuration:\n" + OmegaConf.to_yaml(cfg))
            logger.info("=" * 60 + "\n")

        cfg.compute.device = device

        # Generate experiment ID
        if cfg.logging.auto_generate_exp_id:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            exp_id = f"{cfg.experiment.name}_{timestamp}_parallel"
        else:
            exp_id = f"{cfg.experiment.name}_parallel"

        if global_rank == 0:
            logger.info(f"Experiment ID: {exp_id}\n")

        # Create experiment directory
        # exp_output_dir = Path(cfg.logging.output_dir) / exp_id
        # if global_rank == 0:
        #     exp_output_dir.mkdir(parents=True, exist_ok=True)
        #     # Save a copy of the active config
        #     cfg_path = exp_output_dir / "config.yaml"
        #     with open(cfg_path, "w") as cf:
        #         cf.write(OmegaConf.to_yaml(cfg))
        #     logger.info(f"Experiment directory created: {exp_output_dir}")

        # Wait for all processes to reach this point



        # -------------------------------------------------------------------------
        # 1. Load Dataset (shared across all processes)
        # -------------------------------------------------------------------------
        if global_rank == 0:
            logger.info("Loading dataset...")

        dataset = GlonetDataset(
            data_path=cfg.data.root_path,
            data_files=cfg.data.init_state_files,
            ssh_obs_files=cfg.data.ssh_obs_files if cfg.observations.mode != "full" else None,
            sst_obs_files=cfg.data.sst_obs_files if cfg.observations.mode != "full" else None,
            lazy_load=True,
        )

        # Get total number of samples available
        total_samples = len(dataset.dataset.time)
        
        # First sample index: samples [first_idx, first_idx + batch_size) are processed
        first_idx = cfg.data.get("sample_idx", 0)
        if not 0 <= first_idx < total_samples:
            raise ValueError(f"data.sample_idx={first_idx} out of range [0, {total_samples})")

        # Use batch_size from config or default to remaining samples
        total_batch_size = min(cfg.data.get("batch_size", total_samples), total_samples - first_idx)
        
        if global_rank == 0:
            logger.info(f"Total samples available: {total_samples}")
            logger.info(f"First sample index: {first_idx}")
            logger.info(f"Total batch size: {total_batch_size} "
                        f"(samples {first_idx} to {first_idx + total_batch_size - 1})")

        # -------------------------------------------------------------------------
        # 2. Initialize Shared Components
        # -------------------------------------------------------------------------
        if global_rank == 0:
            logger.info("Initializing shared components...")

        # Initialize observation operator
        obs_operator = ObservationOperator(device=device)

        # Initialize mask builder
        mask_builder = MaskBuilder(device=device)

        # Initialize forward model (shared across processes)
        forward_model = ForwardModel(
            model_path=str(Path(cfg.model.location) / cfg.model.checkpoint_files.part1),
            normalizer_path=cfg.model.location,
            device=device,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
            ocean_mask=None,  # Will be set per sample
        )

        # Initialize gradient filter
        gradient_filter = GradientFilter(
            enable=cfg.optimization.use_gradient_smoothing,
            downsampling_method=cfg.optimization.downsampling_method,
            device=device
        )

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

        # Initialize metrics computer
        metrics_computer = MetricsComputer(
            ocean_mask=None,  # Will be set per sample
            device=device,
            stats_field=None
        )

        # Initialize loss function (will be updated per sample)
        loss_fn = ObservationLoss(
            obs_mask=None,  # Will be set per sample
            loss_weighting=cfg.loss.weighting,
            manual_weights=cfg.loss.manual_weights if cfg.loss.weighting == "manual" else None,
            use_structural_loss=cfg.loss.use_structural_loss,
            structural_operator=cfg.loss.structural_operator,
            structural_loss_weight=cfg.loss.structural_loss_weight,
            device=device,
        )

        if global_rank == 0:
            logger.info("Initialized shared components")

        # -------------------------------------------------------------------------
        # 3. Calculate Batch Indices for This Process
        # -------------------------------------------------------------------------
        batch_start_idx, batch_end_idx = get_batch_indices(
            total_batch_size, world_size, global_rank, first_idx
        )
        
        if global_rank == 0:
            logger.info(f"Rank {global_rank}: processing samples {batch_start_idx} to {batch_end_idx-1}")

        # -------------------------------------------------------------------------
        # 4. Run Parallel Optimization (Memory-Safe Design)
        # -------------------------------------------------------------------------
        # DESIGN NOTE: This implementation is memory-safe because:
        # 1. Each GPU process gets its own batch of samples via get_batch_indices()
        # 2. Within each GPU, samples are processed SEQUENTIALLY in a for loop
        # 3. Memory is explicitly cleaned up between samples (torch.cuda.empty_cache(), gc.collect())
        # 4. This prevents OOM errors while still achieving parallelism across GPUs
        #
        # Example with 50 samples and 4 GPUs:
        # - GPU 0: samples 0-12 (13 samples, processed one by one)
        # - GPU 1: samples 13-25 (13 samples, processed one by one)
        # - GPU 2: samples 26-38 (13 samples, processed one by one)
        # - GPU 3: samples 39-49 (11 samples, processed one by one)
        # All 4 GPUs work simultaneously, but each does sequential processing.
        
        if global_rank == 0:
            logger.info("Starting parallel optimization...")
            logger.info(f"Memory-safe design: {batch_end_idx - batch_start_idx} samples will be processed sequentially on this GPU")

        results = run_optimization_on_batch(
            cfg=cfg,
            dataset=dataset,
            obs_operator=obs_operator,
            mask_builder=mask_builder,
            forward_model=forward_model,
            loss_fn=loss_fn,
            gradient_filter=gradient_filter,
            scheduled_pooling=scheduled_pooling,
            metrics_computer=metrics_computer,
            device=device,
            batch_start_idx=batch_start_idx,
            batch_end_idx=batch_end_idx,
            global_rank=global_rank,
            local_rank=local_rank,
            exp_id=exp_id,
        )

        # -------------------------------------------------------------------------
        # 5. Gather and Report Results
        # -------------------------------------------------------------------------
        if global_rank == 0:
            logger.info("\n" + "=" * 60)
            logger.info("Parallel Optimization Complete!")
            logger.info("=" * 60)
            
            # Calculate and report statistics
            total_samples_processed = 0
            total_improvement = 0.0
            
            for sample_key, sample_results in results.items():
                improvement = (1 - sample_results["final_loss"] / sample_results["initial_loss"]) * 100
                total_samples_processed += 1
                total_improvement += improvement
                logger.info(f"{sample_key}: Initial={sample_results['initial_loss']:.6f}, "
                           f"Final={sample_results['final_loss']:.6f}, "
                           f"Improvement={improvement:.2f}%")
            
            if total_samples_processed > 0:
                avg_improvement = total_improvement / total_samples_processed
                logger.info(f"Average improvement: {avg_improvement:.2f}%")
            
            logger.info("=" * 60 + "\n")





        if global_rank == 0:
            # logger.info(f"Results saved to: {exp_output_dir}")
            logger.info(f"Results saved in hydra experience directory")


    finally:
        pass




if __name__ == "__main__":
    main()
