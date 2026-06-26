#!/usr/bin/env python3
"""
Main entry script for IC optimization.

Usage:
    python run_optimization.py
    python run_optimization.py experiment.name=test_run
    python run_optimization.py observations.mode=simulated optimization.num_iterations=500
    
Hydra automatically manages configuration and logging.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import logging
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gd_optimic import (
    GlonetDataset,
    ObservationOperator,
    ObservationLoss,
    GradientFilter,
    ICOptimizer,
    MetricsComputer,
    MaskBuilder,
    ForwardModel,
)
from gd_optimic.gradient import ScheduledPooling


@hydra.main(version_base=None, config_path="../configs", config_name="optimize_ic")
def main(cfg: DictConfig):
    """
    Main optimization function.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("\n" + "="*60)
    logger.info("IC Optimization with Hydra Configuration")
    logger.info("="*60)
    logger.debug("Configuration:\n" + OmegaConf.to_yaml(cfg))
    logger.info("="*60 + "\n")
    
    # Generate experiment ID
    if cfg.logging.auto_generate_exp_id:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_id = f"{cfg.experiment.name}_{timestamp}"
    else:
        exp_id = cfg.experiment.name
    
    logger.info(f"Experiment ID: {exp_id}\n")
    
    # Set random seeds for reproducibility (A5)
    torch.manual_seed(42)
    np.random.seed(42)
    if cfg.compute.device == "cuda":
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = cfg.compute.device
    
    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    logger.info("Loading dataset...")
    dataset = GlonetDataset(
        data_path=cfg.data.root_path,
        data_files=cfg.data.init_state_files,
        ssh_obs_files=cfg.data.ssh_obs_files if cfg.observations.mode != 'full' else None,
        sst_obs_files=cfg.data.sst_obs_files if cfg.observations.mode != 'full' else None,
        lazy_load=True
    )
    
    # Extract input sequence (initial condition)
    input_sequence = dataset.get_sequence(
        start_idx=cfg.data.sample_idx,
        length=cfg.data.sequence_length
    )
    
    # Extract target sequence (observations for optimization)
    target_start_idx = cfg.data.sample_idx + cfg.data.sequence_length
    target_end_idx = target_start_idx + cfg.data.observation_length
    target_sequence = dataset.get_sequence(
        start_idx=target_start_idx,
        length=cfg.data.observation_length
    )
    
    logger.info(f"Loaded dataset")
    logger.debug(f"  Input sequence: {input_sequence['data'].shape}")
    logger.debug(f"  Target sequence: {target_sequence['data'].shape}")
    
    # -------------------------------------------------------------------------
    # 2. Apply Observation Operators
    # -------------------------------------------------------------------------
    logger.info("Applying observation operators...")
    obs_operator = ObservationOperator(device=device)
    
    # SSH observations
    ssh_mask = None
    if cfg.observations.mode != 'full':
        ssh_obs = dataset.get_ssh_obs(target_start_idx, cfg.data.observation_length)
        mdt = None  # Load MDT if obs_mode='real'
        
        target_sequence, ssh_mask = obs_operator.apply_ssh_operator(
            target_sequence,
            ssh_obs,
            cfg.observations.mode,
            mdt=mdt
        )
    
    # SST observations
    sst_mask = None
    if cfg.observations.mode != 'full':
        sst_obs = dataset.get_sst_obs(target_start_idx, cfg.data.observation_length)
        
        target_sequence, sst_mask = obs_operator.apply_sst_operator(
            target_sequence,
            sst_obs,
            cfg.observations.mode
        )
    
    logger.info(f"Applied observation operators (mode: {cfg.observations.mode})")
    
    # -------------------------------------------------------------------------
    # 3. Create Masks
    # -------------------------------------------------------------------------
    logger.info("Creating masks...")
    mask_builder = MaskBuilder(device=device)
    
    # Sample data for ocean mask
    sample_data = dataset.dataset.isel(time=cfg.data.sample_idx)['data'].values
    ocean_mask = mask_builder.build_ocean_mask(sample_data)
    
    # Observation mask
    obs_mask = mask_builder.build_obs_mask(
        ocean_mask,
        cfg.data.observation_length,
        ssh_nanmask=ssh_mask,
        sst_nanmask=sst_mask,
        obs_mode=cfg.observations.mode
    )
    
    logger.info(f"Created masks")
    logger.debug(f"  Ocean mask: {ocean_mask.shape}")
    logger.debug(f"  Observation mask: {obs_mask.shape}")
    
    # -------------------------------------------------------------------------
    # 4. Prepare Data Tensors
    # -------------------------------------------------------------------------
    logger.info("Preparing data tensors...")
    
    # Input (initial condition): [B, T=2, C, H, W]
    input_data = input_sequence['data'].values
    input_data = np.nan_to_num(input_data, nan=0.0)
    x0_init = torch.from_numpy(input_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
    
    # Target (observations): [B, T_obs, C, H, W]
    target_data = target_sequence['data'].values
    target_data = np.nan_to_num(target_data, nan=0.0)
    target_tensor = torch.from_numpy(target_data[:, 0:5, :, :].copy()).float().unsqueeze(0).to(device)
    
    logger.info(f"Prepared tensors")
    logger.debug(f"  Initial condition: {x0_init.shape}")
    logger.debug(f"  Target observations: {target_tensor.shape}")
    
    # -------------------------------------------------------------------------
    # 5. Initialize Forward Model
    # -------------------------------------------------------------------------
    logger.info("Initializing forward model...")
    forward_model = ForwardModel(
        model_path=str(Path(cfg.model.location) / cfg.model.checkpoint_files.part1),
        normalizer_path=cfg.model.location,
        device=device,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing
    )
    
    logger.info(f"Initialized forward model (gradient checkpointing: {cfg.model.use_gradient_checkpointing})")
    
    # -------------------------------------------------------------------------
    # 6. Initialize Loss Function
    # -------------------------------------------------------------------------
    logger.info("Initializing loss function...")
    loss_fn = ObservationLoss(
        obs_mask=obs_mask,
        loss_weighting=cfg.loss.weighting,
        manual_weights=cfg.loss.manual_weights if cfg.loss.weighting == 'manual' else None,
        device=device
    )
    
    logger.info(f"Initialized loss function (weighting: {cfg.loss.weighting})")
    
    # -------------------------------------------------------------------------
    # 7. Initialize Gradient Filter
    # -------------------------------------------------------------------------
    logger.info("Initializing gradient filter...")
    gradient_filter = GradientFilter(
        filter_type=cfg.optimization.gradient_filter,
        device=device
    )
    
    # Scheduled pooling (if enabled)
    scheduled_pooling = None
    if cfg.optimization.use_scheduled_pooling:
        scheduled_pooling = ScheduledPooling(
            schedule_type=cfg.optimization.pooling_schedule.type,
            num_iterations=cfg.optimization.num_iterations,
            initial_kernel=cfg.optimization.pooling_schedule.initial_kernel,
            final_kernel=cfg.optimization.pooling_schedule.final_kernel,
            schedule_steps=cfg.optimization.pooling_schedule.schedule_steps,
            kernel_sizes=cfg.optimization.pooling_schedule.kernel_sizes
        )
    
    logger.info(f"Initialized gradient filter (type: {cfg.optimization.gradient_filter})")
    if scheduled_pooling is not None:
        logger.debug(f"  Scheduled pooling: {scheduled_pooling.schedule_type}")
    
    # -------------------------------------------------------------------------
    # 8. Initialize Metrics Computer
    # -------------------------------------------------------------------------
    logger.info("Initializing metrics computer...")
    metrics_computer = MetricsComputer(
        ocean_mask=ocean_mask,
        device=device
    )
    
    # Regional masks (for basin-stratified RMSE)
    regional_masks = None
    if cfg.metrics.compute_rmse_basin:
        # Build regional masks
        # This would require implementing regional mask building logic
        # For now, set to None (can be added in Phase P)
        regional_masks = None
        logger.debug("  Note: Regional masks deferred to Phase P")
    
    logger.info(f"Initialized metrics computer")
    
    # -------------------------------------------------------------------------
    # 9. Initialize Optimizer
    # -------------------------------------------------------------------------
    logger.info("Initializing optimizer...")
    
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
        scheduled_pooling=scheduled_pooling
    )
    
    logger.info("Initialized optimizer")
    logger.debug(f"  Learning rate: {cfg.optimization.learning_rate}")
    logger.debug(f"  Num iterations: {cfg.optimization.num_iterations}")
    
    # -------------------------------------------------------------------------
    # 10. Run Optimization
    # -------------------------------------------------------------------------
    logger.info("Starting optimization...")
    
    best_x0, results = optimizer.optimize(
        x0_init=x0_init,
        target_sequence=target_tensor,
        ocean_mask=ocean_mask,
        exp_id=exp_id,
        regional_masks=regional_masks
    )
    
    logger.info("\n" + "="*60)
    logger.info("Optimization Complete!")
    logger.info("="*60)
    logger.info(f"Best loss: {results['best_loss']:.6f} (iteration {results['best_iteration']})")
    logger.info(f"Initial loss: {results['initial_loss']:.6f}")
    logger.info(f"Final loss: {results['final_loss']:.6f}")
    logger.info(f"Improvement: {(1 - results['final_loss'] / results['initial_loss']) * 100:.2f}%")
    logger.info("="*60 + "\n")
    
    exp_output_dir = Path(cfg.logging.output_dir) / exp_id
    logger.info(f"Results saved to: {exp_output_dir}")
    logger.info(f"\nTo view TensorBoard:")
    logger.info(f"  tensorboard --logdir {exp_output_dir}/{cfg.logging.tensorboard_subdir}")
    

if __name__ == "__main__":
    main()
