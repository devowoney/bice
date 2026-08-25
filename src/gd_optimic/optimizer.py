"""
Main IC optimizer class with TensorBoard logging.

Provides:
    - ICOptimizer: Main optimization loop for initial condition optimization
    
Implements:
- Gradient descent on initial conditions
- Multi-step rollout with frozen model
- Observation loss (J_obs only in Phase 1)
- Scheduled gradient filtering (multigrid)
- TensorBoard logging following R8 hierarchy
- Automatic checkpointing

Phase 1 scope: J_obs only
Phase 1.b: Will add J_b + J_q terms
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import gc
import logging

from .utils import ForwardModel, MaskBuilder

logger = logging.getLogger(__name__)
from .loss import ObservationLoss
from .gradient import GradientFilter, ScheduledPooling
from .metrics import MetricsComputer
from .output_handler import OutputHandler
from .meta_learner import BiLevelICOptimizer, UNetMetaGrad2D, NetworkSConfig, MetaLearnerCheckpointManager
import xarray as xr


class ICOptimizer:
    """
    Initial condition optimizer using gradient descent.

    Implements ML-4DVar optimization with frozen pretrained model (glonet v1).
    Follows research decisions:
    - R2: SSH along-track + SST gridded observation operators
    - R4: Per-variable + basin-stratified RMSE metrics
    - R8: TensorBoard hierarchy for experiment tracking
    - CS1: Human-readable, well-commented code

    Scientific integrity constraints:
    - A1: Frozen-model invariant (no weight updates)
    - A2: Strict obs/eval separation (observation mode enforcement)
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        loss_fn: ObservationLoss,
        gradient_filter: GradientFilter,
        metrics_computer: MetricsComputer,
        learning_rate: float,
        num_iterations: int,
        device: str = "cuda",
        output_dir: str = "runs",
        tensorboard_subdir: str = "tb",
        checkpoints_subdir: str = "checkpoints",
        metrics_subdir: str = "metrics",
        save_frequency: int = 100,
        log_frequency: int = 1,
        histogram_frequency: int = 50,
        scheduled_pooling: Optional[ScheduledPooling] = None,
        forecast_horizon: int = 28,
        use_meta_learner: bool = False,
        meta_learner_config: Optional[Dict] = None,
        downsampling_method: str = "average_pooling",
    ):
        """
        Initialize IC optimizer (standard gradient descent or meta-learning).

        Args:
            forward_model: Forward model wrapper for loss computation
            loss_fn: Loss function (observation - forecast)²
            gradient_filter: Gradient filtering object for preprocessing gradients
            metrics_computer: Metrics computation object for diagnostics
            learning_rate: Gradient descent learning rate for IC optimization
            num_iterations: Total number of optimization iterations (outer loop)
            device: PyTorch device ('cuda' or 'cpu')
            output_dir: Base output directory (typically .tmp/outputs)
            tensorboard_subdir: Subdirectory for TensorBoard logs
            checkpoints_subdir: Subdirectory for checkpoints
            metrics_subdir: Subdirectory for metrics
            save_frequency: Save checkpoint every N iterations
            log_frequency: Log metrics every N iterations
            histogram_frequency: Log histograms/embeddings every N iterations
            scheduled_pooling: Optional scheduled pooling object for multigrid optimization
            forecast_horizon: Number of forecast steps for diagnostics (default: 28)
            use_meta_learner: Whether to use bi-level meta-learning (Phase P = meta-learned IC optimization)
            meta_learner_config: Configuration dict for BiLevelICOptimizer with keys:
                - enabled: bool, whether meta-learning is active (default: True)
                - meta_lr: Learning rate for network_s (default: 1e-3, typically 1e-4)
                - num_meta_steps: Number of IC updates per outer iteration (default: 10)
                - grad_align_steps: Number of ALIGN phase steps (default: 6)
                - w_align: Weight on gradient alignment loss (ALIGN phase, default: 1.0)
                - w_perf: Weight on performance/forecast loss (PERF phase, default: 1.0)
                - lambda_reg: Weight on L2 regularization (default: 1e-4, prevents parameter explosion)
                - base_channels: UNet base channels for network_s (default: 32)
                - output_scale: Scaling factor for network_s output (default: 0.01)
                
                CURRICULUM LEARNING STRATEGY:
                - ALIGN (m < grad_align_steps): Trains network_s to predict gradient-aligned updates
                  * Loss: l_align + l_reg, where l_align = ||network_s(x) - ∇_x||_2
                  * Goal: Learn physical patterns that match observed gradients
                  * Expected: l_align DECREASES as network learns gradient shape
                
                - PERF (m >= grad_align_steps): Refines updates to maximize loss reduction
                  * Loss: w_perf * L_perf + [optional: weak l_align] + l_reg
                  * Goal: Make IC updates that actually improve forecasts
                  * Expected: L_perf DECREASES, improvement POSITIVE (forecasts get better)
        """
        self.forward_model = forward_model
        self.loss_fn = loss_fn
        self.gradient_filter = gradient_filter
        self.metrics_computer = metrics_computer
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.device = device
        self.output_dir = Path(output_dir)
        self.tensorboard_subdir = tensorboard_subdir
        self.checkpoints_subdir = checkpoints_subdir
        self.metrics_subdir = metrics_subdir
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.histogram_frequency = histogram_frequency
        self.scheduled_pooling = scheduled_pooling
        self.forecast_horizon = forecast_horizon
        self.downsampling_method = downsampling_method

        # Tracking variables
        self.history = []
        self.best_loss = float("inf")
        self.best_iteration = 0
        self.best_x0 = None
        self.best_predictions = None
          
        # Store xarray datasets for NetCDF output (set during optimize())
        self.input_sequence_xr = None
        self.target_sequence_xr = None
        self.ground_truth_sequence_xr = None

        # Meta-learner setup (Phase P)
        self.use_meta_learner = use_meta_learner
        self.meta_learner = None
        self.meta_learner_config = meta_learner_config or {}
        self.checkpoint_manager = None  # Will be initialized in optimize()
        if self.use_meta_learner:
            logger.info("Meta-learner (Phase P) is ENABLED")
            # BiLevelICOptimizer will be instantiated in optimize() once we know the IC shape
        else:
            logger.info("Using standard gradient descent (no meta-learner)")

        # Writer and output handler initialized later after exp_id is set
        self.writer = None
        self.output_handler = None

    def setup_directories(self, exp_id: str):
        """
        Set up experiment-specific output directories.

        Creates:
        - .tmp/outputs/{exp_id}/config.yaml
        - .tmp/outputs/{exp_id}/checkpoints/
        - .tmp/outputs/{exp_id}/metrics/
        - .tmp/outputs/{exp_id}/tb/          (TensorBoard)

        Args:
            exp_id: Experiment identifier (e.g., "refIC_fullobs_2026-06-22_15-30-45")
        """
        self.exp_id = exp_id

        # Create experiment directory structure
        # self.exp_dir = self.output_dir / exp_id
        self.exp_dir = Path(".")
        self.checkpoint_dir = self.exp_dir / self.checkpoints_subdir
        self.metrics_dir = self.exp_dir / self.metrics_subdir
        self.tensorboard_dir = self.exp_dir / self.tensorboard_subdir

        # Create all directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer for this experiment
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Initialize output handler for NetCDF diagnostics
        self.output_handler = OutputHandler(exp_dir=self.exp_dir, device=self.device)

        logger.info(f"Output directory: {self.exp_dir}")
        logger.info(f"TensorBoard logs: {self.tensorboard_dir}")
        logger.info(f"Checkpoints: {self.checkpoint_dir}")

    def optimize(
        self,
        x0_init: torch.Tensor,
        target_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
        exp_id: str,
        regional_masks: Optional[Dict[str, np.ndarray]] = None,
        input_sequence_xr: Optional[xr.Dataset] = None,
        target_sequence_xr: Optional[xr.Dataset] = None,
        ground_truth_sequence_xr: Optional[xr.Dataset] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Run optimization loop.

        Args:
            x0_init: Initial condition [B, T=2, C, H, W]
            target_sequence: Target observations [B, T_obs, C, H, W] (assimilation window only)
            ocean_mask: Ocean mask [C, H, W]
            exp_id: Experiment identifier for output directory
            regional_masks: Regional masks for basin-stratified metrics (optional)
            input_sequence_xr: xarray Dataset with input sequence (for NetCDF output)
            target_sequence_xr: xarray Dataset with target sequence (assimilation window, for coordinates)
            ground_truth_sequence_xr: xarray Dataset with full ground truth (T=forecast_horizon, for RMSE diagnostics)

        Returns:
            best_x0: Optimized initial condition [B, T=2, C, H, W]
            results: Dictionary with optimization results and history
        """
        # Setup directories for this experiment
        self.setup_directories(exp_id)
          
        # Store xarray datasets for NetCDF output
        self.input_sequence_xr = input_sequence_xr
        self.target_sequence_xr = target_sequence_xr
        self.ground_truth_sequence_xr = ground_truth_sequence_xr
        self.regional_masks = regional_masks
         
        # Convert ground truth xarray to tensor for RMSE diagnostics
        self.ground_truth_tensor = None
        if ground_truth_sequence_xr is not None:
            ground_truth_data = ground_truth_sequence_xr["data"].values
            ground_truth_data = np.nan_to_num(ground_truth_data, nan=0.0)
            self.ground_truth_tensor = (
                torch.from_numpy(ground_truth_data[:, 0:5, :, :].copy()).float().to(x0_init.device)
            )

        # Initialize optimization variables
        x0_current = x0_init.clone().detach().requires_grad_(True)
        x0_reference = x0_init.clone().detach()  # Reference IC (never updated)

        # Store ocean_mask on self for logging and other uses
        self._ocean_mask = ocean_mask.to(self.device) if ocean_mask is not None else None
        # Placeholder for last applied IC update (learning_rate * masked_gradients)
        self._last_ic_update = None

        # Number of forward steps to roll out (for assimilation window)
        num_forecast_steps = target_sequence.shape[1]

        # Initialize meta-learner if enabled (Phase P)
        if self.use_meta_learner:
            logger.info("\nInitializing BiLevelICOptimizer ...")
            # Create NetworkS config from meta_learner_config
            # T=2 (temporal steps for IC), C=5 (channels)
            network_s_config = NetworkSConfig(
                base_channels=self.meta_learner_config.get('base_channels', 32),
                num_groups=self.meta_learner_config.get('num_groups', 8),
                output_scale=self.meta_learner_config.get('output_scale', 0.01),
                temporal_steps=2,  # IC has T=2 steps
                input_channels=5,  # SSH, T, S, U, V
                output_channels=5,
            )
            
            # Instantiate UNetMetaGrad2D
            network_s = UNetMetaGrad2D(network_s_config).to(self.device)
            logger.info(f"--------------------------------------------")
            logger.info(f"UNetMetaGrad2D parameters: {network_s.get_parameter_count():,}")
            logger.info(f"--------------------------------------------")
            
            # Instantiate BiLevelICOptimizer
            # BiLevelICOptimizer signature: (network_s, ocean_mask, device='cuda', config=None)
            # Pass forward_model and loss_fn so meta-learner can recompute loss inside meta loop
            self.meta_learner = BiLevelICOptimizer(
                network_s,
                ocean_mask,
                forward_model=self.forward_model,
                loss_fn=self.loss_fn,
                num_forecast_steps=num_forecast_steps,
                device=self.device,
                config=self.meta_learner_config,
                writer=self.writer,
            )
            logger.info("  BiLevelICOptimizer initialized successfully\n")

            # Initialize checkpoint manager
            self.checkpoint_manager = MetaLearnerCheckpointManager(
                checkpoint_dir=self.checkpoint_dir,
                device=self.device
            )
            
            # Handle meta-learner mode selection (training, fine-tuning, inference)
            meta_learner_mode = self.meta_learner_config.get('mode', 'training')
            load_checkpoint = self.meta_learner_config.get('load_checkpoint')
            
            logger.info(f"Meta-learner mode: {meta_learner_mode}")
            
            if meta_learner_mode == 'training':
                logger.info("Training meta-learner from scratch")
                # Network initialized with random weights above
                
            elif meta_learner_mode == 'fine_tune':
                logger.info("Fine-tuning meta-learner from pre-trained checkpoint")
                
                if load_checkpoint:
                    # Load checkpoint from explicit path
                    checkpoint_info = self.checkpoint_manager.load_meta_learner(
                        network_s, meta_optimizer=self.meta_learner.meta_optimizer, checkpoint_path=load_checkpoint
                    )
                    logger.info(f"Loaded checkpoint from {load_checkpoint}")
                else:
                    logger.warning("No checkpoint specified for fine-tuning. Starting from scratch.")
                
                # Unfreeze for fine-tuning
                self.checkpoint_manager.unfreeze_meta_learner(network_s)
                
            elif meta_learner_mode == 'inference':
                logger.info("Loading pre-trained meta-learner for inference (frozen)")
                
                if load_checkpoint:
                    # Load checkpoint from explicit path
                    checkpoint_info = self.checkpoint_manager.load_meta_learner(
                        network_s, meta_optimizer=self.meta_learner.meta_optimizer, checkpoint_path=load_checkpoint
                    )
                    logger.info(f"Loaded checkpoint from {load_checkpoint}")
                else:
                    logger.warning("No checkpoint specified for inference. Using scratch weights (unfrozen).")
                
                # Freeze for inference
                self.checkpoint_manager.freeze_meta_learner(network_s)
                
            else:
                raise ValueError(f"Unknown meta mode: {meta_learner_mode}. Must be 'training', 'fine_tune', or 'inference'.")

            # Track meta-learning metrics in history
            self._j_prev = None

        logger.info(f"\n{'='*60}")
        logger.info("Starting IC Optimization")
        logger.info(f"{'='*60}")
        logger.info(f"Initial condition shape: {x0_init.shape}")
        logger.info(f"Target sequence shape: {target_sequence.shape}")
        logger.info(f"Forecast steps: {num_forecast_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        if self.use_meta_learner:
            logger.info(f"Meta-learner: ENABLED (Phase P - Bi-level optimization)")
            logger.info(f"  Two-phase curriculum learning:")
            logger.info(f"  - ALIGN phase (first {self.meta_learner_config.get('grad_align_steps', 6)} steps) "
                        f"Learn gradient-aligned patterns")
            logger.info(f"  - PERF phase (remaining steps): Maximize loss reduction")
            logger.info(f"  Configuration:")
            logger.info(f"    * Meta LR: {self.meta_learner_config.get('meta_lr', 1e-3)}")
            logger.info(f"    * Num meta-steps per iteration: {self.meta_learner_config.get('num_meta_steps', 10)}")
            logger.info(f"    * Weights: w_align={self.meta_learner_config.get('w_align', 1.0)}, "
                        f"w_perf={self.meta_learner_config.get('w_perf', 1.0)}, "
                        f"lambda_reg={self.meta_learner_config.get('lambda_reg', 1e-4)}")
            logger.info(f"  Expected metrics:")
            logger.info(f"    * ALIGN: l_align DECREASES (learning gradient shape)")
            logger.info(f"    * PERF: L_perf DECREASES, improvement POSITIVE (IC updates working)")
        else:
            logger.info(f"Meta-learner: DISABLED (standard gradient descent only)")
        logger.info(f"Num iterations: {self.num_iterations}")
        if self.scheduled_pooling is not None:
            logger.info(f"Scheduled pooling: {self.scheduled_pooling.schedule_type}")
        logger.info(f"{'='*60}\n")

        # Compute initial predictions (for comparison)
        with torch.no_grad():
            _, y_hat_init_steps = self.forward_model.forward(x0_init, num_forecast_steps)
            initial_loss, initial_details = self.loss_fn(
                y_hat_init_steps, target_sequence, return_details=True
            )
            logger.info(f"Initial loss: {initial_loss.item():.6f}\n")

        # Main optimization loop
        for iteration in range(self.num_iterations):
            # Zero gradients
            if x0_current.grad is not None:
                x0_current.grad.zero_()

            # Get current pooling kernel size (for scheduled pooling)
            if self.scheduled_pooling is not None:
                current_kernel = self.scheduled_pooling.get_kernel_size(iteration)
                self.gradient_filter.set_kernel_size(current_kernel)
            else:
                current_kernel = 1  # No pooling

            # Forward pass: rollout model for num_forecast_steps
            _, y_hat_steps = self.forward_model.forward(x0_current, num_forecast_steps)

            # Compute loss
            loss, loss_details = self.loss_fn(y_hat_steps, target_sequence, return_details=True)

            should_log = (iteration + 1) % self.log_frequency == 0

            # Backward pass: compute gradients
            gradients = torch.autograd.grad(loss, x0_current, create_graph=False)[0]

            # Apply gradient filtering
            filtered_gradients = self.gradient_filter(
                gradients.detach(),
                kernel_size=current_kernel
            )

            # Apply ocean mask to gradients (zero out land gradients)
            # Adjust ocean_mask channel dimension if necessary
            if ocean_mask.dim() == 3 and ocean_mask.shape[0] != filtered_gradients.shape[2]:
                mask_ch = int(ocean_mask.shape[0])
                grad_ch = int(filtered_gradients.shape[2])
                logger.warning(
                    "Ocean mask channels (%d) != gradients channels (%d). Adjusting mask.", mask_ch, grad_ch
                )
                if mask_ch >= grad_ch:
                    ocean_mask = ocean_mask[:grad_ch, :, :]
                else:
                    raise ValueError(
                        f"Ocean mask has fewer channels ({mask_ch}) than gradients ({grad_ch})."
                    )

            # Compute masked gradients (no grad needed)
            masked_gradients = filtered_gradients * ocean_mask.unsqueeze(0).unsqueeze(0)

            # Default history meta loss
            history_meta_loss = None

            # If meta-learner disabled: apply standard gradient descent update inside no_grad
            if not self.use_meta_learner:
                with torch.no_grad():
                    ic_update = self.learning_rate * masked_gradients
                    # Save last update on CPU for TensorBoard logging
                    self._last_ic_update = ic_update.detach().cpu().clone()
                    x0_current = x0_current - ic_update
                    # Ensure x0_current requires grad for next iteration
                    x0_current = x0_current.detach().requires_grad_(True)
            else:
                # Meta-learner path: predict IC update without graph
                with torch.no_grad():
                    ic_update = self.meta_learner.predict_update(x0_current.detach()).requires_grad_(True)
                    self._last_ic_update = ic_update.detach().cpu().clone()
                    x0_current = x0_current - ic_update

                # Compute loss at new IC for meta-learner
                with torch.no_grad():
                    _, y_hat_steps_new = self.forward_model.forward(x0_current, num_forecast_steps)
                    loss_current, _ = self.loss_fn(
                        y_hat_steps_new, target_sequence, return_details=True
                    )

                # Initialize loss history on first iteration
                if self._j_prev is None:
                    self._j_prev = loss.item()

                # Multi-step meta-optimization of network S parameters
                self.meta_learner.target_sequence = target_sequence
                meta_diags = self.meta_learner.step(
                    first_update=ic_update,
                    x_current=x0_current,
                    loss_prev=self._j_prev,
                    gradient_prev=masked_gradients,
                    iteration=iteration,
                )

                # Update loss history for next iteration
                self._j_prev = float(loss_current.detach().cpu().item())

                # Extract meta-loss for logging
                history_meta_loss = meta_diags.get("L_meta", None) if isinstance(meta_diags, dict) else None

                # Log meta-loss scalars to TensorBoard (L_meta and components)
                try:
                    if isinstance(meta_diags, dict) and self.writer is not None:
                        if meta_diags.get("L_align") is not None:
                            self.writer.add_scalar("meta/L_align", float(meta_diags.get("L_align")), iteration + 1)
                        if meta_diags.get("L_perf") is not None:
                            self.writer.add_scalar("meta/L_perf", float(meta_diags.get("L_perf")), iteration + 1)
                        if meta_diags.get("L_reg") is not None:
                            self.writer.add_scalar("meta/L_reg", float(meta_diags.get("L_reg")), iteration + 1)
                except Exception:
                    logger.exception("Failed to write meta loss components to TensorBoard")

                # Detach x0_current for next outer iteration to avoid graph growth
                x0_current = x0_current.detach().requires_grad_(True)

            # Compute metrics (with no_grad to save memory)
            with torch.no_grad():
                # Use last forecast step for evaluation
                y_hat_final = y_hat_steps[:, -1, :, :, :]
                target_final = target_sequence[:, -1, :, :, :]

                metrics = self.metrics_computer.compute_all_metrics(
                    y_hat_final,
                    target_final,
                    x0_current,
                    x0_reference,
                    regional_masks=regional_masks,
                )

            # Track best solution
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_iteration = iteration + 1
                self.best_x0 = x0_current.detach().clone()
                self.best_predictions = y_hat_steps.detach().clone()

            # Store history
            history_entry = {
                "iteration": iteration + 1,
                "loss": loss.item(),
                "kernel_size": current_kernel,
                "weighted_per_variable_loss": {
                    k: v.mean().item() if torch.is_tensor(v) else v
                    for k, v in loss_details["weighted_per_variable"].items()
                },
                "rmse_global": metrics["rmse_global"],
                "ic_rmse": metrics["ic_rmse"],
            }

            # Add meta-learning metrics if enabled
            if self.use_meta_learner and history_meta_loss is not None:
                history_entry["meta_loss"] = history_meta_loss

            if self._last_ic_update is not None and self._ocean_mask is not None:
                finite_diff_total, _ = self._compute_finite_difference_ic_update(
                    self._last_ic_update[0, -1],
                    self._ocean_mask.detach().cpu(),
                )
                history_entry["gradient_finite_difference_ic_update"] = finite_diff_total

            if regional_masks is not None:
                history_entry["rmse_basin"] = metrics["rmse_basin"]

            self.history.append(history_entry)

            # Logging to TensorBoard
            if should_log:
                # Pass current and reference IC to TensorBoard logger so cumulative correction (from zero) can be visualized
                self._log_to_tensorboard(
                    iteration + 1,
                    loss_details,
                    metrics,
                    current_kernel,
                    x0_current=x0_current,
                    x0_reference=x0_reference,
                )

            # Log histograms/embeddings less frequently
            if (iteration + 1) % self.histogram_frequency == 0:
                self._log_histograms_to_tensorboard(
                    iteration + 1, masked_gradients, x0_current, ocean_mask
                )

            # Print progress
            if (iteration + 1) % self.save_frequency == 0 or iteration == 0:
                self._print_progress(iteration + 1, loss.item(), metrics, current_kernel)

            # Save checkpoint
            if (iteration + 1) % self.save_frequency == 0:
                self._save_checkpoint(iteration + 1, x0_current, y_hat_steps)
                
                # Save meta-learner checkpoint if enabled
                if self.use_meta_learner and self.meta_learner is not None:
                    self.checkpoint_manager.save_meta_learner(
                        network_s=self.meta_learner.network_s,
                        meta_optimizer=self.meta_learner.meta_optimizer,
                        iteration=iteration + 1,
                        meta_loss=self.meta_learner.loss_history[-1] if self.meta_learner.loss_history else float('inf'),
                        meta_loss_history=self.meta_learner.loss_history,
                    )

            # Memory cleanup
            del y_hat_steps, loss, gradients, filtered_gradients, masked_gradients
            if iteration % 50 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Final logging
        logger.info(f"\n{'='*60}")
        logger.info("Optimization Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Best loss: {self.best_loss:.6f} (iteration {self.best_iteration})")
        logger.info(f"Final loss: {self.history[-1]['loss']:.6f}")
        logger.info(f"{'='*60}\n")

        # Save final results
        results = {
            "history": self.history,
            "best_loss": self.best_loss,
            "best_iteration": self.best_iteration,
            "initial_loss": initial_loss.item(),
            "final_loss": self.history[-1]["loss"],
        }

        self._save_final_results(
            results,
            x0_init,
            target_sequence,
            self.ground_truth_tensor,
            ocean_mask,
            regional_masks,
        )

        # Close TensorBoard writer
        self.writer.close()

        return self.best_x0, results

    def _log_to_tensorboard(
        self,
        iteration: int,
        loss_details: Dict,
        metrics: Dict,
        kernel_size: int,
        x0_current: Optional[torch.Tensor] = None,
        x0_reference: Optional[torch.Tensor] = None,
    ):

        """
        Log metrics to TensorBoard.

        Follows R8 decision on TensorBoard hierarchy:
        loss/J_obs/{total,ssh,sst,uo,vo}
        metrics/rmse/{global,basin}/{var}
        state/ic/{norm,update_magnitude}
        """
        """
        Log metrics to TensorBoard.

        Follows R8 decision on TensorBoard hierarchy:
        loss/J_obs/{total,ssh,sst,uo,vo}
        metrics/rmse/{global,basin}/{var}
        state/ic/{norm,update_magnitude}
        """
        # Loss metrics (per-variable)
        per_var = loss_details["weighted_per_variable"]
        self.writer.add_scalar("loss/J_obs/total", per_var["total"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/SSH", per_var["ssh"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/SST", per_var["sst"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/SSS", per_var["sss"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/UO", per_var["uo"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/VO", per_var["vo"].mean().item(), iteration)

        # Structural loss metrics (per-variable) if enabled
        if "struct_mse_per_var_weighted" in loss_details:
            struct_per_var = loss_details["struct_mse_per_var_weighted"]
            self.writer.add_scalar("loss/S_loss/total", struct_per_var["total"].mean().item(), iteration)
            self.writer.add_scalar("loss/S_loss/SSH", struct_per_var["ssh"].mean().item(), iteration)
            self.writer.add_scalar("loss/S_loss/SST", struct_per_var["sst"].mean().item(), iteration)
            self.writer.add_scalar("loss/S_loss/SSS", struct_per_var["sss"].mean().item(), iteration)
            self.writer.add_scalar("loss/S_loss/UO", struct_per_var["uo"].mean().item(), iteration)
            self.writer.add_scalar("loss/S_loss/VO", struct_per_var["vo"].mean().item(), iteration)

        # RMSE metrics (global)
        for var, rmse in metrics["rmse_global"].items():
            self.writer.add_scalar(f"metrics/rmse/global/{var}", rmse, iteration)

        # RMSE metrics (basin-stratified)
        if "rmse_basin" in metrics:
            for region, var_dict in metrics["rmse_basin"].items():
                for var, rmse in var_dict.items():
                    self.writer.add_scalar(f"metrics/rmse/{region}/{var}", rmse, iteration)

        # IC RMSE
        for var, rmse in metrics["ic_rmse"].items():
            self.writer.add_scalar(f"metrics/ic_rmse/{var}", rmse, iteration)

        # Finite-difference IC update norm on the step correction field
        if self._last_ic_update is not None and self._ocean_mask is not None:
            finite_diff_total, finite_diff_per_channel = self._compute_finite_difference_ic_update(
                self._last_ic_update[0, -1],
                self._ocean_mask.detach().cpu(),
            )
            self.writer.add_scalar(
                "diag/finite_difference_ic_update/total",
                finite_diff_total,
                iteration,
            )
            for var, norm in finite_diff_per_channel.items():
                self.writer.add_scalar(
                    f"diag/finite_difference_ic_update/per_channel/{var}",
                    norm,
                    iteration,
                )

            # Also measure the cumulative correction relative to the original IC.
            if x0_current is not None and x0_reference is not None:
                cumulative_update = x0_current[0, -1] - x0_reference[0, -1]
                cumulative_total, cumulative_per_channel = (
                    self._compute_finite_difference_ic_update(
                        cumulative_update,
                        self._ocean_mask.detach().cpu(),
                    )
                )
                self.writer.add_scalar(
                    "diag/finite_difference_ic_update/cumulative/total",
                    cumulative_total,
                    iteration,
                )
                for var, norm in cumulative_per_channel.items():
                    self.writer.add_scalar(
                        f"diag/finite_difference_ic_update/cumulative/per_channel/{var}",
                        norm,
                        iteration,
                    )

        # Pooling kernel size
        self.writer.add_scalar("optimizer/pooling_kernel", kernel_size, iteration)

        # IC correction/update visualizations (per-step and cumulative, coordinate-aware)
        # _last_ic_update: [B, T, C, H, W] on CPU
        if self._last_ic_update is not None and self._ocean_mask is not None:
            import numpy as _np

            var_names = ["SSH", "T", "S", "U", "V"]
            units_map = {"SSH": "m", "T": "°C", "S": "psu", "U": "m/s", "V": "m/s"}

            # Last applied IC update: [C, H, W]
            last_update = self._last_ic_update[0, -1]
            masked_update_all = last_update * self._ocean_mask.detach().cpu()
            update_images = [("step/ic_update", masked_update_all, "per-step")]

            if x0_current is not None and x0_reference is not None:
                total_update = x0_current[0, -1] - x0_reference[0, -1]
                masked_total_update = total_update.detach().cpu() * self._ocean_mask.detach().cpu()
                update_images.append(("total/ic_update", masked_total_update, "cumulative"))

            if masked_update_all.numel() > 0:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                extent = None
                origin = "lower"
                if (
                    self.input_sequence_xr is not None
                    and "lat" in self.input_sequence_xr.coords
                    and "lon" in self.input_sequence_xr.coords
                ):
                    lat = self.input_sequence_xr.coords["lat"].values
                    lon = self.input_sequence_xr.coords["lon"].values
                    extent = (
                        float(_np.min(lon)),
                        float(_np.max(lon)),
                        float(_np.min(lat)),
                        float(_np.max(lat)),
                    )
                    origin = "lower" if float(lat[0]) < float(lat[-1]) else "upper"

                for image_tag, update_image, update_kind in update_images:
                    try:
                        fig, axes = plt.subplots(
                            5, 1, figsize=(11, 18), dpi=100, constrained_layout=True
                        )
                        if not isinstance(axes, _np.ndarray):
                            axes = _np.array([axes])

                        fig.suptitle(
                            f"IC {update_kind} update (north up, lon/lat coordinates)",
                            fontsize=13,
                        )

                        for ch_idx, ax in enumerate(axes):
                            arr = update_image[ch_idx].detach().cpu().numpy()
                            var_name = var_names[ch_idx] if ch_idx < len(var_names) else f"ch{ch_idx}"
                            ch_abs = float(_np.nanmax(_np.abs(arr))) if arr.size > 0 else 1.0
                            if not _np.isfinite(ch_abs) or ch_abs == 0.0:
                                ch_abs = 1.0

                            im = ax.imshow(
                                arr,
                                cmap="seismic",
                                vmin=-ch_abs,
                                vmax=ch_abs,
                                interpolation="nearest",
                                origin=origin,
                                aspect="auto",
                                extent=extent,
                            )
                            ax.set_title(
                                f"{var_name} update ({units_map.get(var_name, '')})",
                                fontsize=10,
                            )
                            ax.set_xlabel("lon")
                            ax.set_ylabel("lat")
                            ax.tick_params(labelsize=8)
                            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

                        legend_text = (
                            "IC update = learning_rate × filtered_gradient × ocean_mask\n"
                            f"Color shows the {update_kind} correction in physical units."
                        )
                        fig.text(
                            0.01,
                            0.005,
                            legend_text,
                            fontsize=8,
                            color="black",
                            bbox=dict(facecolor="white", alpha=0.75, pad=3, edgecolor="none"),
                        )

                        fig.canvas.draw()
                        img_buf = _np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                        self.writer.add_image(
                            image_tag, img_buf, iteration, dataformats="HWC"
                        )
                    except Exception:
                        # Fallback: normalized raster if plotting fails.
                        update_np = update_image.detach().cpu().numpy()
                        for ch_idx, var_name in enumerate(var_names):
                            ch_arr = update_np[ch_idx]
                            ch_abs = float(_np.nanmax(_np.abs(ch_arr)))
                            if not _np.isfinite(ch_abs) or ch_abs == 0.0:
                                ch_abs = 1.0
                            norm_ch = ((ch_arr / (2.0 * ch_abs)) + 0.5).clip(0.0, 1.0)
                            self.writer.add_image(
                                f"{image_tag}/{var_name}",
                                norm_ch[None, :, :],
                                iteration,
                                dataformats="CHW",
                            )
                    finally:
                        plt.close("all")

    def _log_histograms_to_tensorboard(
        self,
        iteration: int,
        gradients: torch.Tensor,
        x0_current: torch.Tensor,
        ocean_mask: torch.Tensor,
    ):
        """
        Log histograms and embeddings to TensorBoard.

        Args:
            iteration: Current iteration
            gradients: Filtered gradients [B, T, C, H, W]
            x0_current: Current IC [B, T, C, H, W]
            ocean_mask: Ocean mask [C, H, W]
        """
        # Gradient histograms (per channel, using last timestep)
        var_names = ["SSH", "T", "S", "U", "V"]
        for ch_idx, var_name in enumerate(var_names):
            grad_ch = gradients[0, -1, ch_idx]  # [H, W]
            ocean_points = grad_ch[ocean_mask[ch_idx] > 0]

            if ocean_points.numel() > 0:
                self.writer.add_histogram(
                    f"gradients/{var_name}", ocean_points.detach().cpu().numpy(), iteration
                )

        # IC histograms (per channel, using last timestep)
        for ch_idx, var_name in enumerate(var_names):
            ic_ch = x0_current[0, -1, ch_idx]  # [H, W]
            ocean_points = ic_ch[ocean_mask[ch_idx] > 0]

            if ocean_points.numel() > 0:
                self.writer.add_histogram(
                    f"state/ic/{var_name}", ocean_points.detach().cpu().numpy(), iteration
                )

    def _compute_finite_difference_ic_update(
        self,
        field: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute finite-difference IC update norms for a [C, H, W] field."""
        var_names = ["SSH", "T", "S", "U", "V"]
        total_sq = 0.0
        per_channel = {}

        for ch_idx, var_name in enumerate(var_names):
            img = field[ch_idx]
            mask = ocean_mask[ch_idx].to(img)

            amplitude_sq = float((img.pow(2) * mask).sum().item())
            finite_difference_sq = 0.0

            if img.shape[1] > 1:
                gx = img[:, 1:] - img[:, :-1]
                gx_mask = mask[:, 1:] * mask[:, :-1]
                finite_difference_sq += float((gx.pow(2) * gx_mask).sum().item())

            if img.shape[0] > 1:
                gy = img[1:, :] - img[:-1, :]
                gy_mask = mask[1:, :] * mask[:-1, :]
                finite_difference_sq += float((gy.pow(2) * gy_mask).sum().item())

            # Divide by the field amplitude so uniformly scaling an update does
            # not change its pixelization score.
            channel_sq = (
                finite_difference_sq / amplitude_sq if amplitude_sq > 0.0 else 0.0
            )
            per_channel[var_name] = float(np.sqrt(channel_sq))
            total_sq += channel_sq

        return float(np.sqrt(total_sq)), per_channel

    def _print_progress(self, iteration: int, loss: float, metrics: Dict, kernel_size: int):
        """
        Print optimization progress.

        Args:
            iteration: Current iteration
            loss: Current loss value
            metrics: Metrics dictionary
            kernel_size: Current pooling kernel size
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration}/{self.num_iterations}")
        logger.info(f"Pooling Kernel: {kernel_size}")
        logger.info(f"{'='*60}")
        logger.info(f"Loss: {loss:.6f}")
        logger.info(f"Best Loss: {self.best_loss:.6f} (iteration {self.best_iteration})")
        logger.info("Global RMSE:")
        for var, rmse in metrics["rmse_global"].items():
            logger.info(f"  {var:8s}: {rmse:.6e}")
        logger.info("IC RMSE:")
        for var, rmse in metrics["ic_rmse"].items():
            logger.info(f"  {var:8s}: {rmse:.6e}")
        logger.info(f"{'='*60}\n")

    def _save_checkpoint(self, iteration: int, x0_current: torch.Tensor, predictions: torch.Tensor):
        """
        Save checkpoint to experiment's checkpoints/ directory.

        Args:
            iteration: Current iteration
            x0_current: Current IC
            predictions: Current predictions
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter{iteration}.pt"

        torch.save(
            {
                "iteration": iteration,
                "x0": x0_current.detach().cpu(),
                "predictions": predictions.detach().cpu(),
                "loss": self.history[-1]["loss"],
                "best_loss": self.best_loss,
                "best_iteration": self.best_iteration,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def _save_final_results(
        self,
        results: Dict,
        x0_init: torch.Tensor,
        target_sequence: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
        regional_masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Save final optimization results to experiment directory.

        Args:
            results: Results dictionary
            x0_init: Initial condition [B, T=2, C, H, W]
            target_sequence: Target observations [B, T_obs, C, H, W] (assimilation window)
            ground_truth_sequence: Full ground truth [B, T=forecast_horizon, C, H, W]
            ocean_mask: Ocean mask [C, H, W]
        """
        # Save history as JSON to metrics/
        history_path = self.metrics_dir / "optimization_history.json"
        with open(history_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved history: {history_path}")

        # Compute full forecasts for diagnostics (T=forecast_horizon)
        logger.info(f"Computing full forecast ({self.forecast_horizon} steps) for diagnostics...")
        with torch.no_grad():
            _, reference_forecast = self.forward_model.forward(x0_init, self.forecast_horizon)
            _, full_forecast = self.forward_model.forward(self.best_x0, self.forecast_horizon)
        logger.info(f"Reference forecast shape: {reference_forecast.shape}")
        logger.info(f"Optimized forecast shape: {full_forecast.shape}")

        # Save best IC and predictions to exp root (legacy .pt format)
        best_path = self.exp_dir / "best_solution.pt"
        torch.save(
            {
                "x0": self.best_x0.cpu(),
                "predictions": self.best_predictions.cpu(),  # Assimilation window predictions (T=7)
                "reference_forecast": reference_forecast.cpu(),
                "full_forecast": full_forecast.cpu(),  # Full forecast (T=forecast_horizon)
                "iteration": self.best_iteration,
                "loss": self.best_loss,
            },
            best_path,
        )

        logger.info(f"Saved best solution: {best_path}")
         
        # Save state NetCDF + visualization diagnostics (if xarray datasets are provided)
        if self.input_sequence_xr is not None and self.target_sequence_xr is not None:
            try:
                self.output_handler.save_outputs(
                    x0_init=x0_init,
                    x0_optimized=self.best_x0,
                    reference_forecast=reference_forecast,
                    full_forecast=full_forecast,  # Full forecast (T=forecast_horizon) for state/plots
                    target_sequence=target_sequence,  # Assimilation window (T=7)
                    ground_truth_sequence=ground_truth_sequence,  # Full ground truth for RMSE evolution
                    input_sequence=self.input_sequence_xr,
                    target_sequence_xr=self.target_sequence_xr,
                    ground_truth_sequence_xr=self.ground_truth_sequence_xr,
                    regional_masks=regional_masks,
                    ocean_mask=ocean_mask,
                    best_iteration=self.best_iteration,
                    best_loss=self.best_loss,
                    mean_field=self.metrics_computer.mean_field,
                )
            except Exception as e:
                logger.error(f"Failed to save outputs: {e}", exc_info=True)
                logger.warning("Continuing without NetCDF outputs...")
        else:
            logger.warning("Skipping NetCDF diagnostics (xarray datasets not provided)")
