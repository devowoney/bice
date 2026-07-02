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
    ):
        """
        Initialize IC optimizer.

        Args:
            forward_model: Forward model wrapper
            loss_fn: Loss function (observation term)
            gradient_filter: Gradient filtering object
            metrics_computer: Metrics computation object
            learning_rate: Gradient descent learning rate
            num_iterations: Total number of optimization iterations
            device: PyTorch device ('cuda' or 'cpu')
            output_dir: Base output directory (typically .tmp/outputs)
            tensorboard_subdir: Subdirectory for TensorBoard logs (inside exp_id dir)
            checkpoints_subdir: Subdirectory for checkpoints (inside exp_id dir)
            metrics_subdir: Subdirectory for metrics (inside exp_id dir)
            save_frequency: Save checkpoint every N iterations
            log_frequency: Log metrics every N iterations
            histogram_frequency: Log histograms/embeddings every N iterations
            scheduled_pooling: Optional scheduled pooling object for multigrid optimization
            forecast_horizon: Number of forecast steps for diagnostics (default: 28)
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

        # Number of forward steps to roll out (for assimilation window)
        num_forecast_steps = target_sequence.shape[1]

        logger.info(f"\n{'='*60}")
        logger.info("Starting IC Optimization")
        logger.info(f"{'='*60}")
        logger.info(f"Initial condition shape: {x0_init.shape}")
        logger.info(f"Target sequence shape: {target_sequence.shape}")
        logger.info(f"Forecast steps: {num_forecast_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
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

            # Backward pass: compute gradients
            gradients = torch.autograd.grad(loss, x0_current, create_graph=False)[0]

            # Apply gradient filtering
            filtered_gradients = self.gradient_filter(
                gradients,
                x0_current=x0_current,
                x0_reference=x0_reference,
                kernel_size=current_kernel,
            )

            # Apply ocean mask to gradients (zero out land gradients)
            with torch.no_grad():
                # Ensure ocean_mask matches channel dimension of gradients
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

                masked_gradients = filtered_gradients * ocean_mask.unsqueeze(0).unsqueeze(0)

                # Gradient descent update
                x0_current = x0_current - self.learning_rate * masked_gradients

                # Ensure x0_current requires grad for next iteration
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
                "per_variable_loss": {
                    k: v.mean().item() if torch.is_tensor(v) else v
                    for k, v in loss_details["per_variable"].items()
                },
                "rmse_global": metrics["rmse_global"],
                "ic_rmse": metrics["ic_rmse"],
            }

            if regional_masks is not None:
                history_entry["rmse_basin"] = metrics["rmse_basin"]

            self.history.append(history_entry)

            # Logging to TensorBoard
            if (iteration + 1) % self.log_frequency == 0:
                self._log_to_tensorboard(iteration + 1, loss_details, metrics, current_kernel)

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
        self, iteration: int, loss_details: Dict, metrics: Dict, kernel_size: int
    ):
        """
        Log metrics to TensorBoard.

        Follows R8 decision on TensorBoard hierarchy:
        loss/J_obs/{total,ssh,sst,uo,vo}
        metrics/rmse/{global,basin}/{var}
        metrics/gradient/norm/{total,per_channel}
        state/ic/{norm,update_magnitude}
        """
        # Loss metrics (per-variable)
        per_var = loss_details["per_variable"]
        self.writer.add_scalar("loss/J_obs/total", per_var["total"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/ssh", per_var["ssh"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/sst", per_var["sst"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/sss", per_var["sss"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/uo", per_var["uo"].mean().item(), iteration)
        self.writer.add_scalar("loss/J_obs/vo", per_var["vo"].mean().item(), iteration)

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

        # Pooling kernel size
        self.writer.add_scalar("state/pooling_kernel", kernel_size, iteration)

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
                    best_loss=self.best_loss
                )
            except Exception as e:
                logger.error(f"Failed to save outputs: {e}", exc_info=True)
                logger.warning("Continuing without NetCDF outputs...")
        else:
            logger.warning("Skipping NetCDF diagnostics (xarray datasets not provided)")
