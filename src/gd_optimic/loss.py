"""
Loss computation for IC optimization.

Provides:
    - ObservationLoss: MSE loss with observation masking
    
Phase 1 scope: J_obs only (observation term).
Phase 1.b will add J_b (background) and J_q (model error) terms.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class ObservationLoss:
    """
    Observation term of the cost function: J_obs
    
    Implements:
    - Per-timestep MSE between forecast and observations
    - Observation masking (only compute loss where obs exist)
    - Dynamic or manual loss weighting across variables
    - Per-variable loss tracking for TensorBoard
    
    Phase 1: J_obs only (no background or model error terms)
    Phase 1.b: Will add J_b + J_q
    """
    
    def __init__(
        self,
        obs_mask: torch.Tensor,
        loss_weighting: str = "dynamic",
        manual_weights: list = None,
        device: str = "cuda"
    ):
        """
        Initialize observation loss function.
        
        Args:
            obs_mask: Observation mask [T, C, H, W] - 1 where obs exist, 0 elsewhere
            loss_weighting: Loss weighting strategy ('dynamic' or 'manual')
                - 'dynamic': Automatically balance loss magnitudes across variables
                - 'manual': Use user-specified weights
            manual_weights: Manual weights for each channel [SSH, T, S, U, V] if loss_weighting='manual'
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.obs_mask = obs_mask
        self.loss_weighting = loss_weighting
        self.manual_weights = manual_weights
        self.device = device
        
        # MSE loss function
        self.mse_fn = nn.MSELoss(reduction='none')
    
    def compute_mse_per_step(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-timestep, per-channel MSE with observation masking.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            
        Returns:
            mse_per_step: MSE for each timestep and channel [B, T, C]
        """
        # Compute element-wise MSE: [B, T, C, H, W]
        mse_raw = self.mse_fn(predictions, targets)
        
        # Apply observation mask: only compute loss where observations exist
        # obs_mask is [T, C, H, W], broadcast to [B, T, C, H, W]
        mse_masked = mse_raw * self.obs_mask.unsqueeze(0)
        
        # Count valid observation points per channel
        # obs_mask_sum: [T, C] - number of valid obs pixels per channel per timestep
        obs_mask_sum = self.obs_mask.sum(dim=(-2, -1))  # Sum over spatial dims
        
        # Normalize by number of valid observations
        # This gives mean MSE over observed pixels only
        # mse_per_step: [B, T, C]
        mse_per_step = mse_masked.sum(dim=(-2, -1)) / (obs_mask_sum.unsqueeze(0) + 1e-10)
        
        return mse_per_step
    
    def compute_normalized_mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute normalized MSE (NMSE) with time-aggregated statistics.
        
        NMSE normalization: Divide MSE by variance of target to get dimensionless metric.
        This helps balance loss contributions across variables with different scales.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            
        Returns:
            nmse_per_step: Normalized MSE per step [B, T, C]
            nloss_sum_intime: Sum of NMSE over time [B, C]
            nloss_mean_intime: Mean NMSE over time [B, C]
            total_nloss: Total scalar loss [B]
        """
        # Compute per-step MSE: [B, T, C]
        mse_per_step = self.compute_mse_per_step(predictions, targets)
        
        # Compute target variance per channel (for normalization)
        # Apply obs mask to only consider observed pixels
        targets_masked = targets * self.obs_mask.unsqueeze(0)
        
        # Variance per channel: [C]
        # We compute variance over all valid observations (B, T, H, W dims)
        obs_mask_expanded = self.obs_mask.unsqueeze(0).expand_as(targets)
        valid_count = obs_mask_expanded.sum(dim=(0, 1, 3, 4))  # Count per channel
        
        # Mean per channel
        target_mean = targets_masked.sum(dim=(0, 1, 3, 4)) / (valid_count + 1e-10)
        
        # Variance per channel: E[(x - mean)^2]
        targets_centered = (targets - target_mean.view(1, 1, -1, 1, 1)) * self.obs_mask.unsqueeze(0)
        target_var = (targets_centered ** 2).sum(dim=(0, 1, 3, 4)) / (valid_count + 1e-10)
        
        # Normalize MSE by target variance: [B, T, C]
        nmse_per_step = mse_per_step / (target_var.unsqueeze(0).unsqueeze(0) + 1e-10)
        
        # Time aggregation
        nloss_sum_intime = nmse_per_step.sum(dim=1)  # [B, C] - sum over time
        nloss_mean_intime = nmse_per_step.mean(dim=1)  # [B, C] - mean over time
        
        # Total loss (scalar)
        total_nloss = nloss_sum_intime.sum(dim=1)  # [B]
        
        return nmse_per_step, nloss_sum_intime, nloss_mean_intime, total_nloss
    
    def compute_weighted_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted loss with dynamic or manual weighting.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            
        Returns:
            weighted_loss: Weighted scalar loss [B]
            weights: Weights used for each channel [B, C]
        """
        # Compute normalized MSE aggregated over time: [B, C]
        _, nloss_sum_intime, _, _ = self.compute_normalized_mse(predictions, targets)
        
        if self.loss_weighting == 'dynamic':
            # Dynamic weights: balance loss magnitudes
            # Larger loss → smaller weight (to prevent domination)
            # This gives equal attention to all variables regardless of scale
            
            loss_mag = nloss_sum_intime.detach().abs() + 1e-10
            weights = loss_mag.sum(dim=1, keepdim=True) / loss_mag  # [B, C]
            
        elif self.loss_weighting == 'manual':
            # Manual weights from configuration
            if self.manual_weights is None:
                # Default: equal weights
                self.manual_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
            
            manual_weights_tensor = torch.tensor(
                self.manual_weights,
                dtype=nloss_sum_intime.dtype,
                device=self.device
            ).view(1, -1)  # [1, C]
            
            # Normalize weights to sum to 1
            weights = manual_weights_tensor / (manual_weights_tensor.sum(dim=1, keepdim=True) + 1e-10)
            
            # Expand to batch size
            weights = weights.expand(nloss_sum_intime.shape[0], -1)  # [B, C]
        
        else:
            raise ValueError(f"Unknown loss_weighting: {self.loss_weighting}. Must be 'dynamic' or 'manual'")
        
        # Apply weights: [B]
        weighted_loss = (weights * nloss_sum_intime).sum(dim=1)
        
        return weighted_loss, weights
    
    def compute_per_variable_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss per variable for TensorBoard logging.
        
        Follows R8 decision on TensorBoard hierarchy:
        loss/J_obs/{total,ssh,sst,uo,vo}
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            
        Returns:
            loss_dict: Dictionary with per-variable losses
                Keys: 'total', 'ssh', 'sst', 'sss', 'uo', 'vo'
        """
        # Compute normalized MSE aggregated over time: [B, C]
        _, nloss_sum_intime, _, _ = self.compute_normalized_mse(predictions, targets)
        
        # Extract per-variable losses
        # Channel mapping: 0=SSH, 1=T, 2=S, 3=U, 4=V
        loss_dict = {
            'total': nloss_sum_intime.sum(dim=1),  # [B] - total loss
            'ssh': nloss_sum_intime[:, 0],  # [B] - SSH loss
            'sst': nloss_sum_intime[:, 1],  # [B] - Temperature (SST) loss
            'sss': nloss_sum_intime[:, 2],  # [B] - Salinity (SSS) loss
            'uo': nloss_sum_intime[:, 3],  # [B] - Eastward velocity loss
            'vo': nloss_sum_intime[:, 4],  # [B] - Northward velocity loss
        }
        
        return loss_dict
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: compute loss.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            return_details: If True, return detailed loss breakdown
            
        Returns:
            loss: Scalar loss (for backprop)
            details: Dictionary with loss breakdown (if return_details=True)
        """
        # Compute weighted loss
        weighted_loss, weights = self.compute_weighted_loss(predictions, targets)
        
        # Take mean over batch
        loss = weighted_loss.mean()
        
        if return_details:
            # Compute detailed breakdown for logging
            nmse_per_step, nloss_sum_intime, nloss_mean_intime, total_nloss = \
                self.compute_normalized_mse(predictions, targets)
            
            per_var_losses = self.compute_per_variable_loss(predictions, targets)
            
            details = {
                'loss': loss.item(),
                'nmse_per_step': nmse_per_step,  # [B, T, C]
                'nloss_sum_intime': nloss_sum_intime,  # [B, C]
                'nloss_mean_intime': nloss_mean_intime,  # [B, C]
                'total_nloss': total_nloss,  # [B]
                'weights': weights,  # [B, C]
                'per_variable': per_var_losses,  # Dict
            }
            
            return loss, details
        else:
            return loss, {}
