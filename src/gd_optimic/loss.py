"""
Loss computation for IC optimization.

Provides:
    - ObservationLoss: MSE loss with observation masking
    - Combined loss with optional structural consistency term
    
Phase 1 scope: J_obs only (observation term).
Phase 2: Add J_struct (structure consistency term) with optional operators.
Phase 1.b will add J_b (background) and J_q (model error) terms.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from .structural_loss import StructureConsistencyLoss


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
        use_structural_loss: bool = False,
        structural_operator: str = "gradient",
        structural_loss_weight: float = 0.01,
        device: str = "cuda"
    ):
        """
        Initialize observation loss function with optional structural consistency term.
        
        Args:
            obs_mask: Observation mask [T, C, H, W] - 1 where obs exist, 0 elsewhere
            loss_weighting: Loss weighting strategy ('dynamic' or 'manual')
                - 'dynamic': Automatically balance loss magnitudes across variables
                - 'manual': Use user-specified weights
            manual_weights: Manual weights for each channel [SSH, T, S, U, V] if loss_weighting='manual'
            use_structural_loss: Enable structural consistency loss term J_struct
            structural_operator: Type of structural operator ('gradient' or 'laplacian')
            structural_loss_weight: Weight for structural loss term (typical range: 0.01-0.1)
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.obs_mask = obs_mask
        self.loss_weighting = loss_weighting
        self.manual_weights = manual_weights
        self.device = device
        
        # MSE loss function
        self.mse_fn = nn.MSELoss(reduction='none')
        
        # Structural consistency loss (optional)
        self.use_structural_loss = use_structural_loss
        self.structural_loss_weight = structural_loss_weight
        
        if use_structural_loss:
            self.structural_loss_fn = StructureConsistencyLoss(
                operator_type=structural_operator,
                obs_mask=obs_mask,
                device=device
            )
        else:
            self.structural_loss_fn = None
    
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
        se_raw = self.mse_fn(predictions, targets)
        
        # Apply observation mask: only compute loss where observations exist
        # obs_mask is [T, C, H, W], broadcast to [B, T, C, H, W]
        se_masked = se_raw * self.obs_mask.unsqueeze(0)
        
        # Count valid observation points per channel
        # obs_mask_sum: [T, C] - number of valid obs pixels per channel per timestep
        obs_mask_sum = self.obs_mask.sum(dim=(-2, -1))  # Sum over spatial dims
        
        # Normalize by number of valid observations
        # This gives mean MSE over observed pixels only
        # mse_per_step: [B, T, C]
        mse_per_step = se_masked.sum(dim=(-2, -1)) / (obs_mask_sum.unsqueeze(0) + 1e-10)
        
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
        
        # Compute (or reuse cached) target variance per channel for normalization.
        # The target variance depends only on the targets and obs_mask — cache it to avoid
        # recomputing every optimization iteration which saves time.
        if not hasattr(self, "_cached_target_var") or self._cached_target_var is None:
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
            # target_var = torch.tensor(1, device=self.device) 
            # Cache on CPU to avoid holding GPU memory
            self._cached_target_var = target_var.detach().cpu()
        else:
            # Move cached variance to the device of mse_per_step for computation
            target_var = self._cached_target_var.to(mse_per_step.device)

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
            weights = self._compute_dynamic_weights(nloss_sum_intime)
            
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

    @staticmethod
    def _compute_dynamic_weights(loss_per_var: torch.Tensor) -> torch.Tensor:
        """Compute inverse-magnitude weights from per-variable loss values."""
        loss_mag = loss_per_var.detach().abs() + 1e-10
        return loss_mag.sum(dim=1, keepdim=True) / loss_mag
    
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
        Forward pass: compute combined loss (observation + structural consistency).
        
        Total loss: L = J_obs + weight_struct * J_struct.
        Manual weighting uses the configured weights for both terms. Dynamic
        weighting is computed independently from each term's channel magnitudes.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            return_details: If True, return detailed loss breakdown
            
        Returns:
            loss: Scalar loss (for backprop)
            details: Dictionary with loss breakdown (if return_details=True)
        """
        # Compute observation loss with weighting
        weighted_loss, weights = self.compute_weighted_loss(predictions, targets)
        
        # Take mean over batch
        obs_loss = weighted_loss.mean()
        
        # Compute structural consistency loss if enabled
        struct_loss_value = torch.tensor(0.0, device=obs_loss.device)
        struct_loss_details = {}
        struct_mse_per_var = None
        struct_weights = None
        
        if self.use_structural_loss and self.structural_loss_fn is not None:
            struct_loss_value, struct_loss_details = self.structural_loss_fn(
                predictions,
                targets,
                weight=1.0
            )
            struct_mse_per_var = struct_loss_details.get('struct_mse_per_var', None)  # [B, C]
            
            if struct_mse_per_var is not None:
                if self.loss_weighting == 'dynamic':
                    # Structural dynamic weights must reflect structural loss
                    # magnitudes, not observation loss magnitudes.
                    struct_weights = self._compute_dynamic_weights(struct_mse_per_var)
                else:
                    # Manual weights are configured once and apply directly to
                    # both observation and structural terms.
                    struct_weights = weights

                weighted_struct_per_var_for_loss = struct_mse_per_var * struct_weights  # [B, C]
                # Aggregate over variables and apply structural loss weight
                struct_loss_per_batch = weighted_struct_per_var_for_loss.sum(dim=1)  # [B]
                struct_loss_value = (self.structural_loss_weight * struct_loss_per_batch).mean()
        
        # Total loss
        total_loss = obs_loss + struct_loss_value
        
        if return_details:
            # Compute detailed breakdown for logging
            nmse_per_step, nloss_sum_intime, nloss_mean_intime, total_nloss = \
                self.compute_normalized_mse(predictions, targets)
            
            wloss = weights * nloss_sum_intime
            wloss_dict = {
                'total': wloss.sum(dim=1),  # [B] - total loss
                'ssh': wloss[:, 0],  # [B] - SSH loss
                'sst': wloss[:, 1],  # [B] - Temperature (SST) loss
                'sss': wloss[:, 2],  # [B] - Salinity (SSS) loss
                'uo': wloss[:, 3],  # [B] - Eastward velocity loss
                'vo': wloss[:, 4],  # [B] - Northward velocity loss
            }
            
            # Compute per-variable structural loss if available
            struct_loss_per_var_dict = {}
            if struct_mse_per_var is not None:
                # struct_mse_per_var shape: [B, C]
                # Apply structural weighting and structural loss scaling.
                weighted_by_var = struct_mse_per_var * struct_weights  # [B, C]
                # Then: weight by structural loss scaling parameter
                weighted_struct_per_var = weighted_by_var * self.structural_loss_weight
                struct_loss_per_var_dict = {
                    'total': weighted_struct_per_var.sum(dim=1),  # [B]
                    'ssh': weighted_struct_per_var[:, 0],  # [B]
                    'sst': weighted_struct_per_var[:, 1],  # [B]
                    'sss': weighted_struct_per_var[:, 2],  # [B]
                    'uo': weighted_struct_per_var[:, 3],  # [B]
                    'vo': weighted_struct_per_var[:, 4],  # [B]
                }
                        
            details = {
                'loss': total_loss.item(),
                'obs_loss': obs_loss.item(),
                'nmse_per_step': nmse_per_step,  # [B, T, C]
                'nloss_sum_intime': nloss_sum_intime,  # [B, C]
                'nloss_mean_intime': nloss_mean_intime,  # [B, C]
                'total_nloss': total_nloss,  # [B]
                'weights': weights,  # [B, C]
                'weighted_per_variable': wloss_dict,  # Dict
            }
            
            # Add structural loss details if enabled
            if self.use_structural_loss:
                details['struct_loss'] = struct_loss_value.item()
                details['struct_loss_weight'] = self.structural_loss_weight
                details['struct_weights'] = struct_weights  # [B, C]
                details['struct_mse_per_var_weighted'] = struct_loss_per_var_dict  # Per-variable breakdown
                details.update({
                    f'struct_{k}': v for k, v in struct_loss_details.items()
                    if k not in ['struct_mse_per_var']  # Exclude per_var as we already added it
                })
            
            return total_loss, details
        else:
            return total_loss, {}
