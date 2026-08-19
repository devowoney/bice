"""
Structural operators for structure consistency loss.

Provides:
    - StructuralOperator: Base class for spatial derivative operators
    - GradientOperator: First-order gradient (∇)
    - LaplacianOperator: Second-order Laplacian (∇²)
    - StructureConsistencyLoss: Main loss term J_struct

Structure Consistency Loss:
    J_struct = ||S(ψ_obs) - S(G(ψ))||² 
    where S(·) is a structural operator (gradient, Laplacian, etc.)
    and G(ψ) is the forecast state
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class StructuralOperator(ABC):
    """Base class for spatial derivative operators."""
    
    @abstractmethod
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply structural operator to field.
        
        Args:
            field: Input field [B, T, C, H, W] or [B, C, H, W]
            
        Returns:
            Output after applying operator
        """
        pass
    
    @abstractmethod
    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        """Implement operator call."""
        pass


class GradientOperator(StructuralOperator):
    """
    First-order gradient operator (∇).
    
    Computes spatial gradients in x and y directions.
    Output: Concatenates [∂f/∂x, ∂f/∂y] along channel dimension.
    
    Kernel:
        ∂f/∂x ≈ (f[i,j+1] - f[i,j-1]) / 2  (central difference)
        ∂f/∂y ≈ (f[i+1,j] - f[i-1,j]) / 2  (central difference)
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize gradient operator.
        
        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
    
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute first-order gradient.
        
        Args:
            field: Input field [B, T, C, H, W] or [B, C, H, W]
            
        Returns:
            Gradient field with doubled channels: [B, T, 2*C, H, W] or [B, 2*C, H, W]
        """
        is_temporal = (field.ndim == 5)
        
        if is_temporal:
            B, T, C, H, W = field.shape
            # Reshape to [B*T, C, H, W] for batch processing
            field_reshaped = field.reshape(B * T, C, H, W)
        else:
            B, C, H, W = field.shape
            field_reshaped = field
        
        # Compute gradients using central difference
        # ∂f/∂x: (f[i,j+1] - f[i,j-1]) / 2
        grad_x = (field_reshaped[:, :, :, 2:] - field_reshaped[:, :, :, :-2]) / 2.0
        
        # ∂f/∂y: (f[i+1,j] - f[i-1,j]) / 2
        grad_y = (field_reshaped[:, :, 2:, :] - field_reshaped[:, :, :-2, :]) / 2.0
        
        # Pad gradients to match original spatial dimensions
        # Central difference reduces size by 2 in each direction, so pad by 1 on each side
        grad_x_padded = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')  # Pad x-direction
        grad_y_padded = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')  # Pad y-direction
        
        # Concatenate along channel dimension: [B*T, 2*C, H, W]
        grad_combined = torch.cat([grad_x_padded, grad_y_padded], dim=1)
        
        if is_temporal:
            # Reshape back to [B, T, 2*C, H, W]
            grad_combined = grad_combined.reshape(B, T, 2*C, H, W)
        
        return grad_combined
    
    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        """Compute first-order gradient."""
        return self.apply(field)


class LaplacianOperator(StructuralOperator):
    """
    Second-order Laplacian operator (∇²).
    
    Computes Laplacian (second spatial derivatives).
    Output: Laplacian field, same shape as input.
    
    Kernel:
        ∇²f ≈ (f[i-1,j] + f[i+1,j] + f[i,j-1] + f[i,j+1] - 4*f[i,j])
        
    This is the discrete Laplacian approximation using 4-neighbor stencil.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Laplacian operator.
        
        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
    
    def apply(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian.
        
        Args:
            field: Input field [B, T, C, H, W] or [B, C, H, W]
            
        Returns:
            Laplacian field, same shape as input [B, T, C, H, W] or [B, C, H, W]
        """
        is_temporal = (field.ndim == 5)
        
        if is_temporal:
            B, T, C, H, W = field.shape
            # Reshape to [B*T, C, H, W] for batch processing
            field_reshaped = field.reshape(B * T, C, H, W)
        else:
            B, C, H, W = field.shape
            field_reshaped = field
        
        # 4-neighbor Laplacian stencil: ∇²f = f_left + f_right + f_top + f_bottom - 4*f_center
        laplacian = (
            field_reshaped[:, :, :-2, 1:-1] +  # Top neighbor (i-1,j)
            field_reshaped[:, :, 2:, 1:-1] +   # Bottom neighbor (i+1,j)
            field_reshaped[:, :, 1:-1, :-2] +  # Left neighbor (i,j-1)
            field_reshaped[:, :, 1:-1, 2:] -   # Right neighbor (i,j+1)
            4 * field_reshaped[:, :, 1:-1, 1:-1]  # Center point (i,j)
        )
        
        # Pad back to original spatial dimensions
        laplacian_padded = F.pad(laplacian, (1, 1, 1, 1), mode='replicate')
        
        if is_temporal:
            # Reshape back to [B, T, C, H, W]
            laplacian_padded = laplacian_padded.reshape(B, T, C, H, W)
        
        return laplacian_padded
    
    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian."""
        return self.apply(field)


class StructureConsistencyLoss:
    """
    Structure consistency loss term for IC optimization.
    
    Measures consistency between structural properties of observations and forecasts:
        J_struct = ||S(ψ_obs) - S(G(ψ))||²_mask
    
    where:
    - S(·) is a structural operator (gradient, Laplacian, etc.)
    - ψ_obs is the observed state
    - G(ψ) is the forecast from the model
    - ||·||²_mask is MSE with observation masking
    
    Physical Motivation:
    - Gradients enforce that local slopes match observations
    - Laplacian enforces curvature/smoothness consistency
    - These structural constraints improve physical realism
    """
    
    def __init__(
        self,
        operator_type: str = "gradient",
        obs_mask: torch.Tensor = None,
        device: str = "cuda"
    ):
        """
        Initialize structure consistency loss.
        
        Args:
            operator_type: Type of structural operator ('gradient' or 'laplacian')
            obs_mask: Observation mask [T, C, H, W] (1 where obs exist, 0 elsewhere)
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
        self.operator_type = operator_type
        self.obs_mask = obs_mask
        
        # Initialize operator
        if operator_type == "gradient":
            self.operator = GradientOperator(device=device)
        elif operator_type == "laplacian":
            self.operator = LaplacianOperator(device=device)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
        
        logger.info(f"StructureConsistencyLoss initialized with {operator_type} operator")
    
    def compute_structure_mse(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute structure consistency loss with structural operator.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            
        Returns:
            struct_mse_per_step: Structure consistency MSE per timestep [B, T]
            weighted_loss: Weighted loss per batch [B]
            struct_mse_per_var: Structure MSE per variable [B, C] for logging
        """
        B, T, C, H, W = predictions.shape
        
        # Apply structural operator to both predictions and targets
        S_pred = self.operator(predictions)  # [B, T, C', H, W]
        S_target = self.operator(targets)     # [B, T, C', H, W]
        
        # Compute MSE between structural fields
        struct_se = (S_pred - S_target) ** 2  # Squared error
        
        # If observation mask exists, apply it (need to handle channel dimension change)
        if self.obs_mask is not None:
            # For gradient operator, obs_mask needs to be duplicated for 2*C channels
            if self.operator_type == "gradient":
                obs_mask_expanded = torch.cat([self.obs_mask, self.obs_mask], dim=1)
            else:
                obs_mask_expanded = self.obs_mask
            
            # Expand batch dimension
            obs_mask_expanded = obs_mask_expanded.unsqueeze(0)
            
            # Apply mask
            struct_se_masked = struct_se * obs_mask_expanded
            
            # Normalize by number of valid observations
            obs_count = obs_mask_expanded.sum(dim=(-2, -1)).clamp(min=1e-10)
            struct_mse_all_channels = struct_se_masked.sum(dim=(-2, -1)) / obs_count
        else:
            # No masking, compute MSE over spatial dimensions
            struct_mse_all_channels = struct_se.mean(dim=(-2, -1))  # [B, T, C']
        
        # Aggregate over time for per-variable logging: [B, C'] -> [B, C]
        # For gradient operator: C' = 2*C, so combine (x_grad, y_grad) back to original C variables
        if self.operator_type == "gradient":
            # Reshape [B, T, 2*C] -> [B, T, C, 2] to separate x and y gradients
            struct_mse_reshaped = struct_mse_all_channels.reshape(B, T, C, 2)
            # Combine x and y gradients: average or RMS combination
            # Using RMS: sqrt(grad_x^2 + grad_y^2) provides physical magnitude
            struct_mse_combined = torch.sqrt((struct_mse_reshaped ** 2).sum(dim=-1))  # [B, T, C]
        else:
            # Laplacian: already per-channel, no reshaping needed
            struct_mse_combined = struct_mse_all_channels  # [B, T, C]
        
        # Per-timestep loss: [B, T]
        struct_mse_per_step = struct_mse_combined.mean(dim=-1)
        
        # Per-variable loss (aggregated over time): [B, C]
        struct_mse_per_var = struct_mse_combined.mean(dim=1)
        
        # Total loss per batch (unweighted): [B]
        unweighted_loss = struct_mse_per_step.sum(dim=1)
        
        return struct_mse_per_step, struct_mse_per_var, unweighted_loss
    
    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted structure consistency loss.
        
        Args:
            predictions: Model predictions [B, T, C, H, W]
            targets: Target observations [B, T, C, H, W]
            weight: Weight for this loss term (default: 1.0)
            
        Returns:
            loss: Scalar weighted loss
            details: Dictionary with detailed metrics including per-variable losses
        """
        struct_mse_per_step, struct_mse_per_var, unweighted_loss = self.compute_structure_mse(
            predictions, targets
        )
        
        # Apply weight and take mean over batch
        loss = (weight * unweighted_loss).mean()
        
        details = {
            'struct_mse_per_step': struct_mse_per_step,  # [B, T]
            'unweighted_loss': unweighted_loss,  # [B]
            'struct_mse_per_var': struct_mse_per_var,  # [B, C] - per-variable for logging
            'loss': loss.item(),
            'operator_type': self.operator_type,
            'weight': weight,
        }
        
        return loss, details
