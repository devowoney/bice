"""
Gradient filtering for IC optimization.

Provides:
    - GradientFilter: Apply spatial filtering to gradients
    - ScheduledPooling: Multi-resolution gradient filtering (coarse-to-fine)
    
Gradient filtering helps optimization by:
- Smoothing gradients spatially (reduce high-frequency noise)
- Multi-scale optimization (start coarse, refine fine)
- Avoiding local minima
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GradientFilter:
    """
    Apply spatial filtering to gradients during IC optimization.
    
    Supports:
    - 'none': No filtering (raw gradients)
    - 'weight': Weight by variance (emphasize high-variance regions)
    - 'delta': Filter by IC perturbation magnitude
    - 'pooling': Average pooling (spatial smoothing)
    - 'pooling+weight': Pooling + variance weighting
    
    Scheduled pooling: Coarse-to-fine optimization (multigrid)
    - Start with large kernel (global optimization)
    - Gradually reduce to small kernel (local refinement)
    - Avoids local minima early in optimization
    """
    
    def __init__(
        self,
        filter_type: str = "pooling",
        device: str = "cuda"
    ):
        """
        Initialize gradient filter.
        
        Args:
            filter_type: Filtering strategy ('none', 'weight', 'delta', 'pooling', 'pooling+weight')
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.filter_type = filter_type
        self.device = device
        
        # Pooling kernel is set dynamically (for scheduled pooling)
        self.current_kernel = None
    
    def set_kernel_size(self, kernel_size: int):
        """
        Set pooling kernel size (for scheduled pooling).
        
        Args:
            kernel_size: Kernel size for average pooling
        """
        self.current_kernel = kernel_size
    
    def apply_pooling(
        self,
        gradients: torch.Tensor,
        kernel_size: int
    ) -> torch.Tensor:
        """
        Apply average pooling to gradients.
        
        Average pooling spatially smooths gradients, removing high-frequency noise.
        Larger kernels = more smoothing (coarse optimization).
        Smaller kernels = less smoothing (fine optimization).
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            kernel_size: Pooling kernel size (must divide H and W evenly)
            
        Returns:
            pooled_gradients: Smoothed gradients [B, T, C, H, W]
        """
        if kernel_size == 1:
            # No pooling needed
            return gradients
        
        batch_size, time_steps, channels, height, width = gradients.shape
        
        # Reshape to [B*T*C, 1, H, W] for 2D pooling
        gradients_reshaped = gradients.reshape(batch_size * time_steps * channels, 1, height, width)
        
        # Apply average pooling
        # stride=kernel_size for non-overlapping pooling
        pooled = F.avg_pool2d(
            gradients_reshaped,
            kernel_size=kernel_size,
            stride=kernel_size
        )
        
        # Upsample back to original resolution
        # Use nearest-neighbor to preserve pooled values
        upsampled = F.interpolate(
            pooled,
            size=(height, width),
            mode='nearest'
        )
        
        # Reshape back to [B, T, C, H, W]
        pooled_gradients = upsampled.reshape(batch_size, time_steps, channels, height, width)
        
        return pooled_gradients
    
    def apply_weight_filter(
        self,
        gradients: torch.Tensor,
        variance_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Weight gradients by variance map.
        
        High-variance regions get more weight (more dynamic, more important).
        Low-variance regions get less weight (less signal).
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            variance_map: Variance map [H, W] or [C, H, W]
            
        Returns:
            weighted_gradients: Variance-weighted gradients [B, T, C, H, W]
        """
        # Normalize variance map to [0, 1] range
        var_min = variance_map.min()
        var_max = variance_map.max()
        variance_norm = (variance_map - var_min) / (var_max - var_min + 1e-10)
        
        # Expand variance map to match gradient shape
        if variance_norm.dim() == 2:
            # [H, W] → [1, 1, 1, H, W]
            variance_norm = variance_norm.view(1, 1, 1, *variance_norm.shape)
        elif variance_norm.dim() == 3:
            # [C, H, W] → [1, 1, C, H, W]
            variance_norm = variance_norm.view(1, 1, *variance_norm.shape)
        
        # Apply weighting
        weighted_gradients = gradients * variance_norm
        
        return weighted_gradients
    
    def apply_delta_filter(
        self,
        gradients: torch.Tensor,
        x0_current: torch.Tensor,
        x0_reference: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter gradients by IC perturbation magnitude.
        
        Emphasize gradients where IC has changed significantly.
        This focuses updates on regions that are actively being optimized.
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            x0_current: Current IC [B, T, C, H, W]
            x0_reference: Reference IC [B, T, C, H, W]
            
        Returns:
            filtered_gradients: Delta-weighted gradients [B, T, C, H, W]
        """
        # Compute IC perturbation magnitude
        delta = torch.abs(x0_current - x0_reference)
        
        # Normalize delta to [0, 1] per channel
        delta_min = delta.min(dim=(-2, -1), keepdim=True)[0]
        delta_max = delta.max(dim=(-2, -1), keepdim=True)[0]
        delta_norm = (delta - delta_min) / (delta_max - delta_min + 1e-10)
        
        # Apply weighting
        filtered_gradients = gradients * delta_norm
        
        return filtered_gradients
    
    def __call__(
        self,
        gradients: torch.Tensor,
        x0_current: torch.Tensor = None,
        x0_reference: torch.Tensor = None,
        variance_map: torch.Tensor = None,
        kernel_size: int = None
    ) -> torch.Tensor:
        """
        Apply gradient filtering.
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            x0_current: Current IC (for delta filtering)
            x0_reference: Reference IC (for delta filtering)
            variance_map: Variance map (for weight filtering)
            kernel_size: Pooling kernel size (overrides self.current_kernel if provided)
            
        Returns:
            filtered_gradients: Filtered gradients [B, T, C, H, W]
        """
        if self.filter_type == "none":
            # No filtering - return raw gradients
            return gradients
        
        elif self.filter_type == "pooling":
            # Average pooling only
            if kernel_size is None:
                kernel_size = self.current_kernel if self.current_kernel is not None else 1
            
            return self.apply_pooling(gradients, kernel_size)
        
        elif self.filter_type == "weight":
            # Variance weighting only
            if variance_map is None:
                raise ValueError("variance_map required for 'weight' filtering")
            
            return self.apply_weight_filter(gradients, variance_map)
        
        elif self.filter_type == "delta":
            # Delta filtering only
            if x0_current is None or x0_reference is None:
                raise ValueError("x0_current and x0_reference required for 'delta' filtering")
            
            return self.apply_delta_filter(gradients, x0_current, x0_reference)
        
        elif self.filter_type == "pooling+weight":
            # Pooling + variance weighting
            if variance_map is None:
                raise ValueError("variance_map required for 'pooling+weight' filtering")
            
            if kernel_size is None:
                kernel_size = self.current_kernel if self.current_kernel is not None else 1
            
            # Apply pooling first
            pooled = self.apply_pooling(gradients, kernel_size)
            
            # Then apply variance weighting
            filtered = self.apply_weight_filter(pooled, variance_map)
            
            return filtered
        
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")


class ScheduledPooling:
    """
    Manage scheduled pooling (multigrid optimization).
    
    Supports:
    - 'step': Change kernel at specific iterations (recommended)
    - 'linear': Linear interpolation from initial to final kernel
    - 'exponential': Exponential decay from initial to final kernel
    
    Coarse-to-fine optimization:
    1. Start with large kernel (coarse grid, global optimization)
    2. Gradually reduce kernel (fine grid, local refinement)
    3. Avoids local minima early, converges to fine details later
    """
    
    def __init__(
        self,
        schedule_type: str,
        num_iterations: int,
        initial_kernel: int = 24,
        final_kernel: int = 1,
        schedule_steps: list = None,
        kernel_sizes: list = None
    ):
        """
        Initialize scheduled pooling.
        
        Args:
            schedule_type: Schedule type ('step', 'linear', 'exponential')
            num_iterations: Total number of optimization iterations
            initial_kernel: Starting kernel size (coarse)
            final_kernel: Ending kernel size (fine)
            schedule_steps: Iteration numbers to change kernel (for 'step' schedule)
            kernel_sizes: Kernel sizes at each step (for 'step' schedule)
        """
        self.schedule_type = schedule_type
        self.num_iterations = num_iterations
        self.initial_kernel = initial_kernel
        self.final_kernel = final_kernel
        self.schedule_steps = schedule_steps
        self.kernel_sizes = kernel_sizes
        
        # Validate step schedule
        if schedule_type == 'step':
            if schedule_steps is None or kernel_sizes is None:
                raise ValueError("schedule_steps and kernel_sizes required for 'step' schedule")
            
            if len(kernel_sizes) != len(schedule_steps) + 1:
                raise ValueError(f"kernel_sizes must have {len(schedule_steps) + 1} elements")
    
    def get_kernel_size(self, iteration: int) -> int:
        """
        Get pooling kernel size for current iteration.
        
        Args:
            iteration: Current iteration number (0-indexed)
            
        Returns:
            kernel_size: Pooling kernel size for this iteration
        """
        if self.schedule_type == 'step':
            # Step schedule: discrete changes at specific iterations
            # Find which step we're in
            for idx, step_iter in enumerate(self.schedule_steps):
                if iteration < step_iter:
                    return self.kernel_sizes[idx]
            
            # After all steps, use final kernel
            return self.kernel_sizes[-1]
        
        elif self.schedule_type == 'linear':
            # Linear interpolation from initial to final
            progress = iteration / max(self.num_iterations - 1, 1)
            kernel_size = int(
                self.initial_kernel + progress * (self.final_kernel - self.initial_kernel)
            )
            
            # Ensure kernel is at least 1
            return max(kernel_size, 1)
        
        elif self.schedule_type == 'exponential':
            # Exponential decay from initial to final
            progress = iteration / max(self.num_iterations - 1, 1)
            
            # Exponential decay: k(t) = k_init * (k_final / k_init)^t
            kernel_size = int(
                self.initial_kernel * (self.final_kernel / self.initial_kernel) ** progress
            )
            
            # Ensure kernel is at least 1
            return max(kernel_size, 1)
        
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
