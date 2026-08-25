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
import math


class GaussianLowPassFilter:
    """
    Apply Gaussian low-pass filtering to spatially smooth gradients.
    
    Replaces or complements average pooling with frequency-domain smoothing.
    Better preserves spatial structure while removing high-frequency noise.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize Gaussian low-pass filter.
        
        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
    
    def create_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float = None
    ) -> torch.Tensor:
        """
        Create 2D Gaussian kernel for low-pass filtering.
        
        Args:
            kernel_size: Size of the Gaussian kernel (odd number recommended)
            sigma: Standard deviation of Gaussian (if None, default to kernel_size/6)
            
        Returns:
            kernel: 2D Gaussian kernel [kernel_size, kernel_size]
        """
        if sigma is None:
            # Default: sigma = kernel_size / 6 gives good smoothing
            sigma = kernel_size / 6.0
        
        # Create coordinate grids
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        y = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
        
        # Center coordinates
        x = x - (kernel_size - 1) / 2.0
        y = y - (kernel_size - 1) / 2.0
        
        # Create 2D grid
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Compute Gaussian
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Normalize to sum to 1
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def apply(
        self,
        gradients: torch.Tensor,
        kernel_size: int
    ) -> torch.Tensor:
        """
        Apply Gaussian low-pass filter to gradients.
        
        Handles 5D temporal tensors: [B, T, C, H, W]
        - B: batch size
        - T: time steps
        - C: channels
        - H: latitude
        - W: longitude
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            kernel_size: Size of Gaussian kernel
            
        Returns:
            filtered_gradients: Low-pass filtered gradients [B, T, C, H, W]
        """
        if kernel_size == 1:
            # No filtering needed
            return gradients
        
        # Unpack shape [B, T, C, H, W]
        batch_size, time_steps, channels, height, width = gradients.shape
        
        # Create Gaussian kernel
        gauss_kernel = self.create_gaussian_kernel(kernel_size)
        
        # Reshape for depthwise convolution: [C, 1, kernel_size, kernel_size]
        conv_kernel = gauss_kernel.view(1, 1, kernel_size, kernel_size).repeat(
            channels, 1, 1, 1
        )
        
        # Reshape gradients to [B*T, C, H, W] for grouped convolution
        gradients_reshaped = gradients.reshape(batch_size * time_steps, channels, height, width)
        
        # Pad asymmetrically so even-sized kernels preserve the spatial shape.
        # With an even kernel, symmetric conv2d padding would add one pixel.
        padding_before = (kernel_size - 1) // 2
        padding_after = kernel_size // 2
        gradients_reshaped = F.pad(
            gradients_reshaped,
            (padding_before, padding_after, padding_before, padding_after),
            mode="constant",
            value=0,
        )

        # Apply depthwise convolution (groups=channels for per-channel filtering)
        filtered = F.conv2d(
            gradients_reshaped,
            conv_kernel,
            padding=0,
            groups=channels
        )
        
        # Reshape back to [B, T, C, H, W]
        filtered_gradients = filtered.reshape(batch_size, time_steps, channels, height, width)
        
        return filtered_gradients


class GradientFilter:
    """
    Apply spatial filtering to gradients during IC optimization.
    
    Boolean on/off control with configurable downsampling method:
    - When disabled: Raw gradients (no smoothing)
    - When enabled: Apply smoothing using selected downsampling_method
      - 'average_pooling': Non-overlapping pooling (fast, simple)
      - 'gaussian_lowpass': Gaussian low-pass filter (better structure preservation)
    
    Scheduled filtering: Coarse-to-fine optimization (multigrid)
    - Start with large kernel (global optimization)
    - Gradually reduce to small kernel (local refinement)
    - Works with both downsampling methods
    """
    
    def __init__(
        self,
        enable: bool = False,
        downsampling_method: str = "average_pooling",
        device: str = "cuda"
    ):
        """
        Initialize gradient filter.
        
        Args:
            enable: Boolean to enable/disable gradient smoothing
            downsampling_method: 'average_pooling' or 'gaussian_lowpass'
                - Used when enable=True
                - 'average_pooling' → simple averaging (fast)
                - 'gaussian_lowpass' → Gaussian smoothing (better structure)
            device: PyTorch device ('cuda' or 'cpu')
        """
        if downsampling_method not in ['average_pooling', 'gaussian_lowpass']:
            raise ValueError(f"downsampling_method must be 'average_pooling' or 'gaussian_lowpass', got {downsampling_method}")
        
        self.enable = enable
        self.downsampling_method = downsampling_method
        self.device = device
        
        # Initialize Gaussian filter (used when downsampling_method='gaussian_lowpass')
        if downsampling_method == 'gaussian_lowpass':
            self.gaussian_filter = GaussianLowPassFilter(device=device)
        else:
            self.gaussian_filter = None
        
        # Kernel size for scheduled filtering (set dynamically via set_kernel_size())
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
        kernel_size: int,
        method: str = None
    ) -> torch.Tensor:
        """
        Apply downsampling to gradients (average pooling or Gaussian low-pass).
        
        Average pooling spatially smooths gradients via non-overlapping pooling.
        Gaussian low-pass provides frequency-domain smoothing with better structure preservation.
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            kernel_size: Kernel size for downsampling
            method: Downsampling method ('average_pooling', 'gaussian_lowpass', or None for default)
            
        Returns:
            pooled_gradients: Smoothed gradients [B, T, C, H, W]
        """
        if kernel_size == 1:
            # No pooling needed
            return gradients
        
        # Use provided method or fall back to configured method
        if method is None:
            method = self.downsampling_method
        
        if method == "gaussian_lowpass":
            # Gaussian low-pass filtering
            return self.gaussian_filter.apply(gradients, kernel_size)
        
        elif method == "average_pooling":
            # Average pooling (original method)
            # Shape: [B, T, C, H, W]
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
        
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
    
    def __call__(
        self,
        gradients: torch.Tensor,
        kernel_size: int = None
    ) -> torch.Tensor:
        """
        Apply gradient filtering.
        
        Args:
            gradients: Gradient tensor [B, T, C, H, W]
            kernel_size: Filter kernel size (overrides self.current_kernel if provided)
                        Used for both pooling and Gaussian low-pass filtering
            
        Returns:
            filtered_gradients: Filtered gradients [B, T, C, H, W]
        """
        if not self.enable:
            # No filtering - return raw gradients
            return gradients
        
        # Get kernel size (for scheduled filtering)
        if kernel_size is None:
            kernel_size = self.current_kernel if self.current_kernel is not None else 1
        
        if kernel_size == 1:
            # No filtering at kernel_size=1
            return gradients
        
        # Apply smoothing using selected method
        if self.downsampling_method == "gaussian_lowpass":
            return self.gaussian_filter.apply(gradients, kernel_size)
        elif self.downsampling_method == "average_pooling":
            return self.apply_pooling(gradients, kernel_size, method="average_pooling")
        else:
            raise ValueError(f"Unknown downsampling_method: {self.downsampling_method}")


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
