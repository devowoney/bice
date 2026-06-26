"""
Utility classes for initial condition optimization.

Provides:
    - MaskBuilder: Create ocean/land masks and observation masks
    - ForwardModel: Wrapper for multi-step rollout with gradient checkpointing
    - Normalizers: Load normalization/denormalization functions
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import sys
import logging

logger = logging.getLogger(__name__)

# Import glonet utilities (assumes they are in sys.path or installed)
# Note: These imports match the reference notebook's structure
base = Path(__file__).resolve().parent
lib_dir = (base / '..' / '..' / 'model' / 'glonet').resolve()
sys.path.insert(0, str(lib_dir))
from utility import get_normalizer1, get_denormalizer1
from modelp2 import Glonet


class MaskBuilder:
    """
    Build ocean/land masks and observation masks from dataset.
    
    Masks are used to:
    - Zero out gradients on land pixels (ocean_mask)
    - Apply observation operators only where data exists (obs_mask)
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize mask builder.
        
        Args:
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.device = device
        
    def build_ocean_mask(self, sample_data: np.ndarray) -> torch.Tensor:
        """
        Create ocean mask from sample data.

        Ocean mask is 1 where data is valid (ocean), 0 where NaN (land).

        This dataset may contain a stacked state with many channels (e.g. 85).
        Downselect to the 5 core channels used by the model: [SSH, T, S, U, V].

        Args:
            sample_data: numpy array of shape [C, H, W] or [H, W] where NaN indicates land

        Returns:
            ocean_mask: torch.Tensor of shape [5, H, W], 1=ocean, 0=land
        """
        # Handle different input shapes
        sample_arr = np.asarray(sample_data)

        if sample_arr.ndim == 2:
            # Single-channel H x W -> expand to 5 channels
            land_mask = np.isnan(sample_arr).astype(np.float32)
            ocean_mask = 1.0 - land_mask
            ocean_mask = np.stack([ocean_mask.copy() for _ in range(5)], axis=0)
        elif sample_arr.ndim == 3:
            C = sample_arr.shape[0]
            if C >= 5:
                # Use first 5 channels corresponding to core variables
                land_mask = np.isnan(sample_arr[:5, :, :]).astype(np.float32)
                ocean_mask = 1.0 - land_mask
            else:
                raise ValueError(f"Unexpected channel count in sample_data: {C}. Expected at least 5 channels.")
        else:
            raise ValueError(f"Unsupported sample_data shape: {sample_arr.shape}")

        # Convert to torch and move to device
        ocean_mask_tensor = torch.from_numpy(ocean_mask.copy()).float().to(self.device)

        return ocean_mask_tensor
    
    def build_obs_mask(
        self,
        ocean_mask: torch.Tensor,
        obs_length: int,
        ssh_nanmask: np.ndarray = None,
        sst_nanmask: np.ndarray = None,
        obs_mode: str = "full"
    ) -> torch.Tensor:
        """
        Create observation mask indicating where observations are available.
        
        Args:
            ocean_mask: torch.Tensor [C, H, W] - base ocean mask
            obs_length: int - number of time steps in observation window
            ssh_nanmask: np.ndarray [T, H, W] - mask for SSH observations (1=observed, NaN=not observed)
            sst_nanmask: np.ndarray [T, H, W] - mask for SST observations
            obs_mode: str - observation mode ('full', 'simulated', 'real')
                - 'full': all ocean pixels observed (idealized twin/ceiling check)
                - 'simulated': realistic observation coverage with GLORYS12 truth (OSSE)
                - 'real': real satellite observations (OSE)
        
        Returns:
            obs_mask: torch.Tensor [T, C, H, W] - observation mask
                      1 where observations available, 0 elsewhere
        """
        # Determine channel dimension for observations (we expect 5 core variables)
        # If ocean_mask has more channels (e.g., full stacked state with 85 channels),
        # select the first 5 channels which correspond to [SSH, T, S, U, V].
        if ocean_mask.dim() == 3 and ocean_mask.shape[0] >= 5:
            base_ocean_mask = ocean_mask[:5, :, :]
        elif ocean_mask.dim() == 2:
            # If ocean_mask provided without channel dim (H, W), expand to 5 channels
            base_ocean_mask = ocean_mask.unsqueeze(0).repeat(5, 1, 1)
        else:
            base_ocean_mask = ocean_mask

        # Start with ocean mask repeated over time
        # Shape: [T, C=5, H, W]
        obs_mask = base_ocean_mask.unsqueeze(0).repeat(obs_length, 1, 1, 1).clone()
        
        if obs_mode == "full":
            # All ocean pixels are observed for all variables
            # No modification needed - obs_mask already equals ocean_mask repeated
            pass
        
        elif obs_mode in ["simulated", "real"]:
            # Apply realistic observation coverage
            # Channel 0: SSH - along-track altimetry (sparse)
            # Channel 1: SST - gridded satellite (dense but with gaps)
            # Channels 2-4: SSS, U, V - not directly observed (set to 0)
            
            if ssh_nanmask is not None:
                # SSH observation mask: 1 where observed, 0 where not
                ssh_mask_torch = torch.from_numpy(
                    np.nan_to_num(ssh_nanmask, nan=0.0)
                ).float().to(self.device)
                obs_mask[:, 0, :, :] = ssh_mask_torch
            else:
                # If no SSH mask provided, set SSH to not observed
                obs_mask[:, 0, :, :] = 0.0
            
            if sst_nanmask is not None:
                # SST observation mask: 1 where observed, 0 where not
                sst_mask_torch = torch.from_numpy(
                    np.nan_to_num(sst_nanmask, nan=0.0)
                ).float().to(self.device)
                obs_mask[:, 1, :, :] = sst_mask_torch
            else:
                # If no SST mask provided, set SST to not observed
                obs_mask[:, 1, :, :] = 0.0
            
            # SSS, U, V are not directly observed in satellite data
            obs_mask[:, 2:5, :, :] = 0.0
        
        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}. Must be 'full', 'simulated', or 'real'")
        
        return obs_mask


class ForwardModel:
    """
    Wrapper for multi-step forward rollout using glonet with gradient checkpointing.
    
    This class encapsulates:
    - Model loading and device placement
    - Normalization/denormalization
    - Multi-step autoregressive rollout
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        model_path: str,
        normalizer_path: str,
        device: str = "cuda",
        use_gradient_checkpointing: bool = True
    ):
        """
        Initialize forward model.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            normalizer_path: Path to normalizer statistics
            device: PyTorch device ('cuda' or 'cpu')
            use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
        """
        self.device = device
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Load normalizers
        self.normalizer = get_normalizer1(normalizer_path)
        self.denormalizer = get_denormalizer1(normalizer_path)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to eval mode (frozen model, no dropout/batchnorm randomness)
        
        # Freeze all model parameters (frozen-model invariant A1)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            model: Loaded PyTorch model
        """
        # Import the gradient checkpointing wrapper from model directory
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model" / "glonet"))
        from optimIC_GD_glonetLit import GlonetGradientCheckpointing
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with gradient checkpointing if enabled
        if self.use_gradient_checkpointing:
            model = GlonetGradientCheckpointing(shape_in=(2, 5, 672, 1440))
        else:
            # Fallback: try importing Glonet directly
            try:
                from modelp2 import Glonet
                model = Glonet(shape_in=(2, 5, 672, 1440))
            except ImportError:
                logger.warning("Falling back to GlonetGradientCheckpointing (Glonet import failed)")
                model = GlonetGradientCheckpointing(shape_in=(2, 5, 672, 1440))
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def forward(
        self,
        x0: torch.Tensor,
        num_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step forward rollout with autoregressive prediction.
        
        Model output [B, T=2, C, H, W] goes directly back as input for next step.
        Only the last timestep [:, -1, :, :, :] is extracted for the accumulated predictions.
        
        Args:
            x0: Initial condition [B, T=2, C, H, W]
            num_steps: Number of forecast steps to generate
            
        Returns:
            x0_normalized: Normalized initial condition [B, T=2, C, H, W]
            y_hat_steps: Stacked predictions [B, num_steps, C, H, W] (denormalized, from last timestep)
        """
        # Normalize initial condition
        x0_normalized = self.normalizer(x0)
        
        # First forward pass
        if self.use_gradient_checkpointing:
            y_hat = torch.utils.checkpoint.checkpoint(
                self.model,
                x0_normalized,
                use_reentrant=False
            )
        else:
            y_hat = self.model(x0_normalized)
        
        # y_hat is [B, T=2, C, H, W], extract last timestep and denormalize
        y_hat_steps = [self.denormalizer(y_hat[:, -1, :, :, :])]
        
        # Autoregressive forecasting for remaining steps
        for step_idx in range(num_steps - 1):
            # Feed model output directly back as input for next step
            if self.use_gradient_checkpointing:
                y_hat = torch.utils.checkpoint.checkpoint(
                    self.model,
                    y_hat,
                    use_reentrant=False
                )
            else:
                y_hat = self.model(y_hat)
            
            # Extract last timestep and denormalize
            y_hat_steps.append(self.denormalizer(y_hat[:, -1, :, :, :]))
        
        # Stack all predictions: [B, num_steps, C, H, W]
        y_hat_steps = torch.stack(y_hat_steps, dim=1)
        
        return x0_normalized, y_hat_steps


class NormalizerLoader:
    """
    Load and manage normalization/denormalization functions.
    
    This is a lightweight wrapper around the existing utility functions.
    """
    
    @staticmethod
    def load_normalizers(model_location: str) -> Tuple:
        """
        Load normalizer and denormalizer functions.
        
        Args:
            model_location: Path to directory containing normalization statistics
            
        Returns:
            normalizer: Function to normalize data
            denormalizer: Function to denormalize data
        """
        normalizer = get_normalizer1(model_location)
        denormalizer = get_denormalizer1(model_location)
        
        return normalizer, denormalizer
