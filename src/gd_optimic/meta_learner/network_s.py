"""
Neural network S(θ, x) that learns to predict IC updates.

Architecture: UNet with skip connections for multi-scale feature extraction.

Input: Current IC state [B, C, H, W] or [B, T, C, H, W]
Output: Update direction (full step, no learning rate) [B, C, H, W] or [B, T, C, H, W]
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class NetworkSConfig:
    """Configuration for S(θ, x) UNet network."""
    
    input_channels: int = 5  # SSH, T, S, U, V
    output_channels: int = 5  # Same as input (spatial update)
    spatial_height: int = 330
    spatial_width: int = 360
    
    # UNet architecture
    base_channels: int = 32  # Base channels for encoder/decoder
    num_groups: int = 8  # Groups for GroupNorm
    
    # Temporal dimension (set to 1 for single-step, >1 for trajectory)
    temporal_steps: int = 1
    
    # Output scaling
    output_scale: float = 0.01  # Damping factor to prevent explosive updates


class DoubleConv(nn.Module):
    """Double convolution block with GroupNorm and ReLU."""
    
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetMetaGrad2D(nn.Module):
    """
    UNet-based update predictor: S(θ, x) → δx
    
    Predicts the **full IC update** for one optimization step.
    Multi-scale feature extraction via encoder-decoder with skip connections.
    
    Supports both single-step [B, C, H, W] and temporal [B, T, C, H, W] inputs.
    """
    
    def __init__(self, cfg: NetworkSConfig):
        super().__init__()
        self.cfg = cfg
        self.output_scale = cfg.output_scale
        
        # Input channels accounting for temporal dimension
        in_ch = cfg.temporal_steps * cfg.input_channels
        out_ch = cfg.temporal_steps * cfg.output_channels
        b = cfg.base_channels
        groups = cfg.num_groups
        
        # Encoder
        self.enc1 = DoubleConv(in_ch, b, groups=groups)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(b, b * 2, groups=groups)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(b * 2, b * 4, groups=groups)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(b * 4, b * 8, groups=groups)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(b * 8, b * 16, groups=groups)
        
        # Decoder with skip connections
        # up4 outputs b*8; concat with e3(b*4) -> b*12
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8, groups=groups)
        
        # up3 outputs b*4; concat with e2(b*2) -> b*6
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4, groups=groups)
        
        # up2 outputs b*2; concat with e1(b) -> b*3
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2, groups=groups)
        
        # up1 outputs b; concat with x_input (in_ch channels) -> b + in_ch
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b, groups=groups)
        
        # Output head
        self.out_conv = nn.Conv2d(b, out_ch, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict IC update.
        
        Args:
            x: [B, C, H, W] current IC state (single-step)
               or [B, T, C, H, W] trajectory (multi-step)
        
        Returns:
            update: [B, C, H, W] or [B, T, C, H, W] predicted update
        """
        # Handle temporal dimension if present
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)
            temporal_reshape = True
        else:
            B, C, H, W = x.shape
            temporal_reshape = False
        
        # Encoder with skip connections
        e1 = self.enc1(x)              # (B, b, H, W)
        p1 = self.pool1(e1)            # (B, b, H/2, W/2)
        
        # Apply checkpointing to large encoder blocks to save memory during higher-order grads
        e2 = checkpoint(self.enc2, p1)             # (B, 2b, H/2, W/2)
        p2 = self.pool2(e2)            # (B, 2b, H/4, W/4)
        
        e3 = checkpoint(self.enc3, p2)             # (B, 4b, H/4, W/4)
        p3 = self.pool3(e3)            # (B, 4b, H/8, W/8)
        
        e4 = checkpoint(self.enc4, p3)             # (B, 8b, H/8, W/8)
        p4 = self.pool4(e4)            # (B, 8b, H/16, W/16)
        
        # Bottleneck (checkpoint)
        b_out = checkpoint(self.bottleneck, p4)    # (B, 16b, H/8, W/8)
        
        b = self.cfg.base_channels
        
        # Decoder with skip connections (channels must match concat dims)
        d4 = self.up4(b_out)           # (B, 8b, H/4, W/4)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 16b, H/4, W/4)
        d4 = checkpoint(self.dec4, d4)             # (B, 8b, H/4, W/4)
        
        d3 = self.up3(d4)              # (B, 4b, H/2, W/2)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 8b, H/2, W/2)
        d3 = checkpoint(self.dec3, d3)             # (B, 4b, H/2, W/2)
        
        d2 = self.up2(d3)              # (B, 2b, H, W)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 4b, H, W)
        d2 = checkpoint(self.dec2, d2)             # (B, 2b, H, W)
        
        d1 = self.up1(d2)              # (B, b, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 2b, H, W)
        d1 = checkpoint(self.dec1, d1)             # (B, b, H, W)
        
        # Output head
        out = self.out_conv(d1)        # (B, out_ch, H, W)
        
        # Reshape back to temporal dimension if needed
        if temporal_reshape:
            out = out.reshape(B, self.cfg.temporal_steps, self.cfg.output_channels, H, W)
        
        # Scale to prevent explosive updates
        out = self.output_scale * torch.tanh(out / self.output_scale)
        
        return out
    
    def get_parameter_count(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
