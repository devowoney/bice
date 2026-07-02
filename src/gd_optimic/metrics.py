"""
Metrics computation for IC optimization evaluation.

Provides:
    - MetricsComputer: Compute RMSE (global + basin-stratified) and PSD metrics
    
Follows R4 decisions:
- Per-variable RMSE (SSH, T, S, U, V)
- Basin-stratified RMSE (Gulf Stream, high/low-variance, coastal)
- 2D directional spectrum by wavenumber (PSD)
- Band-energy ratio metric (mesoscale energy / total)

Phase 1: Focus on RMSE metrics
Phase P: Add PSD band-energy ratio after initial runs
"""

import torch
import numpy as np
import xarray as xr
from typing import Dict, Tuple
from scipy import signal


class MetricsComputer:
    """
    Compute evaluation metrics for IC optimization.
    
    Implements R4 decisions:
    - RMSE: Per-variable + basin-stratified
    - PSD: 2D directional spectrum (Phase P)
    - Success threshold: Adaptive (decided after first runs)
    """
    
    def __init__(
        self,
        ocean_mask: torch.Tensor,
        device: str = "cuda"
    ):
        """
        Initialize metrics computer.
        
        Args:
            ocean_mask: Ocean mask [C, H, W] - 1=ocean, 0=land
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.ocean_mask = ocean_mask
        self.device = device
    
    def compute_rmse_per_channel(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RMSE per channel (global average over ocean pixels).
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Target truth [B, C, H, W]
            
        Returns:
            rmse_per_channel: RMSE for each channel [B, C]
        """
        # Compute squared error
        squared_error = (predictions - targets) ** 2
        
        # Apply ocean mask: only compute RMSE over ocean pixels
        # ocean_mask: [C, H, W], broadcast to [B, C, H, W]
        masked_error = squared_error * self.ocean_mask.unsqueeze(0)
        
        # Count valid ocean pixels per channel
        ocean_count = self.ocean_mask.sum(dim=(-2, -1))  # [C]
        
        # Mean squared error per channel
        mse_per_channel = masked_error.sum(dim=(-2, -1)) / (ocean_count.unsqueeze(0) + 1e-10)
        
        # Root mean squared error
        rmse_per_channel = torch.sqrt(mse_per_channel)
        
        return rmse_per_channel
    
    def compute_rmse_global(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute global RMSE per variable.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Target truth [B, C, H, W]
            
        Returns:
            rmse_dict: Dictionary with per-variable RMSE
                Keys: 'SSH', 'T', 'S', 'U', 'V'
        """
        # Compute RMSE per channel: [B, C]
        rmse_per_channel = self.compute_rmse_per_channel(predictions, targets)
        
        # Take mean over batch: [C]
        rmse_mean = rmse_per_channel.mean(dim=0)
        
        # Channel mapping: 0=SSH, 1=T, 2=S, 3=U, 4=V
        rmse_dict = {
            'SSH': float(rmse_mean[0]),
            'T': float(rmse_mean[1]),
            'S': float(rmse_mean[2]),
            'U': float(rmse_mean[3]),
            'V': float(rmse_mean[4]),
        }
        
        return rmse_dict
    
    def compute_rmse_basin_stratified(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regional_masks: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute basin-stratified RMSE.
        
        Follows R4 decision: Per-variable + basin-stratified RMSE
        Basins: Gulf Stream, high-variance, low-variance (coastal deferred to Phase P)
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Target truth [B, C, H, W]
            regional_masks: Dictionary of regional masks
                Keys: 'global', 'gulf_stream', 'high_var', 'low_var'
                Values: boolean masks [C, H, W]
            
        Returns:
            rmse_dict: Nested dictionary {region: {variable: rmse}}
        """
        rmse_dict = {}
        
        # Move predictions and targets to CPU for regional computation
        pred_cpu = predictions.detach().cpu().numpy()
        targ_cpu = targets.detach().cpu().numpy()
        
        for region_name, region_mask in regional_masks.items():
            region_rmse = {}
            
            # Convert mask to torch tensor
            region_mask_torch = torch.from_numpy(region_mask).float().to(self.device)
            
            # Apply regional mask to predictions and targets
            # region_mask: [C, H, W], broadcast to [B, C, H, W]
            pred_regional = predictions * region_mask_torch.unsqueeze(0)
            targ_regional = targets * region_mask_torch.unsqueeze(0)
            
            # Compute squared error
            squared_error = (pred_regional - targ_regional) ** 2
            
            # Count valid pixels per channel in this region
            region_count = region_mask_torch.sum(dim=(-2, -1))  # [C]
            
            # Mean squared error per channel
            mse_per_channel = squared_error.sum(dim=(-2, -1)) / (region_count.unsqueeze(0) + 1e-10)
            
            # Root mean squared error
            rmse_per_channel = torch.sqrt(mse_per_channel)
            
            # Take mean over batch
            rmse_mean = rmse_per_channel.mean(dim=0)
            
            # Store per-variable RMSE
            region_rmse = {
                'SSH': float(rmse_mean[0]),
                'T': float(rmse_mean[1]),
                'S': float(rmse_mean[2]),
                'U': float(rmse_mean[3]),
                'V': float(rmse_mean[4]),
            }
            
            rmse_dict[region_name] = region_rmse
        
        return rmse_dict
    
    def compute_ic_rmse(
        self,
        x0_current: torch.Tensor,
        x0_reference: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute RMSE between current IC and reference IC.
        
        Tracks how much the initial condition has changed during optimization.
        
        Args:
            x0_current: Current IC [B, T, C, H, W]
            x0_reference: Reference IC [B, T, C, H, W]
            
        Returns:
            ic_rmse_dict: Dictionary with per-variable IC RMSE
        """
        # Use last timestep: [B, C, H, W]
        x0_current_last = x0_current[:, -1, :, :, :]
        x0_reference_last = x0_reference[:, -1, :, :, :]
        
        # Compute RMSE per channel
        rmse_per_channel = self.compute_rmse_per_channel(x0_current_last, x0_reference_last)
        
        # Take mean over batch
        rmse_mean = rmse_per_channel.mean(dim=0)
        
        # Store per-variable IC RMSE
        ic_rmse_dict = {
            'SSH': float(rmse_mean[0]),
            'T': float(rmse_mean[1]),
            'S': float(rmse_mean[2]),
            'U': float(rmse_mean[3]),
            'V': float(rmse_mean[4]),
        }
        
        return ic_rmse_dict
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        x0_current: torch.Tensor,
        x0_reference: torch.Tensor,
        regional_masks: Dict[str, np.ndarray] = None
    ) -> Dict:
        """
        Compute all metrics for a single optimization iteration.
        
        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Target truth [B, C, H, W]
            x0_current: Current IC [B, T, C, H, W]
            x0_reference: Reference IC [B, T, C, H, W]
            regional_masks: Regional masks (optional, for basin-stratified RMSE)
            
        Returns:
            metrics_dict: Dictionary with all metrics
        """
        metrics_dict = {}
        
        # Global RMSE
        metrics_dict['rmse_global'] = self.compute_rmse_global(predictions, targets)
        
        # Basin-stratified RMSE (if regional masks provided)
        if regional_masks is not None:
            metrics_dict['rmse_basin'] = self.compute_rmse_basin_stratified(
                predictions, targets, regional_masks
            )
        
        # IC RMSE
        metrics_dict['ic_rmse'] = self.compute_ic_rmse(x0_current, x0_reference)
        
        return metrics_dict


class PSDComputer:
    """
    Compute Power Spectral Density (PSD) metrics.
    
    Implements R4 decision: 2D directional spectrum by wavenumber
    Phase P: Add band-energy ratio metric after initial runs
    
    PSD analysis reveals:
    - How well forecast captures mesoscale features (50-500 km)
    - Energy distribution across wavenumber bands
    - Success metric: band-energy ratio (mesoscale energy / total)
    """
    
    def __init__(
        self,
        grid_resolution_deg: float = 0.25,
        n_bins: int = 30
    ):
        """
        Initialize PSD computer.
        
        Args:
            grid_resolution_deg: Grid resolution in degrees (0.25° for GLORYS12)
            n_bins: Number of logarithmic wavenumber bins
        """
        self.grid_resolution_deg = grid_resolution_deg
        self.n_bins = n_bins
        
        # Nyquist frequency: f_N = 1/(2*dx)
        self.f_nyquist = 1.0 / (2.0 * grid_resolution_deg)
    
    def compute_2d_psd(
        self,
        field: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D Power Spectral Density (isotropic).
        
        Args:
            field: 2D field [H, W]
            mask: Ocean mask [H, W] (True=ocean, False=land)
            
        Returns:
            k_radial: Radial wavenumber bins [n_bins]
            psd_radial: PSD values [n_bins]
        """
        # Zero out land pixels
        field_masked = field.copy()
        field_masked[~mask] = 0.0
        
        # Compute 2D FFT
        fft_2d = np.fft.fft2(field_masked)
        
        # Power spectrum: |F(k)|^2
        power_2d = np.abs(fft_2d) ** 2
        
        # Shift zero frequency to center
        power_2d_shifted = np.fft.fftshift(power_2d)
        
        # Create wavenumber grid
        ny, nx = field.shape
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=self.grid_resolution_deg))
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=self.grid_resolution_deg))
        
        # 2D wavenumber magnitude grid
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_radial_grid = np.sqrt(kx_grid**2 + ky_grid**2)
        
        # Flatten for binning
        k_flat = k_radial_grid.flatten()
        p_flat = power_2d_shifted.flatten()
        
        # Logarithmic wavenumber bins
        k_min = k_flat[k_flat > 0].min()
        k_max = k_flat.max()
        
        if k_max <= k_min:
            return np.array([]), np.array([])
        
        bins = np.logspace(np.log10(k_min), np.log10(k_max), self.n_bins + 1)
        which_bin = np.digitize(k_flat, bins) - 1
        
        psd_binned = np.full(self.n_bins, np.nan, dtype=np.float64)
        k_binned = np.sqrt(bins[:-1] * bins[1:])  # Geometric center for log bins
        
        for b in range(self.n_bins):
            in_bin = which_bin == b
            if np.any(in_bin):
                psd_binned[b] = p_flat[in_bin].mean()
        
        # Keep only valid bins
        valid = np.isfinite(psd_binned) & (psd_binned > 0) & (k_binned > 0)
        
        return k_binned[valid], psd_binned[valid]
    
    def compute_band_energy_ratio(
        self,
        k_radial: np.ndarray,
        psd_radial: np.ndarray,
        mesoscale_band: Tuple[float, float] = (1.0/500.0, 1.0/50.0)
    ) -> float:
        """
        Compute band-energy ratio: mesoscale energy / total energy.
        
        Follows R4 decision: Band-energy ratio metric
        Mesoscale band: 50-500 km wavelength (1/500 to 1/50 cycle/km)
        
        Args:
            k_radial: Radial wavenumber bins [n_bins]
            psd_radial: PSD values [n_bins]
            mesoscale_band: Wavenumber band for mesoscale features (k_min, k_max)
            
        Returns:
            band_energy_ratio: Ratio of mesoscale energy to total energy
        """
        # Tiny local trapezoidal integrator to avoid np.trapz compatibility issues
        def _trapz(y: np.ndarray, x: np.ndarray) -> float:
            if x.size < 2 or y.size < 2:
                return 0.0
            dx = x[1:] - x[:-1]
            return float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))

        # Total energy: integral of PSD over all wavenumbers
        total_energy = _trapz(psd_radial, k_radial)
        
        # Mesoscale energy: integral of PSD over mesoscale band
        k_min, k_max = mesoscale_band
        in_band = (k_radial >= k_min) & (k_radial <= k_max)
        
        if not np.any(in_band):
            return 0.0
        
        mesoscale_energy = _trapz(psd_radial[in_band], k_radial[in_band])
        
        # Band-energy ratio
        band_ratio = mesoscale_energy / (total_energy + 1e-10)
        
        return float(band_ratio)
