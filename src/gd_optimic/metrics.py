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
        device: str = "cuda",
        stats_field: object = None,
    ):
        """
        Initialize metrics computer.

        Args:
            ocean_mask: Ocean mask [C, H, W] - 1=ocean, 0=land
            device: PyTorch device ('cuda' or 'cpu')
            stats_field: Optional mean field per channel [C, H, W].
            Examples: MDT/climatology or SSH stats. If provided, diagnostics use
            anomaly = field - stats_field.
        """
        self.ocean_mask = ocean_mask
        self.device = device
        # Mean field used to compute anomalies for diagnostics. If None, will be computed from input data when available.
        # Stored as torch.Tensor on same device with shape [C, H, W]
        self.mean_field = None
        if stats_field is not None:
            # Accept numpy/xarray/torch array; convert to torch.Tensor
            import numpy as _np
            import torch as _torch
            if isinstance(stats_field, _torch.Tensor):
                self.mean_field = stats_field.to(self.device).float()
            elif hasattr(stats_field, 'values'):
                # xarray DataArray: handle possible time/channel dimensions robustly
                arr = _np.nan_to_num(stats_field.values, nan=0.0)
                # If stats has time dimension (time, ch, lat, lon), average over time
                if arr.ndim == 4:
                    arr = arr.mean(axis=0)  # now [ch, lat, lon]

                # Determine expected spatial shape from ocean_mask if available
                C_expected = self.ocean_mask.shape[0] if self.ocean_mask is not None else None
                H_expected = self.ocean_mask.shape[1] if self.ocean_mask is not None else None
                W_expected = self.ocean_mask.shape[2] if self.ocean_mask is not None else None

                # If arr is [ch, H, W] and matches channels, accept directly
                if arr.ndim == 3 and C_expected is not None and arr.shape[1:] == (H_expected, W_expected):
                    ch_dim = arr.shape[0]
                    if ch_dim == C_expected:
                        mean_arr = arr
                    else:
                        # If stats only contains fewer channels (e.g., SSH only), broadcast into C_expected
                        mean_arr = _np.zeros((C_expected, H_expected, W_expected), dtype=arr.dtype)
                        n_fill = min(ch_dim, C_expected)
                        mean_arr[:n_fill, :, :] = arr[:n_fill, :, :]
                elif (
                    arr.ndim == 2
                    and (H_expected is not None and W_expected is not None)
                    and arr.shape == (H_expected, W_expected)
                ):
                    # Stats provided only SSH spatial field (lat, lon) -> place into SSH channel (0)
                    mean_arr = _np.zeros((C_expected, H_expected, W_expected), dtype=arr.dtype)
                    mean_arr[0, :, :] = arr
                else:
                    # Fallback: try to coerce to (C, H, W) if possible
                    if arr.ndim == 3:
                        # If arr shape matches (H, W, C) transpose
                        if (
                            C_expected is not None
                            and arr.shape[-1] == C_expected
                            and arr.shape[0] == H_expected
                            and arr.shape[1] == W_expected
                        ):
                            mean_arr = arr.transpose(2, 0, 1)
                        else:
                            # Unknown layout - attempt to reshape if sizes match
                            try:
                                mean_arr = arr.reshape((C_expected, H_expected, W_expected))
                            except Exception:
                                raise ValueError(f"Unsupported stats_field shape: {arr.shape}")
                    else:
                        raise ValueError(f"Unsupported stats_field shape: {arr.shape}")

                self.mean_field = _torch.from_numpy(_np.nan_to_num(mean_arr, nan=0.0)).float().to(self.device)
            else:
                self.mean_field = _torch.from_numpy(_np.nan_to_num(_np.array(stats_field), nan=0.0)).float().to(self.device)
    
    def compute_rmse_per_channel(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RMSE per channel (global average over ocean pixels) on anomaly fields.

        If a mean_field is set (per-channel spatial mean [C, H, W]), diagnostics are computed
        on anomalies: field - mean_field. Otherwise, diagnostics use the raw fields.

        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Target truth [B, C, H, W]

        Returns:
            rmse_per_channel: RMSE for each channel [B, C]
        """
        # If mean field is available, compute anomalies
        if self.mean_field is not None:
            preds = predictions - self.mean_field.unsqueeze(0)
            targs = targets - self.mean_field.unsqueeze(0)
        else:
            preds = predictions
            targs = targets

        # Compute squared error on anomaly or full fields
        squared_error = (preds - targs) ** 2

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
        Compute RMSE between current IC and reference IC on anomaly fields if mean_field is set.

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

        # If mean_field is available, subtract it to compute anomalies
        if self.mean_field is not None:
            x0_current_last = x0_current_last - self.mean_field.unsqueeze(0)
            x0_reference_last = x0_reference_last - self.mean_field.unsqueeze(0)

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
    
    def set_mean_from_sequences(
        self,
        input_sequence_xr=None,
        ground_truth_sequence_xr=None,
    ):
        """
        Set the mean_field used for anomaly diagnostics.

        Priority:
          1. If mean_field was provided at init (MDT), keep it.
          2. If ground_truth_sequence_xr is provided, compute mean over time from it.
          3. Else if input_sequence_xr is provided, compute mean over time from it.

        Args:
            input_sequence_xr: xarray Dataset with 'data' variable [time, ch, lat, lon]
            ground_truth_sequence_xr: xarray Dataset with 'data' variable [time, ch, lat, lon]
        """
        if self.mean_field is not None:
            # If mean_field already set (e.g., stats file), attempt to fill only missing channels from sequences.
            try:
                import torch as _torch
                # Compute mean_over_time to use for filling
                data = ds['data'].values  # [time, ch, lat, lon]
                data = _np.nan_to_num(data, nan=0.0)
                mean_over_time = data.mean(axis=0)  # [ch, lat, lon]
                mean_over_time_t = _torch.from_numpy(mean_over_time).float().to(self.device)

                # If shapes match, replace channels that are all-zero in existing mean_field
                if self.mean_field.shape == mean_over_time_t.shape:
                    for ch in range(self.mean_field.shape[0]):
                        if _torch.allclose(self.mean_field[ch], _torch.zeros_like(self.mean_field[ch])):
                            self.mean_field[ch] = mean_over_time_t[ch]
                else:
                    # If shapes differ, try to broadcast based on channel count
                    C_existing = self.mean_field.shape[0]
                    C_new = mean_over_time_t.shape[0]
                    C_min = min(C_existing, C_new)
                    for ch in range(C_min):
                        if _torch.allclose(self.mean_field[ch], _torch.zeros_like(self.mean_field[ch])):
                            self.mean_field[ch] = mean_over_time_t[ch]
            except Exception:
                # If any issue occurs, keep existing mean_field
                return
            return

        import numpy as _np
        import torch as _torch

        ds = None
        if ground_truth_sequence_xr is not None:
            ds = ground_truth_sequence_xr
        elif input_sequence_xr is not None:
            ds = input_sequence_xr

        if ds is None:
            # No data available to compute mean; leave mean_field as None
            return

        # Extract data variable and compute mean over time axis
        data = ds['data'].values  # [time, ch, lat, lon]
        data = _np.nan_to_num(data, nan=0.0)
        mean_over_time = data.mean(axis=0)  # [ch, lat, lon]

        # Store as torch tensor on device
        self.mean_field = _torch.from_numpy(mean_over_time).float().to(self.device)

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

        # Ensure mean_field set if possible (caller should set earlier)
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

    @staticmethod
    def compute_ebcr(
        psd_reference: np.ndarray,
        psd_optimized: np.ndarray,
        psd_truth: np.ndarray,
        denominator_tolerance: float = 1e-12,
    ) -> np.ndarray:
        """
        Compute the energy-bias correction ratio element-wise.

        Values below zero indicate a bias sign flip, values from zero through
        one indicate reduced bias, and values above one indicate increased bias.
        Where the reference/truth bias is numerically zero, EBCR is undefined
        and returned as NaN.
        """
        psd_reference = np.asarray(psd_reference, dtype=np.float64)
        psd_optimized = np.asarray(psd_optimized, dtype=np.float64)
        psd_truth = np.asarray(psd_truth, dtype=np.float64)
        denominator = psd_reference - psd_truth
        scale = max(
            float(np.nanmax(np.abs(psd_reference))) if psd_reference.size else 0.0,
            float(np.nanmax(np.abs(psd_truth))) if psd_truth.size else 0.0,
            1.0,
        )
        ebcr = np.full(np.broadcast_shapes(
            psd_reference.shape, psd_optimized.shape, psd_truth.shape
        ), np.nan, dtype=np.float64)
        valid = np.abs(denominator) > denominator_tolerance * scale
        np.divide(
            psd_optimized - psd_truth,
            denominator,
            out=ebcr,
            where=valid,
        )
        return ebcr
