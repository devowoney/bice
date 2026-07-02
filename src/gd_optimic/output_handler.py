"""
Output handler for IC optimization outputs.

Provides:
    - Ocean state outputs in NetCDF (`states/`)
    - Diagnostic visualizations in PNG (`diagnostics/`)
    - RMSE/PSD comparison plots (reference vs optimized)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Handles saving optimization outputs.

    - State files are saved as NetCDF in `states/`
    - Diagnostics are saved as PNG in `diagnostics/`
    """
    
    # Variable metadata (name, units, long_name)
    VAR_METADATA = {
        0: ('SSH', 'm', 'Sea Surface Height'),
        1: ('T', '°C', 'Sea Surface Temperature'),
        2: ('S', 'psu', 'Sea Surface Salinity'),
        3: ('U', 'm/s', 'Zonal Velocity'),
        4: ('V', 'm/s', 'Meridional Velocity'),
    }
    
    def __init__(self, exp_dir: Path, device: str = 'cuda'):
        """
        Initialize output handler.
        
        Args:
            exp_dir: Experiment directory root
            device: Device for tensor operations
        """
        self.exp_dir = Path(exp_dir)
        self.device = device
        
        # Create diagnostics directory
        self.diagnostics_dir = self.exp_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output handler initialized: {self.diagnostics_dir}")
    
    def save_outputs(
        self,
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
        reference_forecast: torch.Tensor,
        full_forecast: torch.Tensor,
        target_sequence: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        input_sequence: xr.Dataset,
        target_sequence_xr: xr.Dataset,
        ground_truth_sequence_xr: xr.Dataset,
        ocean_mask: torch.Tensor,
        best_iteration: int,
        best_loss: float,
        regional_masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Save outputs:
        1) ocean state NetCDF files
        2) diagnostic visualization figures (PNG)
         
        Args:
            x0_init: Initial condition [B, T=2, C=5, H, W]
            x0_optimized: Optimized initial condition [B, T=2, C=5, H, W]
            reference_forecast: Forecast from initial IC [B, T, C=5, H, W]
            full_forecast: Full forecast from optimized IC [B, T=21, C=5, H, W] (full forecast window)
            target_sequence: Target observations [B, T=7, C=5, H, W] (assimilation window only)
            ground_truth_sequence: Full ground truth [B, T=21, C=5, H, W] (OSSE reanalysis for all timesteps)
            input_sequence: xarray Dataset with input sequence (for coordinates)
            target_sequence_xr: xarray Dataset with target sequence (for coordinates)
            ground_truth_sequence_xr: xarray Dataset with ground truth (for coordinates)
            ocean_mask: Ocean mask [C, H, W]
            best_iteration: Best iteration number
            best_loss: Best loss value
        """
        try:
            logger.info("\n" + "="*60)
            logger.info("Saving state NetCDF and diagnostic visualizations...")
            logger.info("="*60)

            state_dir = self.exp_dir / "states"
            state_dir.mkdir(parents=True, exist_ok=True)

            # Save ocean states as NetCDF
            self._save_state_netcdf(
                state_dir=state_dir,
                x0_init=x0_init,
                x0_optimized=x0_optimized,
                reference_forecast=reference_forecast,
                full_forecast=full_forecast,
                ground_truth_sequence=ground_truth_sequence,
                input_sequence=input_sequence,
                ground_truth_sequence_xr=ground_truth_sequence_xr,
                best_iteration=best_iteration,
                best_loss=best_loss,
            )
            self._save_forecast_animation(
                state_dir=state_dir,
                reference_forecast=reference_forecast,
                optimized_forecast=full_forecast,
                input_sequence=input_sequence,
            )

            # Save diagnostics as visualization files (no NetCDF)
            self._save_ic_correction_figure(x0_init, x0_optimized, input_sequence)
            rmse_data = self._save_rmse_figure(
                x0_init=x0_init,
                x0_optimized=x0_optimized,
                reference_forecast=reference_forecast,
                optimized_forecast=full_forecast,
                ground_truth_sequence=ground_truth_sequence,
                ocean_mask=ocean_mask,
                observation_length=int(target_sequence.shape[1]),
            )
            self._save_psd_figure(
                reference_forecast=reference_forecast,
                optimized_forecast=full_forecast,
                ground_truth_sequence=ground_truth_sequence,
                ocean_mask=ocean_mask,
            )
            regional_summary = {}
            if regional_masks is not None:
                regional_summary = self._save_regional_diagnostics(
                    reference_forecast=reference_forecast,
                    optimized_forecast=full_forecast,
                    ground_truth_sequence=ground_truth_sequence,
                    regional_masks=regional_masks,
                    ocean_mask=ocean_mask,
                    input_sequence=input_sequence,
                    observation_length=int(target_sequence.shape[1]),
                )
                self._save_gulf_stream_animation(
                    state_dir=state_dir,
                    reference_forecast=reference_forecast,
                    optimized_forecast=full_forecast,
                    input_sequence=input_sequence,
                    gulf_stream_mask=regional_masks["gulf_stream"],
                )

            summary_path = self.diagnostics_dir / "diagnostic_summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "best_iteration": best_iteration,
                        "best_loss": float(best_loss),
                        "rmse_horizon": int(rmse_data["horizon"]),
                        "files": {
                            "ic_correction_plot": "diagnostics/ic_correction_comparison.png",
                            "rmse_plot": "diagnostics/rmse_evolution.png",
                            "psd_plot": "diagnostics/psd_comparison.png",
                            "regional_rmse_plot": "diagnostics/rmse_region_*.png",
                            "regional_psd_plot": "diagnostics/psd_region_*.png",
                            "regional_metrics_json": "diagnostics/regional_metrics.json",
                            "forecast_animation": "states/forecast_comparison.gif",
                            "gulf_stream_animation": "states/gulf_stream_forecast_comparison.gif",
                        },
                        "regional_metrics": regional_summary,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"✓ Saved diagnostic summary: {summary_path}")

            logger.info("="*60)
            logger.info("All outputs saved successfully!")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error saving outputs: {e}", exc_info=True)
            raise
    
    def _compute_ic_correction(
        self,
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
        input_sequence: xr.Dataset
    ) -> xr.Dataset:
        """
        Compute IC correction: delta = optimized - initial.
        
        Args:
            x0_init: Initial IC [B, T=2, C=5, H, W]
            x0_optimized: Optimized IC [B, T=2, C=5, H, W]
            input_sequence: xarray Dataset with coordinates
            
        Returns:
            xarray Dataset with IC correction [T=2, C=5, H, W]
        """
        # Compute delta
        delta = (x0_optimized - x0_init).squeeze(0).cpu().numpy()  # [T=2, C=5, H, W]
        
        # Extract coordinates
        lat = input_sequence.coords['lat'].values
        lon = input_sequence.coords['lon'].values
        time = input_sequence.coords['time'].values[:2]  # First 2 time steps
        
        # Create DataArrays for each variable
        data_vars = {}
        for ch_idx in range(delta.shape[1]):
            var_name, units, long_name = self.VAR_METADATA[ch_idx]
            data_vars[var_name] = xr.DataArray(
                delta[:, ch_idx, :, :],
                dims=['time', 'lat', 'lon'],
                coords={'time': time, 'lat': lat, 'lon': lon},
                attrs={'units': units, 'long_name': f'{long_name} Correction', 'description': f'IC correction for {var_name}'}
            )
        
        # Create Dataset
        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'Initial Condition Correction'
        
        return ds

    def _save_state_netcdf(
        self,
        state_dir: Path,
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
        reference_forecast: torch.Tensor,
        full_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        input_sequence: xr.Dataset,
        ground_truth_sequence_xr: xr.Dataset,
        best_iteration: int,
        best_loss: float,
    ) -> None:
        """Save ocean states as NetCDF files."""
        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        ic_time = input_sequence.coords["time"].values[: x0_init.shape[1]]
        fc_time = ground_truth_sequence_xr.coords["time"].values[: full_forecast.shape[1]]

        def _to_dataset(data: torch.Tensor, time_coord, title: str) -> xr.Dataset:
            arr = data.squeeze(0).detach().cpu().numpy()  # [T, C, H, W]
            data_vars = {}
            for ch_idx in range(arr.shape[1]):
                var_name, units, long_name = self.VAR_METADATA[ch_idx]
                data_vars[var_name] = xr.DataArray(
                    arr[:, ch_idx, :, :],
                    dims=["time", "lat", "lon"],
                    coords={"time": time_coord, "lat": lat, "lon": lon},
                    attrs={"units": units, "long_name": long_name},
                )
            ds = xr.Dataset(data_vars)
            ds.attrs["title"] = title
            ds.attrs["best_iteration"] = best_iteration
            ds.attrs["best_loss"] = float(best_loss)
            return ds

        datasets = {
            "initial_condition.nc": _to_dataset(x0_init, ic_time, "Initial Condition"),
            "optimized_initial_condition.nc": _to_dataset(x0_optimized, ic_time, "Optimized Initial Condition"),
            "reference_forecast.nc": _to_dataset(reference_forecast, fc_time, "Reference Forecast (from initial IC)"),
            "optimized_forecast.nc": _to_dataset(full_forecast, fc_time, "Optimized Forecast"),
            "ground_truth.nc": _to_dataset(ground_truth_sequence, fc_time, "Ground Truth (GLORYS12)"),
        }

        for name, ds in datasets.items():
            path = state_dir / name
            ds.to_netcdf(path)
            logger.info(f"✓ Saved state NetCDF: {path}")

    def _get_extent_origin(self, input_sequence: xr.Dataset) -> Tuple[Tuple[float, float, float, float], str]:
        """Return imshow extent and origin based on the input_sequence coords.

        Returns:
            extent: (lon_min, lon_max, lat_min, lat_max)
            origin: 'lower' if lat ascending else 'upper'
        """
        lat = input_sequence.coords['lat'].values
        lon = input_sequence.coords['lon'].values
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
        origin = 'lower' if lat[0] < lat[-1] else 'upper'
        extent = (lon_min, lon_max, lat_min, lat_max)
        return extent, origin

    def _save_forecast_animation(
        self,
        state_dir: Path,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        input_sequence: xr.Dataset,
    ) -> None:
        """Save side-by-side forecast animation (reference vs optimized) as GIF."""
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()   # [T, C, H, W]
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()   # [T, C, H, W]
        horizon = min(ref.shape[0], opt.shape[0])
        ref = ref[:horizon]
        opt = opt[:horizon]

        fig, axes = plt.subplots(5, 2, figsize=(10, 16), constrained_layout=True)
        axes[0, 0].set_title("Reference forecast", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Optimized forecast", fontsize=11, fontweight="bold")

        images = []
        for ch_idx in range(5):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            merged = np.concatenate([ref[:, ch_idx].ravel(), opt[:, ch_idx].ravel()])
            vmax = np.nanpercentile(np.abs(merged), 99)
            vmax = max(vmax, 1e-12)

            extent, origin = self._get_extent_origin(input_sequence)
            im_ref = axes[ch_idx, 0].imshow(
                ref[0, ch_idx], cmap="viridis", vmin=-vmax, vmax=vmax, animated=True,
                extent=extent, origin=origin, aspect='auto'
            )
            im_opt = axes[ch_idx, 1].imshow(
                opt[0, ch_idx], cmap="viridis", vmin=-vmax, vmax=vmax, animated=True,
                extent=extent, origin=origin, aspect='auto'
            )
            images.append((im_ref, im_opt))

            axes[ch_idx, 0].set_ylabel(f"{var_name}\n{long_name}", fontsize=9)
            axes[ch_idx, 0].set_xticks([])
            axes[ch_idx, 0].set_yticks([])
            axes[ch_idx, 1].set_xticks([])
            axes[ch_idx, 1].set_yticks([])

        title = fig.suptitle("Forecast comparison | timestep 0", fontsize=13)

        def _update(frame_idx: int):
            artists = [title]
            title.set_text(f"Forecast comparison | timestep {frame_idx}")
            for ch_idx in range(5):
                im_ref, im_opt = images[ch_idx]
                im_ref.set_data(ref[frame_idx, ch_idx])
                im_opt.set_data(opt[frame_idx, ch_idx])
                artists.extend([im_ref, im_opt])
            return artists

        anim = animation.FuncAnimation(
            fig=fig,
            func=_update,
            frames=horizon,
            interval=500,
            blit=False,
            repeat=True,
        )

        out_path = state_dir / "forecast_comparison.gif"
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=2))
            logger.info(f"✓ Saved forecast animation: {out_path}")
        finally:
            plt.close(fig)

    def _save_ic_correction_figure(self, x0_init: torch.Tensor, x0_optimized: torch.Tensor, input_sequence: xr.Dataset) -> None:
        """Save IC state-map visualization (initial / optimized / correction)."""
        init_np = x0_init.squeeze(0).detach().cpu().numpy()      # [T=2, C, H, W]
        opt_np = x0_optimized.squeeze(0).detach().cpu().numpy()  # [T=2, C, H, W]
        delta_np = opt_np - init_np
        t_idx = init_np.shape[0] - 1  # latest IC state

        # compute extent and origin from input_sequence
        extent, origin = self._get_extent_origin(input_sequence)

        fig, axes = plt.subplots(5, 3, figsize=(15, 18), constrained_layout=True)
        col_titles = ["Initial state", "Optimized state", "Correction (optimized - initial)"]
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")

        for ch_idx in range(5):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            init_field = init_np[t_idx, ch_idx]
            opt_field = opt_np[t_idx, ch_idx]
            delta_field = delta_np[t_idx, ch_idx]

            # Shared color range for initial/optimized panels
            vmax_state = np.nanpercentile(np.abs(np.concatenate([init_field.ravel(), opt_field.ravel()])), 99)
            vmax_state = max(vmax_state, 1e-12)
            vmax_delta = np.nanpercentile(np.abs(delta_field), 99)
            vmax_delta = max(vmax_delta, 1e-12)

            im0 = axes[ch_idx, 0].imshow(
                init_field,
                cmap="viridis",
                vmin=-vmax_state,
                vmax=vmax_state,
                extent=extent,
                origin=origin,
                aspect="auto",
            )
            im1 = axes[ch_idx, 1].imshow(
                opt_field,
                cmap="viridis",
                vmin=-vmax_state,
                vmax=vmax_state,
                extent=extent,
                origin=origin,
                aspect="auto",
            )
            im2 = axes[ch_idx, 2].imshow(
                delta_field,
                cmap="RdBu_r",
                vmin=-vmax_delta,
                vmax=vmax_delta,
                extent=extent,
                origin=origin,
                aspect="auto",
            )

            axes[ch_idx, 0].set_ylabel(f"{var_name}\n{long_name}", fontsize=9)
            for col in range(3):
                axes[ch_idx, col].set_xticks([])
                axes[ch_idx, col].set_yticks([])

            fig.colorbar(im0, ax=axes[ch_idx, 0], fraction=0.046, pad=0.02)
            fig.colorbar(im1, ax=axes[ch_idx, 1], fraction=0.046, pad=0.02)
            fig.colorbar(im2, ax=axes[ch_idx, 2], fraction=0.046, pad=0.02)

        fig.suptitle("IC Correction State Visualization (latest IC timestep)", fontsize=14)
        out_path = self.diagnostics_dir / "ic_correction_comparison.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"✓ Saved IC correction figure: {out_path}")

    def _save_rmse_figure(
        self,
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
        observation_length: int,
    ) -> Dict[str, int]:
        """Save RMSE evolution figure (reference vs optimized)."""
        x0_ref = x0_init.squeeze(0).detach().cpu().numpy()
        x0_opt = x0_optimized.squeeze(0).detach().cpu().numpy()
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        gt = ground_truth_sequence.squeeze(0).detach().cpu().numpy()
        mask = ocean_mask.detach().cpu().numpy()

        horizon = min(ref.shape[0], opt.shape[0], gt.shape[0])
        ref = ref[:horizon]
        opt = opt[:horizon]
        gt = gt[:horizon]
        step_axis = np.arange(horizon + 1)

        rmse_ref = np.zeros((horizon + 1, ref.shape[1]))
        rmse_opt = np.zeros((horizon + 1, opt.shape[1]))

        # Starting point at initial condition
        for c in range(ref.shape[1]):
            valid = mask[c] > 0
            if valid.sum() == 0:
                rmse_ref[0, c] = np.nan
                rmse_opt[0, c] = np.nan
                continue
            ref_ic_err = x0_ref[-1, c] - gt[0, c]
            opt_ic_err = x0_opt[-1, c] - gt[0, c]
            rmse_ref[0, c] = np.sqrt(np.nanmean(ref_ic_err[valid] ** 2))
            rmse_opt[0, c] = np.sqrt(np.nanmean(opt_ic_err[valid] ** 2))

        for t in range(horizon):
            for c in range(ref.shape[1]):
                valid = mask[c] > 0
                if valid.sum() == 0:
                    rmse_ref[t + 1, c] = np.nan
                    rmse_opt[t + 1, c] = np.nan
                    continue
                ref_err = ref[t, c] - gt[t, c]
                opt_err = opt[t, c] - gt[t, c]
                rmse_ref[t + 1, c] = np.sqrt(np.nanmean(ref_err[valid] ** 2))
                rmse_opt[t + 1, c] = np.sqrt(np.nanmean(opt_err[valid] ** 2))

        fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        for ch_idx, ax in enumerate(axes):
            var_name, units, long_name = self.VAR_METADATA[ch_idx]
            ax.plot(step_axis, rmse_ref[:, ch_idx], color="tab:blue", linewidth=2, label="Reference")
            ax.plot(step_axis, rmse_opt[:, ch_idx], color="tab:orange", linewidth=2, label="Optimized")
            border_x = observation_length + 0.5
            ax.axvline(border_x, color="k", linestyle="--", linewidth=1.2, label="Obs/forecast border" if ch_idx == 0 else None)
            ax.set_ylabel(f"RMSE ({units})")
            ax.set_title(f"{long_name}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, horizon)
            if ch_idx == 0:
                ax.legend(loc="best")

        axes[-1].set_xlabel("Timestep (0=initial condition)")
        fig.suptitle(
            f"RMSE Evolution (T={horizon}, includes initial point) - "
            f"obs window: [1,{observation_length}], forecast: [{observation_length+1},{horizon}]",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = self.diagnostics_dir / "rmse_evolution.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"✓ Saved RMSE figure: {out_path}")
        return {"horizon": horizon}

    def _save_psd_figure(
        self,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> None:
        """Save PSD comparison figure (reference vs optimized vs ground truth)."""
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        gt = ground_truth_sequence.squeeze(0).detach().cpu().numpy()
        mask = ocean_mask.detach().cpu().numpy()

        horizon = min(ref.shape[0], opt.shape[0], gt.shape[0])
        ref = ref[:horizon]
        opt = opt[:horizon]
        gt = gt[:horizon]
        C, H, W = ref.shape[1], ref.shape[2], ref.shape[3]
        num_bins = min(H, W) // 2
        k_bins = np.linspace(0, 0.5, num_bins + 1)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2

        def _mean_psd(arr: np.ndarray, channel_idx: int) -> np.ndarray:
            psd_acc = np.zeros(num_bins)
            count = 0
            for t in range(arr.shape[0]):
                field = arr[t, channel_idx] * mask[channel_idx]
                if np.abs(field).max() < 1e-12:
                    continue
                fft2 = np.fft.fft2(field)
                power = np.abs(fft2) ** 2
                kx = np.fft.fftfreq(W, d=1.0)
                ky = np.fft.fftfreq(H, d=1.0)
                kx_grid, ky_grid = np.meshgrid(kx, ky)
                k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
                for i in range(num_bins):
                    in_bin = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
                    if in_bin.sum() > 0:
                        psd_acc[i] += power[in_bin].mean()
                count += 1
            return psd_acc / max(count, 1)

        fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        for ch_idx, ax in enumerate(axes):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            ref_psd = _mean_psd(ref, ch_idx)
            opt_psd = _mean_psd(opt, ch_idx)
            gt_psd = _mean_psd(gt, ch_idx)
            ax.loglog(k_centers[1:], ref_psd[1:] + 1e-20, color="tab:blue", linewidth=2, label="Reference")
            ax.loglog(k_centers[1:], opt_psd[1:] + 1e-20, color="tab:orange", linewidth=2, label="Optimized")
            ax.loglog(
                k_centers[1:],
                gt_psd[1:] + 1e-20,
                color="tab:green",
                linewidth=2,
                linestyle="--",
                label="Ground truth (reanalysis)",
            )
            ax.set_ylabel("PSD")
            ax.set_title(f"{long_name}")
            ax.grid(True, which="both", alpha=0.3)
            if ch_idx == 0:
                ax.legend(loc="best")

        axes[-1].set_xlabel("Normalized wavenumber")
        fig.suptitle("PSD Comparison - Reference vs Optimized vs Ground Truth", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out_path = self.diagnostics_dir / "psd_comparison.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"✓ Saved PSD figure: {out_path}")

    def _save_regional_diagnostics(
        self,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        regional_masks: Dict[str, np.ndarray],
        ocean_mask: torch.Tensor,
        input_sequence: xr.Dataset,
        observation_length: int,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Save notebook-style regional RMSE and PSD diagnostics.

        The reference workflow uses region-specific plots, not a single basin heatmap.
        """
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        gt = ground_truth_sequence.squeeze(0).detach().cpu().numpy()
        regions = ["global", "gulf_stream", "high_var", "low_var"]
        var_names = [self.VAR_METADATA[c][0] for c in range(ref.shape[1])]
        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)

        def _region_mask(region_name: str) -> np.ndarray:
            mask = regional_masks[region_name] > 0
            return mask

        def _rmse_series(arr: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
            horizon = min(arr.shape[0], truth.shape[0])
            series = np.full((horizon + 1, arr.shape[1]), np.nan, dtype=np.float64)
            for c_idx in range(arr.shape[1]):
                valid = mask[c_idx]
                if np.any(valid):
                    init_err = arr[0, c_idx] - truth[0, c_idx]
                    series[0, c_idx] = float(np.sqrt(np.nanmean(init_err[valid] ** 2)))
            for t in range(horizon):
                for c_idx in range(arr.shape[1]):
                    valid = mask[c_idx]
                    if not np.any(valid):
                        continue
                    err = arr[t, c_idx] - truth[t, c_idx]
                    series[t + 1, c_idx] = float(np.sqrt(np.nanmean(err[valid] ** 2)))
            return series

        def _mean_psd(arr: np.ndarray, mask: np.ndarray, channel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            values = arr[:, channel_idx]
            selected = values[:, mask[channel_idx]]
            if selected.size == 0:
                return np.array([]), np.array([])

            h, w = values.shape[1], values.shape[2]
            dlon = float(np.abs(lon[1] - lon[0])) if lon.size > 1 else 0.25
            dlat = float(np.abs(lat[1] - lat[0])) if lat.size > 1 else 0.25
            f_nyquist = min(1.0 / (2.0 * dlon), 1.0 / (2.0 * dlat))
            f_cut = f_nyquist

            sample = values[0]
            ocean_vals = sample[mask[channel_idx]]
            ocean_mean = ocean_vals.mean() if ocean_vals.size > 0 else 0.0
            psd_stack = []
            for t in range(values.shape[0]):
                field = values[t]
                field_centered = np.where(mask[channel_idx], field - ocean_mean, 0.0)
                wy = np.hanning(h)
                wx = np.hanning(w)
                window2d = np.outer(wy, wx)
                field_windowed = field_centered * window2d
                power2d = (np.abs(np.fft.fft2(field_windowed)) ** 2) / (h * w)
                ky = np.fft.fftfreq(h, d=dlat)
                kx = np.fft.fftfreq(w, d=dlon)
                kx_grid_psd, ky_grid_psd = np.meshgrid(kx, ky)
                kr = np.sqrt(kx_grid_psd**2 + ky_grid_psd**2)
                kr_flat = kr.ravel()
                p_flat = power2d.ravel()
                valid = (kr_flat > 0) & (kr_flat <= f_cut)
                kr_flat = kr_flat[valid]
                p_flat = p_flat[valid]
                if kr_flat.size == 0:
                    continue
                bins = np.logspace(np.log10(kr_flat.min()), np.log10(kr_flat.max()), 41)
                which_bin = np.digitize(kr_flat, bins) - 1
                psd_bin = np.full(40, np.nan, dtype=np.float64)
                k_bin = np.sqrt(bins[:-1] * bins[1:])
                for b in range(40):
                    in_bin = which_bin == b
                    if np.any(in_bin):
                        psd_bin[b] = p_flat[in_bin].mean()
                keep = np.isfinite(psd_bin) & (psd_bin > 0) & (k_bin > 0)
                if np.any(keep):
                    psd_stack.append((k_bin[keep], psd_bin[keep]))

            if not psd_stack:
                return np.array([]), np.array([])

            k_ref = psd_stack[0][0]
            psd_vals = np.zeros_like(k_ref)
            count = 0
            for k_vals, p_vals in psd_stack:
                interp = np.interp(k_ref, k_vals, p_vals, left=np.nan, right=np.nan)
                valid = np.isfinite(interp)
                if np.any(valid):
                    psd_vals[valid] += interp[valid]
                    count += 1
            if count == 0:
                return np.array([]), np.array([])
            return k_ref, psd_vals / count

        def _band_ratio(psd: np.ndarray, k_vals: np.ndarray) -> float:
            if psd.size == 0 or k_vals.size == 0:
                return float("nan")
            # local trapezoidal integrator to avoid relying on np.trapz
            def _trapz_local(y: np.ndarray, x: np.ndarray) -> float:
                if x.size < 2 or y.size < 2:
                    return 0.0
                dx = x[1:] - x[:-1]
                return float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))

            total_energy = _trapz_local(psd, k_vals)
            in_band = (k_vals >= 1.0 / 500.0) & (k_vals <= 1.0 / 50.0)
            if not np.any(in_band):
                return float("nan")
            mesoscale_energy = _trapz_local(psd[in_band], k_vals[in_band])
            return float(mesoscale_energy / (total_energy + 1e-10))

        def _save_rmse_plot(region_name: str, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
            ref_series = _rmse_series(ref, gt, mask)
            opt_series = _rmse_series(opt, gt, mask)
            horizon = ref_series.shape[0] - 1
            step_axis = np.arange(horizon + 1)
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            for ch_idx, ax in enumerate(axes):
                var_name, units, long_name = self.VAR_METADATA[ch_idx]
                ax.plot(step_axis, ref_series[:, ch_idx], color="tab:blue", linewidth=2, label="Reference")
                ax.plot(step_axis, opt_series[:, ch_idx], color="tab:orange", linewidth=2, label="Optimized")
                ax.axvline(
                    observation_length + 0.5,
                    color="k",
                    linestyle="--",
                    linewidth=1.2,
                    label="Obs/forecast border" if ch_idx == 0 else None,
                )
                ax.set_ylabel(f"RMSE ({units})")
                ax.set_title(f"{long_name} | {region_name}")
                ax.grid(True, alpha=0.3)
                if ch_idx == 0:
                    ax.legend(loc="best")
            axes[-1].set_xlabel("Timestep (0=initial condition)")
            fig.suptitle(f"RMSE Evolution | {region_name}", fontsize=13)
            out_path = self.diagnostics_dir / f"rmse_region_{region_name}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            return {
                "reference": {var: float(ref_series[-1, idx]) for idx, var in enumerate(var_names)},
                "optimized": {var: float(opt_series[-1, idx]) for idx, var in enumerate(var_names)},
                "file": str(out_path.relative_to(self.exp_dir)),
            }

        def _save_psd_plot(region_name: str, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            summary = {"reference": {}, "optimized": {}, "ground_truth": {}}
            for ch_idx, ax in enumerate(axes):
                var_name, _, long_name = self.VAR_METADATA[ch_idx]
                k_ref, p_ref = _mean_psd(ref, mask, ch_idx)
                k_opt, p_opt = _mean_psd(opt, mask, ch_idx)
                k_gt, p_gt = _mean_psd(gt, mask, ch_idx)
                if k_ref.size > 0:
                    ax.loglog(k_ref, p_ref + 1e-20, color="tab:blue", linewidth=2, label="Reference")
                    summary["reference"][var_name] = _band_ratio(p_ref, k_ref)
                if k_opt.size > 0:
                    ax.loglog(k_opt, p_opt + 1e-20, color="tab:orange", linewidth=2, label="Optimized")
                    summary["optimized"][var_name] = _band_ratio(p_opt, k_opt)
                if k_gt.size > 0:
                    ax.loglog(k_gt, p_gt + 1e-20, color="tab:green", linewidth=2, linestyle="--", label="Ground truth")
                    summary["ground_truth"][var_name] = _band_ratio(p_gt, k_gt)
                ax.set_ylabel("PSD")
                ax.set_title(f"{long_name} | {region_name}")
                ax.grid(True, which="both", alpha=0.3)
                if ch_idx == 0:
                    ax.legend(loc="best")
            axes[-1].set_xlabel("Normalized wavenumber")
            fig.suptitle(f"PSD Comparison | {region_name}", fontsize=13)
            out_path = self.diagnostics_dir / f"psd_region_{region_name}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            summary["file"] = str(out_path.relative_to(self.exp_dir))
            return summary

        summary = {}
        for region_name in regions:
            mask = _region_mask(region_name)
            summary[region_name] = {
                "rmse": _save_rmse_plot(region_name, mask),
                "psd": _save_psd_plot(region_name, mask),
            }

        reg_json = self.diagnostics_dir / "regional_metrics.json"
        with open(reg_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved regional diagnostics: {reg_json}")
        return summary

    def _save_gulf_stream_animation(
        self,
        state_dir: Path,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        input_sequence: xr.Dataset,
        gulf_stream_mask: np.ndarray,
    ) -> None:
        """Save a Gulf Stream cropped animation for reference vs optimized forecasts."""
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)

        # Use a bounding box around the Gulf Stream mask to keep the animation focused.
        region_any = gulf_stream_mask[0] > 0
        y_idx, x_idx = np.where(region_any)
        if y_idx.size == 0 or x_idx.size == 0:
            logger.warning("Gulf Stream mask is empty; skipping region animation.")
            return

        y0, y1 = max(int(y_idx.min()) - 5, 0), min(int(y_idx.max()) + 6, lat.size)
        x0, x1 = max(int(x_idx.min()) - 5, 0), min(int(x_idx.max()) + 6, lon.size)
        lat_crop = lat[y0:y1]
        lon_crop = lon_grid[y0:y1, x0:x1]
        extent = (
            float(np.nanmin(lon_crop)),
            float(np.nanmax(lon_crop)),
            float(np.nanmin(lat_crop)),
            float(np.nanmax(lat_crop)),
        )
        origin = "lower" if lat[0] < lat[-1] else "upper"

        horizon = min(ref.shape[0], opt.shape[0])
        fig, axes = plt.subplots(5, 2, figsize=(10, 16), constrained_layout=True)
        axes[0, 0].set_title("Gulf Stream reference", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Gulf Stream optimized", fontsize=11, fontweight="bold")

        images = []
        for ch_idx in range(5):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            ref_crop = ref[0, ch_idx, y0:y1, x0:x1]
            opt_crop = opt[0, ch_idx, y0:y1, x0:x1]
            merged = np.concatenate([ref_crop.ravel(), opt_crop.ravel()])
            vmax = np.nanpercentile(np.abs(merged), 99)
            vmax = max(vmax, 1e-12)
            im_ref = axes[ch_idx, 0].imshow(
                ref_crop,
                cmap="viridis",
                vmin=-vmax,
                vmax=vmax,
                extent=extent,
                origin=origin,
                aspect="auto",
                animated=True,
            )
            im_opt = axes[ch_idx, 1].imshow(
                opt_crop,
                cmap="viridis",
                vmin=-vmax,
                vmax=vmax,
                extent=extent,
                origin=origin,
                aspect="auto",
                animated=True,
            )
            images.append((im_ref, im_opt))
            axes[ch_idx, 0].set_ylabel(f"{var_name}\n{long_name}", fontsize=9)
            axes[ch_idx, 0].set_xticks([])
            axes[ch_idx, 0].set_yticks([])
            axes[ch_idx, 1].set_xticks([])
            axes[ch_idx, 1].set_yticks([])

        title = fig.suptitle("Gulf Stream comparison | timestep 0", fontsize=13)

        def _update(frame_idx: int):
            title.set_text(f"Gulf Stream comparison | timestep {frame_idx}")
            artists = [title]
            for ch_idx in range(5):
                im_ref, im_opt = images[ch_idx]
                im_ref.set_data(ref[frame_idx, ch_idx, y0:y1, x0:x1])
                im_opt.set_data(opt[frame_idx, ch_idx, y0:y1, x0:x1])
                artists.extend([im_ref, im_opt])
            return artists

        anim = animation.FuncAnimation(fig=fig, func=_update, frames=horizon, interval=500, blit=False, repeat=True)
        out_path = state_dir / "gulf_stream_forecast_comparison.gif"
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=2))
            logger.info(f"✓ Saved Gulf Stream animation: {out_path}")
        finally:
            plt.close(fig)
    
    def _compute_improved_forecast(
        self,
        predictions: torch.Tensor,
        input_sequence: xr.Dataset,
        target_sequence_xr: xr.Dataset
    ) -> xr.Dataset:
        """
        Create improved forecast dataset from predictions.
        
        Args:
            predictions: Forward predictions [B, T=21, C=5, H, W] (full forecast window)
            input_sequence: xarray Dataset with lat/lon coordinates
            target_sequence_xr: xarray Dataset with time coordinates (first T=7 times)
            
        Returns:
            xarray Dataset with predictions [T=21, C=5, H, W]
        """
        # Convert predictions to numpy
        preds = predictions.squeeze(0).cpu().numpy()  # [T=21, C=5, H, W]
        T_forecast = preds.shape[0]
        
        # Extract coordinates
        lat = input_sequence.coords['lat'].values
        lon = input_sequence.coords['lon'].values
        
        # Get time steps: use target times + extrapolate for forecast window
        target_times = target_sequence_xr.coords['time'].values  # [T=7]
        
        if len(target_times) >= 2:
            # Calculate time interval and extrapolate
            dt = target_times[1] - target_times[0]
            time = []
            for i in range(T_forecast):
                time.append(target_times[0] + i * dt)
            time = np.array(time, dtype='datetime64[D]')
        else:
            # Fallback: just use target[0] + increments
            time = []
            for i in range(T_forecast):
                time.append(target_times[0] + i * np.timedelta64(1, 'D'))
            time = np.array(time, dtype='datetime64[D]')
        
        # Create DataArrays for each variable
        data_vars = {}
        for ch_idx in range(preds.shape[1]):
            var_name, units, long_name = self.VAR_METADATA[ch_idx]
            data_vars[var_name] = xr.DataArray(
                preds[:, ch_idx, :, :],
                dims=['time', 'lat', 'lon'],
                coords={'time': time, 'lat': lat, 'lon': lon},
                attrs={'units': units, 'long_name': long_name, 'description': f'Forecast {var_name} from optimized IC'}
            )
        
        # Create Dataset
        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'Improved Forecast from Optimized IC (T=21 forecast window)'
        
        return ds
    
    def _compute_rmse_evolution(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        ocean_mask: torch.Tensor,
        sequence_xr: xr.Dataset
    ) -> xr.Dataset:
        """
        Compute RMSE evolution over time for each variable against ground truth.
        
        Args:
            predictions: Forward predictions [B, T=21, C=5, H, W] (full forecast)
            ground_truth: Ground truth observations [B, T=21, C=5, H, W] (full forecast window)
            ocean_mask: Ocean mask [C, H, W]
            sequence_xr: xarray Dataset with forecast time coordinates (T=21)
            
        Returns:
            xarray Dataset with RMSE [T=21, C=5] showing error evolution throughout forecast window
        """
        # Compute errors across full forecast horizon against ground truth
        errors = predictions - ground_truth  # [B, T=21, C, H, W]
        errors = errors.squeeze(0)  # [T=21, C, H, W]
        
        # Move to CPU and convert to numpy
        errors_np = errors.cpu().numpy()
        ocean_mask_np = ocean_mask.cpu().numpy()
        
        # Compute RMSE for each time step and variable throughout full forecast horizon
        T, C, H, W = errors_np.shape
        rmse = np.zeros((T, C))
        
        for t in range(T):
            for c in range(C):
                # Apply ocean mask
                error_masked = errors_np[t, c, :, :] * ocean_mask_np[c, :, :]
                mask_valid = ocean_mask_np[c, :, :] > 0
                
                # Compute RMSE only over valid ocean points
                if mask_valid.sum() > 0:
                    rmse[t, c] = np.sqrt(np.nanmean(error_masked[mask_valid] ** 2))
                else:
                    rmse[t, c] = np.nan
        
        # Extract time coordinate (full forecast times)
        time = sequence_xr.coords['time'].values
        
        # Create DataArrays for each variable
        data_vars = {}
        for ch_idx in range(C):
            var_name, units, long_name = self.VAR_METADATA[ch_idx]
            data_vars[f'{var_name}_RMSE'] = xr.DataArray(
                rmse[:, ch_idx],
                dims=["time"],
                coords={"time": time},
                attrs={
                    "units": units,
                    "long_name": f"{long_name} RMSE",
                    "description": (
                        f"Root mean square error for {var_name} (full forecast horizon)"
                    ),
                },
            )
        
        # Create Dataset
        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'RMSE Evolution Over Full Forecast Window (T=21) Against Ground Truth'
        
        return ds
    
    def _compute_psd_analysis(
        self,
        predictions: torch.Tensor,
        ocean_mask: torch.Tensor
    ) -> xr.Dataset:
        """
        Compute mean power spectral density for each variable.
        
        Uses 2D FFT, bins by wavenumber k = sqrt(kx^2 + ky^2), averages over time.
        
        Args:
            predictions: Forward predictions [B, T=7, C=5, H, W]
            ocean_mask: Ocean mask [C, H, W]
            
        Returns:
            xarray Dataset with PSD [frequency_bins, C=5]
        """
        # Move to CPU and convert to numpy
        preds_np = predictions.squeeze(0).cpu().numpy()  # [T, C, H, W]
        ocean_mask_np = ocean_mask.cpu().numpy()  # [C, H, W]
        
        T, C, H, W = preds_np.shape
        
        # Compute 2D FFT for each time step and channel
        num_bins = min(H, W) // 2  # Number of wavenumber bins
        psd_mean = np.zeros((num_bins, C))
        psd_count = np.zeros((num_bins, C))
        
        for ch_idx in range(C):
            # Collect PSD across all time steps
            psd_time_accumulated = np.zeros(num_bins)
            count_time = 0
            
            for t in range(T):
                # Get field and apply mask
                field = preds_np[t, ch_idx, :, :]
                mask = ocean_mask_np[ch_idx, :, :]
                
                # Apply mask (set land to zero)
                field_masked = field * mask
                
                # Skip if all zeros
                if np.abs(field_masked).max() < 1e-10:
                    continue
                
                # Compute 2D FFT
                fft_2d = np.fft.fft2(field_masked)
                power = np.abs(fft_2d) ** 2
                
                # Compute wavenumber grid
                kx = np.fft.fftfreq(W, d=1.0)
                ky = np.fft.fftfreq(H, d=1.0)
                kx_grid, ky_grid = np.meshgrid(kx, ky)
                k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
                
                # Bin by wavenumber magnitude
                k_bins = np.linspace(0, k_mag.max(), num_bins + 1)
                
                for i in range(num_bins):
                    mask_bin = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
                    if mask_bin.sum() > 0:
                        psd_time_accumulated[i] += power[mask_bin].mean()
                
                count_time += 1
            
            # Average over time
            if count_time > 0:
                psd_mean[:, ch_idx] = psd_time_accumulated / count_time
        
        # Create wavenumber coordinate (bin centers)
        k_bins = np.linspace(0, 0.5, num_bins + 1)  # Normalized frequency
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        # Create DataArrays for each variable
        data_vars = {}
        for ch_idx in range(C):
            var_name, units, long_name = self.VAR_METADATA[ch_idx]
            data_vars[f'{var_name}_PSD'] = xr.DataArray(
                psd_mean[:, ch_idx],
                dims=['wavenumber'],
                coords={'wavenumber': k_centers},
                attrs={
                    'units': f'({units})^2',
                    'long_name': f'{long_name} Power Spectral Density',
                    'description': f'Mean PSD for {var_name} averaged over forecast time'
                }
            )
        
        # Create Dataset
        ds = xr.Dataset(data_vars)
        ds.attrs['title'] = 'Power Spectral Density Analysis'
        ds.attrs['wavenumber_description'] = 'Normalized wavenumber bins (cycles per grid point)'
        
        return ds
