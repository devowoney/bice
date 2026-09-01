"""
Output handler for IC optimization outputs.

Provides:
    - Ocean state outputs in NetCDF (`states/`)
    - Metrics and summaries in JSON/CSV (`metrics/`)
    - Diagnostic visualizations in PNG (`diagnostics/`)
    - RMSE/PSD comparison plots (reference vs optimized)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import xarray as xr

from .metrics import PSDComputer

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Handles saving optimization outputs.

    - State files are saved as NetCDF in `states/`
    - Metrics and summaries are saved in `metrics/`
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
    PSD_ANALYSIS_TIMESTEP = 7
    EFFECTIVE_WAVELENGTH_MIN_KM = 65.0
    EFFECTIVE_WAVELENGTH_MAX_KM = 500.0

    def _compute_fixed_timestep_metrics(
        self,
        rmse_ref: np.ndarray,
        rmse_opt: np.ndarray,
        horizon: int,
        var_names: list = None,
    ) -> Dict:
        if var_names is None:
            var_names = [self.VAR_METADATA[i][0] for i in range(rmse_ref.shape[1])]
        return {
            f"t_{t}": {
                var: {
                    "reference": float(rmse_ref[t, idx]),
                    "optimized": float(rmse_opt[t, idx]),
                }
                for idx, var in enumerate(var_names)
            }
            for t in range(7, horizon + 1, 7)
        }

    def _compute_horizontal_gain(
        self,
        rmse_ref: np.ndarray,
        rmse_opt: np.ndarray,
        observation_length: int,
        var_names: list = None,
    ) -> Dict:
        """Measure optimized predictability beyond a reference persistence baseline."""
        if var_names is None:
            var_names = [self.VAR_METADATA[i][0] for i in range(rmse_ref.shape[1])]
        horizon = rmse_ref.shape[0] - 1
        boundary = min(max(int(observation_length), 0), horizon)
        result = {}

        for idx, var in enumerate(var_names):
            baseline = float(rmse_ref[boundary, idx])
            crossing = float("nan")
            if np.isfinite(baseline) and np.isfinite(rmse_opt[boundary, idx]):
                if rmse_opt[boundary, idx] >= baseline:
                    crossing = float(boundary)
                else:
                    for t in range(boundary, horizon):
                        y0, y1 = rmse_opt[t, idx], rmse_opt[t + 1, idx]
                        if np.isfinite(y0) and np.isfinite(y1) and y0 < baseline <= y1:
                            crossing = t + (baseline - y0) / (y1 - y0)
                            break
            gain = crossing - boundary if np.isfinite(crossing) else float("nan")
            result[var] = {
                "persistence_rmse_reference": baseline,
                "persistence_timestep": float(boundary),
                "reference_crossing_timestep": float(boundary),
                "optimized_crossing_timestep": float(crossing),
                "forecast_horizon": horizon,
                "horizontal_gain_steps": float(gain),
                "horizontal_gain_hours": float(gain * 6.0),
                "gain_direction": (
                    "optimized_predictability_gain"
                    if np.isfinite(gain) and gain > 0
                    else "no_gain_or_already_above_persistence"
                ),
            }
        return result

    def _save_metrics_to_csv_and_json(self, regional_summary: Dict, output_prefix: str = "rmse_metrics") -> None:
        fixed_path = self.metrics_dir / f"{output_prefix}_fixed_timesteps.csv"
        with open(fixed_path, "w", newline="") as f:
            rows = (
                {
                    "region": region,
                    "timestep": timestep,
                    "variable": variable,
                    "reference_rmse": values["reference"],
                    "optimized_rmse": values["optimized"],
                }
                for region, data in regional_summary.items()
                for timestep, metrics in data.get("rmse", {}).get("fixed_timestep_metrics", {}).items()
                for variable, values in metrics.items()
            )
            writer = csv.DictWriter(
                f, fieldnames=["region", "timestep", "variable", "reference_rmse", "optimized_rmse"]
            )
            writer.writeheader()
            writer.writerows(rows)

        gain_path = self.metrics_dir / f"{output_prefix}_horizontal_gain.csv"
        with open(gain_path, "w", newline="") as f:
            fields = [
                "region", "variable", "persistence_rmse_reference", "persistence_timestep",
                "reference_crossing_timestep", "optimized_crossing_timestep", "forecast_horizon",
                "horizontal_gain_steps", "horizontal_gain_hours", "gain_direction",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for region, data in regional_summary.items():
                for variable, values in data.get("rmse", {}).get("horizontal_gain", {}).items():
                    writer.writerow({"region": region, "variable": variable, **values})

        summary = {
            region: {
                "final_rmse": {
                    "reference": data.get("rmse", {}).get("reference", {}),
                    "optimized": data.get("rmse", {}).get("optimized", {}),
                },
                "fixed_timestep_metrics": data.get("rmse", {}).get("fixed_timestep_metrics", {}),
                "horizontal_gain": data.get("rmse", {}).get("horizontal_gain", {}),
            }
            for region, data in regional_summary.items()
        }
        with open(self.metrics_dir / f"{output_prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def __init__(self, exp_dir: Path, device: str = 'cuda', run_id: Optional[str] = None):
        """
        Initialize output handler.
        
        Args:
            exp_dir: Experiment directory root
            device: Device for tensor operations
        """
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.run_id = run_id or self.exp_dir.name or "current_run"
        
        self.metrics_dir = self.exp_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Create diagnostics directory
        self.diagnostics_dir = self.exp_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output handler initialized: {self.diagnostics_dir}")

    def save_optimization_history(self, results: Dict) -> Path:
        """Save optimization history in the experiment metrics directory."""
        history_path = self.metrics_dir / "optimization_history.json"
        with open(history_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved history: {history_path}")
        return history_path

    def _save_diagnostic_data_exports(self, summary: Dict) -> None:
        """Persist complete regional RMSE and PSD curves for later multi-run analysis."""
        rmse_path = self.metrics_dir / "rmse_evolution.json"
        psd_path = self.metrics_dir / "psd_analysis.json"
        with open(rmse_path, "w") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "regions": {
                        region: {
                            "timesteps": data["rmse"].get("timesteps", []),
                            "reference": data["rmse"].get("reference_series", {}),
                            "optimized": data["rmse"].get("optimized_series", {}),
                            "fixed_timestep_metrics": data["rmse"].get(
                                "fixed_timestep_metrics", {}
                            ),
                            "horizontal_gain": data["rmse"].get("horizontal_gain", {}),
                        }
                        for region, data in summary.items()
                    },
                },
                f,
                indent=2,
            )
        with open(psd_path, "w") as f:
            json.dump(
                {
                    "run_id": self.exp_dir.name,
                    "regions": {
                        region: data["psd"].get("curves", {})
                        for region, data in summary.items()
                    },
                },
                f,
                indent=2,
            )

        with open(self.metrics_dir / "rmse_evolution.csv", "w", newline="") as f:
            fields = ["run_id", "region", "timestep", "variable", "reference_rmse", "optimized_rmse"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            run_id = self.run_id
            for region, data in summary.items():
                rmse = data["rmse"]
                timesteps = rmse.get("timesteps", [])
                reference = rmse.get("reference_series", {})
                optimized = rmse.get("optimized_series", {})
                for idx, timestep in enumerate(timesteps):
                    for variable in reference:
                        writer.writerow({
                            "run_id": run_id,
                            "region": region,
                            "timestep": timestep,
                            "variable": variable,
                            "reference_rmse": reference[variable][idx],
                            "optimized_rmse": optimized[variable][idx],
                        })

        with open(self.metrics_dir / "psd_analysis.csv", "w", newline="") as f:
            fields = ["run_id", "region", "variable", "wavenumber", "reference_psd",
                      "optimized_psd", "ground_truth_psd"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            run_id = self.run_id
            for region, curves in (
                (region, data["psd"].get("curves", {}))
                for region, data in summary.items()
            ):
                for variable, curve in curves.items():
                    k_values = curve.get("wavenumber", [])
                    for idx, wavenumber in enumerate(k_values):
                        writer.writerow({
                            "run_id": run_id,
                            "region": region,
                            "variable": variable,
                            "wavenumber": wavenumber,
                            "reference_psd": curve.get("reference", [])[idx],
                            "optimized_psd": curve.get("optimized", [])[idx],
                            "ground_truth_psd": curve.get("ground_truth", [])[idx],
                        })

    @staticmethod
    def _compute_windowed_psd_grid(
        field: np.ndarray,
        mask: np.ndarray,
        dlat: float,
        dlon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a shifted 2D PSD grid for one global field."""
        valid = mask > 0
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([[]])

        centered = np.where(valid, field - np.nanmean(field[valid]), 0.0)
        window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
        power = np.fft.fftshift(np.abs(np.fft.fft2(centered * window)) ** 2)
        ky = np.fft.fftshift(np.fft.fftfreq(field.shape[0], d=dlat))
        kx = np.fft.fftshift(np.fft.fftfreq(field.shape[1], d=dlon))
        return kx, ky, power

    def _save_global_psd_ebcr(
        self,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
        input_sequence: xr.Dataset,
    ) -> Optional[Dict]:
        """
        Compute and save EBCR at every available 7-step forecast checkpoint.

        Forecast arrays are indexed from zero, so physical timestep t is stored
        at index t - 1. The visualization and saved numerical data use radial
        PSD curves in wavelength space for efficient multi-run aggregation.
        """
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        truth = ground_truth_sequence.squeeze(0).detach().cpu().numpy()
        mask = ocean_mask.detach().cpu().numpy()
        horizon = min(ref.shape[0], opt.shape[0], truth.shape[0])
        timesteps = list(range(self.PSD_ANALYSIS_TIMESTEP, horizon + 1, 7))
        if not timesteps:
            logger.warning(
                "Skipping EBCR analysis: forecast horizon %d is shorter than "
                "the first timestep %d.",
                horizon,
                self.PSD_ANALYSIS_TIMESTEP,
            )
            return None

        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        dlat = float(np.abs(lat[1] - lat[0])) if lat.size > 1 else 0.25
        dlon = float(np.abs(lon[1] - lon[0])) if lon.size > 1 else 0.25

        def _radial_curve(
            kx: np.ndarray, ky: np.ndarray, power: np.ndarray, n_bins: int = 40
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Reduce a 2D spectrum to a common logarithmic wavenumber curve."""
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            radial = np.sqrt(kx_grid**2 + ky_grid**2)
            positive = radial > 0
            if not np.any(positive):
                return np.array([]), np.array([])
            bins = np.logspace(
                np.log10(radial[positive].min()),
                np.log10(radial.max()),
                n_bins + 1,
            )
            centers = np.sqrt(bins[:-1] * bins[1:])
            curve = np.full(n_bins, np.nan, dtype=np.float64)
            for bin_idx in range(n_bins):
                selected = (radial >= bins[bin_idx]) & (radial < bins[bin_idx + 1])
                if np.any(selected):
                    curve[bin_idx] = np.nanmean(power[selected])
            keep = np.isfinite(curve) & (curve > 0)
            # Wavelength is more useful than wavenumber when comparing runs.
            return 1.0 / centers[keep][::-1], curve[keep][::-1]

        ebcr_data = {"timesteps": {}, "plots": {}}
        ebcr_rows = []
        psd_rows = []

        for timestep in timesteps:
            forecast_idx = timestep - 1
            figure = plt.figure(figsize=(15, 18), constrained_layout=True)
            axes = figure.subplots(5, 2)
            timestep_data = {}
            for ch_idx in range(len(self.VAR_METADATA)):
                psd_ax = axes[ch_idx, 0]
                ebcr_ax = axes[ch_idx, 1]
                var_name, _, long_name = self.VAR_METADATA[ch_idx]
                kx, ky, psd_ref_grid = self._compute_windowed_psd_grid(
                    ref[forecast_idx, ch_idx], mask[ch_idx], dlat, dlon
                )
                _, _, psd_opt_grid = self._compute_windowed_psd_grid(
                    opt[forecast_idx, ch_idx], mask[ch_idx], dlat, dlon
                )
                _, _, psd_truth_grid = self._compute_windowed_psd_grid(
                    truth[forecast_idx, ch_idx], mask[ch_idx], dlat, dlon
                )
                if psd_ref_grid.size == 0:
                    psd_ax.set_title(f"{long_name} | no valid ocean pixels")
                    psd_ax.axis("off")
                    ebcr_ax.axis("off")
                    timestep_data[var_name] = {"wavelength": [], "psd": {}, "ebcr": []}
                    continue

                wavelength, psd_ref = _radial_curve(kx, ky, psd_ref_grid)
                _, psd_opt = _radial_curve(kx, ky, psd_opt_grid)
                _, psd_truth = _radial_curve(kx, ky, psd_truth_grid)
                # Compute EBCR after radial averaging so negative ratios are
                # preserved instead of being discarded as invalid PSD values.
                ebcr = PSDComputer.compute_ebcr(psd_ref, psd_opt, psd_truth)
                psd_ax.loglog(wavelength, psd_ref + 1e-30, color="tab:blue", label="Reference")
                psd_ax.loglog(wavelength, psd_opt + 1e-30, color="tab:orange", label="Optimized")
                psd_ax.loglog(
                    wavelength,
                    psd_truth + 1e-30,
                    color="tab:green",
                    linestyle="--",
                    label="Truth",
                )
                psd_ax.set_ylabel("Radially averaged PSD")
                psd_ax.set_title(f"{var_name}: {long_name}")
                psd_ax.grid(True, which="both", alpha=0.3)
                if ch_idx == 0:
                    psd_ax.legend(loc="best")

                finite_ebcr = np.isfinite(ebcr)
                if np.any(finite_ebcr):
                    ebcr_ax.semilogx(
                        wavelength[finite_ebcr],
                        ebcr[finite_ebcr],
                        color="tab:purple",
                        linewidth=1.8,
                    )
                ebcr_ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=1.0)
                ebcr_ax.axhline(1.0, color="tab:blue", linestyle=":", linewidth=1.0)
                ebcr_ax.set_ylabel("EBCR")
                ebcr_ax.set_title(f"{var_name}: radial EBCR")
                ebcr_ax.grid(True, which="both", alpha=0.3)
                timestep_data[var_name] = {
                    "wavelength_deg": wavelength.tolist(),
                    "wavenumber_cycles_per_deg": (
                        (1.0 / wavelength).tolist() if wavelength.size else []
                    ),
                    "psd": {
                        "reference": psd_ref.tolist(),
                        "optimized": psd_opt.tolist(),
                        "truth": psd_truth.tolist(),
                    },
                    "ebcr": ebcr.tolist(),
                }
                for idx, wavelength_value in enumerate(wavelength):
                    psd_rows.append({
                        "run_id": self.run_id,
                        "timestep": timestep,
                        "variable": var_name,
                        "wavelength_deg": float(wavelength_value),
                        "wavenumber_cycles_per_deg": float(1.0 / wavelength_value),
                        "psd_reference": float(psd_ref[idx]),
                        "psd_optimized": float(psd_opt[idx]),
                        "psd_truth": float(psd_truth[idx]),
                    })
                    ebcr_rows.append({
                        "run_id": self.run_id,
                        "timestep": timestep,
                        "variable": var_name,
                        "wavelength_deg": float(wavelength_value),
                        "ebcr": (
                            float(ebcr[idx])
                            if idx < len(ebcr) and np.isfinite(ebcr[idx])
                            else ""
                        ),
                    })

            axes[-1, 0].set_xlabel("Wavelength (degrees)")
            axes[-1, 1].set_xlabel("Wavelength (degrees)")
            figure.suptitle(
                f"Global radially averaged PSD and EBCR | t={timestep}",
                fontsize=14,
            )
            plot_path = self.diagnostics_dir / f"psd_ebcr_global_t{timestep}.png"
            figure.savefig(plot_path, dpi=200)
            plt.close(figure)
            ebcr_data["timesteps"][str(timestep)] = timestep_data
            ebcr_data["plots"][str(timestep)] = str(plot_path.relative_to(self.exp_dir))

        metadata = {
            "run_id": self.run_id,
            "timesteps": timesteps,
            "definition": "(psd_optimized - psd_truth) / (psd_reference - psd_truth)",
            "classification": {
                "flipped": "EBCR < 0",
                "improved": "0 <= EBCR <= 1",
                "worse": "EBCR > 1",
                "undefined": "reference PSD equals truth PSD",
            },
            "grid": {"dlat_deg": dlat, "dlon_deg": dlon},
            "analysis": ebcr_data,
        }
        with open(self.metrics_dir / "psd_ebcr_timesteps.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(self.metrics_dir / "psd_ebcr_timesteps.csv", "w", newline="") as f:
            fields = [
                "run_id", "timestep", "variable", "wavelength_deg",
                "wavenumber_cycles_per_deg", "ebcr",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(ebcr_rows)
        with open(self.metrics_dir / "psd_wavelength_timesteps.csv", "w", newline="") as f:
            fields = [
                "run_id", "timestep", "variable", "wavelength_deg",
                "wavenumber_cycles_per_deg",
                "psd_reference", "psd_optimized", "psd_truth",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(psd_rows)
        with open(self.metrics_dir / "psd_wavelength_timesteps.json", "w") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "timesteps": timesteps,
                    "domain": "wavelength (degrees) and wavenumber (cycles per degree)",
                    "variables": ebcr_data["timesteps"],
                },
                f,
                indent=2,
            )
        logger.info("Saved global EBCR and wavelength-domain PSD analyses.")
        return metadata

    def _save_local_patch_ebcr(
        self,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        ground_truth_sequence: torch.Tensor,
        ocean_mask: torch.Tensor,
        input_sequence: xr.Dataset,
    ) -> Optional[Dict]:
        """
        Compute local-patch effective resolution from EBCR.

        Patches are approximately 200 km wide. Within each patch, wavenumbers
        are checked from finest to coarser scales and the smallest wavenumber
        with EBCR < 1 is converted to a wavelength for global mapping.
        """
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()
        truth = ground_truth_sequence.squeeze(0).detach().cpu().numpy()
        mask = ocean_mask.detach().cpu().numpy() > 0
        horizon = min(ref.shape[0], opt.shape[0], truth.shape[0])
        timesteps = list(range(self.PSD_ANALYSIS_TIMESTEP, horizon + 1, 7))
        if not timesteps:
            logger.warning("Skipping local EBCR: forecast horizon is shorter than t=7.")
            return None

        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        dlat = float(np.abs(lat[1] - lat[0])) if lat.size > 1 else 0.25
        dlon = float(np.abs(lon[1] - lon[0])) if lon.size > 1 else 0.25
        km_per_degree = 111.2
        patch_km = 200.0
        patch_height = max(2, int(round(patch_km / (km_per_degree * dlat))))

        def _radial_curve(
            kx: np.ndarray, ky: np.ndarray, power: np.ndarray, n_bins: int = 30
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Return wavelength-sorted radial PSD values for one patch."""
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            radial = np.sqrt(kx_grid**2 + ky_grid**2)
            positive = radial > 0
            if not np.any(positive):
                return np.array([]), np.array([])
            bins = np.logspace(
                np.log10(radial[positive].min()),
                np.log10(radial.max()),
                n_bins + 1,
            )
            centers = np.sqrt(bins[:-1] * bins[1:])
            values = np.full(n_bins, np.nan, dtype=np.float64)
            for bin_idx in range(n_bins):
                selected = (radial >= bins[bin_idx]) & (radial < bins[bin_idx + 1])
                if np.any(selected):
                    values[bin_idx] = np.nanmean(power[selected])
            keep = np.isfinite(values) & (values > 0)
            return 1.0 / centers[keep], values[keep]

        def _patch_psd(field: np.ndarray, patch_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Compute a radial PSD curve inside one local patch."""
            if np.count_nonzero(patch_mask) < 4:
                return np.array([]), np.array([])
            return _radial_curve(
                *self._compute_windowed_psd_grid(
                    field, patch_mask, dlat, dlon
                )
            )  # type: ignore[arg-type]

        all_results = {"timesteps": {}, "patch_size_km": patch_km}
        csv_rows = []
        for timestep in timesteps:
            timestep_idx = timestep - 1
            timestep_results = {}
            for ch_idx, (var_name, _, _) in self.VAR_METADATA.items():
                patch_results = []
                figure, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
                value_grid = np.full((len(lat), len(lon)), np.nan, dtype=np.float64)
                for y0 in range(0, len(lat), patch_height):
                    y1 = min(y0 + patch_height, len(lat))
                    center_lat = float(np.mean(lat[y0:y1]))
                    patch_width = max(
                        2,
                        int(round(patch_km / (
                            km_per_degree * max(np.cos(np.deg2rad(center_lat)), 0.1) * dlon
                        ))),
                    )
                    for x0 in range(0, len(lon), patch_width):
                        x1 = min(x0 + patch_width, len(lon))
                        patch_mask = mask[ch_idx, y0:y1, x0:x1]
                        k_ref, psd_ref = _patch_psd(
                            ref[timestep_idx, ch_idx, y0:y1, x0:x1], patch_mask
                        )
                        k_opt, psd_opt = _patch_psd(
                            opt[timestep_idx, ch_idx, y0:y1, x0:x1], patch_mask
                        )
                        k_truth, psd_truth = _patch_psd(
                            truth[timestep_idx, ch_idx, y0:y1, x0:x1], patch_mask
                        )
                        if (
                            k_ref.size == 0
                            or psd_opt.size == 0
                            or psd_truth.size == 0
                            or k_opt.size == 0
                            or k_truth.size == 0
                        ):
                            continue
                        # Different fields can lose different empty radial
                        # bins; compare all spectra on the reference grid.
                        if psd_opt.size != k_ref.size or not np.allclose(k_opt, k_ref):
                            psd_opt = np.interp(
                                k_ref[::-1],
                                k_opt[::-1],
                                psd_opt[::-1],
                                left=np.nan,
                                right=np.nan,
                            )[::-1]
                        if psd_truth.size != k_ref.size or not np.allclose(k_truth, k_ref):
                            psd_truth = np.interp(
                                k_ref[::-1],
                                k_truth[::-1],
                                psd_truth[::-1],
                                left=np.nan,
                                right=np.nan,
                            )[::-1]
                        ebcr = PSDComputer.compute_ebcr(psd_ref, psd_opt, psd_truth)
                        wavelength_km = k_ref * km_per_degree
                        valid = (
                            np.isfinite(ebcr)
                            & (ebcr < 1.0)
                            & (wavelength_km >= self.EFFECTIVE_WAVELENGTH_MIN_KM)
                            & (wavelength_km <= self.EFFECTIVE_WAVELENGTH_MAX_KM)
                        )
                        if not np.any(valid):
                            continue

                        # The effective cutoff is the smallest wavenumber that
                        # still satisfies EBCR < 1.
                        wavenumber = 1.0 / k_ref
                        selected = np.flatnonzero(valid)
                        selected = selected[np.argmin(wavenumber[selected])]
                        effective_wavelength_deg = float(k_ref[selected])
                        effective_wavelength_km = float(wavelength_km[selected])
                        center_y = min((y0 + y1) // 2, len(lat) - 1)
                        center_x = min((x0 + x1) // 2, len(lon) - 1)
                        # Fill the full patch so neighboring patch values are
                        # readable at global scale instead of appearing as dots.
                        value_grid[y0:y1, x0:x1] = effective_wavelength_km
                        patch_result = {
                            "row": int(center_y),
                            "column": int(center_x),
                            "latitude": float(lat[center_y]),
                            "longitude": float(lon[center_x]),
                            "wavelength_km": effective_wavelength_km,
                            "wavelength_deg": effective_wavelength_deg,
                            "wavenumber_cycles_per_deg": float(1.0 / effective_wavelength_deg),
                        }
                        patch_results.append(patch_result)
                        csv_rows.append({
                            "run_id": self.run_id,
                            "timestep": timestep,
                            "variable": var_name,
                            **patch_result,
                        })

                cmap = plt.get_cmap("RdBu_r").copy()
                cmap.set_bad("#d3d3d3")
                image = ax.imshow(
                    np.ma.masked_invalid(value_grid),
                    origin="lower" if lat[0] < lat[-1] else "upper",
                    extent=(
                        float(lon.min()),
                        float(lon.max()),
                        float(lat.min()),
                        float(lat.max()),
                    ),
                    aspect="auto",
                    cmap=cmap,
                    vmin=self.EFFECTIVE_WAVELENGTH_MIN_KM,
                    vmax=self.EFFECTIVE_WAVELENGTH_MAX_KM,
                    interpolation="none",
                )
                figure.colorbar(image, ax=ax, label="Effective wavelength (km)")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(f"Local-patch effective resolution | {var_name} | t={timestep}")
                plot_path = self.diagnostics_dir / (
                    f"psd_local_patch_ebcr_{var_name}_t{timestep}.png"
                )
                figure.savefig(plot_path, dpi=200)
                plt.close(figure)
                timestep_results[var_name] = {
                    "plot": str(plot_path.relative_to(self.exp_dir)),
                    "patches": patch_results,
                }
            all_results["timesteps"][str(timestep)] = timestep_results

        metadata = {
            "run_id": self.run_id,
            "timesteps": timesteps,
            "patch_size_km": patch_km,
            "wavelength_limits_km": {
                "minimum": self.EFFECTIVE_WAVELENGTH_MIN_KM,
                "maximum": self.EFFECTIVE_WAVELENGTH_MAX_KM,
            },
            "definition": "(psd_optimized - psd_truth) / (psd_reference - psd_truth)",
            "selection": (
                "smallest wavenumber with EBCR < 1, scanning from finest "
                "to coarser scales, restricted to 65-500 km wavelength"
            ),
            "grid": {"dlat_deg": dlat, "dlon_deg": dlon},
            "analysis": all_results,
        }
        with open(self.metrics_dir / "psd_local_patch_ebcr.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(self.metrics_dir / "psd_local_patch_ebcr.csv", "w", newline="") as f:
            fields = [
                "run_id", "timestep", "variable", "row", "column",
                "latitude", "longitude", "wavelength_km", "wavelength_deg",
                "wavenumber_cycles_per_deg",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Saved local-patch EBCR effective-resolution analysis.")
        return metadata

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
        mean_field: Optional[torch.Tensor] = None,
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
                mean_field=mean_field,
            )

            # Save diagnostics as visualization files (no NetCDF)
            self._save_ic_correction_figure(
                x0_init=x0_init,
                x0_optimized=x0_optimized,
                input_sequence=input_sequence,
                mean_field=mean_field,
            )
            ebcr_summary = self._save_local_patch_ebcr(
                reference_forecast=reference_forecast,
                optimized_forecast=full_forecast,
                ground_truth_sequence=ground_truth_sequence,
                ocean_mask=ocean_mask,
                input_sequence=input_sequence,
            )

            # Determine RMSE horizon from available forecasts/ground-truth (no separate global rmse figure)
            rmse_horizon = int(
                min(
                    int(reference_forecast.shape[1]),
                    int(full_forecast.shape[1]),
                    int(ground_truth_sequence.shape[1]),
                )
            )

            regional_summary = {}
            if regional_masks is not None:
                regional_summary = self._save_regional_diagnostics(
                    x0_init=x0_init,
                    x0_optimized=x0_optimized,
                    reference_forecast=reference_forecast,
                    optimized_forecast=full_forecast,
                    ground_truth_sequence=ground_truth_sequence,
                    regional_masks=regional_masks,
                    ocean_mask=ocean_mask,
                    input_sequence=input_sequence,
                    observation_length=int(target_sequence.shape[1]),
                )
                self._save_metrics_to_csv_and_json(regional_summary)
                self._save_gulf_stream_animation(
                    state_dir=state_dir,
                    reference_forecast=reference_forecast,
                    optimized_forecast=full_forecast,
                    input_sequence=input_sequence,
                    gulf_stream_mask=regional_masks["gulf_stream"],
                )
            else:
                # Keep the global PSD plot as a fallback when no regional masks are available.
                self._save_psd_figure(
                    reference_forecast=reference_forecast,
                    optimized_forecast=full_forecast,
                    ground_truth_sequence=ground_truth_sequence,
                    ocean_mask=ocean_mask,
                )

            summary_path = self.diagnostics_dir / "diagnostic_summary.json"
            with open(summary_path, "w") as f:
                json.dump(
                    {
                        "best_iteration": best_iteration,
                        "best_loss": float(best_loss),
                        "rmse_horizon": rmse_horizon,
                        "files": {
                            "ic_correction_plot": "diagnostics/ic_correction_comparison.png",
                            "ic_correction_anomaly_plot": "diagnostics/ic_correction_anomaly_comparison.png",
                            "regional_rmse_plot": "diagnostics/rmse_region_*.png",
                            "regional_psd_plot": "diagnostics/psd_region_*.png",
                            "regional_metrics_json": "metrics/regional_metrics.json",
                            "metrics_fixed_timesteps_csv": "metrics/rmse_metrics_fixed_timesteps.csv",
                            "metrics_horizontal_gain_csv": "metrics/rmse_metrics_horizontal_gain.csv",
                            "metrics_summary_json": "metrics/rmse_metrics_summary.json",
                            "rmse_evolution_json": "metrics/rmse_evolution.json",
                            "rmse_evolution_csv": "metrics/rmse_evolution.csv",
                            "psd_analysis_json": "metrics/psd_analysis.json",
                            "psd_analysis_csv": "metrics/psd_analysis.csv",
                            "psd_local_patch_ebcr_json": (
                                "metrics/psd_local_patch_ebcr.json"
                                if ebcr_summary is not None
                                else None
                            ),
                            "psd_local_patch_ebcr_csv": (
                                "metrics/psd_local_patch_ebcr.csv"
                                if ebcr_summary is not None
                                else None
                            ),
                            "psd_local_patch_ebcr_plots": (
                                "diagnostics/psd_local_patch_ebcr_{variable}_t{7,14,21,...}.png"
                                if ebcr_summary is not None
                                else None
                            ),
                            "forecast_animation": "states/forecast_comparison.gif",
                            "forecast_animation_anomaly": "states/forecast_comparison_anomaly.gif",
                            "gulf_stream_animation": "states/gulf_stream_forecast_comparison.gif",
                        },
                        "psd_plot": None if regional_masks is not None else "diagnostics/psd_comparison.png",
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

    def _to_anomaly(self, data: np.ndarray, mean_field: Optional[torch.Tensor]) -> np.ndarray:
        """Subtract the mean field when available to form anomaly fields."""
        if mean_field is None:
            return data
        mean_np = mean_field.detach().cpu().numpy()
        return data - mean_np[None, :, :, :]

    def _save_forecast_animation(
        self,
        state_dir: Path,
        reference_forecast: torch.Tensor,
        optimized_forecast: torch.Tensor,
        input_sequence: xr.Dataset,
        mean_field: Optional[torch.Tensor] = None,
    ) -> None:
        """Save side-by-side forecast animation (reference vs optimized) as GIF."""
        ref = reference_forecast.squeeze(0).detach().cpu().numpy()   # [T, C, H, W]
        opt = optimized_forecast.squeeze(0).detach().cpu().numpy()   # [T, C, H, W]
        ref_anom = self._to_anomaly(ref, mean_field)
        opt_anom = self._to_anomaly(opt, mean_field)
        horizon = min(ref.shape[0], opt.shape[0])
        ref = ref[:horizon]
        opt = opt[:horizon]
        ref_anom = ref_anom[:horizon]
        opt_anom = opt_anom[:horizon]

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
                extent=extent, origin=origin, aspect='auto', interpolation="none"
            )
            im_opt = axes[ch_idx, 1].imshow(
                opt[0, ch_idx], cmap="viridis", vmin=-vmax, vmax=vmax, animated=True,
                extent=extent, origin=origin, aspect='auto', interpolation="none"
            )
            # Add colorbars for reference and optimized panels so animation color scale is visible
            try:
                fig.colorbar(im_ref, ax=axes[ch_idx, 0], fraction=0.046, pad=0.02)
                fig.colorbar(im_opt, ax=axes[ch_idx, 1], fraction=0.046, pad=0.02)
            except Exception:
                # Fallback: ignore colorbar errors (rare in headless environments)
                pass
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

        fig, axes = plt.subplots(5, 2, figsize=(10, 16), constrained_layout=True)
        axes[0, 0].set_title("Reference anomaly", fontsize=11, fontweight="bold")
        axes[0, 1].set_title("Optimized anomaly", fontsize=11, fontweight="bold")

        images = []
        for ch_idx in range(5):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            merged = np.concatenate([ref_anom[:, ch_idx].ravel(), opt_anom[:, ch_idx].ravel()])
            vmax = np.nanpercentile(np.abs(merged), 99)
            vmax = max(vmax, 1e-12)

            extent, origin = self._get_extent_origin(input_sequence)
            im_ref = axes[ch_idx, 0].imshow(
                ref_anom[0, ch_idx],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                animated=True,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )
            im_opt = axes[ch_idx, 1].imshow(
                opt_anom[0, ch_idx],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                animated=True,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )
            try:
                fig.colorbar(im_ref, ax=axes[ch_idx, 0], fraction=0.046, pad=0.02)
                fig.colorbar(im_opt, ax=axes[ch_idx, 1], fraction=0.046, pad=0.02)
            except Exception:
                pass
            images.append((im_ref, im_opt))

            axes[ch_idx, 0].set_ylabel(f"{var_name}\n{long_name}", fontsize=9)
            axes[ch_idx, 0].set_xticks([])
            axes[ch_idx, 0].set_yticks([])
            axes[ch_idx, 1].set_xticks([])
            axes[ch_idx, 1].set_yticks([])

        title = fig.suptitle("Forecast anomaly comparison | timestep 0", fontsize=13)

        def _update_anom(frame_idx: int):
            title.set_text(f"Forecast anomaly comparison | timestep {frame_idx}")
            artists = [title]
            for ch_idx in range(5):
                im_ref, im_opt = images[ch_idx]
                im_ref.set_data(ref_anom[frame_idx, ch_idx])
                im_opt.set_data(opt_anom[frame_idx, ch_idx])
                artists.extend([im_ref, im_opt])
            return artists

        anim = animation.FuncAnimation(
            fig=fig,
            func=_update_anom,
            frames=horizon,
            interval=500,
            blit=False,
            repeat=True,
        )

        out_path = state_dir / "forecast_comparison_anomaly.gif"
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=2))
            logger.info(f"✓ Saved forecast anomaly animation: {out_path}")
        finally:
            plt.close(fig)

    def _save_ic_correction_figure(
        self,
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
        input_sequence: xr.Dataset,
        mean_field: Optional[torch.Tensor] = None,
    ) -> None:
        """Save IC state-map visualization (raw and anomaly views)."""
        init_np = x0_init.squeeze(0).detach().cpu().numpy()      # [T=2, C, H, W]
        opt_np = x0_optimized.squeeze(0).detach().cpu().numpy()  # [T=2, C, H, W]
        delta_np = opt_np - init_np
        init_anom = self._to_anomaly(init_np, mean_field)
        opt_anom = self._to_anomaly(opt_np, mean_field)
        delta_anom = opt_anom - init_anom
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
                interpolation="none",
            )
            im1 = axes[ch_idx, 1].imshow(
                opt_field,
                cmap="viridis",
                vmin=-vmax_state,
                vmax=vmax_state,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )
            im2 = axes[ch_idx, 2].imshow(
                delta_field,
                cmap="RdBu_r",
                vmin=-vmax_delta,
                vmax=vmax_delta,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
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

        fig, axes = plt.subplots(5, 3, figsize=(15, 18), constrained_layout=True)
        col_titles = ["Initial anomaly", "Optimized anomaly", "Correction anomaly"]
        for col_idx, title in enumerate(col_titles):
            axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")

        for ch_idx in range(5):
            var_name, _, long_name = self.VAR_METADATA[ch_idx]
            init_field = init_anom[t_idx, ch_idx]
            opt_field = opt_anom[t_idx, ch_idx]
            delta_field = delta_anom[t_idx, ch_idx]

            vmax_state = np.nanpercentile(np.abs(np.concatenate([init_field.ravel(), opt_field.ravel()])), 99)
            vmax_state = max(vmax_state, 1e-12)
            vmax_delta = np.nanpercentile(np.abs(delta_field), 99)
            vmax_delta = max(vmax_delta, 1e-12)

            im0 = axes[ch_idx, 0].imshow(
                init_field,
                cmap="RdBu_r",
                vmin=-vmax_state,
                vmax=vmax_state,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )
            im1 = axes[ch_idx, 1].imshow(
                opt_field,
                cmap="RdBu_r",
                vmin=-vmax_state,
                vmax=vmax_state,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )
            im2 = axes[ch_idx, 2].imshow(
                delta_field,
                cmap="RdBu_r",
                vmin=-vmax_delta,
                vmax=vmax_delta,
                extent=extent,
                origin=origin,
                aspect="auto",
                interpolation="none",
            )

            axes[ch_idx, 0].set_ylabel(f"{var_name}\n{long_name}", fontsize=9)
            for col in range(3):
                axes[ch_idx, col].set_xticks([])
                axes[ch_idx, col].set_yticks([])

            fig.colorbar(im0, ax=axes[ch_idx, 0], fraction=0.046, pad=0.02)
            fig.colorbar(im1, ax=axes[ch_idx, 1], fraction=0.046, pad=0.02)
            fig.colorbar(im2, ax=axes[ch_idx, 2], fraction=0.046, pad=0.02)

        fig.suptitle("IC Correction Anomaly Visualization (latest IC timestep)", fontsize=14)
        out_path = self.diagnostics_dir / "ic_correction_anomaly_comparison.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info(f"✓ Saved IC correction anomaly figure: {out_path}")

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
        x0_init: torch.Tensor,
        x0_optimized: torch.Tensor,
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
        # Convert initial conditions to numpy to compute initial RMSE consistently with global plot
        x0_ref = x0_init.squeeze(0).detach().cpu().numpy()  # [T=2, C, H, W]
        x0_opt = x0_optimized.squeeze(0).detach().cpu().numpy()  # [T=2, C, H, W]
        regions = ["global", "gulf_stream", "high_var", "low_var"]
        var_names = [self.VAR_METADATA[c][0] for c in range(ref.shape[1])]
        lat = input_sequence.coords["lat"].values
        lon = input_sequence.coords["lon"].values
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)

        def _region_mask(region_name: str) -> np.ndarray:
            mask = regional_masks[region_name] > 0
            return mask

        def _rmse_series(arr: np.ndarray, truth: np.ndarray, mask: np.ndarray, init_arr: np.ndarray = None) -> np.ndarray:
            """Compute RMSE series including initial condition at index 0.

            If init_arr is provided, its last timestep (init_arr[-1]) is used as the
            initial state to compute the RMSE at t=0 — matching the global RMSE plot.
            Otherwise falls back to using arr[0] as before.
            """
            horizon = min(arr.shape[0], truth.shape[0])
            series = np.full((horizon + 1, arr.shape[1]), np.nan, dtype=np.float64)
            for c_idx in range(arr.shape[1]):
                valid = mask[c_idx]
                if np.any(valid):
                    if init_arr is not None:
                        init_field = init_arr[-1, c_idx]
                    else:
                        init_field = arr[0, c_idx]
                    init_err = init_field - truth[0, c_idx]
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
            ref_series = _rmse_series(ref, gt, mask, init_arr=x0_ref)
            opt_series = _rmse_series(opt, gt, mask, init_arr=x0_opt)
            horizon = ref_series.shape[0] - 1
            step_axis = np.arange(horizon + 1)
            fixed_ts_metrics = self._compute_fixed_timestep_metrics(
                ref_series, opt_series, horizon, var_names
            )
            horizontal_gain_metrics = self._compute_horizontal_gain(
                ref_series, opt_series, observation_length, var_names
            )
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            for ch_idx, ax in enumerate(axes):
                var_name, units, long_name = self.VAR_METADATA[ch_idx]
                ax.plot(step_axis, ref_series[:, ch_idx], color="tab:blue", linewidth=2, label="Reference")
                ax.plot(step_axis, opt_series[:, ch_idx], color="tab:orange", linewidth=2, label="Optimized")
                boundary_idx = min(max(int(observation_length), 0), horizon)
                boundary_rmse = ref_series[boundary_idx, ch_idx]
                crossing_idx = horizontal_gain_metrics[var_name]["optimized_crossing_timestep"]
                ax.axhline(
                    boundary_rmse,
                    color="tab:blue",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Reference persistence" if ch_idx == 0 else None,
                )
                ax.plot(boundary_idx, boundary_rmse, "bo", markersize=5, alpha=0.8)
                if np.isfinite(crossing_idx):
                    gain_steps = horizontal_gain_metrics[var_name]["horizontal_gain_steps"]
                    ax.plot(
                        [boundary_idx, crossing_idx],
                        [boundary_rmse, boundary_rmse],
                        color="tab:red",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.8,
                        label="Predictability gain" if ch_idx == 0 else None,
                    )
                    ax.plot(crossing_idx, boundary_rmse, "ro", markersize=5, alpha=0.8)
                    ax.annotate(
                        f"Gain: {gain_steps:.2f} steps",
                        xy=((boundary_idx + crossing_idx) / 2.0, boundary_rmse),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        fontsize=9,
                        color="tab:red",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
                    )
                ax.axvline(
                    boundary_idx,
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
            fig.suptitle(f"RMSE Evolution with Predictability Gain | {region_name}", fontsize=13)
            out_path = self.diagnostics_dir / f"rmse_region_{region_name}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            return {
                "reference": {var: float(ref_series[-1, idx]) for idx, var in enumerate(var_names)},
                "optimized": {var: float(opt_series[-1, idx]) for idx, var in enumerate(var_names)},
                "file": str(out_path.relative_to(self.exp_dir)),
                "timesteps": step_axis.tolist(),
                "reference_series": {
                    var: ref_series[:, idx].tolist() for idx, var in enumerate(var_names)
                },
                "optimized_series": {
                    var: opt_series[:, idx].tolist() for idx, var in enumerate(var_names)
                },
                "fixed_timestep_metrics": fixed_ts_metrics,
                "horizontal_gain": horizontal_gain_metrics,
            }

        def _save_psd_plot(region_name: str, mask: np.ndarray) -> Dict[str, Dict[str, float]]:
            fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
            summary = {"reference": {}, "optimized": {}, "ground_truth": {}}
            curves = {}
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
                curves[var_name] = {
                    "wavenumber": k_ref.tolist() if k_ref.size > 0 else [],
                    "reference": p_ref.tolist() if k_ref.size == p_ref.size else [],
                    "optimized": p_opt.tolist() if k_ref.size == p_opt.size else [],
                    "ground_truth": p_gt.tolist() if k_ref.size == p_gt.size else [],
                }
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
            summary["curves"] = curves
            return summary

        summary = {}
        for region_name in regions:
            mask = _region_mask(region_name)
            summary[region_name] = {
                "rmse": _save_rmse_plot(region_name, mask),
                "psd": _save_psd_plot(region_name, mask),
            }

        reg_json = self.metrics_dir / "regional_metrics.json"
        with open(reg_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved regional diagnostics: {reg_json}")
        self._save_diagnostic_data_exports(summary)
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
                interpolation="none",
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
                interpolation="none",
                animated=True,
            )
            # Add colorbars for gulf stream cropped panels
            try:
                fig.colorbar(im_ref, ax=axes[ch_idx, 0], fraction=0.046, pad=0.02)
                fig.colorbar(im_opt, ax=axes[ch_idx, 1], fraction=0.046, pad=0.02)
            except Exception:
                pass
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
