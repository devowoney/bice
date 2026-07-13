"""
Data loading and observation operators for IC optimization.

Provides:
    - GlonetDataset: Load GLORYS12 init states and observations
    - ObservationOperator: Apply observation operators (SSH along-track, SST gridded)
"""

import xarray as xr
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xesmf as xe


class GlonetDataset:
    """
    Dataset class for loading GLORYS12 initial states and observations.

    Handles:
    - Loading multi-file GLORYS12 datasets
    - Time slicing for input/target sequences
    - Lazy loading with xarray
    """

    def __init__(
        self,
        data_path: str,
        data_files: List[str],
        ssh_obs_files: List[str] = None,
        sst_obs_files: List[str] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Root path to data directory
            data_files: List of GLORYS12 init state files (relative to data_path)
            ssh_obs_files: List of SSH observation files (optional)
            sst_obs_files: List of SST observation files (optional)
            lazy_load: If True, use dask chunks for lazy loading
        """
        self.data_path = Path(data_path)
        self.data_files = data_files
        self.ssh_obs_files = ssh_obs_files
        self.sst_obs_files = sst_obs_files
        self.lazy_load = lazy_load

        # Load datasets
        self.dataset = self._load_dataset(self.data_files)

        # Load observations if provided
        self.ssh_obs_dataset = None
        self.sst_obs_dataset = None

        if ssh_obs_files is not None:
            self.ssh_obs_dataset = self._load_ssh_observations(ssh_obs_files)

        if sst_obs_files is not None:
            self.sst_obs_dataset = self._load_sst_observations(sst_obs_files)

    def _load_dataset(self, file_list: List[str]) -> xr.Dataset:
        """
        Load and concatenate GLORYS12 datasets.

        Args:
            file_list: List of file paths relative to data_path

        Returns:
            dataset: xarray Dataset with concatenated data
        """
        datasets = []

        for fname in file_list:
            file_path = self.data_path / fname

            if self.lazy_load:
                # Use chunks for lazy loading
                ds = xr.open_dataset(file_path, chunks={"time": 10, "lat": 336, "lon": 720})
            else:
                ds = xr.open_dataset(file_path)

            datasets.append(ds)

        # Concatenate along time dimension
        combined = xr.concat(datasets, dim="time", combine_attrs="override").sortby("time")

        return combined

    def _load_ssh_observations(self, file_list: List[str]) -> xr.Dataset:
        """
        Load SSH observations (altimetry).

        Args:
            file_list: List of altimetry file paths

        Returns:
            dataset: xarray Dataset with SSH observations
        """
        datasets = []

        for fname in file_list:
            file_path = Path(fname)  # SSH files use absolute paths

            if self.lazy_load:
                ds = xr.open_dataset(file_path, chunks={"time": 10, "lat": 336, "lon": 720})
            else:
                ds = xr.open_dataset(file_path)

            datasets.append(ds)

        combined = xr.concat(datasets, dim="time", combine_attrs="override").sortby("time")

        return combined

    def _load_sst_observations(self, file_list: List[str]) -> xr.Dataset:
        """
        Load SST observations (ODYSSEA).

        Args:
            file_list: List of SST file paths relative to data_path

        Returns:
            dataset: xarray Dataset with SST observations
        """
        datasets = []

        for fname in file_list:
            file_path = self.data_path / fname

            if self.lazy_load:
                ds = xr.open_dataset(file_path, chunks={"time": 10, "lat": 336, "lon": 720})
            else:
                ds = xr.open_dataset(file_path)

            datasets.append(ds)

        combined = xr.concat(datasets, dim="time", combine_attrs="override").sortby("time")

        return combined

    @classmethod
    def align_grid(cls, source_ds, target_ds):
        """
        Align source dataset/dataarray to target dataset grid using xESMF regridding.

        This follows the reference notebook strategy:
        - compare coordinate shapes first
        - return the original object when grids are already identical
        - otherwise build an explicit target grid and regrid with xESMF

        Args:
            source_ds: Source xarray Dataset or DataArray to be aligned
            target_ds: Target xarray Dataset with desired grid coordinates

        Returns:
            Aligned dataset if coordinates differ, otherwise original dataset
        """
        if isinstance(source_ds, xr.Dataset):
            source_lat = source_ds["lat"].values
            source_lon = source_ds["lon"].values
        else:
            source_lat = source_ds.coords["lat"].values
            source_lon = source_ds.coords["lon"].values

        if isinstance(target_ds, xr.Dataset):
            target_lat = target_ds["lat"].values
            target_lon = target_ds["lon"].values
        else:
            target_lat = target_ds.coords["lat"].values
            target_lon = target_ds.coords["lon"].values

        shapes_match = (source_lat.shape == target_lat.shape) and (
            source_lon.shape == target_lon.shape
        )
        if shapes_match:
            lat_same = np.allclose(source_lat, target_lat)
            lon_same = np.allclose(source_lon, target_lon)
        else:
            lat_same = False
            lon_same = False

        if lat_same and lon_same:
            return source_ds

        if isinstance(source_ds, xr.DataArray):
            source_is_dataarray = True
            source_obj = source_ds.to_dataset(name="data")
        else:
            source_is_dataarray = False
            source_obj = source_ds

        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        target_grid = xr.Dataset(
            {
                "lat": (["y", "x"], lat2d),
                "lon": (["y", "x"], lon2d),
            }
        )

        regridder = xe.Regridder(source_obj, target_grid, method="bilinear", periodic=False)
        aligned = regridder(source_obj)

        if source_is_dataarray:
            aligned = aligned["data"]

        return aligned

    def get_sequence(self, start_idx: int, length: int) -> xr.Dataset:
        """
        Extract a time sequence from the dataset.

        Args:
            start_idx: Starting time index
            length: Number of time steps to extract

        Returns:
            sequence: xarray Dataset slice
        """
        return self.dataset.isel(time=slice(start_idx, start_idx + length))

    def get_ssh_obs(self, start_idx: int, length: int) -> Optional[xr.DataArray]:
        """
        Extract SSH observations for a time window.

        Args:
            start_idx: Starting time index
            length: Number of time steps

        Returns:
            ssh_obs: SSH observations (or None if not loaded)
        """
        if self.ssh_obs_dataset is None:
            return None

        return self.ssh_obs_dataset["sla"].isel(time=slice(start_idx, start_idx + length))

    def get_sst_obs(self, start_idx: int, length: int) -> Optional[xr.DataArray]:
        """
        Extract SST observations for a time window.

        Args:
            start_idx: Starting time index
            length: Number of time steps

        Returns:
            sst_obs: SST observations (or None if not loaded)
        """
        if self.sst_obs_dataset is None:
            return None

        # ODYSSEA SST is in Kelvin, convert to Celsius
        sst_kelvin = self.sst_obs_dataset["adjusted_sea_surface_temperature"].isel(
            time=slice(start_idx, start_idx + length)
        )

        return sst_kelvin - 273.15


class ObservationOperator:
    """
    Apply observation operators to model state.

    Implements:
    - SSH: Along-track altimetry → nearest-neighbor interpolation to grid
    - SST: Gridded satellite → direct pixel match (1:1 colocation)
    - QC: Light (trust CMEMS L3 pre-filtering)
    - OSSE twin: GLORYS12 as truth

    This follows R2 decisions on observation handling.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize observation operator.

        Args:
            device: PyTorch device for tensors
        """
        self.device = device

    @classmethod
    def align_grid(self, source_ds, target_ds):
        """
        Align source dataset/dataarray to target dataset grid using xESMF regridding.

        This follows the reference notebook strategy:
        - compare coordinate shapes first
        - return the original object when grids are already identical
        - otherwise build an explicit target grid and regrid with xESMF

        Args:
            source_ds: Source xarray Dataset or DataArray to be aligned
            target_ds: Target xarray Dataset with desired grid coordinates

        Returns:
            Aligned dataset if coordinates differ, otherwise original dataset
        """
        if isinstance(source_ds, xr.Dataset):
            source_lat = source_ds["lat"].values
            source_lon = source_ds["lon"].values
        else:
            source_lat = source_ds.coords["lat"].values
            source_lon = source_ds.coords["lon"].values

        if isinstance(target_ds, xr.Dataset):
            target_lat = target_ds["lat"].values
            target_lon = target_ds["lon"].values
        else:
            target_lat = target_ds.coords["lat"].values
            target_lon = target_ds.coords["lon"].values

        shapes_match = (source_lat.shape == target_lat.shape) and (
            source_lon.shape == target_lon.shape
        )
        if shapes_match:
            lat_same = np.allclose(source_lat, target_lat)
            lon_same = np.allclose(source_lon, target_lon)
        else:
            lat_same = False
            lon_same = False

        if lat_same and lon_same:
            return source_ds

        if isinstance(source_ds, xr.DataArray):
            source_is_dataarray = True
            source_obj = source_ds.to_dataset(name="data")
        else:
            source_is_dataarray = False
            source_obj = source_ds

        lon2d, lat2d = np.meshgrid(target_lon, target_lat)
        target_grid = xr.Dataset(
            {
                "lat": (["y", "x"], lat2d),
                "lon": (["y", "x"], lon2d),
            }
        )

        regridder = xe.Regridder(source_obj, target_grid, method="bilinear", periodic=False)
        aligned = regridder(source_obj)

        if source_is_dataarray:
            aligned = aligned["data"]

        return aligned

    def apply_ssh_operator(
        self,
        model_state: xr.Dataset,
        ssh_obs: xr.DataArray,
        obs_mode: str,
        mdt: Optional[xr.DataArray] = None,
    ) -> Tuple[xr.Dataset, np.ndarray]:
        """
        Apply SSH observation operator.

        SSH operator follows R2 decision:
        - Along-track altimetry → nearest-neighbor interpolation to grid
        - Preserves realistic satellite trace (sparse obs)

        Args:
            model_state: Model state dataset (target grid)
            ssh_obs: SSH observations (SLA - sea level anomaly)
            obs_mode: Observation mode ('full', 'simulated', 'real')
            mdt: Mean dynamic topography (required for 'real' mode to convert SLA→SSH)

        Returns:
            updated_state: Model state with SSH channel replaced by observations
            ssh_mask: Mask indicating where SSH observations exist [T, H, W]
        """
        if obs_mode == "full":
            # Full observations: all ocean pixels observed (idealized)
            # No modification needed - model state SSH is already truth
            ssh_mask = np.ones_like(model_state["data"][:, 0, :, :].values)
            return model_state, ssh_mask

        elif obs_mode == "simulated":
            # Simulated observations: GLORYS12 truth with realistic obs coverage (OSSE)
            # Preserve real satellite trace - keep only pixels that were observed

            # Create mask from observations: 1 where observed, NaN where not
            ssh_nanmask = xr.where(np.isfinite(ssh_obs), 1.0, np.nan)

            # Regrid mask to target grid
            ssh_nanmask_regridded = self.align_grid(ssh_nanmask, model_state)

            # Apply mask: keep GLORYS12 truth only where satellite observed
            ssh_data = model_state["data"][:, 0, :, :].values  # Channel 0 = SSH
            valid_mask = np.isfinite(ssh_nanmask_regridded.values)

            # Where satellite observed: use truth (already in model_state)
            # Where not observed: set to 0 (will be masked out in loss)
            model_state["data"][:, 0, :, :] = np.where(valid_mask, ssh_data, 0.0)

            # Return mask as numpy array
            ssh_mask = np.nan_to_num(ssh_nanmask_regridded.values, nan=0.0)

            return model_state, ssh_mask

        elif obs_mode == "real":
            # Real observations: actual satellite SSH observations (OSE)

            if mdt is None:
                raise ValueError("MDT (mean dynamic topography) required for real SSH observations")

            # Regrid observations to target grid
            obs_sla_regridded = self.align_grid(ssh_obs, model_state)
            mdt_regridded = self.align_grid(mdt, model_state)

            # Build SSH from observed SLA + MDT
            obs_ssh_full = obs_sla_regridded.values + mdt_regridded.values

            # Create mask: 1 where observed, NaN where not
            ssh_nanmask = xr.where(np.isfinite(ssh_obs), 1.0, np.nan)
            ssh_nanmask_regridded = self.align_grid(ssh_nanmask, model_state)

            # Keep only observed points
            valid_mask = np.isfinite(ssh_nanmask_regridded.values)
            ssh_obs_gridded = np.where(valid_mask, obs_ssh_full, 0.0)

            # Replace model state SSH with observations
            model_state["data"][:, 0, :, :] = ssh_obs_gridded

            # Return mask as numpy array
            ssh_mask = np.nan_to_num(ssh_nanmask_regridded.values, nan=0.0)

            return model_state, ssh_mask

        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")

    def apply_sst_operator(
        self, model_state: xr.Dataset, sst_obs: xr.DataArray, obs_mode: str
    ) -> Tuple[xr.Dataset, np.ndarray]:
        """
        Apply SST observation operator.

        SST operator follows R2 decision:
        - Pre-gridded L3 satellite → direct pixel match (1:1 colocation)
        - No interpolation needed (already on compatible grid)

        Args:
            model_state: Model state dataset (target grid)
            sst_obs: SST observations (already in Celsius)
            obs_mode: Observation mode ('full', 'simulated', 'real')

        Returns:
            updated_state: Model state with SST channel replaced by observations
            sst_mask: Mask indicating where SST observations exist [T, H, W]
        """
        if obs_mode == "full":
            # Full observations: all ocean pixels observed (idealized)
            sst_mask = np.ones_like(model_state["data"][:, 1, :, :].values)
            return model_state, sst_mask

        elif obs_mode == "simulated":
            # Simulated observations: GLORYS12 truth with realistic obs coverage (OSSE)

            # Create mask from observations
            sst_nanmask = xr.where(np.isfinite(sst_obs), 1.0, np.nan)

            # Regrid mask to target grid
            sst_nanmask_regridded = self.align_grid(sst_nanmask, model_state)

            # Apply mask: keep GLORYS12 truth only where satellite observed
            sst_data = model_state["data"][:, 1, :, :].values  # Channel 1 = SST
            valid_mask = np.isfinite(sst_nanmask_regridded.values)

            model_state["data"][:, 1, :, :] = np.where(valid_mask, sst_data, 0.0)

            # SSS, U, V are not observed in satellite data (channels 2-4)
            model_state["data"][:, 2:5, :, :] = 0.0

            # Return mask
            sst_mask = np.nan_to_num(sst_nanmask_regridded.values, nan=0.0)

            return model_state, sst_mask

        elif obs_mode == "real":
            # Real observations: actual satellite SST observations (OSE)

            # Regrid observations to target grid
            obs_sst_regridded = self.align_grid(sst_obs, model_state)

            # Create mask
            sst_nanmask = xr.where(np.isfinite(sst_obs), 1.0, np.nan)
            sst_nanmask_regridded = self.align_grid(sst_nanmask, model_state)

            # Keep only observed points
            valid_mask = np.isfinite(sst_nanmask_regridded.values)
            sst_obs_gridded = np.where(valid_mask, obs_sst_regridded.values, 0.0)

            # Replace model state SST with observations
            model_state["data"][:, 1, :, :] = sst_obs_gridded

            # SSS, U, V are not observed (channels 2-4)
            model_state["data"][:, 2:5, :, :] = 0.0

            # Return mask
            sst_mask = np.nan_to_num(sst_nanmask_regridded.values, nan=0.0)

            return model_state, sst_mask

        else:
            raise ValueError(f"Unknown obs_mode: {obs_mode}")
