#!/usr/bin/env python3
"""
Unified Multirun Statistics Aggregation (v2 - Combined Mode)
Aggregates statistics ACROSS all input directories together.

Usage:
  python3 aggregate_all_v2.py table1_refRun*                 # Combine all runs from matched dirs
  python3 aggregate_all_v2.py dir1 dir2 dir3                 # Combine all runs from 3 dirs
  python3 aggregate_all_v2.py table1_* -o ./output           # Output to custom path
  python3 aggregate_all_v2.py table1_* table2_* --output ./out

Output: Single aggregated_metrics/, aggregated_states/, aggregated_diagnostics/
         with statistics computed across ALL input directories
         CSV/JSON exports for all statistics
         GIF time-series for forecast anomalies by region and variable
"""

import sys, json, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from matplotlib.animation import PillowWriter
import warnings
warnings.filterwarnings('ignore')

# ============ CSV AGGREGATION (FIXED) ============

def process_csv(file_name, all_run_dirs):
    """Process CSV - handles files with/without index columns."""
    dfs = [pd.read_csv(run_dir / "metrics" / file_name) for run_dir in all_run_dirs 
           if (run_dir / "metrics" / file_name).exists()]
    
    if not dfs: return None
    
    # Find index & numeric columns
    index_cols = [c for c in dfs[0].columns if dfs[0][c].dtype == 'object']
    numeric_cols = [c for c in dfs[0].columns if dfs[0][c].dtype in ['int64', 'float64']]
    
    # If NO index cols, use non-_mean/_std cols as index
    if not index_cols:
        index_cols = [c for c in dfs[0].columns 
                      if not (c.endswith('_mean') or c.endswith('_std'))]
    
    if not numeric_cols:
        return None
    
    # Check row counts
    row_counts = [len(df) for df in dfs]
    
    if len(set(row_counts)) == 1:
        # Same length: direct aggregation
        result = dfs[0][index_cols].copy() if index_cols else dfs[0].iloc[:, :1].copy()
        for col in numeric_cols:
            data = np.array([df[col].values for df in dfs])
            result[f"{col}_mean"] = np.mean(data, axis=0)
            result[f"{col}_std"] = np.std(data, axis=0)
        return result
    else:
        # Variable length: groupby merge
        all_df = []
        for rid, df in enumerate(dfs):
            df = df.copy()
            df['_run'] = rid
            all_df.append(df)
        
        combined = pd.concat(all_df, ignore_index=True)
        result = dfs[0][index_cols].drop_duplicates().reset_index(drop=True) if index_cols \
                 else dfs[0].iloc[:, :1].drop_duplicates().reset_index(drop=True)
        
        for col in numeric_cols:
            if index_cols:
                grp = combined.groupby(index_cols)[col].agg(['mean', 'std']).reset_index()
                grp.rename(columns={'mean': f"{col}_mean", 'std': f"{col}_std"}, inplace=True)
                result = result.merge(grp, on=index_cols, how='left')
            else:
                result[f"{col}_mean"] = combined[col].mean()
                result[f"{col}_std"] = combined[col].std()
        
        return result

def process_json(file_name, all_run_dirs):
    """Process JSON files recursively."""
    def agg_json_rec(jsons):
        if not jsons: return None
        result = {}
        for key in jsons[0].keys():
            vals = [j.get(key) for j in jsons]
            if all(isinstance(v, (int, float)) for v in vals):
                result[f"{key}_mean"] = float(np.mean(vals))
                result[f"{key}_std"] = float(np.std(vals))
            elif all(isinstance(v, dict) for v in vals):
                result[key] = agg_json_rec(vals)
            elif all(isinstance(v, list) and len(v) == len(vals[0]) for v in vals):
                result[key] = []
                for idx in range(len(vals[0])):
                    items = [v[idx] for v in vals]
                    if all(isinstance(i, (int, float)) for i in items):
                        result[key].append({"mean": float(np.mean(items)), "std": float(np.std(items))})
                    elif all(isinstance(i, dict) for i in items):
                        result[key].append(agg_json_rec(items))
                    else:
                        result[key].append(vals[0][idx])
            else:
                result[key] = vals[0]
        return result
    
    jsons = []
    for run_dir in all_run_dirs:
        jp = run_dir / "metrics" / file_name
        if jp.exists():
            try:
                with open(jp) as f: jsons.append(json.load(f))
            except: pass
    
    return agg_json_rec(jsons) if jsons else None

def aggregate_metrics(output_dir, all_run_dirs):
    print("\nSTEP 1: METRICS AGGREGATION")
    print("-" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = set()
    json_files = set()
    for run_dir in all_run_dirs:
        md = run_dir / "metrics"
        if md.exists():
            csv_files.update(f.name for f in md.glob("*.csv"))
            json_files.update(f.name for f in md.glob("*.json"))
    
    print("CSV Files:")
    for fn in sorted(csv_files):
        print(f"  {fn}...", end=" ", flush=True)
        res = process_csv(fn, all_run_dirs)
        if res is not None:
            (output_dir / f"{Path(fn).stem}_stats.csv").write_text(res.to_csv(index=False))
            print(f"✓ ({res.shape[0]}×{res.shape[1]})")
        else:
            print("✗")
    
    print("JSON Files:")
    for fn in sorted(json_files):
        print(f"  {fn}...", end=" ", flush=True)
        res = process_json(fn, all_run_dirs)
        if res is not None:
            (output_dir / f"{Path(fn).stem}_stats.json").write_text(json.dumps(res, indent=2))
            print("✓")
        else:
            print("✗")

# ============ STATES AGGREGATION ============

REGIONS = {'global': (-90, 90, -180, 180), 'gulf_stream': (30, 45, -85, -30),
           'kuroshio': (25, 45, 130, 180), 'agulhas': (-35, -20, 20, 60)}

def create_land_mask(all_run_dirs):
    try:
        ds = xr.open_dataset(all_run_dirs[0] / "states" / "initial_condition.nc")
        land = (ds['SSH'].values[0] == 0.0)
        print(f"  Land mask: {land.sum()} px (~{100*land.sum()/land.size:.1f}%)")
        return land
    except Exception as e:
        print(f"  ✗ Land mask error: {e}")
        return None

def accumulate_diffs(all_run_dirs, files, land_mask):
    lat, lon, time = None, None, None
    cnt, mean_sum, m2 = 0, {}, {}
    
    for run_dir in all_run_dirs:
        sd = run_dir / "states"
        if (sd / files[0]).exists() and (sd / files[1]).exists():
            try:
                ds1, ds2 = xr.open_dataset(sd / files[0]), xr.open_dataset(sd / files[1])
                if lat is None:
                    lat, lon, time = ds1.lat.values, ds1.lon.values, ds1.time.values
                for var in ds1.data_vars:
                    diff = (ds1[var].values - ds2[var].values).astype(np.float64)
                    if land_mask is not None:
                        for t in range(diff.shape[0]): diff[t, land_mask] = np.nan
                    if var not in mean_sum:
                        mean_sum[var], m2[var] = np.zeros_like(diff), np.zeros_like(diff)
                    delta = diff - mean_sum[var]
                    mean_sum[var] += delta / (cnt + 1)
                    m2[var] += delta * (diff - mean_sum[var])
                cnt += 1
            except: pass
    
    if cnt == 0: return None
    return {'coords': {'lat': lat, 'lon': lon, 'time': time}, 
            **{var: {'mean': mean_sum[var], 'std': np.sqrt(m2[var] / cnt)} for var in mean_sum}}

def plot_spatial(var_name, mean, std, lat, lon, region, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mean_m, std_m = np.ma.masked_invalid(mean), np.ma.masked_invalid(std)
    vm = np.nanpercentile(np.abs(mean), 99)
    im0 = axes[0].contourf(lon, lat, mean_m, levels=20, cmap='RdBu_r', vmin=-vm, vmax=vm, extend='both')
    axes[0].set_title(f'{var_name} - Mean (Region: {region})')
    axes[0].set_xlabel('Lon'); axes[0].set_ylabel('Lat')
    plt.colorbar(im0, ax=axes[0], label=var_name)
    vs = np.nanpercentile(std, 99)
    im1 = axes[1].contourf(lon, lat, std_m, levels=20, cmap='viridis', vmax=vs, extend='max')
    axes[1].set_title(f'{var_name} - Std (Region: {region})')
    axes[1].set_xlabel('Lon'); axes[1].set_ylabel('Lat')
    plt.colorbar(im1, ax=axes[1], label='Std')
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()

def crop_region(data, lat, lon, bounds):
    lm = (lat >= bounds[0]) & (lat <= bounds[1])
    lnm = (lon >= bounds[2]) & (lon <= bounds[3])
    return data[lm, :][:, lnm], lat[lm], lon[lnm]

def plot_forecast_anomaly_frame(var_name, mean_t, std_t, lat, lon, region, ax_mean, ax_std):
    """Plot mean and std on provided axes for a single time step."""
    mean_m = np.ma.masked_invalid(mean_t)
    std_m = np.ma.masked_invalid(std_t)
    
    vm = np.nanpercentile(np.abs(mean_t), 99)
    im0 = ax_mean.contourf(lon, lat, mean_m, levels=20, cmap='RdBu_r', vmin=-vm, vmax=vm, extend='both')
    ax_mean.set_xlabel('Lon'); ax_mean.set_ylabel('Lat')
    ax_mean.set_title(f'{var_name} Mean')
    if not hasattr(ax_mean, '_cbar_mean'):
        cbar0 = plt.colorbar(im0, ax=ax_mean)
        ax_mean._cbar_mean = cbar0
    
    vs = np.nanpercentile(std_t, 99)
    im1 = ax_std.contourf(lon, lat, std_m, levels=20, cmap='viridis', vmax=vs, extend='max')
    ax_std.set_xlabel('Lon'); ax_std.set_ylabel('Lat')
    ax_std.set_title(f'{var_name} Std')
    if not hasattr(ax_std, '_cbar_std'):
        cbar1 = plt.colorbar(im1, ax=ax_std)
        ax_std._cbar_std = cbar1

def create_forecast_anomaly_gif(output_dir, all_run_dirs, diffs_dict, lat, lon, region, bnds, var_name):
    """Create GIF for forecast anomaly stats over time for a specific variable and region."""
    if var_name not in diffs_dict:
        return False
    
    try:
        from PIL import Image
        from io import BytesIO
    except ImportError:
        print(f"    ⚠ PIL not available, skipping GIF for {var_name} ({region})")
        return False
    
    times = diffs_dict['coords']['time']
    means = diffs_dict[var_name]['mean']  # shape: (time, lat, lon)
    stds = diffs_dict[var_name]['std']    # shape: (time, lat, lon)
    
    # Get region indices (2D)
    lat_idx = (lat >= bnds[0]) & (lat <= bnds[1])
    lon_idx = (lon >= bnds[2]) & (lon <= bnds[3])
    lat_c = lat[lat_idx]
    lon_c = lon[lon_idx]
    
    frames = []
    for t in range(len(times)):
        # Crop this timestep to region (2D indexing for each time step)
        mean_t = means[t, :, :]  # shape: (lat, lon)
        std_t = stds[t, :, :]    # shape: (lat, lon)
        
        mean_c = mean_t[lat_idx, :][:, lon_idx]  # Crop to region
        std_c = std_t[lat_idx, :][:, lon_idx]
        
        # Create figure
        fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(14, 5))
        
        mean_m = np.ma.masked_invalid(mean_c)
        std_m = np.ma.masked_invalid(std_c)
        
        vm = np.nanpercentile(np.abs(mean_c), 99) if np.isfinite(mean_c).any() else 1.0
        im0 = ax_mean.contourf(lon_c, lat_c, mean_m, levels=15, cmap='RdBu_r', vmin=-vm, vmax=vm, extend='both')
        ax_mean.set_xlabel('Lon'); ax_mean.set_ylabel('Lat')
        ax_mean.set_title(f'{var_name} Mean - T{t:02d}')
        plt.colorbar(im0, ax=ax_mean, label=var_name)
        
        vs = np.nanpercentile(std_c, 99) if np.isfinite(std_c).any() else 1.0
        im1 = ax_std.contourf(lon_c, lat_c, std_m, levels=15, cmap='viridis', vmax=vs, extend='max')
        ax_std.set_xlabel('Lon'); ax_std.set_ylabel('Lat')
        ax_std.set_title(f'{var_name} Std - T{t:02d}')
        plt.colorbar(im1, ax=ax_std, label='Std')
        
        plt.tight_layout()
        
        # Save figure to BytesIO buffer and load with PIL
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        frames.append(Image.open(buffer).copy())
        buffer.close()
        plt.close(fig)
    
    # Write frames to GIF
    if frames:
        gif_path = output_dir / f"forecast_anomaly_{var_name}_{region}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0, optimize=False)
        print(f"    ✓ {gif_path.name} ({len(frames)} frames)")
        return True
    return False

def aggregate_states_with_gif(output_dir, all_run_dirs):
    """Aggregate states and create forecast anomaly GIFs."""
    print("\nSTEP 2: STATES AGGREGATION")
    print("-" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Creating land mask...")
    land = create_land_mask(all_run_dirs)
    
    # Store forecast anomaly data for GIF generation
    forecast_data = None
    spatial_stats_list = []
    
    for name, files in [("I.C. Correction", ('optimized_initial_condition.nc', 'initial_condition.nc')),
                        ("Forecast Anomaly", ('optimized_forecast.nc', 'reference_forecast.nc'))]:
        print(f"\n{name}:")
        res = accumulate_diffs(all_run_dirs, files, land)
        if res:
            lat, lon = res['coords']['lat'], res['coords']['lon']
            time = res['coords']['time']
            print(f"  Variables: {', '.join([k for k in res if k != 'coords'])}")
            
            # Store forecast data for GIF generation
            if name == "Forecast Anomaly":
                forecast_data = res
            
            # Generate spatial plots for each region and collect stats for CSV/JSON
            for var in res:
                if var == 'coords': continue
                mn, sd = res[var]['mean'], res[var]['std']
                
                # Collect stats for CSV export (spatial statistics per variable/region/time)
                for t in range(mn.shape[0]):
                    for reg, bnds in REGIONS.items():
                        mn_c, la_c, ln_c = crop_region(mn[t], lat, lon, bnds)
                        sd_c, _, _ = crop_region(sd[t], lat, lon, bnds)
                        
                        # Compute regional statistics
                        mean_val = np.nanmean(mn_c)
                        std_val = np.nanstd(mn_c)
                        std_of_std = np.nanmean(sd_c)
                        
                        spatial_stats_list.append({
                            'type': name.lower().replace(' ', '_'),
                            'variable': var,
                            'region': reg,
                            'timestep': int(t),
                            'mean': float(mean_val),
                            'std': float(std_val),
                            'mean_of_std': float(std_of_std)
                        })
                        
                        # Only generate PNG for first timestep to avoid excessive files
                        if t == 0:
                            # Clean filename: replace dots and spaces with underscores
                            clean_name = name.lower().replace('.', '').replace(' ', '_')
                            out_p = output_dir / f"{clean_name}_{var}_{reg}.png"
                            plot_spatial(var, mn_c, sd_c, la_c, ln_c, reg, out_p)
                
                print(f"    ✓ {var} ({len(REGIONS)} regions × {mn.shape[0]} timesteps)")
    
    # Export spatial statistics to CSV and JSON
    if spatial_stats_list:
        stats_df = pd.DataFrame(spatial_stats_list)
        stats_csv = output_dir / "spatial_stats.csv"
        stats_df.to_csv(stats_csv, index=False)
        print(f"\n  ✓ spatial_stats.csv ({len(spatial_stats_list)} rows)")
        
        stats_json = output_dir / "spatial_stats.json"
        with open(stats_json, 'w') as f:
            json.dump(spatial_stats_list, f, indent=2)
        print(f"  ✓ spatial_stats.json")
    
    # Create GIFs for forecast anomaly
    if forecast_data:
        print(f"\nGenerating Forecast Anomaly GIFs...")
        for var in forecast_data:
            if var == 'coords': continue
            for reg, bnds in REGIONS.items():
                create_forecast_anomaly_gif(output_dir, all_run_dirs, forecast_data, 
                                          lat, lon, reg, bnds, var)

# ============ DIAGNOSTICS ============

def aggregate_diagnostics(output_dir, all_run_dirs):
    print("\nSTEP 3: DIAGNOSTICS AGGREGATION")
    print("-" * 60)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== RMSE TIME EVOLUTION =====
    print("\n1. RMSE Evolution (over time)...")
    
    # Load rmse_evolution.csv from all runs
    rmse_dfs = []
    for run_dir in all_run_dirs:
        # all_run_dirs already points to individual runs (0, 1, 2, ...)
        rmse_csv = run_dir / "metrics" / "rmse_evolution.csv"
        if rmse_csv.exists():
            df = pd.read_csv(rmse_csv)
            rmse_dfs.append(df)
    
    if rmse_dfs:
        rmse_combined = pd.concat(rmse_dfs, ignore_index=True)
        
        # Aggregate by region, timestep, variable, metric
        rmse_agg = rmse_combined.groupby(['region', 'timestep', 'variable']).agg({
            'reference_rmse': ['mean', 'std'],
            'optimized_rmse': ['mean', 'std']
        }).reset_index()
        
        rmse_agg.columns = ['region', 'timestep', 'variable', 
                            'ref_rmse_mean', 'ref_rmse_std', 
                            'opt_rmse_mean', 'opt_rmse_std']
        
        # Export CSV
        rmse_csv_out = output_dir / "rmse_evolution_aggregated.csv"
        rmse_agg.to_csv(rmse_csv_out, index=False)
        print(f"  ✓ rmse_evolution_aggregated.csv ({len(rmse_agg)} rows)")
        
        # Export JSON
        rmse_json_out = output_dir / "rmse_evolution_aggregated.json"
        with open(rmse_json_out, 'w') as f:
            json.dump(rmse_agg.to_dict('records'), f, indent=2)
        print(f"  ✓ rmse_evolution_aggregated.json")
        
        # Plot RMSE evolution with error bands
        for region in rmse_agg['region'].unique():
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'RMSE Evolution - {region}', fontsize=14, fontweight='bold')
            
            vars_list = rmse_agg['variable'].unique()
            for idx, var in enumerate(vars_list[:6]):  # Up to 6 variables
                ax = axes.flat[idx]
                var_data = rmse_agg[(rmse_agg['region'] == region) & (rmse_agg['variable'] == var)]
                
                # Sort by timestep
                var_data = var_data.sort_values('timestep')
                t = var_data['timestep'].values
                
                # Plot reference
                ax.plot(t, var_data['ref_rmse_mean'].values, 'o-', label='Reference', linewidth=2)
                ax.fill_between(t, 
                                var_data['ref_rmse_mean'].values - var_data['ref_rmse_std'].values,
                                var_data['ref_rmse_mean'].values + var_data['ref_rmse_std'].values,
                                alpha=0.3)
                
                # Plot optimized
                ax.plot(t, var_data['opt_rmse_mean'].values, 's-', label='Optimized', linewidth=2)
                ax.fill_between(t, 
                                var_data['opt_rmse_mean'].values - var_data['opt_rmse_std'].values,
                                var_data['opt_rmse_mean'].values + var_data['opt_rmse_std'].values,
                                alpha=0.3)
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel('RMSE')
                ax.set_title(f'{var}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(vars_list), 6):
                axes.flat[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"rmse_evolution_{region}.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ rmse_evolution_{region}.png")
    else:
        print("  ⚠ No rmse_evolution.csv files found")
    
    # ===== PSD BY WAVENUMBER =====
    print("\n2. PSD Analysis (by wavenumber)...")
    
    # Load psd_analysis.csv from all runs
    psd_dfs = []
    for run_dir in all_run_dirs:
        # all_run_dirs already points to individual runs (0, 1, 2, ...)
        psd_csv = run_dir / "metrics" / "psd_analysis.csv"
        if psd_csv.exists():
            df = pd.read_csv(psd_csv)
            psd_dfs.append(df)
    
    if psd_dfs:
        psd_combined = pd.concat(psd_dfs, ignore_index=True)
        
        # Aggregate by region, wavenumber, variable, metric
        psd_agg = psd_combined.groupby(['region', 'wavenumber', 'variable']).agg({
            'reference_psd': ['mean', 'std'],
            'optimized_psd': ['mean', 'std'],
            'ground_truth_psd': ['mean', 'std']
        }).reset_index()
        
        psd_agg.columns = ['region', 'wavenumber', 'variable',
                           'ref_psd_mean', 'ref_psd_std',
                           'opt_psd_mean', 'opt_psd_std',
                           'gt_psd_mean', 'gt_psd_std']
        
        # Export CSV
        psd_csv_out = output_dir / "psd_analysis_aggregated.csv"
        psd_agg.to_csv(psd_csv_out, index=False)
        print(f"  ✓ psd_analysis_aggregated.csv ({len(psd_agg)} rows)")
        
        # Export JSON
        psd_json_out = output_dir / "psd_analysis_aggregated.json"
        with open(psd_json_out, 'w') as f:
            json.dump(psd_agg.to_dict('records'), f, indent=2)
        print(f"  ✓ psd_analysis_aggregated.json")
        
        # Plot PSD with error bands (logarithmic scale)
        for region in psd_agg['region'].unique():
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'PSD Analysis (log scale) - {region}', fontsize=14, fontweight='bold')
            
            vars_list = psd_agg['variable'].unique()
            for idx, var in enumerate(vars_list[:6]):  # Up to 6 variables
                ax = axes.flat[idx]
                var_data = psd_agg[(psd_agg['region'] == region) & (psd_agg['variable'] == var)]
                
                # Sort by wavenumber
                var_data = var_data.sort_values('wavenumber')
                k = var_data['wavenumber'].values
                
                # Plot reference
                ax.loglog(k, var_data['ref_psd_mean'].values, 'o-', label='Reference', linewidth=2)
                ax.fill_between(k,
                                np.maximum(var_data['ref_psd_mean'].values - var_data['ref_psd_std'].values, 1e-10),
                                var_data['ref_psd_mean'].values + var_data['ref_psd_std'].values,
                                alpha=0.3)
                
                # Plot optimized
                ax.loglog(k, var_data['opt_psd_mean'].values, 's-', label='Optimized', linewidth=2)
                ax.fill_between(k,
                                np.maximum(var_data['opt_psd_mean'].values - var_data['opt_psd_std'].values, 1e-10),
                                var_data['opt_psd_mean'].values + var_data['opt_psd_std'].values,
                                alpha=0.3)
                
                # Plot ground truth
                ax.loglog(k, var_data['gt_psd_mean'].values, '^-', label='Ground Truth', linewidth=2)
                ax.fill_between(k,
                                np.maximum(var_data['gt_psd_mean'].values - var_data['gt_psd_std'].values, 1e-10),
                                var_data['gt_psd_mean'].values + var_data['gt_psd_std'].values,
                                alpha=0.3)
                
                ax.set_xlabel('Wavenumber')
                ax.set_ylabel('PSD')
                ax.set_title(f'{var}')
                ax.legend()
                ax.grid(True, alpha=0.3, which='both')
            
            # Hide unused subplots
            for idx in range(len(vars_list), 6):
                axes.flat[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"psd_analysis_{region}.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ psd_analysis_{region}.png")
    else:
        print("  ⚠ No psd_analysis.csv files found")

# ============ MAIN ============

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    output_root = None
    input_patterns = []
    
    # Extract -o/--output if present
    for i, arg in enumerate(args):
        if arg in ['-o', '--output'] and i + 1 < len(args):
            output_root = Path(args[i + 1]).resolve()
            break
    
    # Filter input patterns (remove -o/--output and its argument)
    if output_root:
        idx = args.index('-o') if '-o' in args else args.index('--output')
        input_patterns = args[:idx] + args[idx+2:]
    else:
        input_patterns = args
    
    # Expand wildcards
    parent_dirs = []
    for pat in input_patterns:
        exp = glob(pat)
        parent_dirs.extend(exp if exp else ([pat] if Path(pat).exists() else []))
    
    parent_dirs = sorted(list(set(Path(d).resolve() for d in parent_dirs)))
    
    if not parent_dirs:
        print("✗ No matching directories")
        sys.exit(1)
    
    # Collect ALL runs from ALL parent directories
    all_run_dirs = []
    dir_info = []
    
    for pdir in parent_dirs:
        if not pdir.is_dir():
            print(f"✗ Not a directory: {pdir}")
            continue
        
        rdirs = sorted([d for d in pdir.iterdir() if d.is_dir() and d.name.isdigit()])
        if rdirs:
            all_run_dirs.extend(rdirs)
            dir_info.append(f"{pdir.name} ({len(rdirs)} runs)")
    
    if not all_run_dirs:
        print("✗ No runs found in any input directories")
        sys.exit(1)
    
    # Determine output directory
    if output_root:
        out_base = output_root
    else:
        out_base = Path.cwd()
    
    print(f"\n{'='*80}")
    print(f"COMBINED MULTIRUN STATISTICS")
    print(f"{'='*80}")
    print(f"Input directories ({len(parent_dirs)}):")
    for info in dir_info:
        print(f"  - {info}")
    print(f"Total runs: {len(all_run_dirs)}")
    print(f"Output: {out_base}")
    print(f"{'='*80}")
    
    try:
        aggregate_metrics(out_base / "aggregated_metrics", all_run_dirs)
        aggregate_states_with_gif(out_base / "aggregated_states", all_run_dirs)
        aggregate_diagnostics(out_base / "aggregated_diagnostics", all_run_dirs)
        
        print(f"\n{'='*80}")
        print(f"✅ STATISTICS COMPLETE")
        print(f"Output: {out_base}/aggregated_*")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
