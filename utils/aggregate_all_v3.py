#!/usr/bin/env python3
"""
Unified Multirun Statistics Aggregation (v3 - Combined Mode, paper figures)
Aggregates statistics ACROSS all input directories together.

Usage:
  python3 aggregate_all_v3.py table1_refRun*                 # Combine all runs from matched dirs
  python3 aggregate_all_v3.py dir1 dir2 dir3                 # Combine all runs from 3 dirs
  python3 aggregate_all_v3.py table1_* -o ./output           # Output to custom path
  python3 aggregate_all_v3.py table1_* table2_* --output ./out
  python3 aggregate_all_v3.py .tmp/runs/parallel_table1_*    # Nested layouts are searched recursively

Run discovery: every directory below each input that contains metrics/, states/
or diagnostics/ is one run, at any depth (multiruns/<exp>/<N>/ as well as
runs/parallel_<exp>/<timestamp>/gpu<G>_sample<S>/). Search stops at a run dir.

Output: Single aggregated_metrics/, aggregated_states/, aggregated_diagnostics/
         with statistics computed across ALL input directories
         CSV/JSON exports for all statistics
         GIF time-series for forecast anomalies by region and variable

v3 changes (diagnostics only): publication-style RMSE and PSD figures, one
variable per row, saved as PDF + 300-dpi PNG:
  rmse_evolution_{region}  RMSE vs lead time (days), mean ± 1 std across runs,
                           assimilation window (data.observation_length) shaded
  psd_analysis_{region}    PSD vs wavenumber (cycles/km, 1° = 111.32 km) with
                           wavelength top axis, geometric mean ×/÷ geometric std,
                           mesoscale 50–500 km band shaded
  psd_ratio_{region}       PSD / PSD_GLORYS12 per run, then aggregated (1 = perfect)
  effective_resolution_{region}   lambda_eff vs lead day (pseudo-track PSD score,
                           needs metrics/effective_resolution.csv, see
                           backfill_effective_resolution.py / gd_optimic.spectral_scores)
  psd_score_{region}_t{day}       1 - PSD(err)/PSD(truth) vs wavenumber, lambda_eff marked
  pixelization_{scope}     checkerboard fraction of the IC update vs iteration
                           (step / cumulative), mean ± 1 std, physical level and
                           pixelization threshold marked. Read from
                           metrics/optimization_history.json, or for cumulative from
                           metrics/checkerboard_backfill.csv (backfill_checkerboard.py)
"""

import sys, json, numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from matplotlib.animation import PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============ RUN DISCOVERY ============
RUN_MARKERS = ("metrics", "states", "diagnostics")

def find_run_dirs(root):
    """All run dirs at any depth below root (root included); no descent into a found run."""
    if any((root / m).is_dir() for m in RUN_MARKERS):
        return [root]
    runs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith(('.', '__')):
            runs.extend(find_run_dirs(d))
    return runs

def find_config(run_dir, max_up=2):
    """Nearest config.yaml: the run's own, else up to max_up ancestors' (or their .hydra/), e.g. parallel runs."""
    for d in (run_dir, *list(run_dir.parents)[:max_up]):
        for c in (d / "config.yaml", d / ".hydra" / "config.yaml"):
            if c.exists():
                return c
    return None

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

# ============ PAPER FIGURE STYLE (RMSE / PSD) ============

# Generic journal style: full text width, one panel per row.
FIG_WIDTH_IN = 7.0
ROW_HEIGHT_IN = 1.55
PAPER_RC = {
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'grid.linewidth': 0.4, 'grid.color': '#d9d9d9',
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
}
# Colorblind-safe (Okabe-Ito); ground truth is a neutral dashed reference line.
STYLE = {
    'ref': dict(color='#0072B2', lw=1.4, ls='-', label='Reference'),
    'opt': dict(color='#D55E00', lw=1.4, ls='-', label='Optimized'),
    'gt':  dict(color='#222222', lw=1.1, ls='--', label='GLORYS12'),
}
BAND_ALPHA = 0.22
WINDOW_COLOR = '#9e9e9e'
BORDER_STYLE = dict(color='#333333', lw=1.0, ls=(0, (4, 2)))
MESOSCALE_KM = (50.0, 500.0)
KM_PER_DEG = 111.32  # PSD wavenumber is in cycles/degree (grid spacing in degrees)
VAR_META = {
    'SSH': ('Sea surface height', 'm'), 'T': ('Sea surface temperature', '°C'),
    'S': ('Sea surface salinity', 'psu'), 'U': ('Zonal velocity', 'm s$^{-1}$'),
    'V': ('Meridional velocity', 'm s$^{-1}$'),
}

def _ordered_vars(df):
    present = list(df['variable'].unique())
    return [v for v in VAR_META if v in present] + [v for v in present if v not in VAR_META]

def _save_paper_fig(fig, out_stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {Path(out_stem).name}.pdf/.png")

def _panel_label(ax, idx, var):
    name = VAR_META.get(var, (var, ''))[0]
    ax.set_title(f"({chr(97 + idx)}) {name}", loc='left', fontweight='bold', pad=3)

def _band_handle(n_runs, geometric=False):
    label = f"×/÷ 1 geometric std (N = {n_runs} runs)" if geometric else f"±1 std (N = {n_runs} runs)"
    return Patch(facecolor='#888888', alpha=BAND_ALPHA * 1.5, edgecolor='none', label=label)

def _read_observation_length(all_run_dirs):
    """Assimilation window length (days) from each run's config.yaml; None if unknown."""
    import yaml
    lengths = set()
    for run_dir in all_run_dirs:
        cfg_path = find_config(run_dir)
        if cfg_path:
            try:
                with open(cfg_path) as f:
                    lengths.add(int(yaml.safe_load(f)['data']['observation_length']))
            except Exception:
                pass
    if len(lengths) > 1:
        print(f"  ⚠ Mixed observation_length {sorted(lengths)}; shading up to {max(lengths)}")
    return max(lengths) if lengths else None

def plot_rmse_paper(rmse_agg, region, n_runs, obs_len, out_stem):
    """RMSE vs lead time: one variable per row, mean ± 1 std across runs."""
    reg = rmse_agg[rmse_agg['region'] == region]
    vars_list = _ordered_vars(reg)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.5))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            d = reg[reg['variable'] == var].sort_values('timestep')
            t = d['timestep'].values
            if obs_len:
                ax.axvspan(t.min(), obs_len, color=WINDOW_COLOR, alpha=0.18, lw=0, zorder=0)
                ax.axvline(obs_len, **BORDER_STYLE, zorder=4)
            for key in ('ref', 'opt'):
                m, s = d[f'{key}_rmse_mean'].values, np.nan_to_num(d[f'{key}_rmse_std'].values)
                st = STYLE[key]
                ax.fill_between(t, m - s, m + s, color=st['color'], alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(t, m, color=st['color'], lw=st['lw'], ls=st['ls'], zorder=3)
            ax.set_ylabel(f"RMSE ({VAR_META.get(var, (var, ''))[1]})")
            ax.grid(True, axis='y')
            ax.set_xlim(t.min(), t.max())
            _panel_label(ax, idx, var)
        axes[-1].set_xlabel('Lead time (days)')
        axes[-1].xaxis.set_major_locator(plt.MultipleLocator(7))
        axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        handles = [Line2D([], [], color=STYLE[k]['color'], lw=STYLE[k]['lw'], label=STYLE[k]['label'])
                   for k in ('ref', 'opt')] + [_band_handle(n_runs)]
        if obs_len:
            # column-major fill for ncol=3: row 1 = Ref, Opt, band; row 2 = window, border
            handles = [handles[0],
                       Patch(facecolor=WINDOW_COLOR, alpha=0.35, edgecolor='none',
                             label=f'Assimilation window (days 0–{obs_len})'),
                       handles[1],
                       Line2D([], [], **BORDER_STYLE, label=f'Assimilation / forecast border (day {obs_len})'),
                       handles[2]]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.96 if obs_len else 0.975))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

def _log_stats(values):
    """Geometric mean and ×/÷ 1-std factor across runs (bands stay positive on log axes)."""
    lv = np.log(np.clip(values, 1e-30, None))
    return np.exp(lv.mean()), (np.exp(lv.std(ddof=1)) if len(lv) > 1 else 1.0)

def _psd_log_curves(psd_combined, region, var):
    """Per wavenumber: geometric mean/std factor of PSD and of PSD/PSD_GLORYS12 across runs."""
    d = psd_combined[(psd_combined['region'] == region) & (psd_combined['variable'] == var)].copy()
    d['ref_ratio'] = d['reference_psd'] / d['ground_truth_psd']
    d['opt_ratio'] = d['optimized_psd'] / d['ground_truth_psd']
    rows = []
    for k, g in d.groupby('wavenumber'):
        row = {'k': k}
        for col in ('reference_psd', 'optimized_psd', 'ground_truth_psd', 'ref_ratio', 'opt_ratio'):
            row[f'{col}_gm'], row[f'{col}_gs'] = _log_stats(g[col].values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('k')

def plot_psd_paper(psd_combined, region, n_runs, out_stem, ratio=False):
    """PSD (or PSD/PSD_GLORYS12 if ratio) vs wavenumber: one variable per row."""
    reg = psd_combined[psd_combined['region'] == region]
    vars_list = _ordered_vars(reg)
    series = [('ref', 'ref_ratio'), ('opt', 'opt_ratio')] if ratio else \
             [('gt', 'ground_truth_psd'), ('ref', 'reference_psd'), ('opt', 'optimized_psd')]
    inv = lambda x: 1.0 / np.clip(x, 1e-12, None)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.8))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            c = _psd_log_curves(psd_combined, region, var)
            k = c['k'].values / KM_PER_DEG  # cycles/degree -> cycles/km
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.axvspan(1.0 / MESOSCALE_KM[1], 1.0 / MESOSCALE_KM[0], color=WINDOW_COLOR, alpha=0.15, lw=0, zorder=0)
            if ratio:
                ax.axhline(1.0, color=STYLE['gt']['color'], lw=STYLE['gt']['lw'], ls=STYLE['gt']['ls'], zorder=1)
            for key, col in series:
                gm, gs = c[f'{col}_gm'].values, c[f'{col}_gs'].values
                st = STYLE[key]
                if key != 'gt':
                    ax.fill_between(k, gm / gs, gm * gs, color=st['color'], alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(k, gm, color=st['color'], lw=st['lw'], ls=st['ls'], zorder=3)
            if ratio:
                lo, hi = ax.get_ylim()
                span = max(abs(np.log10(lo)), abs(np.log10(hi)))
                ax.set_ylim(10 ** -span, 10 ** span)
                ticks = [v for v in (0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0)
                         if 10 ** -span <= v <= 10 ** span]
                ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
                ax.yaxis.set_minor_locator(plt.NullLocator())
                ax.set_ylabel('PSD / PSD$_{\\mathrm{GLORYS12}}$')
            else:
                u = VAR_META.get(var, (var, ''))[1]
                ax.set_ylabel((f"PSD (({u})$^2$)" if ' ' in u else f"PSD ({u}$^2$)") if u else 'PSD')
            ax.grid(True, which='major', axis='x' if ratio else 'both')
            ax.set_xlim(k.min(), k.max())
            _panel_label(ax, idx, var)
            if idx == 0:
                ax.secondary_xaxis('top', functions=(inv, inv)).set_xlabel('Wavelength (km)')
        axes[-1].set_xlabel('Wavenumber (cycles km$^{-1}$)')
        handles = [Line2D([], [], color=STYLE[k]['color'], lw=STYLE[k]['lw'], ls=STYLE[k]['ls'],
                          label=STYLE[k]['label'] + (' (ratio = 1)' if ratio and k == 'gt' else ''))
                   for k in ('gt', 'ref', 'opt')]
        extra = [_band_handle(n_runs, geometric=True),
                 Patch(facecolor=WINDOW_COLOR, alpha=0.3, edgecolor='none',
                       label=f'Mesoscale ({MESOSCALE_KM[0]:.0f}–{MESOSCALE_KM[1]:.0f} km)')]
        # Legend fills column-major: row 1 = the three lines, row 2 = the two bands
        handles = [handles[0], extra[0], handles[1], extra[1], handles[2]]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.955))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

def plot_psd_score_paper(score_combined, lambda_combined, region, timestep, n_runs, out_stem):
    """PSD score 1 - PSD(err)/PSD(truth) vs wavenumber at one lead day; lambda_eff marked."""
    reg = score_combined[(score_combined['region'] == region) & (score_combined['timestep'] == timestep)]
    lam = lambda_combined[(lambda_combined['region'] == region) & (lambda_combined['timestep'] == timestep)]
    vars_list = _ordered_vars(reg)
    inv = lambda x: 1.0 / np.clip(x, 1e-12, None)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.8))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            d = reg[reg['variable'] == var].groupby('wavenumber_cpkm').agg(
                ref_m=('score_reference', 'mean'), ref_s=('score_reference', 'std'),
                opt_m=('score_optimized', 'mean'), opt_s=('score_optimized', 'std')).reset_index()
            k = d['wavenumber_cpkm'].values
            ax.set_xscale('log')
            ax.axvspan(1.0 / MESOSCALE_KM[1], 1.0 / MESOSCALE_KM[0], color=WINDOW_COLOR, alpha=0.15, lw=0, zorder=0)
            ax.axhline(0.5, color=STYLE['gt']['color'], lw=0.8, ls=':', zorder=1)
            lv = lam[lam['variable'] == var]
            for key in ('ref', 'opt'):
                st = STYLE[key]
                m, s = d[f'{key}_m'].values, np.nan_to_num(d[f'{key}_s'].values)
                ax.fill_between(k, m - s, m + s, color=st['color'], alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(k, m, color=st['color'], lw=st['lw'], zorder=3)
                col = 'lambda_eff_reference_km' if key == 'ref' else 'lambda_eff_optimized_km'
                lam_mean = np.nanmean(lv[col].values) if lv[col].notna().any() else np.nan
                if np.isfinite(lam_mean):
                    ax.axvline(1.0 / lam_mean, color=st['color'], lw=0.9, ls=(0, (2, 1.5)), zorder=4)
            ax.set_ylim(max(-0.5, ax.get_ylim()[0]), 1.02)
            ax.set_ylabel('PSD score')
            ax.grid(True, which='major')
            ax.set_xlim(k.min(), k.max())
            _panel_label(ax, idx, var)
            if idx == 0:
                ax.secondary_xaxis('top', functions=(inv, inv)).set_xlabel('Wavelength (km)')
        axes[-1].set_xlabel('Wavenumber (cycles km$^{-1}$)')
        handles = [Line2D([], [], color=STYLE[k]['color'], lw=STYLE[k]['lw'], label=STYLE[k]['label'])
                   for k in ('ref', 'opt')]
        handles = [handles[0], _band_handle(n_runs), handles[1],
                   Line2D([], [], color='#555555', lw=0.9, ls=(0, (2, 1.5)), label='$\\lambda_{\\mathrm{eff}}$ (run mean)'),
                   Line2D([], [], color=STYLE['gt']['color'], lw=0.8, ls=':', label='Score = 0.5'),
                   Patch(facecolor=WINDOW_COLOR, alpha=0.3, edgecolor='none',
                         label=f'Mesoscale ({MESOSCALE_KM[0]:.0f}–{MESOSCALE_KM[1]:.0f} km)')]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.955))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

# Resolvable range of the pseudo-track spectra (spectral_scores: 0.25 deg grid, 1000 km segments)
LAMBDA_FINEST_KM = 2 * 0.25 * KM_PER_DEG                       # Nyquist, 55.7 km
LAMBDA_COARSEST_KM = round(1000.0 / (0.25 * KM_PER_DEG)) * 0.25 * KM_PER_DEG  # segment, 1001.9 km

def plot_effective_resolution_paper(lambda_combined, region, n_runs, out_stem):
    """Effective resolution vs lead day: one variable per row, mean ± 1 std over runs.

    Runs whose score never crosses 0.5 are censored at the resolvable bound
    (resolved_all -> Nyquist, unresolved -> segment length); a point with any
    censored run is drawn hollow with a triangle marker at that bound.
    """
    reg = lambda_combined[lambda_combined['region'] == region].copy()
    for key in ('reference', 'optimized'):
        col, status = reg[f'lambda_eff_{key}_km'].copy(), reg[f'status_{key}']
        col[status == 'resolved_all'] = LAMBDA_FINEST_KM
        col[status == 'unresolved'] = LAMBDA_COARSEST_KM
        reg[f'lam_{key}'] = col
        reg[f'cens_{key}'] = status.isin(['resolved_all', 'unresolved'])
    vars_list = _ordered_vars(reg)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.6))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            d = reg[reg['variable'] == var].groupby('timestep').agg(
                ref_m=('lam_reference', 'mean'), ref_s=('lam_reference', 'std'), ref_c=('cens_reference', 'any'),
                opt_m=('lam_optimized', 'mean'), opt_s=('lam_optimized', 'std'), opt_c=('cens_optimized', 'any'),
            ).reset_index()
            t = d['timestep'].values
            for key, marker, dx in (('ref', 'o', -0.25), ('opt', 's', 0.25)):
                st = STYLE[key]
                m, s, cens = d[f'{key}_m'].values, np.nan_to_num(d[f'{key}_s'].values), d[f'{key}_c'].values
                ax.plot(t + dx, m, color=st['color'], lw=st['lw'], zorder=3)
                ax.errorbar(t[~cens] + dx, m[~cens], yerr=s[~cens], color=st['color'], ls='none',
                            marker=marker, ms=4.5, capsize=2.5, elinewidth=0.9, zorder=4)
                for up in (False, True):  # censored points: hollow triangles at the bound
                    sel = cens & ((m >= LAMBDA_COARSEST_KM * 0.999) if up else (m < LAMBDA_COARSEST_KM * 0.999))
                    ax.plot(t[sel] + dx, m[sel], ls='none', marker='^' if up else 'v', ms=5.5,
                            mfc='white', mec=st['color'], mew=1.1, zorder=5)
            for bound in (LAMBDA_FINEST_KM, LAMBDA_COARSEST_KM):
                ax.axhline(bound, color='#888888', lw=0.6, ls=':', zorder=1)
            ax.set_yscale('log')
            ax.set_ylim(LAMBDA_FINEST_KM / 1.15, LAMBDA_COARSEST_KM * 1.15)
            ticks = [60, 100, 200, 500, 1000]
            ax.set_yticks(ticks, [str(v) for v in ticks])
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.set_ylabel('$\\lambda_{\\mathrm{eff}}$ (km)')
            ax.grid(True, axis='y')
            _panel_label(ax, idx, var)
        axes[-1].set_xlabel('Lead time (days)')
        axes[-1].set_xticks(sorted(reg['timestep'].unique()))
        lines = [Line2D([], [], color=STYLE[k]['color'], lw=STYLE[k]['lw'], marker=m, ms=4.5,
                        label=STYLE[k]['label']) for k, m in (('ref', 'o'), ('opt', 's'))]
        handles = [lines[0],
                   Line2D([], [], color='#555555', ls='none', marker='v', ms=5.5, mfc='white', mew=1.1,
                          label=f'Resolved to grid limit (≤ {LAMBDA_FINEST_KM:.0f} km)'),
                   lines[1],
                   Line2D([], [], color='#555555', ls='none', marker='^', ms=5.5, mfc='white', mew=1.1,
                          label=f'Unresolved (≥ {LAMBDA_COARSEST_KM:.0f} km segment)'),
                   Line2D([], [], color='#555555', lw=0.9, marker='|', ms=8, ls='none',
                          label=f'±1 std (N = {n_runs} runs); lower = finer')]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.96))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

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
        
        # Plot RMSE evolution (paper style: one variable per row)
        obs_len = _read_observation_length(all_run_dirs)
        for region in rmse_agg['region'].unique():
            plot_rmse_paper(rmse_agg, region, len(rmse_dfs), obs_len,
                            output_dir / f"rmse_evolution_{region}")
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
        
        # Plot PSD spectra and PSD/GLORYS12 ratio (paper style: one variable per row)
        for region in psd_agg['region'].unique():
            plot_psd_paper(psd_combined, region, len(psd_dfs), output_dir / f"psd_analysis_{region}")
            plot_psd_paper(psd_combined, region, len(psd_dfs), output_dir / f"psd_ratio_{region}", ratio=True)
    else:
        print("  ⚠ No psd_analysis.csv files found")

    # ===== EFFECTIVE RESOLUTION (pseudo-track spectral score) =====
    print("\n3. Effective resolution (pseudo-track PSD score)...")
    score_dfs, lambda_dfs = [], []
    for run_dir in all_run_dirs:
        s_csv, l_csv = run_dir / "metrics" / "psd_segment_scores.csv", run_dir / "metrics" / "effective_resolution.csv"
        if s_csv.exists() and l_csv.exists():
            score_dfs.append(pd.read_csv(s_csv))
            lambda_dfs.append(pd.read_csv(l_csv))
    if lambda_dfs:
        score_combined = pd.concat(score_dfs, ignore_index=True)
        lambda_combined = pd.concat(lambda_dfs, ignore_index=True)
        lam_agg = lambda_combined.groupby(['region', 'timestep', 'variable']).agg(
            n_runs=('run_id', 'count'),
            lambda_eff_ref_mean=('lambda_eff_reference_km', 'mean'),
            lambda_eff_ref_std=('lambda_eff_reference_km', 'std'),
            lambda_eff_opt_mean=('lambda_eff_optimized_km', 'mean'),
            lambda_eff_opt_std=('lambda_eff_optimized_km', 'std')).reset_index()
        lam_agg.to_csv(output_dir / "effective_resolution_aggregated.csv", index=False)
        print(f"  ✓ effective_resolution_aggregated.csv ({len(lam_agg)} rows)")
        for region in lambda_combined['region'].unique():
            plot_effective_resolution_paper(lambda_combined, region, len(lambda_dfs),
                                            output_dir / f"effective_resolution_{region}")
            for t in sorted(score_combined['timestep'].unique()):
                plot_psd_score_paper(score_combined, lambda_combined, region, t, len(score_dfs),
                                     output_dir / f"psd_score_{region}_t{t}")
    else:
        print("  ⚠ No effective_resolution.csv files found (run backfill_effective_resolution.py)")

    # ===== PIXELIZATION (checkerboard fraction) =====
    aggregate_pixelization(output_dir, all_run_dirs)

# ============ PIXELIZATION (checkerboard fraction) ============
# Same values as gd_optimic.metrics (not imported: gd_optimic pulls in xesmf/ESMF).
CHECKERBOARD_PHYSICAL_LEVEL = 0.038
CHECKERBOARD_THRESHOLD = 0.038 + 0.025
# optimization_history.json key -> (scope, quantity)
CHECKERBOARD_HISTORY_KEYS = {
    'checkerboard_fraction': ('step', 'fraction'),
    'checkerboard_fraction_cumulative': ('cumulative', 'fraction'),
    'checkerboard_amplitude_cumulative': ('cumulative', 'amplitude'),
}

def _load_checkerboard(run_dir):
    """Long-format rows (run_id, scope, quantity, variable, iteration, value, is_best) for one run."""
    run_id = f"{run_dir.parent.name}/{run_dir.name}"
    rows = []
    hist_path = run_dir / "metrics" / "optimization_history.json"
    if hist_path.exists():
        try:
            with open(hist_path) as f:
                hist = json.load(f)
        except Exception:
            hist = {}
        best_it = hist.get('best_iteration')
        for entry in hist.get('history', []):
            for key, (scope, quantity) in CHECKERBOARD_HISTORY_KEYS.items():
                for var, val in (entry.get(key) or {}).items():
                    rows.append((run_id, scope, quantity, var, entry['iteration'], val,
                                 entry['iteration'] == best_it))
    # Older runs: cumulative score backfilled from checkpoints + best IC (skip if logged live)
    bf_path = run_dir / "metrics" / "checkerboard_backfill.csv"
    if bf_path.exists() and not any(r[1] == 'cumulative' for r in rows):
        bf = pd.read_csv(bf_path)
        for _, r in bf.iterrows():
            for quantity in ('fraction', 'amplitude'):
                if pd.notna(r[quantity]):
                    rows.append((run_id, r['scope'], quantity, r['variable'], int(r['iteration']),
                                 float(r[quantity]), r['source'] == 'best'))
    return rows

def plot_pixelization_paper(evo, scope, n_runs, out_stem):
    """Checkerboard fraction vs iteration: one variable per row (total last), mean ± 1 std across runs."""
    d_scope = evo[(evo['scope'] == scope) & (evo['quantity'] == 'fraction')]
    vars_list = [v for v in _ordered_vars(d_scope) if v != 'total'] + \
                (['total'] if 'total' in set(d_scope['variable']) else [])
    st = STYLE['opt']
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.5))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            d = d_scope[d_scope['variable'] == var].sort_values('iteration')
            it, m, sd = d['iteration'].values, d['mean'].values, np.nan_to_num(d['std'].values)
            ax.fill_between(it, m - sd, m + sd, color=st['color'], alpha=BAND_ALPHA, lw=0, zorder=2)
            ax.plot(it, m, color=st['color'], lw=st['lw'], marker='o' if len(it) < 30 else None,
                    ms=3, zorder=3)
            ax.axhline(CHECKERBOARD_PHYSICAL_LEVEL, color=WINDOW_COLOR, lw=1.0, zorder=1)
            ax.axhline(CHECKERBOARD_THRESHOLD, **BORDER_STYLE, zorder=4)
            ax.set_ylabel('Checkerboard\nfraction')
            ax.set_ylim(bottom=0)
            ax.grid(True, axis='y')
            if var == 'total':
                ax.set_title(f"({chr(97 + idx)}) Mean over variables", loc='left', fontweight='bold', pad=3)
            else:
                _panel_label(ax, idx, var)
        axes[-1].set_xlabel('Iteration')
        handles = [Line2D([], [], color=st['color'], lw=st['lw'], label=f'{scope.capitalize()} IC update'),
                   _band_handle(n_runs),
                   Line2D([], [], color=WINDOW_COLOR, lw=1.0,
                          label=f'Physical level ({CHECKERBOARD_PHYSICAL_LEVEL:.3f})'),
                   Line2D([], [], **BORDER_STYLE, label=f'Pixelization threshold ({CHECKERBOARD_THRESHOLD:.3f})')]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.96))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

def aggregate_pixelization(output_dir, all_run_dirs):
    print("\n4. Pixelization (checkerboard fraction of the IC update)...")
    rows = [r for run_dir in all_run_dirs for r in _load_checkerboard(run_dir)]
    if not rows:
        print("  ⚠ No checkerboard scores found (optimization_history.json / checkerboard_backfill.csv)")
        return
    cb = pd.DataFrame(rows, columns=['run_id', 'scope', 'quantity', 'variable', 'iteration', 'value', 'is_best'])
    keys = ['run_id', 'scope', 'quantity', 'variable']

    # Iteration-wise statistics across runs
    evo = cb.groupby(['scope', 'quantity', 'variable', 'iteration'])['value'].agg(
        n_runs='count', mean='mean', std='std').reset_index()
    evo.to_csv(output_dir / "pixelization_evolution_aggregated.csv", index=False)
    print(f"  ✓ pixelization_evolution_aggregated.csv ({len(evo)} rows)")

    # Per-run score: last iteration ("final") and best iteration / optimized IC ("best")
    cb = cb.sort_values('iteration', kind='stable')
    final = cb.groupby(keys).tail(1).set_index(keys)
    best = cb[cb['is_best']].groupby(keys).tail(1).set_index(keys)
    per_run = pd.DataFrame({'final_iteration': final['iteration'], 'final': final['value'],
                            'best_iteration': best['iteration'], 'best': best['value']}).reset_index()
    is_frac = per_run['quantity'] == 'fraction'
    for col in ('final', 'best'):
        flag = (per_run[col] > CHECKERBOARD_THRESHOLD).astype(float)
        per_run[f'{col}_pixelized'] = flag.where(is_frac & per_run[col].notna())
    per_run.to_csv(output_dir / "pixelization_per_run.csv", index=False)
    print(f"  ✓ pixelization_per_run.csv ({per_run['run_id'].nunique()} runs)")

    # Summary across runs (*_share_pixelized: fraction of runs above the threshold)
    summary = per_run.groupby(['scope', 'quantity', 'variable']).agg(
        n_runs=('run_id', 'count'),
        final_mean=('final', 'mean'), final_std=('final', 'std'),
        best_mean=('best', 'mean'), best_std=('best', 'std'),
        final_share_pixelized=('final_pixelized', 'mean'),
        best_share_pixelized=('best_pixelized', 'mean')).reset_index()
    summary.to_csv(output_dir / "pixelization_summary.csv", index=False)
    with open(output_dir / "pixelization_summary.json", 'w') as f:
        json.dump({'physical_level': CHECKERBOARD_PHYSICAL_LEVEL, 'threshold': CHECKERBOARD_THRESHOLD,
                   'summary': json.loads(summary.to_json(orient='records'))}, f, indent=2)
    print("  ✓ pixelization_summary.csv/.json")
    for _, r in summary[(summary['quantity'] == 'fraction') & (summary['variable'] == 'total')].iterrows():
        print(f"    {r['scope']:>10} total: final {r['final_mean']:.3f} ± {np.nan_to_num(r['final_std']):.3f}"
              f", pixelized in {np.nan_to_num(r['final_share_pixelized']):.0%} of {r['n_runs']} runs")

    for scope in sorted(evo.loc[evo['quantity'] == 'fraction', 'scope'].unique()):
        n_runs = cb.loc[(cb['scope'] == scope) & (cb['quantity'] == 'fraction'), 'run_id'].nunique()
        plot_pixelization_paper(evo, scope, n_runs, output_dir / f"pixelization_{scope}")

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
    seen_runs = set()
    
    for pdir in parent_dirs:
        if not pdir.is_dir():
            print(f"✗ Not a directory: {pdir}")
            continue
        
        rdirs = [d for d in find_run_dirs(pdir) if d not in seen_runs]
        seen_runs.update(rdirs)
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
