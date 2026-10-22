#!/usr/bin/env python3
"""
Compare RMSE evolution of two aggregated experiments on one figure (paper style).

Reads aggregated_diagnostics/rmse_evolution_aggregated.csv written by
aggregate_all_v3.py and draws 3 curves per variable: Reference (from the
baseline experiment), Optimized baseline, Optimized compared experiment.

Usage:
  python3 compare_rmse_evolution.py BASE_DIR OTHER_DIR -o OUT_DIR \
      [--base-label "Optimized (w/o meta)"] [--other-label "Optimized (meta)"] \
      [--n-runs 50] [--obs-len 7]
  e.g. python3 compare_rmse_evolution.py .tmp/table2_smoothG .tmp/table7_metaInfer \
      -o .tmp/compare_meta_vs_noMeta
"""
import argparse, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_all_v3 import (PAPER_RC, STYLE, BAND_ALPHA, WINDOW_COLOR, BORDER_STYLE,
                              VAR_META, FIG_WIDTH_IN, ROW_HEIGHT_IN, _ordered_vars,
                              _save_paper_fig, _panel_label, _band_handle)

# Okabe-Ito bluish green for the compared experiment
OTHER_STYLE = dict(color='#009E73', lw=1.4, ls='-')

def load_agg(exp_dir):
    return pd.read_csv(Path(exp_dir) / "aggregated_diagnostics" / "rmse_evolution_aggregated.csv")

def plot_rmse_compare(base, other, region, labels, n_runs, obs_len, out_stem):
    """RMSE vs lead time, one variable per row: Reference, Optimized base, Optimized other."""
    b, o = base[base['region'] == region], other[other['region'] == region]
    vars_list = _ordered_vars(b)
    curves = [('ref', b, 'ref', STYLE['ref'], 'Reference'),
              ('base', b, 'opt', STYLE['opt'], labels[0]),
              ('other', o, 'opt', OTHER_STYLE, labels[1])]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(vars_list), 1, sharex=True, squeeze=False,
                                 figsize=(FIG_WIDTH_IN, ROW_HEIGHT_IN * len(vars_list) + 0.6))
        axes = axes[:, 0]
        for idx, (ax, var) in enumerate(zip(axes, vars_list)):
            t = None
            if obs_len:
                ax.axvline(obs_len, **BORDER_STYLE, zorder=4)
            for _, df, key, st, _ in curves:
                d = df[df['variable'] == var].sort_values('timestep')
                t = d['timestep'].values
                m, s = d[f'{key}_rmse_mean'].values, np.nan_to_num(d[f'{key}_rmse_std'].values)
                ax.fill_between(t, m - s, m + s, color=st['color'], alpha=BAND_ALPHA, lw=0, zorder=2)
                ax.plot(t, m, color=st['color'], lw=st['lw'], ls=st['ls'], zorder=3)
            if obs_len:
                ax.axvspan(t.min(), obs_len, color=WINDOW_COLOR, alpha=0.18, lw=0, zorder=0)
            ax.set_ylabel(f"RMSE ({VAR_META.get(var, (var, ''))[1]})")
            ax.grid(True, axis='y')
            ax.set_xlim(t.min(), t.max())
            _panel_label(ax, idx, var)
        axes[-1].set_xlabel('Lead time (days)')
        axes[-1].xaxis.set_major_locator(plt.MultipleLocator(7))
        axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(1))
        handles = [Line2D([], [], color=st['color'], lw=st['lw'], label=lab)
                   for _, _, _, st, lab in curves] + [_band_handle(n_runs)]
        if obs_len:
            handles += [Patch(facecolor=WINDOW_COLOR, alpha=0.35, edgecolor='none',
                              label=f'Assimilation window (days 0–{obs_len})'),
                        Line2D([], [], **BORDER_STYLE, label=f'Assimilation / forecast border (day {obs_len})')]
        fig.align_ylabels(axes)
        fig.tight_layout(h_pad=0.5, rect=(0, 0, 1, 0.95))
        # column-major fill for ncol=3: col 1 = Ref, Opt base; col 2 = Opt other, band; col 3 = window, border
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=1.8)
        _save_paper_fig(fig, out_stem)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('base_dir'); p.add_argument('other_dir')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--base-label', default='Optimized (w/o meta)')
    p.add_argument('--other-label', default='Optimized (meta)')
    p.add_argument('--n-runs', type=int, default=50)
    p.add_argument('--obs-len', type=int, default=7)
    a = p.parse_args()
    base, other = load_agg(a.base_dir), load_agg(a.other_dir)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for region in base['region'].unique():
        plot_rmse_compare(base, other, region, (a.base_label, a.other_label),
                          a.n_runs, a.obs_len, out / f"rmse_evolution_{region}")

if __name__ == "__main__":
    main()
