#!/usr/bin/env python
"""
Plot scatter plots between spinal cord area and spinal canal area (patients' data) for each vertebral level (C2–C7), and report correlation statistics.

Usage:
    python plot_patient_scatter_cord_vs_canal_area.py \
        --cord-csv <T2w_ax_cord_metrics_perlevel.csv> \
        --canal-csv <T2w_ax_canal_metrics_perlevel.csv> \
        -o <output_directory>

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, normaltest, pearsonr

# reuse project's utilities
from plotting.utils import format_pvalue, fetch_participant_and_session

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

VERTEBRAL_LEVELS = [2, 3, 4, 5, 6, 7]  # C2–C7
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}


def load_patient_df(cord_csv, canal_csv):
    cord_df = pd.read_csv(cord_csv)
    canal_df = pd.read_csv(canal_csv)

    # extract participant_id from Filename using project's util
    cord_df['participant_id'] = cord_df['Filename'].astype(str).map(lambda f: fetch_participant_and_session(f)[0])
    canal_df['participant_id'] = canal_df['Filename'].astype(str).map(lambda f: fetch_participant_and_session(f)[0])

    cord_df['VertLevel'] = pd.to_numeric(cord_df['VertLevel'], errors='coerce').astype('Int64')
    canal_df['VertLevel'] = pd.to_numeric(canal_df['VertLevel'], errors='coerce').astype('Int64')

    merged = pd.merge(
        cord_df[['participant_id', 'VertLevel', 'MEAN(area)']],
        canal_df[['participant_id', 'VertLevel', 'MEAN(area)']],
        on=['participant_id', 'VertLevel'], suffixes=('_cord', '_canal')
    )

    merged = merged[(merged['VertLevel'] >= 2) & (merged['VertLevel'] <= 7)].dropna()
    grouped = merged.groupby(['participant_id', 'VertLevel']).mean(numeric_only=True).reset_index()
    return grouped


def plot_scatter_grid(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    results = []

    total_n = df['participant_id'].nunique()
    fig.suptitle(f"Patients' spinal cord vs spinal canal area per level (n={total_n})", fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        x = df_level['MEAN(area)_canal']
        y = df_level['MEAN(area)_cord']

        # normality only when enough samples
        p_x = p_y = np.nan
        if len(x) > 7:
            stat_x, p_x = normaltest(x)
            stat_y, p_y = normaltest(y)

        # correlations when more than one sample
        r_spear = p_spear = r_pear = p_pear = np.nan
        if len(x) > 1:
            r_spear, p_spear = spearmanr(x, y)
            r_pear, p_pear = pearsonr(x, y)

        sns.scatterplot(x=x, y=y, ax=ax, color='blue', alpha=0.6)
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            pfit = np.poly1d(z)
            x_vals = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_vals, pfit(x_vals), color='red', linewidth=2)

        stats_text = (f"Spearman r={np.nan_to_num(r_spear):.2f}, p{format_pvalue(p_spear)}\n"
                      f"Pearson r={np.nan_to_num(r_pear):.2f}, p{format_pvalue(p_pear)}\n"
                      f"Normality canal p{format_pvalue(p_x)}\n"
                      f"Normality cord p{format_pvalue(p_y)}")
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=10)

        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        results.append({'level': LEVEL_TO_LABEL[level], 'spearman_r': r_spear, 'spearman_p': p_spear, 'pearson_r': r_pear, 'pearson_p': p_pear, 'n': len(x)})

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'patient_scatter_cord_vs_canal_area_perlevel.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")

    for res in results:
        print(f"{res['level']}: Spearman r={res['spearman_r']:.2f} (p{format_pvalue(res['spearman_p'])}), Pearson r={res['pearson_r']:.2f} (p{format_pvalue(res['pearson_p'])}), n={res['n']}")


def main():
    parser = argparse.ArgumentParser(description="Plot scatter plots between patient cord and canal area per vertebral level.")
    parser.add_argument('--cord-csv', required=True, help='CSV file with cord metrics per level (e.g. T2w_ax_cord_metrics_perlevel.csv)')
    parser.add_argument('--canal-csv', required=True, help='CSV file with canal metrics per level (e.g. T2w_ax_canal_metrics_perlevel.csv)')
    parser.add_argument('-o', '--out-dir', required=True, help='Output directory for figures')
    args = parser.parse_args()

    df = load_patient_df(os.path.expandvars(args.cord_csv), os.path.expandvars(args.canal_csv))
    plot_scatter_grid(df, os.path.expandvars(args.out_dir))


if __name__ == '__main__':
    main()
