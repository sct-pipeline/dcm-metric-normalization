#!/usr/bin/env python
"""
Plot scatter plots between spinal cord area and spinal canal area (patients' data) for each vertebral level (C2–C3), and report correlation statistics.

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
from generate_figure_PAM50_multiple_subjects import (read_clinical_file, read_c2c3_file_and_apply_exclusions,
                                                     SEX_COLORS_PATIENTS, MYELOPATHY_COLORS)

TICKS_FONT_SIZE = 20
LABELS_FONT_SIZE = TICKS_FONT_SIZE + 2
TITLE_FONT_SIZE = TICKS_FONT_SIZE + 4

VERTEBRAL_LEVELS = [2, 3]  # C2–C3
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3'}


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()
    results = []

    total_n = df['participant_id'].nunique()
    # fig.suptitle(f"Patients' spinal cord vs spinal canal area per level (n={total_n})", fontsize=TITLE_FONT_SIZE + 2)

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

        r_spear, p_spear = spearmanr(x, y)
        r_pear, p_pear = pearsonr(x, y)

        sns.scatterplot(x=x, y=y, ax=ax, color='black', alpha=0.8, s=80)
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            pfit = np.poly1d(z)
            x_vals = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_vals, pfit(x_vals), color='black', linewidth=5)

        stats_text = (f"Spearman\nr={np.nan_to_num(r_spear):.2f}\np{format_pvalue(p_spear, alpha=.05)}"
                      # f"Pearson r={np.nan_to_num(r_pear):.2f}, p{format_pvalue(p_pear)}\n"
                      # f"Normality canal p{format_pvalue(p_x)}\n"
                      # f"Normality cord p{format_pvalue(p_y)}"
                      )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=TICKS_FONT_SIZE-4)

        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        results.append({'level': LEVEL_TO_LABEL[level], 'spearman_r': r_spear, 'spearman_p': p_spear, 'pearson_r': r_pear, 'pearson_p': p_pear, 'n': len(x)})

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'patient_scatter_cord_vs_canal_area_perlevel.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")

    for res in results:
        print(f"{res['level']}: Spearman r={res['spearman_r']:.2f} (p{format_pvalue(res['spearman_p'])}), Pearson r={res['pearson_r']:.2f} (p{format_pvalue(res['pearson_p'])}), n={res['n']}")


def plot_scatter_grid_by_sex(df, output_dir):
    """
    Plot scatter grid split by sex (M/F)
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()
    results = []

    total_n = df['participant_id'].nunique()
    n_m = df[df['sex'] == 'M']['participant_id'].nunique()
    n_f = df[df['sex'] == 'F']['participant_id'].nunique()
    # fig.suptitle(f"Patients' spinal cord vs spinal canal area per level (n={total_n}; M={n_m}, F={n_f})", fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for sex, color in SEX_COLORS_PATIENTS.items():
            df_sex = df_level[df_level.get('sex') == sex]
            x = df_sex['MEAN(area)_canal']
            y = df_sex['MEAN(area)_cord']
            sex_text = 'Male' if sex == 'M' else 'Female'
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, s=80)  # label=sex_text
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                pfit = np.poly1d(z)
                x_vals = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_vals, pfit(x_vals), color=color, linewidth=5)
            r, p = spearmanr(x, y)
            stats_text = f"{sex_text} (n={len(x)})\nr={np.nan_to_num(r):.2f}\np{format_pvalue(p, alpha=.05)}"
            ax.text(0.98, 0.02 if sex == 'M' else 0.15, stats_text, transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0), fontsize=TICKS_FONT_SIZE-4, color=color)
            results.append({'level': LEVEL_TO_LABEL[level], 'sex': sex, 'r': r, 'p': p, 'n': len(x)})

        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        # ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'patient_scatter_cord_vs_canal_area_perlevel_by_sex.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")
    print("\nCorrelation summary per vertebral level and sex:")
    for res in results:
        print(res)


def plot_scatter_grid_by_myelopathy(df, output_dir):
    """
    Plot scatter grid split by myelopathy status.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()
    results = []

    total_n = df['participant_id'].nunique()
    n_yes = df[df['Myelopathy'] == 1]['participant_id'].nunique()
    n_no = df[df['Myelopathy'] == 0]['participant_id'].nunique()
    # fig.suptitle(f"Patients' spinal cord vs spinal canal area per level (n={total_n}; Yes={n_yes}, No={n_no})", fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for myelopathy_status, color in MYELOPATHY_COLORS.items():
            df_myelopathy = df_level[df_level.get('Myelopathy') == myelopathy_status]
            x = df_myelopathy['MEAN(area)_canal']
            y = df_myelopathy['MEAN(area)_cord']
            myelopathy_text = 'Myelopathy yes' if myelopathy_status == 'yes' else 'Myelopathy no'
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, s=80)      # label=myelopathy_text
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                pfit = np.poly1d(z)
                x_vals = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_vals, pfit(x_vals), color=color, linewidth=5)
            r, p = spearmanr(x, y)
            stats_text = f"{myelopathy_text} (n={len(x)})\nr={np.nan_to_num(r):.2f}\np{format_pvalue(p, alpha=.05)}"
            ax.text(0.98, 0.02 if myelopathy_status == 'yes' else 0.15, stats_text, transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0), fontsize=TICKS_FONT_SIZE-4, color=color)
            results.append({'level': LEVEL_TO_LABEL[level], 'myelopathy': myelopathy_status, 'r': r, 'p': p, 'n': len(x)})

        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        # ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'patient_scatter_cord_vs_canal_area_perlevel_by_myelopathy.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")
    print("\nCorrelation summary per vertebral level and myelopathy:")
    for res in results:
        print(res)


def main():
    parser = argparse.ArgumentParser(description="Plot scatter plots between patient cord and canal area per vertebral level.")
    parser.add_argument('-cord-csv', required=True, help='CSV file with cord metrics per level (e.g. T2w_ax_cord_metrics_perlevel.csv)')
    parser.add_argument('-canal-csv', required=True, help='CSV file with canal metrics per level (e.g. T2w_ax_canal_metrics_perlevel.csv)')
    parser.add_argument('-clinical-file', required=True, type=str, help='Path to Excel file with clinical scores (columns like total_mjoa_bl, nurick_bl, ...)')
    parser.add_argument('-participants-to-use', required=True, type=str, help='Path to text file with participant IDs to include (one ID per line)')
    parser.add_argument('-c2c3-file', required=False, type=str, default='$HOME/code/dcm-metric-normalization/scripts/dcm-zurich_T2w_ax_ses-M0_canal_analysis.txt',
                        help="File with list of subjects to use C2 or C3 vert level.")
    parser.add_argument('-o', required=True, help='Output directory for figures')
    args = parser.parse_args()

    # Load cord and canal morphometrics
    df = load_patient_df(os.path.expandvars(args.cord_csv), os.path.expandvars(args.canal_csv))

    # Read text file with levels to use (C3 or C2,C3 or exclude)
    df = read_c2c3_file_and_apply_exclusions(df, os.path.expandvars(args.c2c3_file))

    # Load clinical data
    df_clinical, _ = read_clinical_file(os.path.abspath(args.clinical_file))
    # Rename myelopathy values from 0 to 'myelopathy no' and 1 to 'myelopathy yes'
    df_clinical['myelopathy'] = df_clinical['myelopathy'].map({0: 'myelopathy_no', 1: 'myelopathy_yes'})

    df = df.merge(
        df_clinical[['participant_id', 'sex', 'Myelopathy']],
        on='participant_id', how='inner'
    )

    # Read txt file with participant IDs to include
    with open(args.participants_to_use, 'r') as f:
        participant_ids = [line.strip() for line in f if line.strip()]
    # Keep only requested participants
    df = df[df['participant_id'].isin(participant_ids)].copy()

    plot_scatter_grid(df, os.path.expandvars(args.o))
    plot_scatter_grid_by_sex(df, os.path.expandvars(args.o))
    plot_scatter_grid_by_myelopathy(df, os.path.expandvars(args.o))


if __name__ == '__main__':
    main()
