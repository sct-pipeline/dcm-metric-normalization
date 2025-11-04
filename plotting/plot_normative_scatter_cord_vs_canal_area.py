#!/usr/bin/env python
"""
Plot scatter plots between spinal cord area and canal area (normative data) for each vertebral level (C2–C7), and report correlation statistics.

Usage:
    python plot_normative_scatter_cord_vs_canal_area.py \
        -path-HC <path_to_normative_data_folder> \
        -participants-file-pam50 <path_to_participants_tsv_file> \
        -o <output_directory>
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, normaltest, pearsonr

from utils import format_pvalue

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

VERTEBRAL_LEVELS = [2, 3, 4, 5, 6, 7]  # C2–C7
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}

METRICS_DTYPE = {
    'MEAN(area)': 'float64',
    'VertLevel': 'int64',
    'Slice (I->S)': 'int64',
}


def load_normative_df(normative_dir, participants_file=None):
    # Load cord and canal metrics
    cord_dir = os.path.join(normative_dir, 'spinal_cord')
    canal_dir = os.path.join(normative_dir, 'canal')
    cord_df = pd.DataFrame()
    canal_df = pd.DataFrame()
    for file in os.listdir(cord_dir):
        if 'PAM50.csv' in file:
            df = pd.read_csv(os.path.join(cord_dir, file), dtype=METRICS_DTYPE)
            cord_df = pd.concat([cord_df, df], axis=0, ignore_index=True)
    for file in os.listdir(canal_dir):
        if 'PAM50.csv' in file:
            df = pd.read_csv(os.path.join(canal_dir, file), dtype=METRICS_DTYPE)
            canal_df = pd.concat([canal_df, df], axis=0, ignore_index=True)
    # Add participant_id
    cord_df.insert(0, 'participant_id', cord_df['Filename'].str.split('/').str[0])
    canal_df.insert(0, 'participant_id', canal_df['Filename'].str.split('/').str[0])
    # Merge cord and canal on participant_id, VertLevel, Slice (I->S)
    merged = pd.merge(
        cord_df[['participant_id', 'VertLevel', 'Slice (I->S)', 'MEAN(area)']],
        canal_df[['participant_id', 'VertLevel', 'Slice (I->S)', 'MEAN(area)']],
        on=['participant_id', 'VertLevel', 'Slice (I->S)'],
        suffixes=('_cord', '_canal')
    )
    # Optionally merge participants.tsv info
    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        merged = merged.merge(df_participants[['participant_id', 'age', 'sex']], on='participant_id', how='left')
    # Keep only VertLevel C2–C7
    merged = merged[(merged['VertLevel'] >= 2) & (merged['VertLevel'] <= 7)]
    merged = merged.dropna(subset=['MEAN(area)_cord', 'MEAN(area)_canal'])
    # Compute mean per level for each subject
    grouped = merged.groupby(['participant_id', 'VertLevel']).agg({
        'MEAN(area)_cord': 'mean',
        'MEAN(area)_canal': 'mean',
        'age': 'first',
        'sex': 'first'
    }).reset_index()
    return grouped

def plot_scatter_grid(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    results = []
    normality_results = []

    # Compute unique subject counts per sex for the master title
    total_n = df['participant_id'].nunique()
    suptitle = f"Normative spinal cord vs spinal canal area per level (n={total_n})"
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        x = df_level['MEAN(area)_canal']
        y = df_level['MEAN(area)_cord']
        # Normality test
        stat_x, p_x = normaltest(x)
        stat_y, p_y = normaltest(y)
        normality_results.append({
            'level': LEVEL_TO_LABEL[level],
            'canal_stat': stat_x, 'canal_p': p_x,
            'cord_stat': stat_y, 'cord_p': p_y
        })
        # Spearman and Pearson correlation
        r_spear, p_spear = spearmanr(x, y)
        r_pear, p_pear = pearsonr(x, y)
        sns.scatterplot(x=x, y=y, ax=ax, color='black', alpha=0.6)
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            pfit = np.poly1d(z)
            x_vals = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_vals, pfit(x_vals), color='black', linewidth=2)
        stats_text = (f"Spearman r={r_spear:.2f}, p{format_pvalue(p_spear)}\n"
                      f"Pearson r={r_pear:.2f}, p{format_pvalue(p_pear)}\n"
                      f"Normality canal p{format_pvalue(p_x)}\n"
                      f"Normality cord p{format_pvalue(p_y)}")
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=12)
        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        results.append({
            'level': LEVEL_TO_LABEL[level],
            'spearman_r': r_spear,
            'spearman_p': p_spear,
            'pearson_r': r_pear,
            'pearson_p': p_pear,
            'n': len(x)
        })
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'normative_scatter_cord_vs_canal_area_perlevel.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    print("\nCorrelation summary per vertebral level:")
    for res in results:
        print(f"{res['level']}: Spearman r={res['spearman_r']:.2f} (p{format_pvalue(res['spearman_p'])}), "
              f"Pearson r={res['pearson_r']:.2f} (p{format_pvalue(res['pearson_p'])}), n={res['n']}")
    print("\nNormality test results (D'Agostino and Pearson):")
    for norm_res in normality_results:
        print(f"{norm_res['level']}: Canal p={format_pvalue(norm_res['canal_p'])}, Cord p={format_pvalue(norm_res['cord_p'])}")

def plot_scatter_grid_by_sex(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    results = []
    sex_colors = {'M': 'blue', 'F': 'red'}

    # Compute unique subject counts per sex for the master title
    total_n = df['participant_id'].nunique()
    n_m = df[df['sex'] == 'M']['participant_id'].nunique()
    n_f = df[df['sex'] == 'F']['participant_id'].nunique()
    suptitle = f"Normative spinal cord vs spinal canal area per level (n={total_n}; M={n_m}, F={n_f})"
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for sex, color in sex_colors.items():
            df_sex = df_level[df_level['sex'] == sex]
            x = df_sex['MEAN(area)_canal']
            y = df_sex['MEAN(area)_cord']
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, label=f'{"Male" if sex=="M" else "Female"}')
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                pfit = np.poly1d(z)
                x_vals = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_vals, pfit(x_vals), color=color, linewidth=2)
            r, p = spearmanr(x, y)
            stats_text = (f"{sex}: r={r:.2f}, p{format_pvalue(p)} n={len(x) }")
            ax.text(0.98, 0.02 + 0.08 * (0 if sex == 'M' else 1), stats_text, transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    fontsize=12)
            results.append({
                'level': LEVEL_TO_LABEL[level],
                'sex': sex,
                'r': r,
                'p': p,
                'n': len(x)
            })
        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'normative_scatter_cord_vs_canal_area_perlevel_by_sex.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    print("\nCorrelation summary per vertebral level and sex:")
    for res in results:
        print(res)

def main():
    parser = argparse.ArgumentParser(description="Plot scatter plots between cord and canal area (normative data) per vertebral level.")
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('-o', type=str, required=True,
                        help='Output directory for figure')
    args = parser.parse_args()
    df = load_normative_df(os.path.expandvars(args.path_HC), os.path.expandvars(args.participants_file_pam50))
    plot_scatter_grid(df, os.path.expandvars(args.o))
    plot_scatter_grid_by_sex(df, os.path.expandvars(args.o))

if __name__ == '__main__':
    main()
