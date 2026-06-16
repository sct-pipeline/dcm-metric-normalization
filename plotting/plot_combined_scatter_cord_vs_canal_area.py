#!/usr/bin/env python
"""
Plot scatter plots between spinal cord area and spinal canal area
for both normative and patient cohorts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from utils import fetch_participant_and_session, format_pvalue
from plot_normative_scatter_cord_vs_canal_area import load_normative_df
from plot_patient_scatter_cord_vs_canal_area import load_patient_df
from generate_figure_PAM50_multiple_subjects import (read_clinical_file, read_c2c3_file_and_apply_exclusions,
                                                     NORMATIVE_C2_COLORS, SEX_COLORS_PATIENTS, SEX_COLORS_NORMATIVE,
                                                     MYELOPATHY_COLORS)

LABELS_FONT_SIZE = 20
TICKS_FONT_SIZE = 20
TITLE_FONT_SIZE = 20

VERTEBRAL_LEVELS = [3]  # C2–C3
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3'}

cohort_markers = {'normative': 'o', 'patients': 'X'}

def load_perlevel_df(cord_csv, canal_csv, cohort_label, participants_file=None):
    cord_df = pd.read_csv(cord_csv)
    canal_df = pd.read_csv(canal_csv)

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

    # merge sex if participants file provided
    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        if 'sex' in df_participants.columns:
            grouped = grouped.merge(df_participants[['participant_id', 'sex']], on='participant_id', how='left')
        else:
            grouped['sex'] = np.nan
    else:
        grouped['sex'] = np.nan

    grouped['cohort'] = cohort_label
    return grouped


def plot_combined_by_myelopathy(df, output_dir):
    mpl.rcParams['font.family'] = 'Arial'
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, len(VERTEBRAL_LEVELS), figsize=(6*len(VERTEBRAL_LEVELS), 6), sharey=True)
    if len(VERTEBRAL_LEVELS) == 1:
        axes = [axes]  # Make it a list when there's only one subplot
    else:
        axes = axes.ravel()
    results = []

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    # suptitle = f"Spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    # fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, marker in cohort_markers.items():
            df_cohort = df_level[df_level['cohort'] == cohort]

            # Normative cohort
            if cohort == 'normative':
                df_c = df_level[df_level['cohort'] == cohort]
                x = df_c['MEAN(area)_canal']
                y = df_c['MEAN(area)_cord']
                sns.scatterplot(x=x, y=y, ax=ax, color='black', alpha=0.3, marker=cohort_markers[cohort], s=80)
                                # label=f"{cohort.capitalize()} (n={df_c['participant_id'].nunique()})")

                r_norm, p_norm = spearmanr(x, y)
                stats_text = f"Normative (n={len(x)})\nr={np.nan_to_num(r_norm):.2f}\np{format_pvalue(p_norm, alpha=.05)}"
                ax.text(0.68, 0.02, stats_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0), fontsize=TICKS_FONT_SIZE - 4,
                        color='black')

                # add linear fit (linear regression) per cohort for this level
                x_vals = x.dropna().values
                y_vals = y.dropna().values
                if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                    z = np.polyfit(x_vals, y_vals, 1)
                    pfit = np.poly1d(z)
                    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(xs, pfit(xs), color='black', linewidth=5, alpha=0.5)
            # Patients split by myelopathy status
            else:
                for myelopathy_status, color in MYELOPATHY_COLORS.items():
                    df_plot = df_cohort[df_cohort.get('Myelopathy') == myelopathy_status]
                    x = df_plot['MEAN(area)_canal']
                    y = df_plot['MEAN(area)_cord']
                    myelopathy_text = 'T2w+' if myelopathy_status == 'yes' else 'T2w-'
                    sns.scatterplot(x=x, y=y, ax=ax, color=color, marker=marker, s=80, edgecolor='w', alpha=0.8)
                    # add linear fit (linear regression) for this cohort+sex if enough variation
                    x_vals = x.dropna().values
                    y_vals = y.dropna().values
                    if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                        z = np.polyfit(x_vals, y_vals, 1)
                        pfit = np.poly1d(z)
                        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(xs, pfit(xs), color=color, linewidth=5)
                    r, p = spearmanr(x, y)
                    stats_text = f"{myelopathy_text} (n={len(x)})\nr={np.nan_to_num(r):.2f}\np{format_pvalue(p, alpha=.05)}"
                    ax.text(0.98, 0.02 if myelopathy_status == 'yes' else 0.18, stats_text, transform=ax.transAxes,
                            verticalalignment='bottom', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0), fontsize=TICKS_FONT_SIZE - 4,
                            color=color)
                    results.append(
                        {'level': LEVEL_TO_LABEL[level], 'myelopathy': myelopathy_status, 'r': r, 'p': p, 'n': len(x)})

        # ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # # build custom legend: cohort-specific sex colors and cohort markers
        # # compute per-cohort, per-sex participant counts for this level
        # counts = {}
        # for cohort in cohort_markers:
        #     df_cohort = df_level[df_level['cohort'] == cohort]
        #     for sex_key in ['M', 'F']:
        #         counts[(cohort, sex_key)] = int(df_cohort[df_cohort['sex'] == sex_key]['participant_id'].nunique())
        #
        # handles = [
        #     Line2D([0], [0], marker=cohort_markers['normative'], color='w', markerfacecolor=SEX_COLORS_NORMATIVE['M'], markersize=8, label=f"Normative Male (n={counts.get(('normative','M'),0)})"),
        #     Line2D([0], [0], marker=cohort_markers['normative'], color='w', markerfacecolor=SEX_COLORS_NORMATIVE['F'], markersize=8, label=f"Normative Female (n={counts.get(('normative','F'),0)})"),
        #     Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=SEX_COLORS_PATIENTS['M'], markersize=8, label=f"Patients Male (n={counts.get(('patients','M'),0)})"),
        #     Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=SEX_COLORS_PATIENTS['F'], markersize=8, label=f"Patients Female (n={counts.get(('patients','F'),0)})")]
        # ax.legend(handles=handles)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_by_myelopathy.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")


def plot_combined_persex(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    suptitle = f"Spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, marker in cohort_markers.items():
            df_cohort = df_level[df_level['cohort'] == cohort]
            # plot sexes together within the cohort
            for sex_key in ['M', 'F']:
                mask = df_cohort['sex'] == sex_key

                df_plot = df_cohort[mask]
                if df_plot.empty:
                    continue

                color = SEX_COLORS_NORMATIVE[sex_key] if cohort == 'normative' else SEX_COLORS_PATIENTS[sex_key]

                x = df_plot['MEAN(area)_canal']
                y = df_plot['MEAN(area)_cord']
                sns.scatterplot(x=x, y=y, ax=ax, color=color, marker=marker, s=60, edgecolor='w', alpha=0.8)

                # add linear fit (linear regression) for this cohort+sex if enough variation
                x_vals = x.dropna().values
                y_vals = y.dropna().values
                if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                    z = np.polyfit(x_vals, y_vals, 1)
                    pfit = np.poly1d(z)
                    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(xs, pfit(xs), color=color, linewidth=2)

        ax.set_title(LEVEL_TO_LABEL[level])
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # build custom legend: cohort-specific sex colors and cohort markers
        # compute per-cohort, per-sex participant counts for this level
        counts = {}
        for cohort in cohort_markers:
            df_cohort = df_level[df_level['cohort'] == cohort]
            for sex_key in ['M', 'F']:
                counts[(cohort, sex_key)] = int(df_cohort[df_cohort['sex'] == sex_key]['participant_id'].nunique())

        handles = [
            Line2D([0], [0], marker=cohort_markers['normative'], color='w', markerfacecolor=SEX_COLORS_NORMATIVE['M'], markersize=8, label=f"Normative Male (n={counts.get(('normative','M'),0)})"),
            Line2D([0], [0], marker=cohort_markers['normative'], color='w', markerfacecolor=SEX_COLORS_NORMATIVE['F'], markersize=8, label=f"Normative Female (n={counts.get(('normative','F'),0)})"),
            Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=SEX_COLORS_PATIENTS['M'], markersize=8, label=f"Patients Male (n={counts.get(('patients','M'),0)})"),
            Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=SEX_COLORS_PATIENTS['F'], markersize=8, label=f"Patients Female (n={counts.get(('patients','F'),0)})")]
        ax.legend(handles=handles)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_by_sex.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")

def plot_combined(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    axes = axes.ravel()

    palette = {'normative': 'gray', 'patients': 'black'}

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    # suptitle = f"Spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    # fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, color in palette.items():
            df_c = df_level[df_level['cohort'] == cohort]
            x = df_c['MEAN(area)_canal']
            y = df_c['MEAN(area)_cord']
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.8, marker=cohort_markers[cohort],
                            label=f"{cohort.capitalize()} (n={df_c['participant_id'].nunique()})", s=80)

            # add linear fit (linear regression) per cohort for this level
            x_vals = x.dropna().values
            y_vals = y.dropna().values
            if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                z = np.polyfit(x_vals, y_vals, 1)
                pfit = np.poly1d(z)
                xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(xs, pfit(xs), color=color, linewidth=5)

        ax.set_title(f"{LEVEL_TO_LABEL[level]}", fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_all.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")

def plot_combined_by_normative_c2_area(df, output_dir):
    """
    Plot combined scatter plots split by normative C2 cord area groups for patients.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    suptitle = f"Spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, marker in cohort_markers.items():
            df_cohort = df_level[df_level['cohort'] == cohort]

            # No split for normative cohort
            if cohort == 'normative':
                x = df_cohort['MEAN(area)_canal']
                y = df_cohort['MEAN(area)_cord']
                sns.scatterplot(x=x, y=y, ax=ax, color='gray', marker=marker, s=60, edgecolor='w', alpha=0.8)
                # add linear fit (linear regression) per cohort for this level
                x_vals = x.dropna().values
                y_vals = y.dropna().values
                if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                    z = np.polyfit(x_vals, y_vals, 1)
                    pfit = np.poly1d(z)
                    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax.plot(xs, pfit(xs), color='gray', linewidth=2)
            elif cohort == 'patients':
                for group_key in ['Below normative mean C2 cord area', 'Above normative mean C2 cord area']:
                    mask = df_cohort['normative_mean_c2'] == group_key
                    df_plot = df_cohort[mask]
                    x = df_plot['MEAN(area)_canal']
                    y = df_plot['MEAN(area)_cord']
                    sns.scatterplot(x=x, y=y, ax=ax, color=NORMATIVE_C2_COLORS[group_key], marker=marker,
                                    s=60, edgecolor='w', alpha=0.8)

                    # add linear fit (linear regression) for this cohort+sex if enough variation
                    x_vals = x.dropna().values
                    y_vals = y.dropna().values
                    if x_vals.size >= 2 and (x_vals.max() - x_vals.min()) > 0:
                        z = np.polyfit(x_vals, y_vals, 1)
                        pfit = np.poly1d(z)
                        xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                        ax.plot(xs, pfit(xs), color=NORMATIVE_C2_COLORS[group_key], linewidth=2)

        ax.set_title(LEVEL_TO_LABEL[level])
        ax.set_xlabel('Canal Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Cord Area [mm²]', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # build custom legend: cohort-specific sex colors and cohort markers
        # compute per-cohort, per-sex participant counts for this level
        counts = {}
        for cohort in cohort_markers:
            df_cohort = df_level[df_level['cohort'] == cohort]
            if cohort == 'patients':
                for group_key in ['Below normative mean C2 cord area', 'Above normative mean C2 cord area']:
                    counts[(cohort, group_key)] = int(df_cohort[df_cohort['normative_mean_c2'] == group_key]['participant_id'].nunique())
            elif cohort == 'normative':
                # No split for normative cohort, just count total
                counts[(cohort, 'all')] = int(df_cohort['participant_id'].nunique())

        handles = [
            Line2D([0], [0], marker=cohort_markers['normative'], color='w', markerfacecolor='gray', markersize=8, label=f"Normative (n={counts.get(('normative','all'),0)})"),
            Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=NORMATIVE_C2_COLORS['Below normative mean C2 cord area'], markersize=8, label=f"Patients Below normative mean C2 cord area (n={counts.get(('patients','Below normative mean C2 cord area'),0)})"),
            Line2D([0], [0], marker=cohort_markers['patients'], color='w', markerfacecolor=NORMATIVE_C2_COLORS['Above normative mean C2 cord area'], markersize=8, label=f"Patients Above normative mean C2 cord area (n={counts.get(('patients','Above normative mean C2 cord area'),0)})")
        ]
        ax.legend(handles=handles)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_by_normative_mean_c2.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description='Combine normative and patient per-level cord/canal CSVs and plot together.')
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('--pat-cord-csv', required=True)
    parser.add_argument('--pat-canal-csv', required=True)
    parser.add_argument('-clinical-file', required=True, type=str, help='Path to Excel file with clinical scores (columns like total_mjoa_bl, nurick_bl, ...)')
    parser.add_argument('-participants-to-use', required=True, type=str, help='Path to text file with participant IDs to include (one ID per line)')
    parser.add_argument('-c2c3-file', required=False, type=str, default='$HOME/code/dcm-metric-normalization/scripts/dcm-zurich_T2w_ax_ses-M0_canal_analysis.txt',
                        help="File with list of subjects to use C2 or C3 vert level.")
    parser.add_argument('-o', '--out-dir', required=True)
    args = parser.parse_args()

    norm_df = load_normative_df(os.path.expandvars(args.path_HC), os.path.expandvars(args.participants_file_pam50))
    pat_df = load_patient_df(os.path.expandvars(args.pat_cord_csv), os.path.expandvars(args.pat_canal_csv))

    # Read text file with levels to use (C3 or C2,C3 or exclude)
    pat_df = read_c2c3_file_and_apply_exclusions(pat_df, os.path.expandvars(args.c2c3_file))

    # Load clinical data
    df_clinical, _ = read_clinical_file(os.path.abspath(args.clinical_file))
    # Rename myelopathy values from 0 to 'myelopathy no' and 1 to 'myelopathy yes'
    df_clinical['myelopathy'] = df_clinical['myelopathy'].map({0: 'myelopathy_no', 1: 'myelopathy_yes'})

    pat_df = pat_df.merge(
        df_clinical[['participant_id', 'sex', 'Myelopathy']],
        on='participant_id', how='inner'
    )

    # Read txt file with participant IDs to include
    with open(args.participants_to_use, 'r') as f:
        participant_ids = [line.strip() for line in f if line.strip()]
    # Keep only requested participants
    pat_df = pat_df[pat_df['participant_id'].isin(participant_ids)].copy()

    combined = pd.concat([norm_df.assign(cohort='normative'), pat_df.assign(cohort='patients')], ignore_index=True)

    plot_combined(combined, os.path.expandvars(args.out_dir))
    plot_combined_by_myelopathy(combined, os.path.expandvars(args.out_dir))
    #plot_combined_persex(combined, os.path.expandvars(args.out_dir))
    #plot_combined_by_normative_c2_area(combined, os.path.expandvars(args.out_dir))


if __name__ == '__main__':
    main()
