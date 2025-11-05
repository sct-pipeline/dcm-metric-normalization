#!/usr/bin/env python
"""
Plot scatter plots between spinal cord area and spinal canal area
for both normative and patient cohorts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from utils import fetch_participant_and_session
from plot_normative_scatter_cord_vs_canal_area import load_normative_df
from plot_patient_scatter_cord_vs_canal_area import load_patient_df

VERTEBRAL_LEVELS = [2, 3, 4, 5, 6, 7]
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

SEX_COLORS_NORMATIVE = {
    'M': 'blue',
    'F': 'red',
}

SEX_COLORS_PATIENTS = {
    'M': '#1f77b4',     # light blue
    'F': '#ff7f0e',     # orange
}

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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    palette = {'normative': 'gray', 'patients': 'black'}

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    suptitle = f"Spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE + 2)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, color in palette.items():
            df_c = df_level[df_level['cohort'] == cohort]
            x = df_c['MEAN(area)_canal']
            y = df_c['MEAN(area)_cord']
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, marker=cohort_markers[cohort],
                            label=f"{cohort.capitalize()} (n={df_c['participant_id'].nunique()})")

            # add linear fit (linear regression) per cohort for this level
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
        ax.legend()

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_all.png')
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
    parser.add_argument('--pat-participants-file', required=False, help='Participants TSV for patient cohort (tab-separated)')
    parser.add_argument('-o', '--out-dir', required=True)
    args = parser.parse_args()

    norm_df = load_normative_df(os.path.expandvars(args.path_HC), os.path.expandvars(args.participants_file_pam50))
    pat_df = load_patient_df(os.path.expandvars(args.pat_cord_csv), os.path.expandvars(args.pat_canal_csv), os.path.expandvars(args.pat_participants_file))

    combined = pd.concat([norm_df.assign(cohort='normative'), pat_df.assign(cohort='patients')], ignore_index=True)

    plot_combined(combined, os.path.expandvars(args.out_dir))
    plot_combined_persex(combined, os.path.expandvars(args.out_dir))


if __name__ == '__main__':
    main()
