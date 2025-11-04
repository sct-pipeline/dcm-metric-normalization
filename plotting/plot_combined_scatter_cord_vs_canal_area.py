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

from utils import fetch_participant_and_session
from plot_normative_scatter_cord_vs_canal_area import load_normative_df
from plot_patient_scatter_cord_vs_canal_area import load_patient_df

VERTEBRAL_LEVELS = [2, 3, 4, 5, 6, 7]
LEVEL_TO_LABEL = {2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}


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


def plot_combined(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    palette = {'normative': 'gray', 'patients': 'black'}

    total_counts = df.groupby('cohort')['participant_id'].nunique().to_dict()
    suptitle = f"Combined spinal cord vs spinal canal area per level (n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
    fig.suptitle(suptitle, fontsize=16)

    for i, level in enumerate(VERTEBRAL_LEVELS):
        ax = axes[i]
        df_level = df[df['VertLevel'] == level]
        for cohort, color in palette.items():
            df_c = df_level[df_level['cohort'] == cohort]
            x = df_c['MEAN(area)_canal']
            y = df_c['MEAN(area)_cord']
            sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, label=f"{cohort.capitalize()} (n={df_c['participant_id'].nunique()})")

        ax.set_title(LEVEL_TO_LABEL[level])
        ax.set_xlabel('Canal Area [mm²]')
        ax.set_ylabel('Cord Area [mm²]')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_fig = os.path.join(output_dir, 'combined_scatter_all.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_fig}")


def plot_combined_by_sex(df, output_dir):
    if 'sex' not in df.columns or df['sex'].isna().all():
        print('No sex information available; skipping sex-stratified plots.')
        return

    os.makedirs(output_dir, exist_ok=True)
    sexes = ['M', 'F']
    cohort_palette = {'normative': 'gray', 'patients': 'black'}

    for sex in sexes:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        df_sex = df[df['sex'] == sex]
        if df_sex.empty:
            print(f'No subjects for sex={sex}; skipping.')
            plt.close(fig)
            continue

        total_counts = df_sex.groupby('cohort')['participant_id'].nunique().to_dict()
        suptitle = f"Combined spinal cord vs spinal canal area per level (sex={sex}; n_normative={total_counts.get('normative',0)}, n_patients={total_counts.get('patients',0)})"
        fig.suptitle(suptitle, fontsize=16)

        for i, level in enumerate(VERTEBRAL_LEVELS):
            ax = axes[i]
            df_level = df_sex[df_sex['VertLevel'] == level]
            for cohort, color in cohort_palette.items():
                df_c = df_level[df_level['cohort'] == cohort]
                x = df_c['MEAN(area)_canal']
                y = df_c['MEAN(area)_cord']
                sns.scatterplot(x=x, y=y, ax=ax, color=color, alpha=0.6, label=f"{cohort.capitalize()} (n={df_c['participant_id'].nunique()})")

            ax.set_title(LEVEL_TO_LABEL[level])
            ax.set_xlabel('Canal Area [mm²]')
            ax.set_ylabel('Cord Area [mm²]')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        out_fig = os.path.join(output_dir, f'combined_scatter_by_sex_{sex}.png')
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
    plot_combined_by_sex(combined, os.path.expandvars(args.out_dir))


if __name__ == '__main__':
    main()
