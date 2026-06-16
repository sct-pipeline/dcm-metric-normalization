#!/usr/bin/env python
#
# Plot violin plots showing univariate associations between spinal cord area at specified vert level and mJOA scores
#
# The script reads:
# - Clinical scores from an Excel file (total_mjoa column)
# - Spinal cord area metrics from a CSV file (VertLevel 3, MEAN(area) column)
#
# Example usage:
#   python plot_mjoa_cord_area_violin.py
#       -clinical clinical_scores.xlsx
#       -metrics T2w_ax_cord_metrics_perlevel.csv
#       -o output_directory
#
# Author: Jan Valosek, GitHub Copilot (Claude Sonnet 4)
#

import os
import sys
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

from utils import fetch_participant_and_session

# Font sizes for plots
LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

clinical_scores = [
    'total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth',
    'nurick_bl', 'nurick_6mth', 'nurick_12mth',
    'motor_dysfunction_UE_bl', 'motor_dysfunction_LE_bl', 'sensory_dysfunction_LE_bl', 'sphincter_dysfunction_bl'
]

score_to_label = {
    'total_mjoa_bl': 'mJOA baseline',
    'total_mjoa_6mth': 'mJOA 6 months',
    'total_mjoa_12mth': 'mJOA 12 months',
    'nurick_bl': 'Nurick baseline',
    'nurick_6mth': 'Nurick 6 months',
    'nurick_12mth': 'Nurick 12 months',
    'motor_dysfunction_UE_bl': 'Motor Dysfunction UE baseline',
    'motor_dysfunction_LE_bl': 'Motor Dysfunction LE baseline',
    'sensory_dysfunction_LE_bl': 'Sensory Dysfunction LE baseline',
    'sphincter_dysfunction_bl': 'Sphincter Dysfunction baseline'
}

metrics = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)']
metric_to_title = {
    'MEAN(area)': 'Spinal Cord Area',
    'MEAN(diameter_AP)': 'Diameter AP',
    'MEAN(diameter_RL)': 'Diameter RL'
}
metrics_to_labels = {
    'MEAN(area)': 'Spinal Cord Area [mm²]',
    'MEAN(diameter_AP)': 'Diameter AP [mm]',
    'MEAN(diameter_RL)': 'Diameter RL [mm]'
}

def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot violin plots showing association between spinal cord area at specified vert level and mJOA scores")
    parser.add_argument('-clinical', required=True, type=str,
                        help="Excel file with clinical scores (must contain 'total_mjoa_bl' column)")
    parser.add_argument('-metrics', required=True, type=str,
                        help="CSV file with spinal cord metrics per level (must contain 'VertLevel' and 'MEAN(area)' columns)")
    parser.add_argument('-level', required=False, type=int,
                        help="Spinal level to analyze (default: 2)", default=2)
    parser.add_argument('-o', required=True, type=str,
                        help="Output directory for the figure")

    return parser


def load_clinical_data(clinical_file, subject_col='record_id'):
    """
    Load clinical scores from Excel file

    Args:
        clinical_file: Path to Excel file with clinical scores
        subject_col: Column name for subject IDs

    Returns:
        pandas.DataFrame: Clinical data with subject IDs and mJOA scores
    """
    print(f"Loading clinical data from: {clinical_file}")

    # Try different Excel reading methods
    try:
        df_clinical = pd.read_excel(clinical_file)
    except Exception as e:
        sys.exit(f"Error reading Excel file: {e}")

    # Check required columns
    required_cols = [subject_col] + clinical_scores
    missing_cols = [col for col in required_cols if col not in df_clinical.columns]

    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        sys.exit(f"Available columns: {list(df_clinical.columns)}")

    # # Remove rows with missing mJOA scores
    # df_clinical = df_clinical.dropna(subset=clinical_scores)

    # Keep only relevant columns
    df_clinical = df_clinical[required_cols].copy()

    # Rename subject column
    df_clinical = df_clinical.rename(columns={subject_col: 'participant_id'})

    # Format participant_id column; from `1` to `sub-001`
    df_clinical['participant_id'] = df_clinical['participant_id'].apply(
        lambda x: f"sub-{int(x):03d}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x))

    print(f"Loaded clinical data for {len(df_clinical)} subjects")
    for score in clinical_scores:
        print(f"{score} range: {df_clinical[score].min():.2f} - {df_clinical[score].max():.2f}")

    return df_clinical


def load_cord_metrics(metrics_file, level, structure):
    """
    Load spinal cord metrics and filter for specified level

    Args:
        metrics_file: Path to CSV file with cord metrics
        level: Spinal level to filter (default: 3 for C3)
        structure: Structure to filter (default: 3 for C3)

    Returns:
        pandas.DataFrame: Cord metrics data at specified level
    """
    print(f"Loading cord metrics from: {metrics_file}")

    try:
        df_metrics = pd.read_csv(metrics_file)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

    if structure == 'aSCOR':
        metric_columns = ['aSCOR']
        filename_column = 'Filename_sc'
    else:
        metric_columns = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)']
        filename_column = 'Filename'

    # Check required columns
    required_cols = [filename_column, 'VertLevel'] + metric_columns
    missing_cols = [col for col in required_cols if col not in df_metrics.columns]

    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        sys.exit(f"Available columns: {list(df_metrics.columns)}")

    # Filter for specified level
    df_level = df_metrics[df_metrics['VertLevel'] == level].copy()

    if len(df_level) == 0:
        print(f"No data found for VertLevel {level} (C{level})")
        sys.exit(f"Available VertLevels: {sorted(df_metrics['VertLevel'].unique())}")

    # Remove rows with missing values in any metric column
    df_level = df_level.dropna(subset=metric_columns)

    participant_ids = []
    for file_path in df_level[filename_column]:
        participant_id, _ = fetch_participant_and_session(file_path)
        participant_ids.append(participant_id)
    df_level.insert(0, 'participant_id', participant_ids)

    # Keep only relevant columns
    df_level = df_level[['participant_id'] + metric_columns].copy()

    print(f"Loaded C{level} cord metrics for {len(df_level)} measurements")
    for col in metric_columns:
        print(f"{col} range: {df_level[col].min():.2f} - {df_level[col].max():.2f}")

    return df_level


def merge_data(df_clinical, df_metrics, subject_col='participant_id'):
    """
    Merge clinical and metrics data

    Args:
        df_clinical: Clinical data with mJOA scores
        df_metrics: Cord area data at specified level
        subject_col: Column name for subject IDs

    Returns:
        pandas.DataFrame: Merged data
    """
    # Merge on subject ID
    df_merged = pd.merge(df_clinical, df_metrics, on=subject_col, how='inner')

    print(f"Merged data for {len(df_merged)} subjects")

    if len(df_merged) == 0:
        sys.exit("No matching subjects found between clinical and metrics data")

    return df_merged


def plot_violin_association(df, output_dir, level, structure):
    """
    Create 1x1 boxplot (with stripplot, regression line, stats box) for area vs total_mjoa_bl
    Args:
        df: Merged dataframe with mJOA and cord area data
        output_dir: Output directory for the figure
        level: Spinal level (for title)
        structure: Structure name (for title)
    """
    mpl.rcParams['font.family'] = 'Arial'
    os.makedirs(output_dir, exist_ok=True)

    metric = 'MEAN(area)'
    for score in clinical_scores:
        if metric not in df.columns:
            print(f"Metric column '{metric}' not found in dataframe.")
            return
        df_plot = df[['participant_id', score, metric]].dropna()
        fig, ax = plt.subplots(figsize=(12, 6))
        r, p_value = spearmanr(df_plot[score], df_plot[metric])
        sns.boxplot(data=df_plot, x=score, y=metric,
                    color='lightblue', showcaps=False, medianprops={"color": "black", "linewidth": 3},
                    ax=ax)
        sns.stripplot(data=df_plot, x=score, y=metric, color='darkblue', alpha=0.4, size=4, jitter=True, ax=ax)
        x_numeric = df_plot[score]
        y = df_plot[metric]
        z = np.polyfit(x_numeric, y, 1)
        p = np.poly1d(z)
        unique_mjoa = sorted(df_plot[score].unique())
        x_line_positions = np.linspace(0, len(unique_mjoa) - 1, 100)
        x_line_mjoa = np.interp(x_line_positions, range(len(unique_mjoa)), unique_mjoa)
        y_line_smooth = p(x_line_mjoa)
        ax.plot(x_line_positions, y_line_smooth, color='red', linewidth=2, alpha=0.8)
        x_tick_labels = [f'{mjoa}\n(n={len(df_plot[df_plot[score] == mjoa])})' for mjoa in unique_mjoa]
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel(score_to_label[score], fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel(metrics_to_labels[metric], fontsize=LABELS_FONT_SIZE)
        ax.set_title(f'{score_to_label[score]} vs Baseline {metric_to_title[metric]} at C{level} (n={len(df_plot)} subjects)',
                     fontsize=TITLE_FONT_SIZE)
        stats_text = f'Spearman r = {r:.2f}\np = {p_value:.3f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                fontsize=12)
        ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figure_path = os.path.join(output_dir, f'{score}_{structure}_C{level}_area_violin.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"1x1 area boxplot ({score}) saved to: {figure_path}")


def plot_violin_association_multi(df, output_dir, level, structure):
    """
    Create 3x1 violin plots for area, diameter_AP, diameter_RL if structure is spinal_cord or canal
    Args:
        df: Merged dataframe with mJOA and cord metrics
        output_dir: Output directory for the figure
        level: Spinal level (for title)
        structure: Structure name (for title)
    """
    mpl.rcParams['font.family'] = 'Arial'
    os.makedirs(output_dir, exist_ok=True)

    for score in clinical_scores:
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        for i, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            # Create df_plot for current metric and remove rows with missing values
            df_plot = df[['participant_id', score, metric]].dropna()
            ax = axes[i]
            r, p_value = spearmanr(df_plot[score], df_plot[metric])
            # sns.violinplot(data=df_plot, x=score, y=metric, color='lightblue', alpha=0.4, scale="width", ax=ax)
            sns.boxplot(data=df_plot, x=score, y=metric,
                        color='lightblue', showcaps=False, medianprops={"color": "black", "linewidth": 3},
                        ax=ax)
            sns.stripplot(data=df_plot, x=score, y=metric, color='darkblue', alpha=0.4, size=4, jitter=True, ax=ax)
            x_numeric = df_plot[score]
            y = df_plot[metric]
            z = np.polyfit(x_numeric, y, 1)
            p = np.poly1d(z)
            unique_mjoa = sorted(df_plot[score].unique())
            x_line_positions = np.linspace(0, len(unique_mjoa) - 1, 100)
            x_line_mjoa = np.interp(x_line_positions, range(len(unique_mjoa)), unique_mjoa)
            y_line_smooth = p(x_line_mjoa)
            ax.plot(x_line_positions, y_line_smooth, color='red', linewidth=2, alpha=0.8)
            x_tick_labels = [f'{mjoa}\n(n={len(df_plot[df_plot[score] == mjoa])})' for mjoa in unique_mjoa]
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel(score_to_label[score], fontsize=LABELS_FONT_SIZE)
            ax.set_ylabel(metrics_to_labels[metric], fontsize=LABELS_FONT_SIZE)
            ax.set_title(f'{score_to_label[score]} vs Baseline {metric_to_title[metric]} at C{level} (n={len(df_plot)} subjects)',
                         fontsize=TITLE_FONT_SIZE)
            stats_text = f'Spearman r = {r:.2f}\np = {p_value:.3f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    fontsize=12)
            ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE)
            ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figure_path = os.path.join(output_dir, f'{score}_{structure}_C{level}_multi_violin.png')
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        # Close figure
        plt.close(fig)
        print(f"3x1 violin plot ({score}) saved to: {figure_path}")


def plot_scatter_csa_vs_mjoa(df, output_dir, level, structure):
    """
    Create scatter plot showing association between CSA at C2 and mJOA scores

    Args:
        df: Merged dataframe with mJOA and cord area data
        output_dir: Output directory for the figure
        level: Spinal level (for title)
        structure: Structure name (for title)
    """
    mpl.rcParams['font.family'] = 'Arial'

    if structure == 'aSCOR':
        metric_column = 'aSCOR'
    else:
        metric_column = 'MEAN(area)'

    os.makedirs(output_dir, exist_ok=True)

    x = df[metric_column]
    y = df['total_mjoa_bl']

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(x=x, y=y, color='blue', alpha=0.6)

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    x_vals = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_vals, p(x_vals), color='red', linewidth=2, label='Regression line')

    # Correlation
    r, p_value = spearmanr(x, y)

    plt.xlabel(f'Spinal Cord Area at C{level} [mm²]', fontsize=LABELS_FONT_SIZE)
    plt.ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    plt.title(f'Scatter plot: CSA at C{level} vs. mJOA', fontsize=TITLE_FONT_SIZE)
    plt.legend()

    stats_text = f'Spearman r = {r:.2f}\np = {p_value:.3f}'
    plt.text(0.98, 0.02, stats_text, transform=ax.transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             fontsize=12)

    plt.tight_layout()

    figure_path = os.path.join(output_dir, f'scatter_csa_C{level}_vs_mjoa.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {figure_path}")
    print(f"Spearman correlation: r = {r:.3f}, p = {p_value:.3f}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.clinical):
        raise FileNotFoundError(f"Clinical scores file not found: {args.clinical}")

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    if 'cord' in args.metrics:
        structure = 'spinal_cord'
    elif 'canal' in args.metrics:
        structure = 'canal'
    elif 'aSCOR' in args.metrics:
        structure = 'aSCOR'

    # Load data
    df_clinical = load_clinical_data(args.clinical)
    df_metrics = load_cord_metrics(args.metrics, args.level, structure)

    # Merge data
    df_merged = merge_data(df_clinical, df_metrics)

    # Multi-metric violin plot for spinal_cord or canal
    if structure in ['spinal_cord', 'canal']:
        # plot_violin_association_multi(df_merged, args.o, args.level, structure)
        plot_violin_association(df_merged, args.o, args.level, structure)
    else:
        plot_violin_association(df_merged, args.o, args.level, structure)

    # # Add scatter plot for C2 CSA vs mJOA
    # if args.level == 2:
    #     plot_scatter_csa_vs_mjoa(df_merged, args.o, args.level, structure)


if __name__ == "__main__":
    main()
