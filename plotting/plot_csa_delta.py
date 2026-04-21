#!/usr/bin/env python
#
# Plot cross-sectional area (CSA) delta trajectories
#
# This script:
# 1. Loads cord metrics at each timepoint
# 2. Identifies maximum stenosis level for each subject from participants.tsv
# 3. Extracts MEAN(area) at the two levels surrounding maximum stenosis
# 4. Creates longitudinal trajectories and pre/post surgery comparisons
#
# Example usage:
#   python plot_csa_delta.py
#       -metrics_dir results/2026-02-26_first_metric_computation/raw_results/per_timepoint
#       -participants data/dcm-zurich/participants.tsv
#       -surgery baseline_surgery_dates_merged_clean.csv
#       -o results/2026-04-07_analysis
#
# Author: Kahina Baouche

import os
import sys
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Font sizes for plots
LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

# Map timepoint codes to months
TIMEPOINT_TO_MONTHS = {
    'M0': 0,
    'M6': 6,
    'M12': 12,
    'M24': 24,
    'M36': 36,
    'M48': 48,
    'M60': 60
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot CSA delta trajectories")
    parser.add_argument('-metrics_dir', required=True, type=str,
                        help="Directory containing per_timepoint cord metrics CSV files")
    parser.add_argument('-participants', required=True, type=str,
                        help="TSV file with participants and maximum stenosis levels")
    parser.add_argument('-surgery', required=True, type=str,
                        help="CSV file with surgery dates")
    parser.add_argument('-o', required=True, type=str,
                        help="Output directory for results")
    return parser


def extract_subject_from_filename(filename):
    """Extract subject ID from BIDS filename"""
    match = re.search(r'sub-(\d+)', filename)
    if match:
        return f"sub-{match.group(1)}"
    return None


def load_participants(participants_file):
    """Load participants data with maximum stenosis levels"""
    print(f"Loading participants from: {participants_file}")
    
    df = pd.read_csv(participants_file, sep='\t')
    
    # Extract stenosis level from maximum_stenosis column (e.g., "C4/C5" -> 4)
    def extract_stenosis_level(stenosis_str):
        if pd.isna(stenosis_str):
            return None
        # Extract the first level from stenosis (e.g., "C4/C5" -> 4)
        match = re.search(r'C(\d)', str(stenosis_str))
        if match:
            return int(match.group(1))
        return None
    
    df['stenosis_level'] = df['maximum_stenosis'].apply(extract_stenosis_level)
    
    print(f"Loaded {len(df)} participants")
    print(f"Participants with stenosis level: {df['stenosis_level'].notna().sum()}")
    
    return df[['participant_id', 'stenosis_level', 'surgery_date']].copy()


def load_metrics_by_timepoint(metrics_dir):
    """Load all cord metrics files by timepoint"""
    print(f"Loading metrics from: {metrics_dir}")
    
    metrics_by_timepoint = {}
    
    # Find all perlevel metric files
    for timepoint_code in TIMEPOINT_TO_MONTHS.keys():
        filepath = os.path.join(metrics_dir, f'T2w_ax_cord_metrics_perlevel_{timepoint_code}.csv')
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            # Extract subject ID from Filename column
            df['subject'] = df['Filename'].apply(extract_subject_from_filename)
            
            # Add timepoint information
            df['timepoint'] = timepoint_code
            df['months'] = TIMEPOINT_TO_MONTHS[timepoint_code]
            
            metrics_by_timepoint[timepoint_code] = df
            print(f"Loaded {timepoint_code}: {len(df)} rows")
    
    return metrics_by_timepoint


def extract_csa_at_stenosis_levels(metrics_by_timepoint, participants_df):
    """
    Extract CSA at the two vertebral levels surrounding maximum stenosis
    
    Args:
        metrics_by_timepoint: Dict of dataframes with metrics at each timepoint
        participants_df: DataFrame with stenosis levels for each subject
    
    Returns:
        DataFrame with subject, timepoint, mean_area at stenosis levels
    """
    print("\n=== Extracting CSA at stenosis levels ===")
    
    csa_data = []
    
    for subject, row in participants_df.iterrows():
        subject_id = row['participant_id']
        stenosis_level = row['stenosis_level']
        
        if pd.isna(stenosis_level):
            continue
        
        # Get the two levels surrounding stenosis
        levels_to_extract = [int(stenosis_level), int(stenosis_level) + 1]
        
        # Extract data for each timepoint
        for timepoint_code, df_metrics in metrics_by_timepoint.items():
            # Filter for this subject
            df_subj = df_metrics[df_metrics['subject'] == subject_id]
            
            if len(df_subj) == 0:
                continue
            
            # Extract areas at the stenosis levels
            areas = []
            for level in levels_to_extract:
                df_level = df_subj[df_subj['VertLevel'] == level]
                if len(df_level) > 0:
                    # Take mean if multiple slices per level
                    area = df_level['MEAN(area)'].mean()
                    areas.append(area)
            
            # Only include if we have data from both levels
            if len(areas) >= 1:  # Can be 1 or 2
                mean_area = np.mean(areas)
                
                csa_data.append({
                    'subject': subject_id,
                    'stenosis_level': stenosis_level,
                    'timepoint': timepoint_code,
                    'months': TIMEPOINT_TO_MONTHS[timepoint_code],
                    'mean_area': mean_area,
                    'n_levels': len(areas)
                })
    
    df_csa = pd.DataFrame(csa_data)
    
    print(f"Extracted CSA data for {df_csa['subject'].nunique()} subjects")
    print(f"Total measurements: {len(df_csa)}")
    
    return df_csa


def calculate_csa_delta(df_csa, output_dir):
    """
    Calculate baseline CSA and delta from baseline for each subject and timepoint
    
    Args:
        df_csa: CSA data at each timepoint
        output_dir: Output directory
    
    Returns:
        DataFrame with baseline CSA, follow-up CSA, and delta
    """
    print("\n=== Calculating CSA delta from baseline ===")
    
    csa_delta = []
    
    for subject in df_csa['subject'].unique():
        df_subj = df_csa[df_csa['subject'] == subject].sort_values('months')
        
        # Get baseline (month 0)
        df_baseline = df_subj[df_subj['months'] == 0]
        if len(df_baseline) == 0:
            continue
        
        baseline_area = df_baseline.iloc[0]['mean_area']
        
        # Calculate delta for all timepoints
        for _, row in df_subj.iterrows():
            delta = row['mean_area'] - baseline_area
            
            csa_delta.append({
                'subject': subject,
                'timepoint': row['timepoint'],
                'months': row['months'],
                'baseline_area': baseline_area,
                'followup_area': row['mean_area'],
                'delta_area': delta
            })
    
    df_delta = pd.DataFrame(csa_delta)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'csa_delta_from_baseline.csv')
    df_delta.to_csv(output_file, index=False)
    print(f"Saved CSA delta to: {output_file}")
    
    return df_delta


def calculate_csa_pre_post_surgery(df_csa, participants_df, surgery_df, output_dir):
    """
    Calculate CSA difference before and after surgery
    
    Args:
        df_csa: CSA data at each timepoint
        participants_df: Participants data with surgery dates
        surgery_df: Surgery dates
        output_dir: Output directory
    
    Returns:
        DataFrame with pre-surgery and post-surgery CSA differences
    """
    print("\n=== Calculating pre/post surgery CSA delta ===")
    
    # Merge participants with CSA data
    df_merged = df_csa.merge(participants_df[['participant_id', 'surgery_date']], 
                             left_on='subject', right_on='participant_id', how='left')
    
    # Convert surgery_date to datetime
    df_merged['surgery_date'] = pd.to_datetime(df_merged['surgery_date'], errors='coerce')
    
    pre_post_data = []
    
    for subject in df_merged['subject'].unique():
        df_subj = df_merged[df_merged['subject'] == subject].sort_values('months')
        
        # Get surgery date
        surgery_date = df_subj['surgery_date'].iloc[0]
        
        if pd.isna(surgery_date):
            continue
        
        # Get baseline (pre-surgery, month 0)
        df_baseline = df_subj[df_subj['months'] == 0]
        if len(df_baseline) == 0:
            continue
        
        baseline_area = df_baseline.iloc[0]['mean_area']
        
        # For each post-surgery measurement, calculate delta
        for _, row in df_subj[df_subj['months'] > 0].iterrows():
            # Calculate delta as baseline - followup (positive = improvement/reduction)
            delta = baseline_area - row['mean_area']
            
            pre_post_data.append({
                'subject': subject,
                'surgery_date': surgery_date,
                'timepoint': row['timepoint'],
                'months': row['months'],
                'baseline_area': baseline_area,
                'followup_area': row['mean_area'],
                'delta_area': delta  # baseline - followup
            })
    
    df_pre_post = pd.DataFrame(pre_post_data)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'csa_pre_post_surgery_delta.csv')
    df_pre_post.to_csv(output_file, index=False)
    print(f"Saved pre/post surgery CSA delta to: {output_file}")
    print(f"Found {df_pre_post['subject'].nunique()} subjects with surgery data")
    
    return df_pre_post


def plot_csa_delta_trajectories(df_delta, output_dir):
    """Plot CSA delta trajectories from baseline"""
    print("\n=== Creating CSA delta trajectory plot ===")
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique subjects
    subjects = df_delta['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
    
    # Plot individual trajectories
    for i, subject in enumerate(subjects):
        df_subj = df_delta[df_delta['subject'] == subject].sort_values('months')
        ax.plot(df_subj['months'], df_subj['delta_area'],
               color=colors[i], alpha=0.3, linewidth=1, zorder=1)
    
    # Calculate mean and std by timepoint
    grouped = df_delta.groupby('months')['delta_area'].agg(['mean', 'std', 'count'])
    
    # Plot mean line
    ax.plot(grouped.index, grouped['mean'], color='red', linewidth=4, 
           marker='o', markersize=8, label='Mean', zorder=10)
    
    # Plot standard deviation band
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='red', alpha=0.2, label='±1 SD', zorder=5)
    
    # Add horizontal line at 0 (no change from baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('CSA Delta (mm²)', fontsize=LABELS_FONT_SIZE)
    ax.set_title('Spinal Cord Area Change from Baseline', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Add sample sizes
    for months in grouped.index:
        n = int(grouped.loc[months, 'count'])
        ax.text(months, grouped.loc[months, 'mean'] - grouped.loc[months, 'std'] - 0.3,
               f'n={n}', ha='center', fontsize=9, color='darkred')
    
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(output_dir, 'csa_delta_trajectories.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")
    
    # Print summary
    print(f"\nSummary Statistics:")
    print(grouped[['mean', 'std', 'count']])
    
    plt.close()


def plot_csa_pre_post_surgery_delta(df_pre_post, output_dir):
    """Plot CSA delta pre and post surgery"""
    print("\n=== Creating pre/post surgery CSA delta plot ===")
    
    if len(df_pre_post) == 0:
        print("No pre/post surgery data available")
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get unique subjects
    subjects = df_pre_post['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
    
    # Plot individual trajectories
    for i, subject in enumerate(subjects):
        df_subj = df_pre_post[df_pre_post['subject'] == subject].sort_values('months')
        ax.plot(df_subj['months'], df_subj['delta_area'],
               color=colors[i], alpha=0.3, linewidth=1, zorder=1)
    
    # Calculate mean and std by timepoint
    grouped = df_pre_post.groupby('months')['delta_area'].agg(['mean', 'std', 'count'])
    
    # Plot mean line
    ax.plot(grouped.index, grouped['mean'], color='darkgreen', linewidth=4,
           marker='o', markersize=8, label='Mean', zorder=10)
    
    # Plot standard deviation band
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='green', alpha=0.2, label='±1 SD', zorder=5)
    
    # Add horizontal line at 0 (no change)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Months from Surgery', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('CSA Delta (mm²) - Baseline minus Follow-up', fontsize=LABELS_FONT_SIZE)
    ax.set_title('Spinal Cord Area Change: Pre-Surgery vs Post-Surgery', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    
    # Add sample sizes
    for months in grouped.index:
        n = int(grouped.loc[months, 'count'])
        ax.text(months, grouped.loc[months, 'mean'] - grouped.loc[months, 'std'] - 0.2,
               f'n={n}', ha='center', fontsize=9, color='darkgreen')
    
    # Add shaded region for post-surgery
    ax.axvspan(0, grouped.index.max() + 5, alpha=0.05, color='green', label='Post-surgery')
    
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(output_dir, 'csa_pre_post_surgery_delta.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")
    
    # Print summary
    print(f"\nSummary Statistics (Post-Surgery):")
    print(grouped[['mean', 'std', 'count']])
    
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.o, exist_ok=True)
    
    # Validate input files
    if not os.path.exists(args.metrics_dir):
        raise FileNotFoundError(f"Metrics directory not found: {args.metrics_dir}")
    
    if not os.path.exists(args.participants):
        raise FileNotFoundError(f"Participants file not found: {args.participants}")
    
    if not os.path.exists(args.surgery):
        raise FileNotFoundError(f"Surgery dates file not found: {args.surgery}")
    
    print(f"Output directory: {args.o}")
    print("=" * 80)
    
    # Load data
    participants_df = load_participants(args.participants)
    metrics_by_timepoint = load_metrics_by_timepoint(args.metrics_dir)
    surgery_df = pd.read_csv(args.surgery)
    
    # Extract CSA at stenosis levels
    df_csa = extract_csa_at_stenosis_levels(metrics_by_timepoint, participants_df)
    
    # Calculate CSA delta from baseline
    df_delta = calculate_csa_delta(df_csa, args.o)
    
    # Calculate pre/post surgery delta
    df_pre_post = calculate_csa_pre_post_surgery(df_csa, participants_df, surgery_df, args.o)
    
    # Create visualizations
    plot_csa_delta_trajectories(df_delta, args.o)
    plot_csa_pre_post_surgery_delta(df_pre_post, args.o)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {args.o}")


if __name__ == "__main__":
    main()
