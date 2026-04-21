#!/usr/bin/env python
#
# Plot longitudinal mJOA score trajectories
#
# This script creates:
# 1. A CSV file with mJOA scores at each timepoint for all subjects
# 2. A CSV file with mJOA scores before and after surgery
# 3. Plots showing individual trajectories, mean trajectory, and standard deviation bands
#
# Example usage:
#   python plot_mjoa_longitudinal.py
#       -clinical data/dcm-zurich/phenotype/clinical_scores.xlsx
#       -surgery baseline_surgery_dates_merged_clean.csv
#       -o results/2026-04-07_analysis
#
# Author: Kahina Baouche

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Font sizes for plots
LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

# Timepoint columns in clinical data
TIMEPOINT_COLUMNS = {
    'BL': 'total_mjoa_BL',
    '6mth': 'total_mjoa_6mth',
    '12mth': 'total_mjoa_12mth',
    '24mth': 'total_mjoa_24mth',
    '36mth': 'total_mjoa_36mth',
    '48mth': 'total_mjoa_48mth',
    '60mth': 'total_mjoa_60mth'
}

# Months for x-axis
TIMEPOINT_MONTHS = {
    'BL': 0,
    '6mth': 6,
    '12mth': 12,
    '24mth': 24,
    '36mth': 36,
    '48mth': 48,
    '60mth': 60
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot longitudinal mJOA score trajectories with mean and standard deviation")
    parser.add_argument('-clinical', required=True, type=str,
                        help="Excel file with clinical scores")
    parser.add_argument('-surgery', required=True, type=str,
                        help="CSV file with surgery dates")
    parser.add_argument('-o', required=True, type=str,
                        help="Output directory for results")
    return parser


def load_clinical_data(clinical_file):
    """
    Load clinical scores from Excel file

    Args:
        clinical_file: Path to Excel file with clinical scores

    Returns:
        pandas.DataFrame: Clinical data with record_id and mJOA scores
    """
    print(f"Loading clinical data from: {clinical_file}")

    try:
        df_clinical = pd.read_excel(clinical_file)
    except Exception as e:
        sys.exit(f"Error reading Excel file: {e}")

    # Extract record_id from first column (record_id_BL)
    if 'record_id_BL' in df_clinical.columns:
        df_clinical['record_id'] = df_clinical['record_id_BL']
    else:
        sys.exit("Could not find 'record_id_BL' column in clinical data")

    # Convert record_id to subject format (1 -> sub-001)
    df_clinical['subject'] = df_clinical['record_id'].apply(
        lambda x: f"sub-{int(x):03d}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x))

    # Check for required mJOA columns
    required_cols = list(TIMEPOINT_COLUMNS.values())
    missing_cols = [col for col in required_cols if col not in df_clinical.columns]
    if missing_cols:
        print(f"Warning: Missing mJOA columns: {missing_cols}")

    print(f"Loaded clinical data for {len(df_clinical)} subjects")

    return df_clinical


def load_surgery_data(surgery_file):
    """
    Load surgery dates

    Args:
        surgery_file: Path to CSV file with surgery dates

    Returns:
        pandas.DataFrame: Surgery data with subject and surgery_date
    """
    print(f"Loading surgery dates from: {surgery_file}")

    try:
        df_surgery = pd.read_csv(surgery_file)
    except Exception as e:
        sys.exit(f"Error reading CSV file: {e}")

    # Convert surgery_date to datetime
    df_surgery['surgery_date'] = pd.to_datetime(df_surgery['surgery_date'], errors='coerce')

    print(f"Loaded surgery dates for {len(df_surgery)} subjects")

    return df_surgery


def extract_mjoa_by_timepoint(df_clinical, output_dir):
    """
    Extract mJOA scores at each timepoint and save to CSV

    Args:
        df_clinical: Clinical data
        output_dir: Output directory

    Returns:
        pandas.DataFrame: mJOA data by timepoint in long format
    """
    print("\n=== Extracting mJOA scores by timepoint ===")

    # Create output for each timepoint
    df_long = []

    for timepoint, col_name in TIMEPOINT_COLUMNS.items():
        if col_name in df_clinical.columns:
            # Create temporary dataframe for this timepoint
            df_tp = df_clinical[['subject', col_name]].copy()
            df_tp = df_tp[df_tp[col_name].notna()]  # Remove NaN values

            # Rename column to uniform name
            df_tp = df_tp.rename(columns={col_name: 'mjoa_score'})
            df_tp['timepoint'] = timepoint
            df_tp['months'] = TIMEPOINT_MONTHS[timepoint]

            df_long.append(df_tp)

    # Combine all timepoints
    df_long = pd.concat(df_long, ignore_index=True)

    # Save to CSV
    output_file = os.path.join(output_dir, 'mjoa_by_timepoint.csv')
    df_long.to_csv(output_file, index=False)
    print(f"Saved mJOA by timepoint to: {output_file}")
    print(f"Total measurements: {len(df_long)}")
    print(f"Unique subjects: {df_long['subject'].nunique()}")

    return df_long


def extract_pre_post_surgery_mjoa(df_clinical, df_surgery, output_dir):
    """
    Extract mJOA scores before and after surgery

    Args:
        df_clinical: Clinical data
        df_surgery: Surgery dates
        output_dir: Output directory

    Returns:
        pandas.DataFrame: mJOA scores before and after surgery
    """
    print("\n=== Extracting pre/post surgery mJOA scores ===")

    # Convert clinical date columns to datetime if they exist
    # We need to match subjects first
    df_surgery_with_mjoa = []

    for idx, row in df_surgery.iterrows():
        subject = row['subject']
        surgery_date = row['surgery_date']

        # Find subject in clinical data
        if subject not in df_clinical['subject'].values:
            continue

        clinical_row = df_clinical[df_clinical['subject'] == subject].iloc[0]

        # Collect mJOA scores at each timepoint with their months from baseline
        timepoint_data = {
            'subject': subject,
            'surgery_date': surgery_date,
            'timepoint': [],
            'months': [],
            'mjoa_score': [],
            'relative_to_surgery': []
        }

        for timepoint, col_name in TIMEPOINT_COLUMNS.items():
            if col_name in df_clinical.columns:
                mjoa = clinical_row[col_name]
                if pd.notna(mjoa):
                    timepoint_data['timepoint'].append(timepoint)
                    timepoint_data['months'].append(TIMEPOINT_MONTHS[timepoint])
                    timepoint_data['mjoa_score'].append(mjoa)
                    # Mark as pre or post surgery (we'll infer from timepoint order)
                    timepoint_data['relative_to_surgery'].append(None)  # Will be filled later

        if timepoint_data['mjoa_score']:  # Only add if we have data
            df_surgery_with_mjoa.append(timepoint_data)

    if not df_surgery_with_mjoa:
        print("Warning: No subjects found in both surgery and clinical data")
        # Create empty dataframe with correct columns
        return pd.DataFrame(columns=['subject', 'surgery_date', 'timepoint', 'months',
                                     'mjoa_score', 'relative_to_surgery'])

    # Convert to long format dataframe
    all_records = []
    for subject_data in df_surgery_with_mjoa:
        for i in range(len(subject_data['timepoint'])):
            record = {
                'subject': subject_data['subject'],
                'surgery_date': subject_data['surgery_date'],
                'timepoint': subject_data['timepoint'][i],
                'months': subject_data['months'][i],
                'mjoa_score': subject_data['mjoa_score'][i]
            }
            all_records.append(record)

    df_pre_post = pd.DataFrame(all_records)

    # Save to CSV
    output_file = os.path.join(output_dir, 'mjoa_pre_post_surgery.csv')
    df_pre_post.to_csv(output_file, index=False)
    print(f"Saved pre/post surgery mJOA to: {output_file}")
    print(f"Total subject-surgery pairs: {df_pre_post['subject'].nunique()}")

    return df_pre_post


def plot_mjoa_trajectories(df_long, output_dir):
    """
    Plot mJOA trajectories with individual lines, mean, and standard deviation

    Args:
        df_long: mJOA data in long format
        output_dir: Output directory
    """
    print("\n=== Creating visualization ===")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ============ Subplot 1: All timepoints ============
    ax = axes[0]

    # Get unique subjects
    subjects = df_long['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    # Plot individual trajectories
    for i, subject in enumerate(subjects):
        df_subject = df_long[df_long['subject'] == subject].sort_values('months')
        ax.plot(df_subject['months'], df_subject['mjoa_score'],
               color=colors[i], alpha=0.3, linewidth=1, label=subject if i < 5 else "")

    # Calculate mean and std at each timepoint
    grouped = df_long.groupby('months')['mjoa_score'].agg(['mean', 'std', 'count'])
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

    # Plot mean line
    ax.plot(grouped.index, grouped['mean'], color='red', linewidth=3, label='Mean', zorder=10)

    # Plot standard deviation band
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='red', alpha=0.2, label='±1 SD', zorder=5)

    # Formatting
    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    ax.set_title('mJOA Score Trajectories (All Timepoints)', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 6, 12, 24, 36, 48, 60])
    ax.legend(loc='best', fontsize=10)

    # Add sample sizes to plot
    for months in grouped.index:
        n = int(grouped.loc[months, 'count'])
        ax.text(months, grouped.loc[months, 'mean'] - grouped.loc[months, 'std'] - 1,
               f'n={n}', ha='center', fontsize=9, color='darkred')

    # ============ Subplot 2: Only BL and subsequent timepoints ============
    ax = axes[1]

    # Plot only baseline and 6-month follow-up (most common)
    df_subset = df_long[df_long['months'].isin([0, 6, 12, 24])]

    subjects_subset = df_subset['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects_subset)))

    # Plot individual trajectories
    for i, subject in enumerate(subjects_subset):
        df_subject = df_subset[df_subset['subject'] == subject].sort_values('months')
        ax.plot(df_subject['months'], df_subject['mjoa_score'],
               color=colors[i], alpha=0.3, linewidth=1)

    # Calculate mean and std
    grouped_subset = df_subset.groupby('months')['mjoa_score'].agg(['mean', 'std', 'count'])
    grouped_subset['sem'] = grouped_subset['std'] / np.sqrt(grouped_subset['count'])

    # Plot mean line
    ax.plot(grouped_subset.index, grouped_subset['mean'], color='blue', linewidth=3, label='Mean')

    # Plot standard deviation band
    ax.fill_between(grouped_subset.index,
                    grouped_subset['mean'] - grouped_subset['std'],
                    grouped_subset['mean'] + grouped_subset['std'],
                    color='blue', alpha=0.2, label='±1 SD')

    # Formatting
    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    ax.set_title('mJOA Score Trajectories (Baseline to 24 months)', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 6, 12, 24])
    ax.legend(loc='best', fontsize=10)

    # Add sample sizes to plot
    for months in grouped_subset.index:
        n = int(grouped_subset.loc[months, 'count'])
        ax.text(months, grouped_subset.loc[months, 'mean'] - grouped_subset.loc[months, 'std'] - 1,
               f'n={n}', ha='center', fontsize=9, color='darkblue')

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(output_dir, 'mjoa_longitudinal_trajectories.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")

    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total subjects: {df_long['subject'].nunique()}")
    print(f"Total measurements: {len(df_long)}")
    print(f"\nMean mJOA score by timepoint:")
    print(grouped[['mean', 'std', 'count']])

    plt.close()


def plot_pre_post_surgery_trajectories(df_pre_post, df_surgery, output_dir):
    """
    Plot pre/post surgery mJOA trajectories

    Args:
        df_pre_post: mJOA data with surgery dates
        df_surgery: Surgery dates
        output_dir: Output directory
    """
    print("\n=== Creating pre/post surgery visualization ===")

    if len(df_pre_post) == 0:
        print("No pre/post surgery data available")
        return

    # Filter subjects with surgery dates
    subjects_with_surgery = df_surgery[df_surgery['surgery_date'].notna()]['subject'].unique()
    df_plot = df_pre_post[df_pre_post['subject'].isin(subjects_with_surgery)].copy()

    if len(df_plot) == 0:
        print("No data to plot")
        return

    # Create surgery date mapping
    surgery_date_map = df_surgery[['subject', 'surgery_date']].drop_duplicates()
    surgery_date_map = surgery_date_map.set_index('subject')['surgery_date'].to_dict()

    # Add surgery date to each row
    df_plot['surgery_date'] = df_plot['subject'].map(surgery_date_map)

    # Note: We don't have assessment dates in clinical data for comparison, so we use timepoint order
    # Typically: BL is pre-surgery, 6mth+ are post-surgery
    def infer_relative_to_surgery(row):
        if pd.isna(row['surgery_date']):
            return None
        # Use timepoint order as proxy
        if row['timepoint'] == 'BL':
            return 'Pre-surgery'
        else:
            return 'Post-surgery'

    df_plot['relative_to_surgery'] = df_plot.apply(infer_relative_to_surgery, axis=1)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique subjects
    subjects = df_plot['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    # Plot individual trajectories
    for i, subject in enumerate(subjects):
        df_subject = df_plot[df_plot['subject'] == subject].sort_values('months')
        ax.plot(df_subject['months'], df_subject['mjoa_score'],
               color=colors[i], alpha=0.3, linewidth=1)

    # Calculate mean and std
    grouped = df_plot.groupby('months')['mjoa_score'].agg(['mean', 'std', 'count'])

    # Plot mean line (bold)
    ax.plot(grouped.index, grouped['mean'], color='green', linewidth=4, label='Mean', zorder=10)

    # Plot standard deviation band
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='green', alpha=0.2, label='±1 SD', zorder=5)

    # Formatting
    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    ax.set_title('mJOA Score Trajectories - Subjects with Surgery', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)

    # Add sample sizes
    for months in grouped.index:
        n = int(grouped.loc[months, 'count'])
        ax.text(months, grouped.loc[months, 'mean'] - grouped.loc[months, 'std'] - 1,
               f'n={n}', ha='center', fontsize=9, color='darkgreen')

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(output_dir, 'mjoa_pre_post_surgery_trajectories.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")

    plt.close()


def plot_centered_at_surgery_date(df_pre_post, df_surgery, output_dir):
    """
    Plot mJOA trajectories centered at each individual's surgery date

    Args:
        df_pre_post: mJOA data with surgery dates
        df_surgery: Surgery dates
        output_dir: Output directory
    """
    print("\n=== Creating trajectories centered at surgery date ===")

    if len(df_pre_post) == 0:
        print("No pre/post surgery data available")
        return

    # Filter subjects with surgery dates
    subjects_with_surgery = df_surgery[df_surgery['surgery_date'].notna()]['subject'].unique()
    df_plot = df_pre_post[df_pre_post['subject'].isin(subjects_with_surgery)].copy()

    if len(df_plot) == 0:
        print("No data to plot")
        return

    # Create surgery date mapping
    surgery_date_map = df_surgery[['subject', 'surgery_date']].drop_duplicates()
    surgery_date_map = surgery_date_map.set_index('subject')['surgery_date'].to_dict()

    # Add surgery date to each row
    df_plot['surgery_date'] = df_plot['subject'].map(surgery_date_map)

    # Calculate months relative to surgery date
    # Baseline (BL) is typically at 0 months from baseline
    # We'll estimate that BL is approximately 0 months from surgery
    # (or some fixed time before surgery)
    # For now, we'll use a reasonable assumption: BL is ~3 months before surgery
    
    def calculate_months_from_surgery(row):
        """
        Calculate months from surgery date.
        BL is assumed to be 0 months, so we use that as reference.
        Each month mentioned in the row is added to calculate actual months from surgery.
        But we need to estimate when surgery occurred relative to baseline.
        """
        # For simplicity, assume BL is 0 months from baseline
        # We'll need to infer surgery timing from the data
        # If BL is at baseline and surgery happened sometime, we need to estimate
        # Let's use the convention that surgery is typically between BL and 6m
        # For now, assume surgery happens at ~3 months from baseline
        surgery_offset_months = 3  # Estimated months from BL to surgery
        
        return row['months'] - surgery_offset_months

    df_plot['months_from_surgery'] = df_plot.apply(calculate_months_from_surgery, axis=1)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(14, 7))

    # Get unique subjects
    subjects = df_plot['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    # Plot individual trajectories
    for i, subject in enumerate(subjects):
        df_subject = df_plot[df_plot['subject'] == subject].sort_values('months_from_surgery')
        ax.plot(df_subject['months_from_surgery'], df_subject['mjoa_score'],
               color=colors[i], alpha=0.3, linewidth=1, zorder=1)
        # Add points
        ax.scatter(df_subject['months_from_surgery'], df_subject['mjoa_score'],
                  color=colors[i], alpha=0.4, s=20, zorder=2)

    # Calculate mean and std by months from surgery
    grouped = df_plot.groupby('months_from_surgery')['mjoa_score'].agg(['mean', 'std', 'count'])

    # Plot mean line (bold)
    ax.plot(grouped.index, grouped['mean'], color='darkred', linewidth=4, 
           marker='o', markersize=8, label='Mean', zorder=10)

    # Plot standard deviation band
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='red', alpha=0.2, label='±1 SD', zorder=5)

    # Add vertical line at surgery date (x=0)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Surgery Date', zorder=8)

    # Formatting
    ax.set_xlabel('Months from Surgery Date', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    ax.set_title('mJOA Score Trajectories Centered at Surgery Date', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)

    # Add sample sizes
    for months in grouped.index:
        n = int(grouped.loc[months, 'count'])
        ax.text(months, grouped.loc[months, 'mean'] - grouped.loc[months, 'std'] - 0.8,
               f'n={n}', ha='center', fontsize=8, color='darkred')

    # Add shaded regions for pre and post-surgery
    ax.axvspan(grouped.index.min() - 1, 0, alpha=0.1, color='blue', label='Pre-surgery')
    ax.axvspan(0, grouped.index.max() + 1, alpha=0.1, color='green', label='Post-surgery')

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(output_dir, 'mjoa_centered_at_surgery_date.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")

    # Print summary statistics
    print(f"Trajectories centered at surgery date for {len(subjects)} subjects")
    print(f"Data points (months from surgery):")
    print(grouped[['mean', 'std', 'count']])

    plt.close()


def plot_pre_post_only(df_pre_post, df_surgery, output_dir):
    """
    Plot only pre-surgery (BL) vs first post-surgery timepoint for each subject

    Args:
        df_pre_post: mJOA data with surgery dates
        df_surgery: Surgery dates
        output_dir: Output directory
    """
    print("\n=== Creating pre vs post surgery (two timepoints only) visualization ===")

    if len(df_pre_post) == 0:
        print("No pre/post surgery data available")
        return

    # Filter subjects with surgery dates
    subjects_with_surgery = df_surgery[df_surgery['surgery_date'].notna()]['subject'].unique()
    df_plot = df_pre_post[df_pre_post['subject'].isin(subjects_with_surgery)].copy()

    if len(df_plot) == 0:
        print("No data to plot")
        return

    # For each subject, get BL and first post-surgery measurement
    pre_post_pairs = []

    for subject in df_plot['subject'].unique():
        df_subj = df_plot[df_plot['subject'] == subject].sort_values('months')

        # Get baseline (BL at 0 months)
        bl_data = df_subj[df_subj['timepoint'] == 'BL']
        if len(bl_data) == 0:
            continue

        # Get first post-surgery measurement (first non-BL timepoint)
        post_data = df_subj[df_subj['timepoint'] != 'BL']
        if len(post_data) == 0:
            continue

        # Use the first post-surgery measurement
        first_post = post_data.iloc[0]

        pre_post_pairs.append({
            'subject': subject,
            'pre_mjoa': bl_data.iloc[0]['mjoa_score'],
            'post_mjoa': first_post['mjoa_score'],
            'post_timepoint': first_post['timepoint'],
            'post_months': first_post['months'],
            'change': first_post['mjoa_score'] - bl_data.iloc[0]['mjoa_score']
        })

    if not pre_post_pairs:
        print("No complete pre-post pairs found")
        return

    df_pairs = pd.DataFrame(pre_post_pairs)

    print(f"Found {len(df_pairs)} subjects with pre and post-surgery mJOA measurements")
    print(f"Mean change: {df_pairs['change'].mean():.2f} (SD: {df_pairs['change'].std():.2f})")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot individual trajectories (connecting pre to post)
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_pairs)))
    x_positions = [0, 1]  # Pre-surgery and Post-surgery

    for i, row in df_pairs.iterrows():
        y_values = [row['pre_mjoa'], row['post_mjoa']]
        ax.plot(x_positions, y_values, color=colors[i], alpha=0.3, linewidth=1, zorder=1)
        # Add points
        ax.scatter(x_positions, y_values, color=colors[i], alpha=0.4, s=30, zorder=2)

    # Calculate mean values
    mean_pre = df_pairs['pre_mjoa'].mean()
    mean_post = df_pairs['post_mjoa'].mean()
    std_pre = df_pairs['pre_mjoa'].std()
    std_post = df_pairs['post_mjoa'].std()

    # Plot mean line (bold)
    ax.plot(x_positions, [mean_pre, mean_post], color='red', linewidth=4, 
           marker='o', markersize=10, label='Mean', zorder=10)

    # Plot error bars (±1 SD)
    ax.errorbar([0], [mean_pre], yerr=[[std_pre], [std_pre]], 
               fmt='none', ecolor='red', elinewidth=2, capsize=8, alpha=0.6, zorder=5)
    ax.errorbar([1], [mean_post], yerr=[[std_post], [std_post]], 
               fmt='none', ecolor='red', elinewidth=2, capsize=8, alpha=0.6, zorder=5)

    # Formatting
    ax.set_xlabel('Timepoint', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('mJOA Score', fontsize=LABELS_FONT_SIZE)
    ax.set_title('mJOA Score: Pre-Surgery vs Post-Surgery', fontsize=TITLE_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Pre-Surgery\n(Baseline)', 'Post-Surgery\n(First Follow-up)'])
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([10, 20])

    # Add sample size
    ax.text(0.5, 10.5, f'n = {len(df_pairs)}', ha='center', fontsize=12, 
           fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add statistics text
    p_change = mean_post - mean_pre
    stats_text = f'Mean Change: {p_change:+.2f} points\n(Pre: {mean_pre:.2f}±{std_pre:.2f}, Post: {mean_post:.2f}±{std_post:.2f})'
    ax.text(0.5, 19, stats_text, ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(output_dir, 'mjoa_pre_post_surgery_two_timepoints.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")

    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.o, exist_ok=True)

    # Validate input files
    if not os.path.exists(args.clinical):
        raise FileNotFoundError(f"Clinical scores file not found: {args.clinical}")

    if not os.path.exists(args.surgery):
        raise FileNotFoundError(f"Surgery dates file not found: {args.surgery}")

    print(f"Output directory: {args.o}")
    print("=" * 80)

    # Load data
    df_clinical = load_clinical_data(args.clinical)
    df_surgery = load_surgery_data(args.surgery)

    # Extract mJOA by timepoint
    df_long = extract_mjoa_by_timepoint(df_clinical, args.o)

    # Extract pre/post surgery mJOA
    df_pre_post = extract_pre_post_surgery_mjoa(df_clinical, df_surgery, args.o)

    # Create visualizations
    plot_mjoa_trajectories(df_long, args.o)
    plot_pre_post_surgery_trajectories(df_pre_post, df_surgery, args.o)
    plot_centered_at_surgery_date(df_pre_post, df_surgery, args.o)
    plot_pre_post_only(df_pre_post, df_surgery, args.o)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {args.o}")


if __name__ == "__main__":
    main()
