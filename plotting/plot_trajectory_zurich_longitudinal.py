#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate trajectory plots for longitudinal DCM-Zurich spinal cord morphometric data.

Three visualization:

Option 1 - Native Space Per-Level Trajectories:
    - x-axis: Timepoints (baseline, M3, M6, M12, etc.)
    - y-axis: Morphometric metrics (area, AP_diameter, RL_diameter, etc.)
    - Different colors for vertebral levels (C2-C7)
    - Data: Per-level averaged data (one value per level per subject per timepoint)
    - Shows: Mean ± STD + individual subject trajectories

Option 2 - PAM50 Per-Slice Trajectories:
    - x-axis: Individual slices in the PAM50 template space
    - y-axis: Normalized morphometric metrics
    - Different colors for timepoints (baseline, M3, M6, M12, etc.)
    - Data: PAM50-normalized per-slice data
    - Shows: Mean ± STD + individual subject trajectories

Option 3 - Native Space Per-Slice Trajectories:
    - x-axis: Timepoints (baseline, M3, M6, M12, etc.)
    - y-axis: Morphometric metrics (area, AP_diameter, RL_diameter, etc.)
    - Different colors for vertebral levels (C2-C7)
    - Data: Native space per-slice data (all slices, not averaged by level)
    - Shows: Mean ± STD + individual subject trajectories

Option 4 - Per-Level by Timepoint Trajectories:
    - x-axis: Vertebral levels (C2, C3, C4, C5, C6, C7)
    - y-axis: Morphometric metrics (area, AP_diameter, RL_diameter, etc.)
    - Different colors for timepoints (baseline, M3, M6, M12, etc.)
    - Data: Per-level averaged data (native space)
    - Shows: Mean ± STD + individual subject trajectories

Option 5 - Therapeutic Group Per-Level Trajectories:
    - Separate plots for conservative and operative identified subjects
    - Conservative plot: x-axis uses original longitudinal timepoints
    - Operative plot: each subject's first post-operative timepoint is aligned to T0
    - Data: Per-level averaged data (native space) + time-aware labels
    - Shows: Mean ± STD + individual subject trajectories

Usage example:
    python plot_trajectory_zurich_longitudinal.py \\
        --data-dir /path/to/timepoint_data \\
        --output-dir /path/to/output \\
        --structure cord \\
        --timepoints M0 M6 M12 \\
        --option all

Authors: Kahina Baouche,
Generatlization of generate_figure_PAM50_multiple_subjects.py
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

# ============================================================================
# Constants
# ============================================================================

# Metric names mapping
METRIC_NAMES = {
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(diameter_RL)': 'RL Diameter',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
    'aSCOR': 'aSCOR (Area Spinal Cord Occupation Ratio)',
}

# Metric units
METRIC_UNITS = {
    'MEAN(area)': 'mm²',
    'MEAN(diameter_AP)': 'mm',
    'MEAN(diameter_RL)': 'mm',
    'MEAN(eccentricity)': 'a.u.',
    'MEAN(solidity)': '%',
    'aSCOR': 'ratio',
}

# Vertebral level colors (C2-C7)
VERT_COLORS = {
    2: '#d62728',
    4: '#2ca02c',
    5: '#1f77b4',
    6: '#9467bd',
    7: '#8c564b',
}

VERT_LABELS = {
    2: 'C2',
    3: 'C3',
    4: 'C4',
    5: 'C5',
    6: 'C6',
    7: 'C7',
}

# Timepoint colors (M0, M3, M6, M12, M24, M36, M48, M60)
TIMEPOINT_COLORS = {
    'M0': '#1f77b4',
    'M3': '#ff7f0e',
    'M6': '#2ca02c',
    'M12': '#d62728',
    'M24': '#9467bd',
    'M36': '#8c564b',
    'M48': '#e377c2',
    'M60': '#7f7f7f',
}

# Timepoint to months mapping
TIMEPOINT_MONTHS = {
    'M0': 0,
    'M3': 3,
    'M6': 6,
    'M12': 12,
    'M24': 24,
    'M36': 36,
    'M48': 48,
    'M60': 60,
}

# Font sizes
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9

# Figure sizes
FIG_SIZE_SINGLE = (10, 6)
FIG_SIZE_GRID = (18, 12)


# ============================================================================
# Data loading functions
# ============================================================================

def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'sub-001')."""
    import re
    match = re.search(r'sub-\d+', filename)
    return match.group(0) if match else None


def load_perlevel_data(data_dir, structure='cord', timepoints=None):
    """
    Load per-level native space data for all timepoints.
    
    Returns:
        pd.DataFrame with columns: subject, timepoint, VertLevel, and metrics
    """
    if timepoints is None:
        timepoints = ['M0', 'M6', 'M12']
    
    all_data = []
    
    for tp in timepoints:
        pattern = f'T2w_ax_{structure}_metrics_perlevel_{tp}_data.csv'
        file_path = os.path.join(data_dir, pattern)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract subject ID from Filename if not present
        if 'subject' not in df.columns and 'Filename' in df.columns:
            df['subject'] = df['Filename'].apply(extract_subject_id)
        
        # Add timepoint column if not present
        if 'timepoint' not in df.columns:
            df['timepoint'] = tp
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data files found for timepoints: {timepoints}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to cervical levels only (C2-C7 = VertLevel 2-7)
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)]
    
    return combined


def load_perslice_native_data(data_dir, structure='cord', timepoints=None):
    """
    Load per-slice native space data for all timepoints.
    
    Returns:
        pd.DataFrame with columns: subject, timepoint, Slice (I->S), VertLevel, and metrics
    """
    if timepoints is None:
        timepoints = ['M0', 'M6', 'M12']
    
    all_data = []
    
    for tp in timepoints:
        pattern = f'T2w_ax_{structure}_metrics_perslice_{tp}_data.csv'
        file_path = os.path.join(data_dir, pattern)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract subject ID from Filename if not present
        if 'subject' not in df.columns:
            if 'Filename' in df.columns:
                df['subject'] = df['Filename'].apply(extract_subject_id)
            elif 'Filename_sc' in df.columns:  # For aSCOR
                df['subject'] = df['Filename_sc'].apply(extract_subject_id)
        
        # Add timepoint column if not present
        if 'timepoint' not in df.columns:
            df['timepoint'] = tp
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No native perslice data files found for timepoints: {timepoints}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to cervical levels only
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)]
    
    # Remove rows with missing metric values - check for aSCOR or MEAN(area)
    if structure == 'aSCOR':
        combined = combined.dropna(subset=['aSCOR'])
    else:
        combined = combined.dropna(subset=['MEAN(area)'])
    
    return combined


def load_perslice_pam50_data(data_dir, structure='cord', timepoints=None):
    """
    Load per-slice PAM50 normalized data for all timepoints.
    
    Returns:
        pd.DataFrame with columns: subject, timepoint, Slice (I->S), VertLevel, and metrics
    """
    if timepoints is None:
        timepoints = ['M0', 'M6', 'M12']
    
    all_data = []
    
    for tp in timepoints:
        pattern = f'T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv'
        file_path = os.path.join(data_dir, pattern)
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract subject ID from Filename if not present
        if 'subject' not in df.columns:
            if 'Filename' in df.columns:
                df['subject'] = df['Filename'].apply(extract_subject_id)
            elif 'Filename_sc' in df.columns:  # For aSCOR
                df['subject'] = df['Filename_sc'].apply(extract_subject_id)
        
        # Add timepoint column if not present
        if 'timepoint' not in df.columns:
            df['timepoint'] = tp
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No PAM50 data files found for timepoints: {timepoints}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to cervical levels only
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)]
    
    # Remove rows with missing metric values - check for aSCOR or MEAN(area)
    if structure == 'aSCOR':
        combined = combined.dropna(subset=['aSCOR'])
    else:
        combined = combined.dropna(subset=['MEAN(area)'])
    
    return combined


def infer_labels_dir(data_dir):
    """Infer the directory containing time-aware therapeutic labels."""
    candidate = (Path(data_dir).resolve().parent /
                 'longitudinal_statistics_all_timepoints' /
                 'timepoint_aware_label_verification')
    return str(candidate) if candidate.exists() else None


def load_timepoint_aware_labels(labels_dir, timepoints):
    """
    Load per-timepoint therapeutic labels generated by the longitudinal stats pipeline.

    Returns:
        pd.DataFrame with columns: participant_id, timepoint, therapeutic_group, surgery_date
    """
    all_labels = []

    for tp in timepoints:
        file_path = Path(labels_dir) / f'labels_{tp}.csv'
        if not file_path.exists():
            print(f"Warning: Label file not found: {file_path}")
            continue

        print(f"Loading labels {file_path}...")
        df = pd.read_csv(file_path)
        if 'participant_id' not in df.columns or 'timepoint_aware_label' not in df.columns:
            print(f"Warning: Skipping malformed label file: {file_path}")
            continue

        df = df[['participant_id', 'timepoint_aware_label'] +
                [c for c in ['surgery_date'] if c in df.columns]].copy()
        df['timepoint'] = tp
        df = df.rename(columns={'timepoint_aware_label': 'therapeutic_group'})
        df['therapeutic_group'] = df['therapeutic_group'].str.lower()
        all_labels.append(df)

    if not all_labels:
        raise ValueError(f"No label files found for timepoints: {timepoints}")

    return pd.concat(all_labels, ignore_index=True)


def attach_timepoint_aware_labels(df, labels_df):
    """Merge time-aware therapeutic labels into morphometric data."""
    merged = df.merge(
        labels_df,
        left_on=['subject', 'timepoint'],
        right_on=['participant_id', 'timepoint'],
        how='left'
    )

    missing = merged['therapeutic_group'].isna().sum()
    if missing:
        print(f"Warning: {missing} rows missing therapeutic_group after label merge")

    merged = merged.drop(columns=['participant_id'], errors='ignore')
    merged = merged.dropna(subset=['therapeutic_group']).copy()
    merged['timepoint_month'] = merged['timepoint'].map(TIMEPOINT_MONTHS)
    return merged


def prepare_grouped_trajectory_data(df):
    """
    Split data into conservative and operative groups.

    Operative rows are re-aligned so each subject's first post-operative timepoint is T0.
    """
    conservative_df = df[df['therapeutic_group'] == 'conservative'].copy()
    conservative_df['aligned_month'] = conservative_df['timepoint_month']
    conservative_df['aligned_timepoint'] = conservative_df['timepoint']

    operative_df = df[df['therapeutic_group'] == 'operative'].copy()
    if not operative_df.empty:
        first_postop_month = operative_df.groupby('subject')['timepoint_month'].transform('min')
        operative_df['aligned_month'] = operative_df['timepoint_month'] - first_postop_month
        operative_df['aligned_timepoint'] = operative_df['aligned_month'].astype(int).map(lambda x: f'T{x}')
    else:
        operative_df['aligned_month'] = []
        operative_df['aligned_timepoint'] = []

    return conservative_df, operative_df


def save_operative_postop_counts_table(operative_df, output_dir):
    """
    Save a table with the number of operative subjects per post-op aligned timepoint.

    The alignment is identical to operative plotting:
    each subject's first post-op scan is T0.
    """
    if operative_df.empty:
        print("Warning: No operative rows available for post-op counts table")
        return

    counts_df = (
        operative_df[['subject', 'aligned_month', 'aligned_timepoint']]
        .drop_duplicates()
        .groupby(['aligned_month', 'aligned_timepoint'])['subject']
        .nunique()
        .reset_index(name='Operative')
        .sort_values('aligned_month')
    )

    counts_df['Total'] = counts_df['Operative']
    counts_df = counts_df.rename(columns={
        'aligned_timepoint': 'PostOp_Timepoint',
        'aligned_month': 'Months_After_First_PostOp_Scan'
    })

    # Keep the same style as subject_counts_per_timepoint.csv
    counts_df = counts_df[['PostOp_Timepoint', 'Months_After_First_PostOp_Scan', 'Operative', 'Total']]

    out_dir = os.path.join(output_dir, 'therapeutic_groups')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'subject_counts_operative_postop_aligned.csv')
    counts_df.to_csv(out_path, index=False)
    print(f"Saved operative post-op subject count table: {out_path}")


# ============================================================================
# Option 1: Native Space Per-Level Trajectories
# ============================================================================

def plot_perlevel_trajectories(df, metric, output_dir):
    """
    Plot trajectories for a single metric across timepoints, colored by vertebral level.
    Shows mean ± STD + individual subject trajectories.
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, VertLevel, metric]
        metric: Name of the metric to plot (e.g., 'MEAN(area)')
        output_dir: Directory to save the plot
    """
    # Sort timepoints
    timepoints = sorted(df['timepoint'].unique(), key=lambda x: TIMEPOINT_MONTHS.get(x, 999))
    
    # Verbose for debugging
    print(f"Timepoints found: {timepoints}")
    if len(timepoints) < 2:
        print(f"Warning: Need at least 2 timepoints for trajectories, found {len(timepoints)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    # Map timepoints to x-axis positions (months)
    tp_to_x = {tp: TIMEPOINT_MONTHS[tp] for tp in timepoints if tp in TIMEPOINT_MONTHS}
    
    # Plot for each vertebral level
    for vert_level in sorted(VERT_COLORS.keys()):
        df_level = df[df['VertLevel'] == vert_level]
        
        if df_level.empty:
            continue
        
        color = VERT_COLORS[vert_level]
        label = VERT_LABELS[vert_level]
        
        # Calculate mean and std for this level at each timepoint
        means = []
        stds = []
        x_values = []
        
        for tp in timepoints:
            df_tp = df_level[df_level['timepoint'] == tp]
            if not df_tp.empty and metric in df_tp.columns:
                values = df_tp[metric].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    x_values.append(tp_to_x.get(tp, 0))
        
        if len(means) >= 2:
            # Plot mean line with error bars
            ax.errorbar(x_values, means, yerr=stds, 
                       color=color, linewidth=2.5, marker='o', markersize=8,
                       label=label, capsize=5, capthick=2, alpha=0.9)
            
            # Plot individual subject trajectories
            subjects = df_level['subject'].unique()
            for subj in subjects:
                df_subj = df_level[df_level['subject'] == subj]
                subj_x = []
                subj_y = []
                
                for tp in timepoints:
                    df_tp_subj = df_subj[df_subj['timepoint'] == tp]
                    if not df_tp_subj.empty and metric in df_tp_subj.columns:
                        value = df_tp_subj[metric].dropna().values
                        if len(value) > 0:
                            subj_x.append(tp_to_x.get(tp, 0))
                            subj_y.append(value[0])
                
                if len(subj_x) >= 2:
                    ax.plot(subj_x, subj_y, color=color, linewidth=0.5, 
                           alpha=0.3, linestyle='-', zorder=1)
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    metric_unit = METRIC_UNITS.get(metric, '')
    
    ax.set_xlabel('Time (months)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{metric_unit}]', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'Longitudinal Trajectories: {metric_name} by Vertebral Level', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.legend(title='Vertebral Level', fontsize=LEGEND_FONT_SIZE, 
             title_fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    
    # Set x-ticks to timepoints
    ax.set_xticks([tp_to_x[tp] for tp in timepoints if tp in tp_to_x])
    ax.set_xticklabels(timepoints)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_perlevel_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_option1_plots(df, output_dir, metrics=None):
    """
    Generate all Option 1 plots (per-level native space trajectories).
    """
    if metrics is None:
        metrics = list(METRIC_NAMES.keys())
    
    print("\n" + "="*70)
    print("OPTION 1: Native Space Per-Level Trajectories")
    print("="*70)
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nPlotting {metric}...")
            plot_perlevel_trajectories(df, metric, output_dir)
        else:
            print(f"Warning: Metric '{metric}' not found in data")


# ============================================================================
# Option 2: PAM50 Per-Slice Trajectories
# ============================================================================

def get_disc_junction_slices(df, margin=4, return_junction_info=False):
    """
    Identify slices near vertebral disc junctions (level transitions) in PAM50 space.

    The disc position corresponds to where VertLevel changes as slices are traversed
    from inferior to superior. These transitions cause discontinuities in the plots.
    Excludes `margin` slices on each side of each transition.

    Returns:
        set: Slice numbers to exclude
        list[dict] (optional): Junction metadata if return_junction_info=True
    """
    # Determine the most common VertLevel at each slice across all subjects
    slice_levels = df.groupby('Slice (I->S)')['VertLevel'].agg(lambda x: x.mode()[0])
    slice_levels_sorted = slice_levels.sort_index()

    slices = slice_levels_sorted.index.tolist()
    levels = slice_levels_sorted.values.tolist()

    slices_to_exclude = set()
    junction_info = []

    for i in range(1, len(slices)):
        if levels[i] != levels[i - 1]:
            # Junction between slices[i-1] and slices[i]
            lbl_before = VERT_LABELS.get(levels[i - 1], str(levels[i - 1]))
            lbl_after = VERT_LABELS.get(levels[i], str(levels[i]))
            junction_info.append({
                'slice_before': slices[i - 1],
                'slice_after': slices[i],
                'label_before': lbl_before,
                'label_after': lbl_after,
                'junction_x': (slices[i - 1] + slices[i]) / 2.0,
                'label': f'{lbl_before}/{lbl_after}',
            })
            for j in range(max(0, i - margin), min(len(slices), i + margin)):
                slices_to_exclude.add(slices[j])

    if junction_info:
        for j in junction_info:
            print(f"  Disc junction {j['label_before']}/{j['label_after']}: "
                  f"slices {j['slice_before']}-{j['slice_after']}, excluding ±{margin} slices")
    print(f"  Total slices excluded near junctions: {len(slices_to_exclude)}")

    if return_junction_info:
        return slices_to_exclude, junction_info
    return slices_to_exclude


def plot_perslice_pam50_trajectories(df, metric, output_dir):
    """
    Plot trajectories for a single metric across PAM50 slices, colored by timepoint.
    Shows mean ± STD + individual subject trajectories.
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, Slice (I->S), VertLevel, metric]
        metric: Name of the metric to plot (e.g., 'MEAN(area)')
        output_dir: Directory to save the plot
    """
    timepoints = sorted(df['timepoint'].unique(), key=lambda x: TIMEPOINT_MONTHS.get(x, 999))
    
    if len(timepoints) < 2:
        print(f"Warning: Need at least 2 timepoints for trajectories, found {len(timepoints)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    # Get slice range, excluding slices near vertebral disc junctions
    print("Identifying disc junction slices to exclude:")
    slices_to_exclude, junction_info = get_disc_junction_slices(
        df, margin=4, return_junction_info=True
    )
    all_slices = sorted([s for s in df['Slice (I->S)'].unique() if s not in slices_to_exclude])

    # Plot for each timepoint
    for tp in timepoints:
        df_tp = df[df['timepoint'] == tp]
        
        if df_tp.empty:
            continue
        
        color = TIMEPOINT_COLORS.get(tp, '#000000')
        
        # Calculate mean and std for each slice
        slice_means = []
        slice_stds = []
        slice_values = []
        
        for slice_num in all_slices:
            df_slice = df_tp[df_tp['Slice (I->S)'] == slice_num]
            if not df_slice.empty and metric in df_slice.columns:
                values = df_slice[metric].dropna()
                if len(values) > 0:
                    slice_means.append(values.mean())
                    slice_stds.append(values.std())
                    slice_values.append(slice_num)
        
        if len(slice_means) > 0:
            # Plot mean line with shaded std
            ax.plot(slice_values, slice_means, color=color, linewidth=2.5,
                   label=f'{tp} (n={len(df_tp["subject"].unique())})', alpha=0.9)
            
            # Add shaded std region
            slice_means_arr = np.array(slice_means)
            slice_stds_arr = np.array(slice_stds)
            ax.fill_between(slice_values, 
                           slice_means_arr - slice_stds_arr,
                           slice_means_arr + slice_stds_arr,
                           color=color, alpha=0.2)
            
            # Plot individual subject trajectories
            subjects = df_tp['subject'].unique()
            for subj in subjects:
                df_subj = df_tp[df_tp['subject'] == subj]
                
                subj_slices = []
                subj_values = []
                
                for slice_num in all_slices:
                    df_slice_subj = df_subj[df_subj['Slice (I->S)'] == slice_num]
                    if not df_slice_subj.empty and metric in df_slice_subj.columns:
                        value = df_slice_subj[metric].dropna().values
                        if len(value) > 0:
                            subj_slices.append(slice_num)
                            subj_values.append(value[0])
                
                if len(subj_slices) > 5:  # Only plot if subject has data for multiple slices
                    ax.plot(subj_slices, subj_values, color=color, linewidth=0.3,
                           alpha=0.2, linestyle='-', zorder=1)
    
    # Draw disc-junction markers (aligned with transitions) + labels
    for j in junction_info:
        ax.axvline(j['junction_x'], color='gray', linestyle='--', alpha=0.5, linewidth=1.2)

    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for j in junction_info:
        ax.text(
            j['junction_x'], 0.97, j['label'],
            transform=text_transform,
            fontsize=8,
            color='dimgray',
            ha='center',
            va='top',
            fontweight='bold'
        )
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    metric_unit = METRIC_UNITS.get(metric, '')
    
    ax.set_xlabel('PAM50 Slice (Superior → Inferior)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{metric_unit}]', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'PAM50 Normalized Trajectories: {metric_name} by Timepoint',
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.legend(title='Timepoint', fontsize=LEGEND_FONT_SIZE,
             title_fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.invert_xaxis()  # Invert to show superior to inferior
    
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_perslice_PAM50_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_option2_plots(df, output_dir, metrics=None):
    """
    Generate all Option 2 plots (per-slice PAM50 trajectories).
    """
    if metrics is None:
        metrics = list(METRIC_NAMES.keys())
    
    print("\n" + "="*70)
    print("OPTION 2: PAM50 Per-Slice Trajectories")
    print("="*70)
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nPlotting {metric}...")
            plot_perslice_pam50_trajectories(df, metric, output_dir)
        else:
            print(f"Warning: Metric '{metric}' not found in data")


# ============================================================================
# Option 3: Native Space Per-Slice Trajectories
# ============================================================================

def plot_perslice_native_trajectories(df, metric, output_dir):
    """
    Plot trajectories for a single metric across timepoints using native space per-slice data,
    colored by vertebral level. Shows mean ± STD + individual subject trajectories.
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, Slice (I->S), VertLevel, metric]
        metric: Name of the metric to plot (e.g., 'MEAN(area)')
        output_dir: Directory to save the plot
    """
    # Sort timepoints
    timepoints = sorted(df['timepoint'].unique(), key=lambda x: TIMEPOINT_MONTHS.get(x, 999))
    
    if len(timepoints) < 2:
        print(f"Warning: Need at least 2 timepoints for trajectories, found {len(timepoints)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    # Map timepoints to x-axis positions (months)
    tp_to_x = {tp: TIMEPOINT_MONTHS[tp] for tp in timepoints if tp in TIMEPOINT_MONTHS}
    
    # Plot for each vertebral level
    for vert_level in sorted(VERT_COLORS.keys()):
        df_level = df[df['VertLevel'] == vert_level]
        
        if df_level.empty:
            continue
        
        color = VERT_COLORS[vert_level]
        label = VERT_LABELS[vert_level]
        
        # Calculate mean and std across all slices in this level at each timepoint
        means = []
        stds = []
        x_values = []
        
        for tp in timepoints:
            df_tp = df_level[df_level['timepoint'] == tp]
            if not df_tp.empty and metric in df_tp.columns:
                # Get all slice values (across all slices and subjects for this level and timepoint)
                values = df_tp[metric].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    x_values.append(tp_to_x.get(tp, 0))
        
        if len(means) >= 2:
            # Plot mean line with error bars
            ax.errorbar(x_values, means, yerr=stds, 
                       color=color, linewidth=2.5, marker='o', markersize=8,
                       label=label, capsize=5, capthick=2, alpha=0.9)
            
            # Plot individual subject trajectories (averaged per subject per level per timepoint)
            subjects = df_level['subject'].unique()
            for subj in subjects:
                df_subj = df_level[df_level['subject'] == subj]
                subj_x = []
                subj_y = []
                
                for tp in timepoints:
                    df_tp_subj = df_subj[df_subj['timepoint'] == tp]
                    if not df_tp_subj.empty and metric in df_tp_subj.columns:
                        # Average across all slices for this subject at this level and timepoint
                        values = df_tp_subj[metric].dropna()
                        if len(values) > 0:
                            subj_x.append(tp_to_x.get(tp, 0))
                            subj_y.append(values.mean())
                
                if len(subj_x) >= 2:
                    ax.plot(subj_x, subj_y, color=color, linewidth=0.5, 
                           alpha=0.3, linestyle='-', zorder=1)
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    metric_unit = METRIC_UNITS.get(metric, '')
    
    ax.set_xlabel('Time (months)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{metric_unit}]', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'Longitudinal Trajectories (Per-Slice Native): {metric_name} by Vertebral Level', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.legend(title='Vertebral Level', fontsize=LEGEND_FONT_SIZE, 
             title_fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    
    # Set x-ticks to timepoints
    ax.set_xticks([tp_to_x[tp] for tp in timepoints if tp in tp_to_x])
    ax.set_xticklabels(timepoints)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_perslice_native_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_option3_plots(df, output_dir, metrics=None):
    """
    Generate all Option 3 plots (per-slice native space trajectories).
    """
    if metrics is None:
        metrics = list(METRIC_NAMES.keys())
    
    print("\n" + "="*70)
    print("OPTION 3: Native Space Per-Slice Trajectories")
    print("="*70)
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nPlotting {metric}...")
            plot_perslice_native_trajectories(df, metric, output_dir)
        else:
            print(f"Warning: Metric '{metric}' not found in data")


# ============================================================================
# Option 4: Per-Level by Timepoint Trajectories
# ============================================================================

def plot_level_profiles_by_timepoint(df, metric, output_dir):
    """
    Plot metric profiles across vertebral levels, colored by timepoint.
    Shows mean ± STD + individual subject trajectories.
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, VertLevel, metric]
        metric: Name of the metric to plot (e.g., 'MEAN(area)')
        output_dir: Directory to save the plot
    """
    # Sort timepoints
    timepoints = sorted(df['timepoint'].unique(), key=lambda x: TIMEPOINT_MONTHS.get(x, 999))
    
    if len(timepoints) < 2:
        print(f"Warning: Need at least 2 timepoints for comparison, found {len(timepoints)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)
    
    # Get vertebral levels for x-axis
    vert_levels = sorted([v for v in df['VertLevel'].unique() if v in VERT_LABELS])
    x_positions = list(range(len(vert_levels)))
    
    # Plot for each timepoint
    for tp in timepoints:
        df_tp = df[df['timepoint'] == tp]
        
        if df_tp.empty:
            continue
        
        color = TIMEPOINT_COLORS.get(tp, '#000000')
        
        # Calculate mean and std for each vertebral level
        means = []
        stds = []
        x_vals = []
        
        for idx, vert_level in enumerate(vert_levels):
            df_level = df_tp[df_tp['VertLevel'] == vert_level]
            if not df_level.empty and metric in df_level.columns:
                values = df_level[metric].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    x_vals.append(idx)
        
        if len(means) >= 2:
            # Plot mean line with error bars
            ax.errorbar(x_vals, means, yerr=stds, 
                       color=color, linewidth=2.5, marker='o', markersize=8,
                       label=f'{tp} (n={len(df_tp["subject"].unique())})', 
                       capsize=5, capthick=2, alpha=0.9)
            
            # Plot individual subject trajectories (lighter, thinner lines)
            subjects = df_tp['subject'].unique()
            for subj in subjects:
                df_subj = df_tp[df_tp['subject'] == subj]
                
                subj_x = []
                subj_y = []
                
                for idx, vert_level in enumerate(vert_levels):
                    df_level_subj = df_subj[df_subj['VertLevel'] == vert_level]
                    if not df_level_subj.empty and metric in df_level_subj.columns:
                        value = df_level_subj[metric].dropna().values
                        if len(value) > 0:
                            subj_x.append(idx)
                            subj_y.append(value[0])
                
                if len(subj_x) >= 2:
                    ax.plot(subj_x, subj_y, color=color, linewidth=0.5, 
                           alpha=0.3, linestyle='-', zorder=1)
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    metric_unit = METRIC_UNITS.get(metric, '')
    
    ax.set_xlabel('Vertebral Level', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{metric_unit}]', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'Vertebral Level Profiles: {metric_name} by Timepoint', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.legend(title='Timepoint', fontsize=LEGEND_FONT_SIZE, 
             title_fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    
    # Set x-ticks to vertebral level labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([VERT_LABELS[v] for v in vert_levels])
    
    plt.tight_layout()
    
    # Save figure
    filename = f'trajectory_bylevel_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_option4_plots(df, output_dir, metrics=None):
    """
    Generate all Option 4 plots (level profiles by timepoint).
    """
    if metrics is None:
        metrics = list(METRIC_NAMES.keys())
    
    print("\n" + "="*70)
    print("OPTION 4: Per-Level by Timepoint Trajectories")
    print("="*70)
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nPlotting {metric}...")
            plot_level_profiles_by_timepoint(df, metric, output_dir)
        else:
            print(f"Warning: Metric '{metric}' not found in data")


# ============================================================================
# Option 5: Therapeutic Group Per-Level Trajectories
# ============================================================================

def plot_group_perlevel_trajectories(df, metric, output_dir, group_name, aligned=False):
    """
    Plot per-level trajectories for a therapeutic group.

    Conservative plots use baseline-aligned timepoints.
    Operative plots use time since the first post-operative timepoint (T0, T6, ...).
    """
    time_col = 'aligned_month' if aligned else 'timepoint_month'
    label_col = 'aligned_timepoint' if aligned else 'timepoint'

    df = df.dropna(subset=[time_col]).copy()
    time_values = sorted(df[time_col].unique())
    if len(time_values) < 1:
        print(f"Warning: No timepoints available for {group_name}")
        return

    tick_labels = (
        df[[time_col, label_col]]
        .drop_duplicates()
        .sort_values(time_col)
        .drop_duplicates(subset=[time_col])
        .set_index(time_col)[label_col]
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    for vert_level in sorted(VERT_COLORS.keys()):
        df_level = df[df['VertLevel'] == vert_level]
        if df_level.empty:
            continue

        color = VERT_COLORS[vert_level]
        label = VERT_LABELS[vert_level]

        means = []
        stds = []
        x_values = []

        for time_value in time_values:
            df_tp = df_level[df_level[time_col] == time_value]
            if not df_tp.empty and metric in df_tp.columns:
                values = df_tp[metric].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    x_values.append(time_value)

        if len(means) >= 1:
            means_arr = np.array(means)
            stds_arr = np.nan_to_num(np.array(stds), nan=0.0)

            ax.plot(x_values, means_arr,
                    color=color, linewidth=2.5, marker='o', markersize=8,
                    label=label, alpha=0.9)
            ax.fill_between(x_values,
                            means_arr - stds_arr,
                            means_arr + stds_arr,
                            color=color, alpha=0.15)

            subjects = df_level['subject'].unique()
            for subj in subjects:
                df_subj = df_level[df_level['subject'] == subj].sort_values(time_col)
                subj_x = []
                subj_y = []

                for time_value in time_values:
                    df_tp_subj = df_subj[df_subj[time_col] == time_value]
                    if not df_tp_subj.empty and metric in df_tp_subj.columns:
                        value = df_tp_subj[metric].dropna().values
                        if len(value) > 0:
                            subj_x.append(time_value)
                            subj_y.append(value[0])

                if len(subj_x) >= 2:
                    ax.plot(subj_x, subj_y, color=color, linewidth=0.5,
                            alpha=0.25, linestyle='-', zorder=1)

    metric_name = METRIC_NAMES.get(metric, metric)
    metric_unit = METRIC_UNITS.get(metric, '')
    n_subjects = df['subject'].nunique()

    xlabel = 'Time since first post-operative scan (months)' if aligned else 'Time (months)'
    title_suffix = 'aligned to first post-op timepoint' if aligned else 'time-aware conservative group'

    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{metric_unit}]', fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'{group_name.capitalize()} Trajectories: {metric_name} ({title_suffix})',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')

    ax.legend(title='Vertebral Level', fontsize=LEGEND_FONT_SIZE,
              title_fontsize=LEGEND_FONT_SIZE, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.set_xticks(time_values)
    ax.set_xticklabels([tick_labels[tv] for tv in time_values])

    plt.tight_layout()

    suffix = 'postop_aligned' if aligned else 'baseline_aligned'
    filename = f'trajectory_perlevel_{group_name}_{suffix}_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_option5_plots(df, output_dir, metrics=None):
    """Generate per-level trajectories split by time-aware therapeutic group."""
    if metrics is None:
        metrics = list(METRIC_NAMES.keys())

    print("\n" + "="*70)
    print("OPTION 5: Therapeutic Group Per-Level Trajectories")
    print("="*70)

    conservative_df, operative_df = prepare_grouped_trajectory_data(df)

    save_operative_postop_counts_table(operative_df, output_dir)

    group_configs = [
        ('conservative', conservative_df, False),
        ('operative', operative_df, True),
    ]

    for group_name, group_df, aligned in group_configs:
        group_output_dir = os.path.join(output_dir, 'therapeutic_groups', group_name)
        os.makedirs(group_output_dir, exist_ok=True)

        print(f"\nGroup: {group_name} | rows={len(group_df)} | subjects={group_df['subject'].nunique()}")
        for metric in metrics:
            if metric in group_df.columns:
                print(f"Plotting {metric}...")
                plot_group_perlevel_trajectories(group_df, metric, group_output_dir, group_name, aligned=aligned)
            else:
                print(f"Warning: Metric '{metric}' not found in data")


# ============================================================================
# Main
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Directory containing timepoint data CSV files'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--structure', default='cord', choices=['cord', 'canal', 'aSCOR'],
        help='Structure to analyze (default: cord)'
    )
    parser.add_argument(
        '--timepoints', nargs='+', default=['M0', 'M6', 'M12'],
        help='List of timepoints to include (default: M0 M6 M12)'
    )
    parser.add_argument(
        '--option', default='all', choices=['1', '2', '3', '4', '5', 'all'],
        help='Which plotting option to use: 1=time trajectories by level, 2=per-slice PAM50, 3=per-slice native, 4=level profiles by timepoint, 5=therapeutic groups, all=all options (default: all)'
    )
    parser.add_argument(
        '--metrics', nargs='+',
        help='Specific metrics to plot (default: all)'
    )
    parser.add_argument(
        '--labels-dir',
        help='Directory containing time-aware label files labels_M0.csv, labels_M6.csv, etc. If omitted, inferred from --data-dir when possible.'
    )
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Longitudinal Trajectory Plotting")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Structure: {args.structure}")
    print(f"Timepoints: {', '.join(args.timepoints)}")
    print(f"Option: {args.option}")

    labels_dir = args.labels_dir or infer_labels_dir(args.data_dir)
    if labels_dir:
        print(f"Labels directory: {labels_dir}")
    
    # Determine which metrics to plot
    if args.metrics:
        metrics_to_plot = [m for m in args.metrics if m in METRIC_NAMES]
    else:
        # Default metrics depend on structure
        if args.structure == 'aSCOR':
            metrics_to_plot = ['aSCOR']
        else:
            metrics_to_plot = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 
                              'MEAN(eccentricity)', 'MEAN(solidity)']
    
    print(f"Metrics: {', '.join(metrics_to_plot)}")
    
    # Option 1: Per-level native space trajectories
    if args.option in ['1', 'all']:
        try:
            print("\nLoading per-level native space data...")
            df_perlevel = load_perlevel_data(args.data_dir, args.structure, args.timepoints)
            print(f"Loaded {len(df_perlevel)} rows for {len(df_perlevel['subject'].unique())} subjects")
            
            generate_option1_plots(df_perlevel, args.output_dir, metrics_to_plot)
        except Exception as e:
            print(f"\nError generating Option 1 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Option 2: Per-slice PAM50 trajectories
    if args.option in ['2', 'all']:
        try:
            print("\nLoading per-slice PAM50 data...")
            df_perslice_pam50 = load_perslice_pam50_data(args.data_dir, args.structure, args.timepoints)
            print(f"Loaded {len(df_perslice_pam50)} rows for {len(df_perslice_pam50['subject'].unique())} subjects")
            
            generate_option2_plots(df_perslice_pam50, args.output_dir, metrics_to_plot)
        except Exception as e:
            print(f"\nError generating Option 2 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Option 3: Per-slice native space trajectories
    if args.option in ['3', 'all']:
        try:
            print("\nLoading per-slice native space data...")
            df_perslice_native = load_perslice_native_data(args.data_dir, args.structure, args.timepoints)
            print(f"Loaded {len(df_perslice_native)} rows for {len(df_perslice_native['subject'].unique())} subjects")
            
            generate_option3_plots(df_perslice_native, args.output_dir, metrics_to_plot)
        except Exception as e:
            print(f"\nError generating Option 3 plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Option 4: Per-level by timepoint trajectories
    if args.option in ['4', 'all']:
        try:
            print("\nLoading per-level native space data for Option 4...")
            df_perlevel_opt4 = load_perlevel_data(args.data_dir, args.structure, args.timepoints)
            print(f"Loaded {len(df_perlevel_opt4)} rows for {len(df_perlevel_opt4['subject'].unique())} subjects")
            
            generate_option4_plots(df_perlevel_opt4, args.output_dir, metrics_to_plot)
        except Exception as e:
            print(f"\nError generating Option 4 plots: {e}")
            import traceback
            traceback.print_exc()

    # Option 5: Therapeutic group per-level trajectories
    if args.option in ['5', 'all']:
        try:
            if not labels_dir:
                raise ValueError('No labels directory provided or inferred for therapeutic group plots')

            print("\nLoading per-level native space data for Option 5...")
            df_perlevel_group = load_perlevel_data(args.data_dir, args.structure, args.timepoints)
            print(f"Loaded {len(df_perlevel_group)} rows for {len(df_perlevel_group['subject'].unique())} subjects")

            labels_df = load_timepoint_aware_labels(labels_dir, args.timepoints)
            print(f"Loaded {len(labels_df)} label rows for {labels_df['participant_id'].nunique()} subjects")

            df_perlevel_group = attach_timepoint_aware_labels(df_perlevel_group, labels_df)
            print(
                "Merged therapeutic labels: "
                f"{len(df_perlevel_group)} rows | "
                f"operative={df_perlevel_group[df_perlevel_group['therapeutic_group'] == 'operative']['subject'].nunique()} subjects | "
                f"conservative={df_perlevel_group[df_perlevel_group['therapeutic_group'] == 'conservative']['subject'].nunique()} subjects"
            )

            generate_option5_plots(df_perlevel_group, args.output_dir, metrics_to_plot)
        except Exception as e:
            print(f"\nError generating Option 5 plots: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Plots saved in:", args.output_dir)
    print("="*70)


if __name__ == '__main__':
    main()
