#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate boxplots and violin plots for spinal cord metrics grouped by vertebral level.

For each metric, creates a figure showing:
- X-axis: Vertebral levels (C2-C7)
- Y-axis: Metric values
- Hue (colors): Timepoints (M0, M6, M12, M24, etc.)
- Shows both boxplots and violin plots

Supports spinal cord, canal, and aSCOR metrics.

Usage example:
    python plot_boxplots_violins_by_level.py \\
        --data-dir /path/to/timepoint_data \\
        --output-dir /path/to/output \\
        --structure cord \\
        --timepoints M0 M6 M12

Authors: Kahina Baouche, 2026
"""

import os
import sys
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Constants
# ============================================================================

# Metric names mapping
METRIC_NAMES = {
    'MEAN(area)': 'Cross-Sectional Area (mm²)',
    'MEAN(diameter_AP)': 'AP Diameter (mm)',
    'MEAN(diameter_RL)': 'RL Diameter (mm)',
    'MEAN(eccentricity)': 'Eccentricity (a.u.)',
    'MEAN(solidity)': 'Solidity (%)',
    'aSCOR': 'aSCOR (Area Spinal Cord Occupation Ratio)',
}

# Vertebral level colors
VERT_COLORS = {
    2: '#d62728',
    3: '#ff7f0e',
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

# Timepoint colors (pastel palette)
TIMEPOINT_COLORS = {
    'M0': '#AED6F1',   
    'M3': '#F8B88B',   
    'M6': '#ABEBC6',   
    'M12': '#F5B7B1',  
    'M24': '#D7BDE2',  
    'M36': '#D5A6BD',  
    'M48': '#F8BBD0',  
    'M60': '#BDC3C7', 
}

# Font sizes
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10

# Figure size
FIG_WIDTH = 14
FIG_HEIGHT = 6


# ============================================================================
# Data loading functions
# ============================================================================

def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'sub-001')."""
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
        raise ValueError(f"No data files found for timepoints: {timepoints}")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Filter to cervical levels only (C2-C7 = VertLevel 2-7)
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)]
    
    # Add vertebral level labels
    combined['Level'] = combined['VertLevel'].map(VERT_LABELS)
    
    return combined


# ============================================================================
# Plotting functions
# ============================================================================

def create_boxplot_figure(df, metric, output_dir):
    """
    Create a boxplot figure showing metric values across vertebral levels
    with different colors for each timepoint (using seaborn's hue).
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, VertLevel, Level, metric]
        metric: Name of the metric to plot
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Get sorted timepoints
    timepoints = sorted(df['timepoint'].unique())
    timepoint_colors = {tp: TIMEPOINT_COLORS.get(tp, '#000000') for tp in timepoints}
    
    # Create boxplot with hue
    sns.boxplot(
        data=df,
        x='Level',
        y=metric,
        hue='timepoint',
        hue_order=timepoints,
        palette=timepoint_colors,
        ax=ax
    )
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    ax.set_xlabel('Vertebral Level', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title(f'Boxplot: {metric_name} by Vertebral Level and Timepoint', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.legend(title='Timepoint', fontsize=TICK_FONT_SIZE, title_fontsize=TICK_FONT_SIZE, 
             loc='best', framealpha=0.95)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    filename = f'boxplot_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def create_violin_figure(df, metric, output_dir):
    """
    Create a violin plot figure showing metric distributions across vertebral levels
    with different colors for each timepoint (hue).
    
    Parameters:
        df: DataFrame with columns [subject, timepoint, VertLevel, Level, metric]
        metric: Name of the metric to plot
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Get sorted timepoints
    timepoints = sorted(df['timepoint'].unique())
    timepoint_colors = {tp: TIMEPOINT_COLORS.get(tp, '#000000') for tp in timepoints}
    
    # Create violin plot with hue
    sns.violinplot(
        data=df,
        x='Level',
        y=metric,
        hue='timepoint',
        hue_order=timepoints,
        palette=timepoint_colors,
        ax=ax,
        split=False
    )
    
    # Formatting
    metric_name = METRIC_NAMES.get(metric, metric)
    ax.set_xlabel('Vertebral Level', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_title(f'Violin Plot: {metric_name} by Vertebral Level and Timepoint', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.legend(title='Timepoint', fontsize=TICK_FONT_SIZE, title_fontsize=TICK_FONT_SIZE,
             loc='best', framealpha=0.95)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    filename = f'violinplot_{metric.replace("(", "").replace(")", "").replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def generate_plots(df, output_dir, metrics=None, plot_type='both'):
    """
    Generate boxplots and/or violin plots for all metrics.
    
    Parameters:
        df: DataFrame with metric data
        output_dir: Directory to save plots
        metrics: List of metrics to plot (default: all available)
        plot_type: 'boxplot', 'violinplot', or 'both'
    """
    if metrics is None:
        # Auto-detect available metrics
        metrics = [m for m in METRIC_NAMES.keys() if m in df.columns]
    
    print(f"\nGenerating plots for {len(metrics)} metrics...")
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data, skipping...")
            continue
        
        print(f"\n  Processing: {metric}")
        
        if plot_type in ['boxplot', 'both']:
            print(f"    Creating boxplot...")
            create_boxplot_figure(df, metric, output_dir)
        
        if plot_type in ['violinplot', 'both']:
            print(f"    Creating violin plot...")
            create_violin_figure(df, metric, output_dir)


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
        '--plot-type', default='both', choices=['boxplot', 'violinplot', 'both'],
        help='Type of plots to generate (default: both)'
    )
    parser.add_argument(
        '--metrics', nargs='+',
        help='Specific metrics to plot (default: all)'
    )
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Boxplots and Violin Plots by Vertebral Level")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Structure: {args.structure}")
    print(f"Timepoints: {', '.join(args.timepoints)}")
    print(f"Plot type: {args.plot_type}")
    
    # Load data and generate plots - verbose with print statements for debugging
    try:
        print(f"\nLoading per-level data")
        df = load_perlevel_data(args.data_dir, args.structure, args.timepoints)
        print(f"Loaded {len(df)} rows for {len(df['subject'].unique())} subjects")
        print(f"Available metrics: {[m for m in METRIC_NAMES.keys() if m in df.columns]}")
        
        # Determine which metrics to plot
        if args.metrics:
            metrics_to_plot = [m for m in args.metrics if m in METRIC_NAMES and m in df.columns]
        else:
            # Default metrics depend on structure
            if args.structure == 'aSCOR':
                metrics_to_plot = ['aSCOR'] if 'aSCOR' in df.columns else []
            else:
                metrics_to_plot = [m for m in ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)',
                                                'MEAN(eccentricity)', 'MEAN(solidity)'] if m in df.columns]
        
        print(f"Metrics to plot: {', '.join(metrics_to_plot)}")
        
        # Generate plots
        generate_plots(df, args.output_dir, metrics_to_plot, args.plot_type)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Plots in:", args.output_dir)
    print("="*70)


if __name__ == '__main__':
    main()
