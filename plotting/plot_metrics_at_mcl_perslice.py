#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute and plot morphometric metrics at the Maximum Compression Level (MCL).

For each subject, a window of N slices is extracted on each side of the MCL
disc junction (N slices from the upper vertebra + N slices from the lower
vertebra). Metrics are averaged over those 2*N slices per timepoint and
plotted as longitudinal trajectories (individual + group mean ± SD).

Supports both native-space and PAM50-normalized per-slice data.

--- Standard mode (default) ---
  X-axis : months from M0 baseline scan.
  Delta  : value_at_tp - value_at_M0.

--- Operative mode (--operative-only) ---
  Filters to subjects who had surgery and have a valid pre/post scan mapping
  in --baseline-surgery-mapping (baseline_surgery_dates_merged_clean.csv).

  How the pre-surgery timepoint is identified
  -------------------------------------------
  The mapping CSV was built by comparing each subject's actual scan dates
  (estimated as date_of_scan + nominal_months) to their surgery_date.
  The column `before_surg_found` gives the timepoint code (e.g. M0, M6)
  of the last scan that occurred BEFORE surgery, confirmed to have data.
  The column `months_diff_scan_to_surgery` gives the number of months
  elapsed between the M0 scan date and the surgery date.

  How months-from-surgery is computed
  ------------------------------------
  For each timepoint tp:
      months_from_surgery = TIMEPOINT_MONTHS[tp] - months_diff_scan_to_surgery

  Example: subject whose surgery was 1.8 months after their M0 scan
      M0  → 0   - 1.8 = -1.8  (1.8 months before surgery)
      M6  → 6   - 1.8 =  4.2  (4.2 months after surgery)
      M12 → 12  - 1.8 = 10.2  etc.

  How delta is computed
  ---------------------
  The reference value is the metric at `before_surg_found`.
  delta_at_tp = value_at_tp - value_at_before_surg_found
  The pre-surgery timepoint therefore always has delta = 0.

Example usage:
    # Standard analysis (all subjects)
    python plot_metrics_at_mcl_perslice.py \\
        --data-dir path/to/timepoint_data \\
        --participants data/dcm-zurich/participants.tsv \\
        --output-dir results/mcl_native --space native --n-slices 3

    # Operative-only analysis (months from surgery)
    python plot_metrics_at_mcl_perslice.py \\
        --data-dir path/to/timepoint_data \\
        --participants data/dcm-zurich/participants.tsv \\
        --baseline-surgery-mapping baseline_surgery_dates_merged_clean.csv \\
        --output-dir results/mcl_operative --space native --n-slices 3 \\
        --operative-only

Author: Kahina Baouche
"""

import os
import re
import argparse
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

TIMEPOINT_ORDER = ['M0', 'M3', 'M6', 'M12', 'M24', 'M36', 'M48', 'M60']

TIMEPOINT_MONTHS = {
    'M0': 0, 'M3': 3, 'M6': 6, 'M12': 12,
    'M24': 24, 'M36': 36, 'M48': 48, 'M60': 60,
}

TIMEPOINT_COLORS = {
    'M0': '#1f77b4', 'M3': '#ff7f0e', 'M6': '#2ca02c', 'M12': '#d62728',
    'M24': '#9467bd', 'M36': '#8c564b', 'M48': '#e377c2', 'M60': '#7f7f7f',
}

METRIC_NAMES = {
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(diameter_RL)': 'RL Diameter',
    'MEAN(compression_ratio)': 'Compression Ratio',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
}

METRIC_UNITS = {
    'MEAN(area)': 'mm²',
    'MEAN(diameter_AP)': 'mm',
    'MEAN(diameter_RL)': 'mm',
    'MEAN(compression_ratio)': 'a.u.',
    'MEAN(eccentricity)': 'a.u.',
    'MEAN(solidity)': '%',
}

LABELS_FONT_SIZE = 13
TICKS_FONT_SIZE = 11
TITLE_FONT_SIZE = 14


# ============================================================================
# Argument parser
# ============================================================================

def get_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--data-dir', required=True,
                   help='Directory containing per-slice metric CSV files')
    p.add_argument('--participants', required=True,
                   help='participants.tsv with participant_id and maximum_stenosis columns')
    p.add_argument('--output-dir', required=True,
                   help='Directory to save results and figures')
    p.add_argument('--space', default='native', choices=['native', 'PAM50'],
                   help='Whether to use native-space or PAM50-normalized per-slice data (default: native)')
    p.add_argument('--structure', default='cord', choices=['cord', 'canal', 'aSCOR'],
                   help='Structure to analyze (default: cord)')
    p.add_argument('--timepoints', nargs='+', default=['M0', 'M6', 'M12'],
                   help='Timepoints to include (default: M0 M6 M12)')
    p.add_argument('--n-slices', type=int, default=3,
                   help='Number of slices to take on each side of the MCL junction (default: 3)')
    p.add_argument('--metrics', nargs='+',
                   default=['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(compression_ratio)'],
                   help='Metrics to plot (default: area, AP diameter, compression ratio)')
    p.add_argument('--operative-only', action='store_true',
                   help='Restrict to operative subjects only; x-axis becomes months from surgery '
                        'and delta is relative to the last pre-surgery scan. '
                        'Requires --baseline-surgery-mapping.')
    p.add_argument('--baseline-surgery-mapping',
                   help='Path to baseline_surgery_dates_merged_clean.csv (columns: subject, '
                        'date_of_scan, surgery_date, months_diff_scan_to_surgery, '
                        'before_surg_found, after_surg_found)')
    return p


# ============================================================================
# Helpers
# ============================================================================

def extract_subject_id(filename):
    """Extract 'sub-XXX' from a BIDS filename string."""
    m = re.search(r'sub-\d+', str(filename))
    return m.group(0) if m else None


def parse_mcl(mcl_str):
    """
    Parse MCL string (e.g. 'C4/C5') into (upper_level, lower_level) integers.
    Returns None if parsing fails.
    """
    if pd.isna(mcl_str) or not isinstance(mcl_str, str):
        return None
    m = re.match(r'\s*C(\d+)\s*/\s*C(\d+)\s*$', mcl_str.strip())
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (min(a, b), max(a, b))


# ============================================================================
# Data loading
# ============================================================================

def load_participants(participants_file):
    """
    Load participant_id and maximum_stenosis from participants.tsv.

    Returns DataFrame with columns: subject, upper_level, lower_level
    (the two vertebral levels flanking the MCL disc).
    """
    df = pd.read_csv(participants_file, sep='\t')
    if 'participant_id' not in df.columns or 'maximum_stenosis' not in df.columns:
        raise ValueError('participants.tsv must contain participant_id and maximum_stenosis')

    df = df[['participant_id', 'maximum_stenosis']].rename(
        columns={'participant_id': 'subject'}
    ).copy()

    parsed = df['maximum_stenosis'].apply(parse_mcl)
    df['upper_level'] = parsed.apply(lambda x: x[0] if x else None)
    df['lower_level'] = parsed.apply(lambda x: x[1] if x else None)

    missing = df['upper_level'].isna().sum()
    if missing:
        print(f'  Warning: {missing} subject(s) with missing/unparseable maximum_stenosis will be skipped')

    df = df.dropna(subset=['upper_level', 'lower_level'])
    df['upper_level'] = df['upper_level'].astype(int)
    df['lower_level'] = df['lower_level'].astype(int)

    print(f'  Loaded MCL for {len(df)} subjects')
    print('  MCL distribution:')
    print(df['maximum_stenosis'].value_counts().to_string())

    return df


def load_perslice_data(data_dir, structure, timepoints, space='native'):
    """
    Load per-slice metric CSVs for all timepoints.

    File naming:
      native : T2w_ax_{structure}_metrics_perslice_{tp}_data.csv
      PAM50  : T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv

    Returns combined DataFrame with subject, timepoint, Slice (I->S),
    VertLevel, and metric columns.
    """
    all_data = []

    for tp in timepoints:
        if space == 'PAM50':
            fname = f'T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv'
        else:
            fname = f'T2w_ax_{structure}_metrics_perslice_{tp}_data.csv'

        fpath = Path(data_dir) / fname
        if not fpath.exists():
            print(f'  Warning: file not found: {fpath}')
            continue

        print(f'  Loading {fpath.name}...')
        df = pd.read_csv(fpath)

        if 'subject' not in df.columns:
            fname_col = 'Filename_sc' if structure == 'aSCOR' else 'Filename'
            if fname_col in df.columns:
                df['subject'] = df[fname_col].apply(extract_subject_id)

        if 'timepoint' not in df.columns:
            df['timepoint'] = tp

        all_data.append(df)

    if not all_data:
        raise ValueError(f'No per-slice data files found in {data_dir}')

    combined = pd.concat(all_data, ignore_index=True)

    # Filter to cervical levels C2–C7
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)].copy()

    # Compute compression ratio from AP and RL diameters
    if 'MEAN(diameter_AP)' in combined.columns and 'MEAN(diameter_RL)' in combined.columns:
        combined['MEAN(compression_ratio)'] = (
            combined['MEAN(diameter_AP)'] / combined['MEAN(diameter_RL)']
        )

    print(f'  Loaded {len(combined)} rows for {combined["subject"].nunique()} subjects '
          f'across {combined["timepoint"].nunique()} timepoint(s)')
    return combined


# ============================================================================
# MCL slice extraction
# ============================================================================

def find_mcl_junction_slices(df_subj_tp, upper_level, lower_level, n_slices):
    """
    Find the N slices closest to the disc junction on each side (upper and lower
    vertebra) for a single (subject, timepoint) dataframe.

    In both native and PAM50 space, higher Slice (I->S) values correspond to
    more superior (closer to head) positions. Therefore:
      - The upper vertebra (e.g. C4) occupies higher slice numbers
      - The lower vertebra (e.g. C5) occupies lower slice numbers
      - The junction is between the lowest slices of the upper vertebra
        and the highest slices of the lower vertebra

    Parameters
    ----------
    df_subj_tp : DataFrame
        Data for one subject at one timepoint; must have Slice (I->S) and VertLevel.
    upper_level : int
        The superior vertebral level (e.g. 4 for C4/C5).
    lower_level : int
        The inferior vertebral level (e.g. 5 for C4/C5).
    n_slices : int
        Number of slices to take from each side of the junction.

    Returns
    -------
    list of int or None
        Slice numbers in the MCL window, or None if not enough data.
    """
    df_upper = df_subj_tp[df_subj_tp['VertLevel'] == upper_level]['Slice (I->S)'].dropna()
    df_lower = df_subj_tp[df_subj_tp['VertLevel'] == lower_level]['Slice (I->S)'].dropna()

    if df_upper.empty or df_lower.empty:
        return None

    # From the upper vertebra: take the n_slices with the lowest slice numbers
    # (most inferior slices of the upper level = closest to the junction)
    upper_slices = sorted(df_upper.unique())[:n_slices]

    # From the lower vertebra: take the n_slices with the highest slice numbers
    # (most superior slices of the lower level = closest to the junction)
    lower_slices = sorted(df_lower.unique())[-n_slices:]

    junction_slices = sorted(set(upper_slices + lower_slices))

    if len(junction_slices) == 0:
        return None

    return junction_slices


def extract_metrics_at_mcl(all_data, participants_df, n_slices, metrics):
    """
    For each (subject, timepoint), extract per-slice metric values in the MCL
    window and average them.

    Returns a long-format DataFrame with one row per (subject, timepoint) and
    columns: subject, maximum_stenosis, timepoint, months, n_slices_used,
    and one averaged-metric column per metric.
    """
    print(f'\n=== Extracting metrics at MCL (±{n_slices} slices around junction) ===')

    records = []
    skipped_no_mcl = 0
    skipped_no_data = 0

    # Iterate over subjects that have MCL info
    for _, part_row in participants_df.iterrows():
        subject = part_row['subject']
        upper_level = part_row['upper_level']
        lower_level = part_row['lower_level']
        mcl_label = part_row['maximum_stenosis']

        df_subj = all_data[all_data['subject'] == subject]

        if df_subj.empty:
            skipped_no_data += 1
            continue

        for tp in df_subj['timepoint'].unique():
            df_subj_tp = df_subj[df_subj['timepoint'] == tp]

            junction_slices = find_mcl_junction_slices(
                df_subj_tp, upper_level, lower_level, n_slices
            )

            if junction_slices is None:
                continue

            # Extract rows for those slices
            df_window = df_subj_tp[df_subj_tp['Slice (I->S)'].isin(junction_slices)]

            if df_window.empty:
                continue

            record = {
                'subject': subject,
                'maximum_stenosis': mcl_label,
                'timepoint': tp,
                'months': TIMEPOINT_MONTHS.get(tp, None),
                'n_slices_used': len(df_window),
            }

            for metric in metrics:
                if metric in df_window.columns:
                    values = df_window[metric].dropna()
                    record[metric] = values.mean() if len(values) > 0 else np.nan
                else:
                    record[metric] = np.nan

            records.append(record)

    df_mcl = pd.DataFrame(records)

    print(f'  Subjects with MCL info: {len(participants_df)}')
    print(f'  Subjects with no data in per-slice files: {skipped_no_data}')
    print(f'  Total subject-timepoint rows extracted: {len(df_mcl)}')
    if not df_mcl.empty:
        print(f'  Unique subjects with data: {df_mcl["subject"].nunique()}')

    return df_mcl


# ============================================================================
# Plotting
# ============================================================================

def plot_metric_trajectories(df_mcl, metric, output_dir, space='native'):
    """
    For a single metric, plot:
      - Individual subject trajectories (faint colored lines)
      - Group mean ± SD (bold line with shaded band)

    X-axis: months from baseline.
    """
    df_plot = df_mcl.dropna(subset=[metric, 'months']).copy()
    if df_plot.empty:
        print(f'  No data for metric {metric}, skipping plot')
        return

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 6))

    subjects = df_plot['subject'].unique()
    subject_colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    for i, subj in enumerate(subjects):
        df_subj = df_plot[df_plot['subject'] == subj].sort_values('months')
        ax.plot(df_subj['months'], df_subj[metric],
                color=subject_colors[i], alpha=0.25, linewidth=1,
                marker='o', markersize=3, zorder=1)

    # Group mean ± SD
    grouped = df_plot.groupby('months')[metric].agg(['mean', 'std', 'count'])
    ax.plot(grouped.index, grouped['mean'],
            color='black', linewidth=3, marker='o', markersize=7,
            label='Group mean', zorder=10)
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='black', alpha=0.15, label='±1 SD', zorder=5)
    ax.axhline(y=grouped['mean'].iloc[0], color='black',
               linestyle='--', linewidth=1, alpha=0.4)

    # Sample sizes below SD band
    for months, row in grouped.iterrows():
        ax.text(months, row['mean'] - row['std'] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={int(row["count"])}', ha='center', fontsize=9, color='dimgray')

    metric_name = METRIC_NAMES.get(metric, metric)
    unit = METRIC_UNITS.get(metric, '')

    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{unit}]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(
        f'{metric_name} at MCL (±{df_mcl["n_slices_used"].median():.0f} slices) [{space} space]\n'
        f'n={df_plot["subject"].nunique()} subjects',
        fontsize=TITLE_FONT_SIZE, fontweight='bold'
    )
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_metric = metric.replace('(', '').replace(')', '').replace(' ', '_')
    fname = f'mcl_{safe_metric}_trajectories.png'
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


def plot_delta_from_baseline(df_mcl, metric, output_dir, space='native'):
    """
    For a single metric, compute delta from each subject's baseline value and plot
    individual trajectories + group mean ± SD.
    """
    df_plot = df_mcl.dropna(subset=[metric, 'months']).copy()
    if df_plot.empty:
        return

    # Compute delta per subject
    delta_records = []
    for subj in df_plot['subject'].unique():
        df_subj = df_plot[df_plot['subject'] == subj].sort_values('months')
        bl = df_subj[df_subj['months'] == 0]
        if bl.empty:
            continue
        baseline_val = bl.iloc[0][metric]
        for _, row in df_subj.iterrows():
            delta_records.append({
                'subject': subj,
                'months': row['months'],
                'delta': row[metric] - baseline_val,
            })

    if not delta_records:
        print(f'  No baseline data for delta plot of {metric}')
        return

    df_delta = pd.DataFrame(delta_records)

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 6))

    subjects = df_delta['subject'].unique()
    subject_colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    for i, subj in enumerate(subjects):
        df_subj = df_delta[df_delta['subject'] == subj].sort_values('months')
        ax.plot(df_subj['months'], df_subj['delta'],
                color=subject_colors[i], alpha=0.25, linewidth=1,
                marker='o', markersize=3, zorder=1)

    grouped = df_delta.groupby('months')['delta'].agg(['mean', 'std', 'count'])
    ax.plot(grouped.index, grouped['mean'],
            color='steelblue', linewidth=3, marker='o', markersize=7,
            label='Group mean', zorder=10)
    ax.fill_between(grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color='steelblue', alpha=0.2, label='±1 SD', zorder=5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='No change from baseline')

    for months, row in grouped.iterrows():
        ax.text(months, row['mean'] - row['std'] - 0.05 * abs(ax.get_ylim()[1] - ax.get_ylim()[0]),
                f'n={int(row["count"])}', ha='center', fontsize=9, color='dimgray')

    metric_name = METRIC_NAMES.get(metric, metric)
    unit = METRIC_UNITS.get(metric, '')

    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(f'Δ {metric_name} [{unit}]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(
        f'Change in {metric_name} from Baseline at MCL [{space} space]\n'
        f'n={df_delta["subject"].nunique()} subjects',
        fontsize=TITLE_FONT_SIZE, fontweight='bold'
    )
    ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_metric = metric.replace('(', '').replace(')', '').replace(' ', '_')
    fname = f'mcl_{safe_metric}_delta_from_baseline.png'
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


def plot_multi_metric_panel(df_mcl, metrics, output_dir, space='native'):
    """
    One figure with one row per metric: absolute value trajectories (left column)
    and delta from baseline (right column). Group mean ± SD + individual lines.
    """
    available = [m for m in metrics if m in df_mcl.columns]
    if not available:
        print('  No metrics available for multi-panel plot')
        return

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.5 * n_rows), sharex='col')
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, metric in enumerate(available):
        df_abs = df_mcl.dropna(subset=[metric, 'months']).copy()
        metric_name = METRIC_NAMES.get(metric, metric)
        unit = METRIC_UNITS.get(metric, '')

        # ---- Left: absolute values ----
        ax_abs = axes[row_idx, 0]
        if not df_abs.empty:
            subjects = df_abs['subject'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
            for i, subj in enumerate(subjects):
                df_s = df_abs[df_abs['subject'] == subj].sort_values('months')
                ax_abs.plot(df_s['months'], df_s[metric],
                            color=colors[i], alpha=0.2, linewidth=1, zorder=1)

            grp = df_abs.groupby('months')[metric].agg(['mean', 'std', 'count'])
            ax_abs.plot(grp.index, grp['mean'], color='black', linewidth=2.5,
                        marker='o', markersize=6, zorder=10)
            ax_abs.fill_between(grp.index,
                                grp['mean'] - grp['std'],
                                grp['mean'] + grp['std'],
                                color='black', alpha=0.12, zorder=5)

        ax_abs.set_ylabel(f'{metric_name}\n[{unit}]', fontsize=LABELS_FONT_SIZE)
        ax_abs.tick_params(labelsize=TICKS_FONT_SIZE)
        ax_abs.grid(True, alpha=0.25)
        if row_idx == 0:
            ax_abs.set_title('Absolute Values at MCL', fontsize=TITLE_FONT_SIZE, fontweight='bold')

        # ---- Right: delta from baseline ----
        ax_dlt = axes[row_idx, 1]
        delta_records = []
        for subj in df_abs['subject'].unique():
            df_s = df_abs[df_abs['subject'] == subj].sort_values('months')
            bl = df_s[df_s['months'] == 0]
            if bl.empty:
                continue
            bl_val = bl.iloc[0][metric]
            for _, r in df_s.iterrows():
                delta_records.append({'subject': subj, 'months': r['months'],
                                      'delta': r[metric] - bl_val})

        if delta_records:
            df_dlt = pd.DataFrame(delta_records)
            subjects = df_dlt['subject'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
            for i, subj in enumerate(subjects):
                df_s = df_dlt[df_dlt['subject'] == subj].sort_values('months')
                ax_dlt.plot(df_s['months'], df_s['delta'],
                            color=colors[i], alpha=0.2, linewidth=1, zorder=1)

            grp_d = df_dlt.groupby('months')['delta'].agg(['mean', 'std', 'count'])
            ax_dlt.plot(grp_d.index, grp_d['mean'], color='steelblue', linewidth=2.5,
                        marker='o', markersize=6, zorder=10,
                        label=f'n={df_dlt["subject"].nunique()}')
            ax_dlt.fill_between(grp_d.index,
                                grp_d['mean'] - grp_d['std'],
                                grp_d['mean'] + grp_d['std'],
                                color='steelblue', alpha=0.15, zorder=5)
            ax_dlt.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
            ax_dlt.legend(fontsize=TICKS_FONT_SIZE - 1, loc='best')

        ax_dlt.set_ylabel(f'Δ {metric_name}\n[{unit}]', fontsize=LABELS_FONT_SIZE)
        ax_dlt.tick_params(labelsize=TICKS_FONT_SIZE)
        ax_dlt.grid(True, alpha=0.25)
        if row_idx == 0:
            ax_dlt.set_title('Change from Baseline at MCL', fontsize=TITLE_FONT_SIZE, fontweight='bold')

    axes[-1, 0].set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    axes[-1, 1].set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)

    n_slices_label = int(df_mcl['n_slices_used'].median())
    fig.suptitle(
        f'Morphometrics at MCL — {n_slices_label}-slice window around MCL junction [{space} space]',
        fontsize=TITLE_FONT_SIZE + 1, fontweight='bold', y=1.002
    )
    plt.tight_layout()

    fpath = os.path.join(output_dir, 'mcl_metrics_panel.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


def plot_by_mcl_group(df_mcl, metric, output_dir, n_slices, space='native'):
    """
    Stratify trajectories by MCL group (e.g. C4/C5, C5/C6) — one line per group.
    Groups with fewer than 3 subjects are merged into 'Other'.
    """
    df_plot = df_mcl.dropna(subset=[metric, 'months']).copy()
    if df_plot.empty:
        return

    group_counts = df_plot.groupby('maximum_stenosis')['subject'].nunique()
    rare_groups = group_counts[group_counts < 3].index.tolist()
    df_plot['mcl_group'] = df_plot['maximum_stenosis'].apply(
        lambda x: 'Other' if x in rare_groups else x
    )

    groups = sorted(df_plot['mcl_group'].unique())
    palette = sns.color_palette('tab10', len(groups))
    group_color = dict(zip(groups, palette))

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(11, 6))

    for grp in groups:
        df_grp = df_plot[df_plot['mcl_group'] == grp]
        n_subj = df_grp['subject'].nunique()
        agg = df_grp.groupby('months')[metric].agg(['mean', 'std'])
        color = group_color[grp]
        ax.plot(agg.index, agg['mean'], color=color, linewidth=2.5,
                marker='o', markersize=7, label=f'{grp} (n={n_subj})', zorder=5)
        ax.fill_between(agg.index,
                        agg['mean'] - agg['std'],
                        agg['mean'] + agg['std'],
                        color=color, alpha=0.15, zorder=3)

    metric_name = METRIC_NAMES.get(metric, metric)
    unit = METRIC_UNITS.get(metric, '')
    ax.set_xlabel('Months from Baseline', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{unit}]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(
        f'{metric_name} at MCL — stratified by MCL group (±{n_slices} slices) [{space} space]',
        fontsize=TITLE_FONT_SIZE, fontweight='bold'
    )
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_metric = metric.replace('(', '').replace(')', '').replace(' ', '_')
    fname = f'mcl_{safe_metric}_by_mcl_group.png'
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


# ============================================================================
# Operative analysis: surgery-relative timeline
# ============================================================================

def load_surgery_mapping(mapping_file):
    """
    Load baseline_surgery_dates_merged_clean.csv.

    Retains only subjects with valid before_surg_found AND after_surg_found
    (i.e. confirmed data exists on both sides of surgery).

    Key columns used downstream:
      before_surg_found          : timepoint code of last pre-surgery scan
      after_surg_found           : timepoint code of first post-surgery scan
      months_diff_scan_to_surgery: months from M0 scan date to surgery date
                                   (used to shift the timepoint axis to
                                    months-from-surgery)
    """
    df = pd.read_csv(mapping_file)

    required = {'subject', 'months_diff_scan_to_surgery', 'before_surg_found', 'after_surg_found'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'baseline_surgery_dates_merged_clean.csv missing columns: {missing}')

    # Standardise subject IDs to sub-XXX format
    df['subject'] = df['subject'].astype(str)
    if not df['subject'].str.startswith('sub-').all():
        df['subject'] = 'sub-' + df['subject'].str.extract(r'(\d+)')[0].str.zfill(3)

    # Keep only subjects with both pre- and post-surgery scans confirmed
    df = df[df['before_surg_found'].notna() & df['after_surg_found'].notna()].copy()

    print(f'  Surgery mapping loaded: {len(df)} subjects with valid pre/post scan pairs')
    print('  Pre-surgery timepoint distribution:')
    print(df['before_surg_found'].value_counts().to_string())

    return df[['subject', 'months_diff_scan_to_surgery',
               'before_surg_found', 'after_surg_found']].copy()


def extract_metrics_operative(all_data, participants_df, surgery_mapping, n_slices, metrics):
    """
    Extract MCL-window metrics for operative subjects only, computing
    months_from_surgery and delta relative to the pre-surgery scan.

    For each subject in surgery_mapping:
      1. Retrieve MCL (upper/lower level) from participants_df.
      2. For each available timepoint, extract the metric window around the
         MCL junction (same logic as extract_metrics_at_mcl).
      3. Compute months_from_surgery = TIMEPOINT_MONTHS[tp] - months_diff_scan_to_surgery
         (positive = after surgery, negative = before surgery).
      4. Compute delta = value_at_tp - value_at_before_surg_found.
         The pre-surgery timepoint therefore has delta = 0 by definition.
      5. Tag each row as 'pre_surgery' or 'post_surgery' for later filtering.

    Returns a long-format DataFrame with one row per (subject, timepoint).
    """
    print(f'\n=== Operative analysis: extracting metrics at MCL (±{n_slices} slices) ===')

    # Merge MCL info into surgery mapping
    mapping = surgery_mapping.merge(
        participants_df[['subject', 'upper_level', 'lower_level', 'maximum_stenosis']],
        on='subject', how='inner'
    )
    print(f'  Subjects with both surgery mapping and MCL info: {len(mapping)}')

    records = []

    for _, row in mapping.iterrows():
        subject = row['subject']
        upper_level = row['upper_level']
        lower_level = row['lower_level']
        mcl_label = row['maximum_stenosis']
        months_diff = row['months_diff_scan_to_surgery']  # months from M0 scan to surgery
        before_tp = row['before_surg_found']              # e.g. 'M0', 'M6'

        df_subj = all_data[all_data['subject'] == subject]
        if df_subj.empty:
            continue

        # --- Step 1: get the reference metric value (at before_surg_found) ---
        df_ref_tp = df_subj[df_subj['timepoint'] == before_tp]
        junction_slices_ref = find_mcl_junction_slices(
            df_ref_tp, upper_level, lower_level, n_slices
        )
        if junction_slices_ref is None:
            print(f'  Warning: {subject} — no MCL slices found at pre-surgery '
                  f'timepoint {before_tp}, skipping')
            continue

        df_ref_window = df_ref_tp[df_ref_tp['Slice (I->S)'].isin(junction_slices_ref)]
        ref_values = {
            m: df_ref_window[m].dropna().mean()
            for m in metrics if m in df_ref_window.columns
        }

        # --- Step 2: loop over all available timepoints ---
        for tp in df_subj['timepoint'].unique():
            df_tp = df_subj[df_subj['timepoint'] == tp]

            junction_slices = find_mcl_junction_slices(
                df_tp, upper_level, lower_level, n_slices
            )
            if junction_slices is None:
                continue

            df_window = df_tp[df_tp['Slice (I->S)'].isin(junction_slices)]
            if df_window.empty:
                continue

            # months_from_surgery: positive = after surgery, negative = before
            months_from_surgery = TIMEPOINT_MONTHS.get(tp, np.nan) - months_diff

            record = {
                'subject': subject,
                'maximum_stenosis': mcl_label,
                'timepoint': tp,
                'months_from_M0': TIMEPOINT_MONTHS.get(tp, np.nan),
                'months_diff_scan_to_surgery': months_diff,
                'months_from_surgery': months_from_surgery,
                'phase': 'pre_surgery' if tp == before_tp else (
                         'post_surgery' if TIMEPOINT_MONTHS.get(tp, 0) >
                                           TIMEPOINT_MONTHS.get(before_tp, 0) else 'earlier'),
                'pre_surgery_tp': before_tp,
                'n_slices_used': len(df_window),
            }

            for m in metrics:
                if m in df_window.columns:
                    val = df_window[m].dropna().mean()
                    record[m] = val
                    # delta = change from the pre-surgery reference scan
                    record[f'delta_{m}'] = val - ref_values.get(m, np.nan)
                else:
                    record[m] = np.nan
                    record[f'delta_{m}'] = np.nan

            records.append(record)

    df_op = pd.DataFrame(records)

    if df_op.empty:
        print('  No operative data extracted — check subject ID matching.')
        return df_op

    print(f'  Subjects extracted: {df_op["subject"].nunique()}')
    print(f'  Total subject-timepoint rows: {len(df_op)}')
    print('  Timepoint distribution:')
    print(df_op['timepoint'].value_counts().sort_index().to_string())

    return df_op


def _operative_group_stats(df_plot, y_col):
    """
    Compute group statistics per timepoint and map them to their mean
    months-from-surgery x-axis position.

    Because each subject's surgery happened at a slightly different time
    relative to their M0 scan, grouping by the continuous months_from_surgery
    value would produce one row per subject (std = NaN). Instead, we group by
    timepoint code and use the mean months_from_surgery for that timepoint
    as the x-axis coordinate. This preserves the "months from surgery" axis
    while producing meaningful group statistics.

    Returns a DataFrame indexed by mean months_from_surgery with columns
    mean, std, count, timepoint, and a xtick_label showing both.
    """
    tp_order = [tp for tp in TIMEPOINT_ORDER if tp in df_plot['timepoint'].unique()]

    rows = []
    for tp in tp_order:
        dtp = df_plot[df_plot['timepoint'] == tp].dropna(subset=[y_col])
        if dtp.empty:
            continue
        rows.append({
            'timepoint': tp,
            'x': dtp['months_from_surgery'].mean(),   # mean x for this timepoint
            'mean': dtp[y_col].mean(),
            'std': dtp[y_col].std(),
            'count': dtp[y_col].count(),
        })

    return pd.DataFrame(rows).set_index('x')


def _add_surgery_bands(ax, x_min, x_max):
    ax.axvline(0, color='crimson', linestyle='--', linewidth=1.8,
               alpha=0.8, label='Surgery', zorder=8)
    ax.axvspan(x_min - 1, 0, alpha=0.05, color='steelblue')
    ax.axvspan(0, x_max + 1, alpha=0.05, color='tomato')


def plot_operative_trajectories(df_op, metric, output_dir, space='native'):
    """
    Plot absolute metric values vs months-from-surgery for operative subjects.

    Individual lines use each subject's exact months_from_surgery.
    Group statistics (mean ± SD) are computed per timepoint and placed at
    the mean months_from_surgery for that timepoint (see _operative_group_stats).
    """
    df_plot = df_op.dropna(subset=[metric, 'months_from_surgery']).copy()
    if df_plot.empty:
        return

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(11, 6))

    subjects = df_plot['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    for i, subj in enumerate(subjects):
        ds = df_plot[df_plot['subject'] == subj].sort_values('months_from_surgery')
        ax.plot(ds['months_from_surgery'], ds[metric],
                color=colors[i], alpha=0.2, linewidth=1,
                marker='o', markersize=3, zorder=1)

    grp = _operative_group_stats(df_plot, metric)
    ax.plot(grp.index, grp['mean'], color='black', linewidth=3,
            marker='o', markersize=7, label='Group mean', zorder=10)
    ax.fill_between(grp.index, grp['mean'] - grp['std'], grp['mean'] + grp['std'],
                    color='black', alpha=0.15, label='±1 SD', zorder=5)

    # Annotate n below the SD band — get ylim after drawing
    fig.canvas.draw()
    for x, row in grp.iterrows():
        y_bot = row['mean'] - (row['std'] if pd.notna(row['std']) else 0)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(x, y_bot - y_range * 0.04,
                f'n={int(row["count"])}', ha='center', fontsize=8, color='dimgray')

    _add_surgery_bands(ax, grp.index.min(), grp.index.max())

    metric_name = METRIC_NAMES.get(metric, metric)
    unit = METRIC_UNITS.get(metric, '')
    ax.set_xlabel('Months from Surgery', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(f'{metric_name} [{unit}]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(
        f'{metric_name} at MCL — operative subjects (n={df_plot["subject"].nunique()}) [{space} space]',
        fontsize=TITLE_FONT_SIZE, fontweight='bold'
    )
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe = metric.replace('(', '').replace(')', '').replace(' ', '_')
    fpath = os.path.join(output_dir, f'operative_mcl_{safe}_trajectories.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


def plot_operative_delta(df_op, metric, output_dir, space='native'):
    """
    Plot delta (change from pre-surgery scan) vs months-from-surgery.
    The pre-surgery timepoint is always at delta=0 (by construction).
    Group statistics are computed per timepoint (see _operative_group_stats).
    """
    delta_col = f'delta_{metric}'
    df_plot = df_op.dropna(subset=[delta_col, 'months_from_surgery']).copy()
    if df_plot.empty:
        return

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(11, 6))

    subjects = df_plot['subject'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))

    for i, subj in enumerate(subjects):
        ds = df_plot[df_plot['subject'] == subj].sort_values('months_from_surgery')
        ax.plot(ds['months_from_surgery'], ds[delta_col],
                color=colors[i], alpha=0.2, linewidth=1,
                marker='o', markersize=3, zorder=1)

    grp = _operative_group_stats(df_plot, delta_col)
    ax.plot(grp.index, grp['mean'], color='steelblue', linewidth=3,
            marker='o', markersize=7, label='Group mean', zorder=10)
    ax.fill_between(grp.index, grp['mean'] - grp['std'], grp['mean'] + grp['std'],
                    color='steelblue', alpha=0.18, label='±1 SD', zorder=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label='No change from pre-surgery scan')
    _add_surgery_bands(ax, grp.index.min(), grp.index.max())

    fig.canvas.draw()
    for x, row in grp.iterrows():
        y_bot = row['mean'] - (row['std'] if pd.notna(row['std']) else 0)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(x, y_bot - y_range * 0.05,
                f'n={int(row["count"])}', ha='center', fontsize=8, color='dimgray')

    metric_name = METRIC_NAMES.get(metric, metric)
    unit = METRIC_UNITS.get(metric, '')
    ax.set_xlabel('Months from Surgery', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(f'Δ {metric_name} [{unit}]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(
        f'Change in {metric_name} relative to pre-surgery scan [{space} space]\n'
        f'Operative subjects (n={df_plot["subject"].nunique()})',
        fontsize=TITLE_FONT_SIZE, fontweight='bold'
    )
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=TICKS_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    safe = metric.replace('(', '').replace(')', '').replace(' ', '_')
    fpath = os.path.join(output_dir, f'operative_mcl_{safe}_delta_from_presurg.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


def plot_operative_panel(df_op, metrics, output_dir, space='native'):
    """
    Summary panel: one row per metric, two columns (absolute | delta).
    X-axis = months from surgery for both columns.
    Group statistics per column use _operative_group_stats (grouped by timepoint).
    """
    available = [m for m in metrics if m in df_op.columns]
    if not available:
        return

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, metric in enumerate(available):
        delta_col = f'delta_{metric}'
        metric_name = METRIC_NAMES.get(metric, metric)
        unit = METRIC_UNITS.get(metric, '')

        for col, (col_label, y_col) in enumerate([
            ('Absolute Values', metric),
            ('Change from Pre-surgery Scan', delta_col),
        ]):
            ax = axes[ri, col]
            df_plot = df_op.dropna(subset=[y_col, 'months_from_surgery']).copy()
            if df_plot.empty:
                continue

            subjects = df_plot['subject'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(subjects)))
            for i, subj in enumerate(subjects):
                ds = df_plot[df_plot['subject'] == subj].sort_values('months_from_surgery')
                ax.plot(ds['months_from_surgery'], ds[y_col],
                        color=colors[i], alpha=0.15, linewidth=0.9, zorder=1)

            grp = _operative_group_stats(df_plot, y_col)
            color = 'black' if col == 0 else 'steelblue'
            ax.plot(grp.index, grp['mean'], color=color, linewidth=2.5,
                    marker='o', markersize=6, zorder=10,
                    label=f'n={df_plot["subject"].nunique()}')
            ax.fill_between(grp.index, grp['mean'] - grp['std'], grp['mean'] + grp['std'],
                            color=color, alpha=0.12, zorder=5)

            if col == 1:
                ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)

            x_vals = grp.index
            _add_surgery_bands(ax, x_vals.min(), x_vals.max())

            prefix = 'Δ ' if col == 1 else ''
            ax.set_ylabel(f'{prefix}{metric_name}\n[{unit}]', fontsize=LABELS_FONT_SIZE)
            ax.tick_params(labelsize=TICKS_FONT_SIZE)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=TICKS_FONT_SIZE - 1, loc='best')

            if ri == 0:
                ax.set_title(col_label, fontsize=TITLE_FONT_SIZE, fontweight='bold')

    axes[-1, 0].set_xlabel('Months from Surgery', fontsize=LABELS_FONT_SIZE)
    axes[-1, 1].set_xlabel('Months from Surgery', fontsize=LABELS_FONT_SIZE)

    n_subj = df_op['subject'].nunique()
    n_slices_label = int(df_op['n_slices_used'].median())
    fig.suptitle(
        f'Operative subjects — MCL metrics ({n_slices_label}-slice window) '
        f'aligned to surgery date  [n={n_subj}] [{space} space]',
        fontsize=TITLE_FONT_SIZE + 1, fontweight='bold', y=1.002
    )
    plt.tight_layout()

    fpath = os.path.join(output_dir, 'operative_mcl_metrics_panel.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    print(f'  Saved: {fpath}')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.operative_only and not args.baseline_surgery_mapping:
        raise ValueError('--operative-only requires --baseline-surgery-mapping')

    print('=' * 70)
    print('Metrics at MCL per-slice analysis')
    print('=' * 70)
    print(f'Data directory  : {args.data_dir}')
    print(f'Participants    : {args.participants}')
    print(f'Output directory: {args.output_dir}')
    print(f'Space           : {args.space}')
    print(f'Structure       : {args.structure}')
    print(f'Timepoints      : {", ".join(args.timepoints)}')
    print(f'N slices/side   : {args.n_slices}')
    print(f'Metrics         : {", ".join(args.metrics)}')
    print(f'Operative only  : {args.operative_only}')
    print()

    print('Loading participants...')
    participants_df = load_participants(args.participants)

    print('\nLoading per-slice data...')
    all_data = load_perslice_data(
        args.data_dir, args.structure, args.timepoints, space=args.space
    )

    # Make sure requested metrics that can be derived are available
    metrics_to_extract = list(args.metrics)
    if 'MEAN(compression_ratio)' in metrics_to_extract:
        if 'MEAN(compression_ratio)' not in all_data.columns:
            print('  Warning: compression ratio could not be computed (missing AP or RL diameter)')
            metrics_to_extract.remove('MEAN(compression_ratio)')

    df_mcl = extract_metrics_at_mcl(all_data, participants_df, args.n_slices, metrics_to_extract)

    if df_mcl.empty:
        print('\nNo data extracted — check that subject IDs match between participants.tsv and metric files.')
        return

    # Save extracted data to CSV
    csv_path = os.path.join(args.output_dir, 'metrics_at_mcl.csv')
    df_mcl.to_csv(csv_path, index=False)
    print(f'\nSaved extracted MCL metrics table: {csv_path}')

    # Print summary statistics per timepoint
    available_metrics = [m for m in metrics_to_extract if m in df_mcl.columns]
    print('\nSummary statistics per timepoint:')
    for tp in sorted(df_mcl['timepoint'].unique(), key=lambda x: TIMEPOINT_MONTHS.get(x, 999)):
        df_tp = df_mcl[df_mcl['timepoint'] == tp]
        vals = {m: f'{df_tp[m].mean():.2f}±{df_tp[m].std():.2f}' for m in available_metrics if m in df_tp}
        print(f'  {tp} (n={df_tp["subject"].nunique()}): ' +
              ', '.join(f'{METRIC_NAMES.get(m, m)}={v}' for m, v in vals.items()))

    # ----------------------------------------------------------------
    # Standard figures (all subjects, months from M0)
    # ----------------------------------------------------------------
    if not args.operative_only:
        print('\nGenerating figures...')
        for metric in available_metrics:
            print(f'\n  {METRIC_NAMES.get(metric, metric)}:')
            plot_metric_trajectories(df_mcl, metric, args.output_dir, space=args.space)
            plot_delta_from_baseline(df_mcl, metric, args.output_dir, space=args.space)
            plot_by_mcl_group(df_mcl, metric, args.output_dir, args.n_slices, space=args.space)

        print('\n  Multi-metric panel figure:')
        plot_multi_metric_panel(df_mcl, available_metrics, args.output_dir, space=args.space)

    # ----------------------------------------------------------------
    # Operative-only figures (months from surgery)
    # ----------------------------------------------------------------
    if args.operative_only:
        print('\nLoading surgery mapping...')
        surgery_mapping = load_surgery_mapping(args.baseline_surgery_mapping)

        df_op = extract_metrics_operative(
            all_data, participants_df, surgery_mapping, args.n_slices, available_metrics
        )

        if df_op.empty:
            print('\nNo operative data extracted — nothing to plot.')
        else:
            # Save extracted table
            op_csv = os.path.join(args.output_dir, 'metrics_at_mcl_operative.csv')
            df_op.to_csv(op_csv, index=False)
            print(f'\nSaved operative MCL metrics table: {op_csv}')

            # Summary stats
            print('\nSummary statistics per timepoint (months from surgery):')
            for tp in sorted(df_op['timepoint'].unique(),
                             key=lambda x: TIMEPOINT_MONTHS.get(x, 999)):
                dtp = df_op[df_op['timepoint'] == tp]
                m_surg = dtp['months_from_surgery'].mean()
                vals = {m: f'{dtp[m].mean():.2f}±{dtp[m].std():.2f}'
                        for m in available_metrics if m in dtp}
                print(f'  {tp} (~{m_surg:+.1f} months from surgery, n={dtp["subject"].nunique()}): '
                      + ', '.join(f'{METRIC_NAMES.get(m, m)}={v}' for m, v in vals.items()))

            print('\nGenerating operative figures...')
            for metric in available_metrics:
                print(f'\n  {METRIC_NAMES.get(metric, metric)}:')
                plot_operative_trajectories(df_op, metric, args.output_dir, space=args.space)
                plot_operative_delta(df_op, metric, args.output_dir, space=args.space)

            print('\n  Operative panel figure:')
            plot_operative_panel(df_op, available_metrics, args.output_dir, space=args.space)

    print('\n' + '=' * 70)
    print(f'Done. Results saved to: {args.output_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
