#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group-wise PAM50 per-slice pre/post surgery plots grouped by maximum stenosis.

Logic for pre/post pairing per subject is derived from baseline_surgery_dates_merged.csv:
- For each subject, uses before_surg and after_surg columns
- For pre-surgery timepoint: finds the closest available timepoint <= before_surg (in decreasing order)
- For post-surgery timepoint: finds the closest available timepoint >= after_surg (in increasing order)
- Updates baseline_surgery_dates_merged.csv with before_surg_found and after_surg_found columns
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
from matplotlib.transforms import blended_transform_factory

METRICS_DTYPE = {'MEAN(area)': float, 'MEAN(diameter_AP)': float, 
                 'MEAN(diameter_RL)': float, 'MEAN(eccentricity)': float, 
                 'MEAN(solidity)': float}


METRIC_NAMES = {
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(diameter_RL)': 'RL Diameter',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
    'aSCOR': 'aSCOR',
}

METRIC_UNITS = {
    'MEAN(area)': 'mm²',
    'MEAN(diameter_AP)': 'mm',
    'MEAN(diameter_RL)': 'mm',
    'MEAN(eccentricity)': 'a.u.',
    'MEAN(solidity)': '%',
    'aSCOR': 'ratio',
}

VERT_LABELS = {2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}

TIMEPOINT_ORDER = ['M0', 'M3', 'M6', 'M12', 'M24', 'M36', 'M48', 'M60']


def extract_subject_id(filename: str) -> str | None:
    m = re.search(r'sub-\d+', str(filename))
    return m.group(0) if m else None


def load_perslice_pam50_data(data_dir: str, structure: str, timepoints: list[str]) -> pd.DataFrame:
    frames = []
    for tp in timepoints:
        p = Path(data_dir) / f'T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'subject' not in df.columns:
            if 'Filename' in df.columns:
                df['subject'] = df['Filename'].apply(extract_subject_id)
            elif 'Filename_sc' in df.columns:
                df['subject'] = df['Filename_sc'].apply(extract_subject_id)
        df['timepoint'] = tp
        frames.append(df)

    if not frames:
        raise ValueError('No PAM50 per-slice files found')

    out = pd.concat(frames, ignore_index=True)
    out = out[(out['VertLevel'] >= 2) & (out['VertLevel'] <= 7)].copy()
    return out


def get_disc_junctions(df: pd.DataFrame) -> list[dict]:
    slice_levels = df.groupby('Slice (I->S)')['VertLevel'].agg(lambda x: x.mode()[0]).sort_index()
    slices = slice_levels.index.to_list()
    levels = slice_levels.values.tolist()

    junctions = []
    for i in range(1, len(slices)):
        if levels[i] != levels[i - 1]:
            lbl_before = VERT_LABELS.get(levels[i - 1], str(levels[i - 1]))
            lbl_after = VERT_LABELS.get(levels[i], str(levels[i]))
            # Create label and normalize to ascending order (e.g., C3/C2 -> C2/C3)
            raw_label = f'{lbl_before}/{lbl_after}'
            normalized_label = normalize_disc_label(raw_label) or raw_label
            junctions.append({'x': (slices[i - 1] + slices[i]) / 2.0, 'label': normalized_label})
    return junctions


def load_participants_max_stenosis(participants_file: str, filter_number_stenosis: float = None) -> pd.DataFrame:
    df = pd.read_csv(participants_file, sep='\t')
    if 'participant_id' not in df.columns or 'maximum_stenosis' not in df.columns:
        raise ValueError('participants.tsv must contain participant_id and maximum_stenosis')
    
    result = df[['participant_id', 'maximum_stenosis']].rename(columns={'participant_id': 'subject'})
    
    # If filtering by number_stenosis, apply the filter
    if filter_number_stenosis is not None:
        if 'number_stenosis' not in df.columns:
            raise ValueError('participants.tsv must contain number_stenosis column for filtering')
        
        # Create filtered df with number_stenosis
        result = df[['participant_id', 'maximum_stenosis', 'number_stenosis']].rename(
            columns={'participant_id': 'subject'}
        )
        
        # Filter to only subjects with the specified number_stenosis value
        n_before = len(result)
        result = result[result['number_stenosis'] == filter_number_stenosis].copy()
        n_after = len(result)
        n_removed = n_before - n_after
        
        if n_removed > 0:
            print(f'  Filtered to number_stenosis={filter_number_stenosis}: removed {n_removed} subject(s)')
        
        # Drop the number_stenosis column as it's no longer needed
        result = result.drop('number_stenosis', axis=1)
    
    return result


def load_baseline_surgery_mapping(mapping_file: str) -> pd.DataFrame:
    """Load baseline_surgery_dates_merged.csv and return subjects with valid before_surg/after_surg.
    
    Excludes subjects where surgery_date is before date_of_scan (invalid chronology).
    """
    df = pd.read_csv(mapping_file)
    if 'subject' not in df.columns or 'before_surg' not in df.columns or 'after_surg' not in df.columns:
        raise ValueError('baseline_surgery_dates_merged.csv must contain subject, before_surg, after_surg')
    
    # Ensure subject is string format (e.g., 'sub-001')
    df['subject'] = df['subject'].astype(str)
    if not df['subject'].str.startswith('sub-').all():
        df['subject'] = 'sub-' + df['subject'].str.extract('(\d+)')[0].str.zfill(3)
    
    # Filter to subjects with both before_surg and after_surg defined
    valid = df[(df['before_surg'].notna()) & (df['after_surg'].notna())].copy()
    
    # Remove subjects where surgery_date is before date_of_scan (invalid chronology)
    if 'surgery_date' in valid.columns and 'date_of_scan' in valid.columns:
        valid['surgery_date'] = pd.to_datetime(valid['surgery_date'], errors='coerce')
        valid['date_of_scan'] = pd.to_datetime(valid['date_of_scan'], errors='coerce')
        
        # Keep only rows where surgery_date >= date_of_scan
        n_before = len(valid)
        valid = valid[valid['surgery_date'] >= valid['date_of_scan']].copy()
        n_after = len(valid)
        n_removed = n_before - n_after
        
        if n_removed > 0:
            print(f'  Removed {n_removed} subject(s) with surgery_date before date_of_scan')
    
    return valid





def build_prepost_pairs_from_mapping(mapping_df: pd.DataFrame, all_data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Build pre/post pairs from baseline_surgery_dates_merged.csv
    Uses exact timepoints specified in before_surg/after_surg columns.
    
    Returns
    -------
    df_pairs : pd.DataFrame
        Combined data for pre/post pairs with phase column and before_tp/after_tp columns
    failed_subjects : list[dict]
        List of subjects that couldn't be paired
    """
    pair_rows = []
    failed_subjects = []
    
    for _, row in mapping_df.iterrows():
        subject = row['subject']
        before_tp = row['before_surg']
        after_tp = row['after_surg']
        maximum_stenosis = row.get('maximum_stenosis', None)
        
        # Skip if no timepoints specified
        if pd.isna(before_tp) or pd.isna(after_tp):
            failed_subjects.append({'subject': subject, 'issue': 'no_timepoints_specified'})
            continue
        
        # Check if data exists for both timepoints (simple check, no searching for alternatives)
        has_before = ((all_data['subject'] == subject) & (all_data['timepoint'] == before_tp)).any()
        has_after = ((all_data['subject'] == subject) & (all_data['timepoint'] == after_tp)).any()
        
        if not has_before or not has_after:
            issues = []
            if not has_before:
                issues.append(f'missing_before_{before_tp}')
            if not has_after:
                issues.append(f'missing_after_{after_tp}')
            failed_subjects.append({'subject': subject, 'issue': ';'.join(issues)})
            continue
        
        # Retrieve data for before phase
        d_before = all_data[(all_data['subject'] == subject) & (all_data['timepoint'] == before_tp)].copy()
        d_before['phase'] = 'before'
        d_before['before_tp'] = before_tp
        d_before['after_tp'] = after_tp
        if maximum_stenosis:
            d_before['maximum_stenosis'] = maximum_stenosis
        
        # Retrieve data for after phase
        d_after = all_data[(all_data['subject'] == subject) & (all_data['timepoint'] == after_tp)].copy()
        d_after['phase'] = 'after'
        d_after['before_tp'] = before_tp
        d_after['after_tp'] = after_tp
        if maximum_stenosis:
            d_after['maximum_stenosis'] = maximum_stenosis
        
        pair_rows.append(d_before)
        pair_rows.append(d_after)
    
    df_pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    
    return df_pairs, failed_subjects


def aggregate_group_prepost(df_pairs: pd.DataFrame, metrics: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for metric in metrics:
        tmp = df_pairs.dropna(subset=[metric]).copy()
        agg = (
            tmp.groupby(['maximum_stenosis', 'phase', 'Slice (I->S)'])[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
            .rename(columns={'mean': 'value_mean', 'std': 'value_std', 'count': 'n'})
        )
        out[metric] = agg
    return out


def sanitize_name(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', str(text)).strip('_')


def normalize_disc_label(label: str) -> str | None:
    """Normalize disc label to ascending form, e.g. C6/C5 -> C5/C6."""
    if not isinstance(label, str):
        return None
    m = re.match(r'\s*C(\d+)\s*/\s*C(\d+)\s*$', label)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = min(a, b), max(a, b)
    return f'C{lo}/C{hi}'


def load_hc_normative_data(path_HC: str, metric: str) -> pd.DataFrame | None:
    """
    Load HC normative data for a specific metric from PAM50 CSV files.
    
    Parameters
    ----------
    path_HC : str
        Path to directory containing HC PAM50 CSV files (searches recursively)
    metric : str
        Metric name (e.g., 'MEAN(area)')
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with HC data or None if not found
    """
    df_list = []
    
    # Search recursively for PAM50 CSV files
    import glob
    pattern = os.path.join(path_HC, '**', '*PAM50.csv')
    csv_files = glob.glob(pattern, recursive=True)
    
    for filepath in csv_files:
        try:
            df_subject = pd.read_csv(filepath, dtype=METRICS_DTYPE)
            if metric in df_subject.columns:
                df_list.append(df_subject)
        except Exception:
            continue
    
    if not df_list:
        return None
    
    df = pd.concat(df_list, ignore_index=True)
    if 'VertLevel' in df.columns:
        df = df[(df['VertLevel'] >= 2) & (df['VertLevel'] <= 7)].copy()
    
    return df


def compute_hc_reference_line(df_HC: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute HC reference mean line and std band per slice.
    
    Parameters
    ----------
    df_HC : pd.DataFrame
        HC data with MEAN and STD columns
    metric : str
        Metric name (e.g., 'MEAN(area)')
    
    Returns
    -------
    tuple of (slices, means, stds_from_column, count) or None
        - slices: slice indices
        - means: mean values across HC subjects per slice
        - stds_from_column: standard deviations per slice (from STD column if available)
        - count: number of HC subjects per slice
    """
    if df_HC is None or df_HC.empty or metric not in df_HC.columns:
        return None
    
    # Extract metric name without MEAN( prefix and ) suffix
    metric_base = metric.replace('MEAN(', '').replace(')', '')
    std_col = f'STD({metric_base})'
    
    # Group by slice and compute mean and std
    ref_data = df_HC[['Slice (I->S)', metric]].dropna()
    if ref_data.empty:
        return None
    
    # Compute group statistics per slice
    grouped = ref_data.groupby('Slice (I->S)')[metric].agg(['mean', 'std', 'count']).reset_index()
    
    # If STD column exists in data, average those values too (within-subject SD)
    if std_col in df_HC.columns:
        std_col_data = df_HC[['Slice (I->S)', std_col]].dropna()
        if not std_col_data.empty:
            std_group = std_col_data.groupby('Slice (I->S)')[std_col].mean().reset_index()
            std_group.rename(columns={std_col: 'within_subject_std'}, inplace=True)
            grouped = grouped.merge(std_group, on='Slice (I->S)', how='left')
            # Use between-subject std when available, within-subject std as fallback
            grouped['final_std'] = grouped['std'].fillna(grouped['within_subject_std'])
        else:
            grouped['final_std'] = grouped['std']
    else:
        grouped['final_std'] = grouped['std']
    
    grouped = grouped.sort_values('Slice (I->S)')
    return grouped['Slice (I->S)'].values, grouped['mean'].values, grouped['final_std'].values, grouped['count'].values


def plot_group(group_name: str, agg_by_metric: dict[str, pd.DataFrame], metrics: list[str],
               junctions: list[dict], output_file: str, n_subjects: int, filter_number_stenosis: float = None,
               hc_ref_data: dict[str, tuple] | None = None, df_pairs: pd.DataFrame | None = None) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    phase_style = {
        'before': {'color': '#1f77b4', 'label': 'Before surgery'},
        'after': {'color': '#d62728', 'label': 'After surgery'},
    }

    target_group = normalize_disc_label(group_name)
    
    # Count exact number of unique subjects used to compute the trajectories
    if df_pairs is not None:
        group_data = df_pairs[df_pairs['maximum_stenosis'] == group_name]
        n_exact_subjects = group_data['subject'].nunique()
    else:
        n_exact_subjects = n_subjects

    for i, metric in enumerate(metrics):
        ax = axes[i]
        agg = agg_by_metric[metric]
        g = agg[agg['maximum_stenosis'] == group_name]

        for phase in ['before', 'after']:
            gp = g[g['phase'] == phase].sort_values('Slice (I->S)')
            if gp.empty:
                continue
            x = gp['Slice (I->S)'].values
            y = gp['value_mean'].values
            s = np.nan_to_num(gp['value_std'].values, nan=0.0)
            
            # Count exact number of unique subjects for this phase
            if df_pairs is not None:
                phase_data = df_pairs[(df_pairs['maximum_stenosis'] == group_name) & 
                                      (df_pairs['phase'] == phase)]
                n_subjects_phase = phase_data['subject'].nunique()
            else:
                n_subjects_phase = int(np.nanmedian(gp['n'].values)) if len(gp['n']) else 0

            ax.plot(x, y,
                    color=phase_style[phase]['color'],
                    linewidth=2.2,
                    label=f"{phase_style[phase]['label']} (n={n_subjects_phase})")
            ax.fill_between(x, y - s, y + s,
                            color=phase_style[phase]['color'], alpha=0.15)
        
        # Plot HC reference line (black) with SD band if available
        if hc_ref_data and metric in hc_ref_data:
            hc_ref_result = hc_ref_data[metric]
            if len(hc_ref_result) == 4:
                hc_slices, hc_means, hc_stds, hc_counts = hc_ref_result
            else:
                hc_slices, hc_means = hc_ref_result
                hc_stds = None
            
            # Plot SD band (light gray shading) - no label
            if hc_stds is not None:
                ax.fill_between(hc_slices, hc_means - hc_stds, hc_means + hc_stds,
                               color='black', alpha=0.08, zorder=5)
            
            # Plot reference line (black) with n_subjects label
            ax.plot(hc_slices, hc_means,
                    color='black', linewidth=2.0, linestyle='-', 
                    label='spine-generic (n=203)', zorder=10, alpha=0.8)

        for j in junctions:
            ax.axvline(j['x'], color='gray', linestyle='--', linewidth=1.0, alpha=0.35)
            if target_group and normalize_disc_label(j['label']) == target_group:
                ax.axvspan(j['x'] - 1.0, j['x'] + 1.0, color='#D32F2F', alpha=0.10, zorder=0)
                ax.axvline(j['x'], color='#D32F2F', linestyle='--', linewidth=2.0, alpha=0.9)

        if i == 0:
            t = blended_transform_factory(ax.transData, ax.transAxes)
            for j in junctions:
                is_target = target_group and normalize_disc_label(j['label']) == target_group
                ax.text(j['x'], 0.97, j['label'], transform=t,
                    ha='center', va='top', fontsize=8,
                    color=('#D32F2F' if is_target else 'dimgray'),
                    fontweight='bold')

        ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)}\n[{METRIC_UNITS.get(metric, '')}]", fontsize=10)
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc='best')

    axes[-1].set_xlabel('PAM50 Slice (Superior → Inferior)', fontsize=11)
    for ax in axes:
        ax.invert_xaxis()

    # Build title with optional filter information
    title = f'Group-wise Pre/Post Trajectories by Maximum Stenosis: {group_name} (n={n_exact_subjects})'
    if filter_number_stenosis is not None:
        title += f' [number_stenosis={filter_number_stenosis}]'
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--participants-file', required=True)
    p.add_argument('--baseline-surgery-mapping', required=True, help='Path to baseline_surgery_dates_merged.csv')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--structure', default='cord', choices=['cord', 'canal', 'aSCOR'])
    p.add_argument('--timepoints', nargs='+', default=['M0', 'M6', 'M12', 'M24', 'M36', 'M48', 'M60'])
    p.add_argument('--metrics', nargs='+', help='Default: standard set present in data')
    p.add_argument('--min-subjects-per-group', type=int, default=2,
                   help='Skip groups with fewer than this number of valid subjects')
    p.add_argument('--filter-number-stenosis', type=float, default=None,
                   help='Filter to only subjects with this number_stenosis value (e.g., 1.0)')
    p.add_argument('--hc-reference-path', type=str, default=None,
                   help='Path to spine-generic PAM50 normalized metrics directory for HC reference data')
    return p


def main() -> None:
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading PAM50 data...')
    df = load_perslice_pam50_data(args.data_dir, args.structure, args.timepoints)

    default_metrics = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)', 'MEAN(solidity)']
    if args.structure == 'aSCOR':
        default_metrics = ['aSCOR']
    metrics = args.metrics if args.metrics else [m for m in default_metrics if m in df.columns]
    if not metrics:
        raise ValueError('No requested metrics found in data')

    # Load HC reference data if path provided
    hc_ref_data = None
    if args.hc_reference_path:
        print(f'Loading HC reference data from {args.hc_reference_path}...')
        hc_ref_data = {}
        for metric in metrics:
            df_HC = load_hc_normative_data(args.hc_reference_path, metric)
            if df_HC is not None:
                result = compute_hc_reference_line(df_HC, metric)
                if result is not None:
                    hc_ref_data[metric] = result
                    print(f'  Loaded HC reference for {metric}')
            else:
                print(f'  No HC data found for {metric}')
        if not hc_ref_data:
            print('  Warning: No HC reference data loaded')
            hc_ref_data = None

    print('Loading baseline/surgery mapping from baseline_surgery_dates_merged.csv...')
    mapping_df = load_baseline_surgery_mapping(args.baseline_surgery_mapping)

    print('Loading maximum stenosis from participants.tsv...')
    part_df = load_participants_max_stenosis(args.participants_file, args.filter_number_stenosis)

    # Merge mapping with maximum stenosis
    subject_df = mapping_df.merge(part_df, on='subject', how='left')
    
    # Remove subjects without maximum stenosis
    subject_df = subject_df.dropna(subset=['maximum_stenosis']).copy()

    # Build pre/post pairs
    print('Building pre/post pairs with closest available timepoints...')
    df_pairs, failed_subjects = build_prepost_pairs_from_mapping(subject_df, df)

    if df_pairs.empty:
        raise ValueError('No valid subject pairs could be created')

    # Save audit tables
    audit_dir = Path(args.output_dir)
    
    included_subjects = (
        df_pairs[['subject', 'maximum_stenosis', 'before_tp', 'after_tp']]
        .drop_duplicates()
        .sort_values('subject')
    )
    included_subjects.to_csv(audit_dir / 'included_subjects_prepost_mapping_v2.csv', index=False)

    failed_df = pd.DataFrame(failed_subjects).drop_duplicates().sort_values(['subject'])
    failed_df.to_csv(audit_dir / 'failed_subjects_prepost_mapping_v2.csv', index=False)

    # Group-wise plots
    junctions = get_disc_junctions(df)
    agg_by_metric = aggregate_group_prepost(df_pairs, metrics)

    group_counts = included_subjects.groupby('maximum_stenosis')['subject'].nunique().sort_values(ascending=False)
    group_counts_df = group_counts.reset_index().rename(columns={'subject': 'n_subjects'})
    group_counts_df.to_csv(audit_dir / 'group_counts_by_maximum_stenosis_v2.csv', index=False)

    print('\nGroups (valid subjects):')
    print(group_counts_df.to_string(index=False))

    plotted = 0
    for group, n_sub in group_counts.items():
        if n_sub < args.min_subjects_per_group:
            continue
        out_file = audit_dir / f'group_prepost_v2_{sanitize_name(group)}.png'
        plot_group(group, agg_by_metric, metrics, junctions, str(out_file), int(n_sub), 
                   args.filter_number_stenosis, hc_ref_data, df_pairs)
        plotted += 1

    print(f'\nSaved {plotted} group plot(s) in: {args.output_dir}')
    print(f'Included subjects: {included_subjects.shape[0]}')
    print(f'Failed subjects: {failed_df.shape[0]}')


if __name__ == '__main__':
    main()
