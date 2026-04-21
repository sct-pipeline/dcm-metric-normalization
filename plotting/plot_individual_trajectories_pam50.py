#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot individual longitudinal PAM50 per-slice trajectories.

For each subject, create one figure with one subplot per metric:
- x-axis: PAM50 slices
- y-axis: metric value
- all available timepoints are overlaid
- each timepoint is labeled as operative or conservative (time-aware label)
- dashed vertical lines mark disc junctions (no slice removal)

Usage example:
    python plot_individual_trajectories_pam50.py \
        --data-dir /path/to/timepoint_data \
        --output-dir /path/to/output \
        --structure cord \
        --timepoints M0 M6 M12 M24 M36 M48 M60

Authors: Kahina Baouche
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


def extract_subject_id(filename: str) -> str | None:
    match = re.search(r'sub-\d+', str(filename))
    return match.group(0) if match else None


def infer_labels_dir(data_dir: str) -> str | None:
    candidate = (Path(data_dir).resolve().parent /
                 'longitudinal_statistics_all_timepoints' /
                 'timepoint_aware_label_verification')
    return str(candidate) if candidate.exists() else None


def load_timepoint_aware_labels(labels_dir: str, timepoints: list[str]) -> pd.DataFrame:
    all_labels = []
    for tp in timepoints:
        file_path = Path(labels_dir) / f'labels_{tp}.csv'
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path)
        if 'participant_id' not in df.columns or 'timepoint_aware_label' not in df.columns:
            continue
        df = df[['participant_id', 'timepoint_aware_label']].copy()
        df['timepoint'] = tp
        df = df.rename(columns={'participant_id': 'subject',
                                'timepoint_aware_label': 'therapeutic_group'})
        df['therapeutic_group'] = df['therapeutic_group'].str.lower()
        all_labels.append(df)

    if not all_labels:
        raise ValueError('No valid time-aware labels found.')

    return pd.concat(all_labels, ignore_index=True)


def load_stenosis_maps(participants_file: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Load subject-level stenosis info from participants.tsv.

    Returns
    -------
    max_map : dict[str, str]
        subject -> maximum_stenosis (single disc label)
    all_map : dict[str, list[str]]
        subject -> list of stenosis disc labels from column `stenosis`
    """
    df = pd.read_csv(participants_file, sep='\t')
    required = {'participant_id', 'maximum_stenosis', 'stenosis'}
    if not required.issubset(df.columns):
        raise ValueError('participants.tsv must contain participant_id, maximum_stenosis, and stenosis columns')

    tmp_max = df[['participant_id', 'maximum_stenosis']].dropna().copy()
    tmp_max['maximum_stenosis'] = tmp_max['maximum_stenosis'].astype(str).str.strip()
    max_map = dict(zip(tmp_max['participant_id'], tmp_max['maximum_stenosis']))

    all_map: dict[str, list[str]] = {}
    for _, row in df[['participant_id', 'stenosis']].dropna().iterrows():
        subject = str(row['participant_id']).strip()
        raw = str(row['stenosis'])
        levels = [x.strip() for x in raw.split(',') if x.strip()]
        all_map[subject] = levels

    return max_map, all_map


def load_perslice_pam50_data(data_dir: str, structure: str,
                             timepoints: list[str]) -> pd.DataFrame:
    all_data = []
    for tp in timepoints:
        file_path = Path(data_dir) / f'T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv'
        if not file_path.exists():
            print(f'Warning: file not found: {file_path}')
            continue

        df = pd.read_csv(file_path)
        if 'subject' not in df.columns:
            if 'Filename' in df.columns:
                df['subject'] = df['Filename'].apply(extract_subject_id)
            elif 'Filename_sc' in df.columns:
                df['subject'] = df['Filename_sc'].apply(extract_subject_id)
        df['timepoint'] = tp
        all_data.append(df)

    if not all_data:
        raise ValueError('No PAM50 per-slice files found.')

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined[(combined['VertLevel'] >= 2) & (combined['VertLevel'] <= 7)].copy()
    return combined


def get_disc_junctions(df: pd.DataFrame) -> list[dict]:
    """Get disc-junction x positions from VertLevel transitions across slices."""
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
            junctions.append({
                'x': (slices[i - 1] + slices[i]) / 2.0,
                'label': normalized_label,
            })
    return junctions


def normalize_disc_label(label: str) -> str | None:
    """Normalize a disc label to ascending form, e.g. C6/C5 -> C5/C6."""
    if not isinstance(label, str):
        return None
    m = re.match(r'\s*C(\d+)\s*/\s*C(\d+)\s*$', label)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = min(a, b), max(a, b)
    return f'C{lo}/C{hi}'


def normalize_disc_list(labels: list[str] | None) -> set[str]:
    """Normalize a list of disc labels and return a deduplicated set."""
    out = set()
    if not labels:
        return out
    for lbl in labels:
        n = normalize_disc_label(lbl)
        if n:
            out.add(n)
    return out


def sort_timepoints(timepoints: list[str]) -> list[str]:
    return sorted(timepoints, key=lambda x: TIMEPOINT_MONTHS.get(x, 999))


def get_std_column(metric: str) -> str | None:
    """Map MEAN(metric) column to STD(metric) column when available."""
    if metric.startswith('MEAN(') and metric.endswith(')'):
        inside = metric[len('MEAN('):-1]
        return f'STD({inside})'
    return None


def plot_subject(df_subject: pd.DataFrame, metrics: list[str],
                 junctions: list[dict], out_file: str,
                 max_stenosis_label: str | None = None,
                 other_stenosis_labels: list[str] | None = None) -> None:
    subject = df_subject['subject'].iloc[0]
    available_tps = sort_timepoints(df_subject['timepoint'].dropna().unique().tolist())

    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    handles = {}
    target_stenosis = normalize_disc_label(max_stenosis_label) if max_stenosis_label else None
    all_stenosis = normalize_disc_list(other_stenosis_labels)
    other_stenosis = {s for s in all_stenosis if s != target_stenosis}

    for i, metric in enumerate(metrics):
        ax = axes[i]

        for tp in available_tps:
            df_tp = df_subject[df_subject['timepoint'] == tp].sort_values('Slice (I->S)')
            if metric not in df_tp.columns:
                continue

            x = df_tp['Slice (I->S)'].values
            y = df_tp[metric].values
            valid = ~pd.isna(y)
            if valid.sum() == 0:
                continue

            group_values = df_tp['therapeutic_group'].dropna().unique().tolist()
            group = group_values[0] if len(group_values) > 0 else 'unknown'

            color = TIMEPOINT_COLORS.get(tp, '#000000')
            linestyle = '-' if group == 'operative' else '--' if group == 'conservative' else ':'
            line, = ax.plot(x[valid], y[valid], color=color, linestyle=linestyle,
                            linewidth=1.8, alpha=0.95,
                            label=f'{tp} ({group})')
            handles[f'{tp} ({group})'] = line

            # If available, add ±STD shaded band for this subject/timepoint.
            std_col = get_std_column(metric)
            if std_col and std_col in df_tp.columns:
                y_std = df_tp[std_col].values
                valid_std = valid & ~pd.isna(y_std)
                if valid_std.sum() > 0:
                    ax.fill_between(
                        x[valid_std],
                        y[valid_std] - y_std[valid_std],
                        y[valid_std] + y_std[valid_std],
                        color=color,
                        alpha=0.12,
                        linewidth=0
                    )

        for j in junctions:
            ax.axvline(j['x'], color='gray', linestyle='--', linewidth=1.0, alpha=0.4)

        # Highlight the maximum stenosis junction for this subject when available
        for j in junctions:
            j_norm = normalize_disc_label(j['label'])
            if not j_norm:
                continue
            if j_norm in other_stenosis:
                ax.axvspan(j['x'] - 0.8, j['x'] + 0.8, color='#FFB74D', alpha=0.10, zorder=0)
                ax.axvline(j['x'], color='#FFB74D', linestyle='--', linewidth=1.6, alpha=0.8)
            if target_stenosis and j_norm == target_stenosis:
                ax.axvspan(j['x'] - 1.0, j['x'] + 1.0, color='#D32F2F', alpha=0.10, zorder=0)
                ax.axvline(j['x'], color='#D32F2F', linestyle='--', linewidth=2.0, alpha=0.9)

        metric_name = METRIC_NAMES.get(metric, metric)
        unit = METRIC_UNITS.get(metric, '')
        ax.set_ylabel(f'{metric_name}\n[{unit}]', fontsize=10)
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(axis='both', labelsize=9)

        if i == 0:
            transform = blended_transform_factory(ax.transData, ax.transAxes)
            for j in junctions:
                j_norm = normalize_disc_label(j['label'])
                if target_stenosis and j_norm == target_stenosis:
                    text_color = '#D32F2F'
                elif j_norm in other_stenosis:
                    text_color = '#F57C00'
                else:
                    text_color = 'dimgray'
                ax.text(j['x'], 0.97, j['label'], transform=transform,
                        fontsize=8, color=text_color, ha='center', va='top', fontweight='bold')

    axes[-1].set_xlabel('PAM50 Slice (Superior → Inferior)', fontsize=11)
    for ax in axes:
        ax.invert_xaxis()

    stenosis_suffix = f' | maximum stenosis: {max_stenosis_label}' if max_stenosis_label else ''
    title = f'{subject} — Individual Longitudinal PAM50 Per-slice Trajectories{stenosis_suffix}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    if handles:
        ordered_labels = sort_timepoints([k.split(' (')[0] for k in handles.keys()])
        ordered = []
        for tp in ordered_labels:
            for lbl, h in handles.items():
                if lbl.startswith(f'{tp} '):
                    ordered.append((lbl, h))
        if ordered:
            labels, hs = zip(*ordered)
            fig.legend(hs, labels, title='Timepoint (group)', ncol=min(4, len(labels)),
                       loc='upper center', bbox_to_anchor=(0.5, 0.955), fontsize=9, title_fontsize=9)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing T2w_ax_{structure}_metrics_perslice_PAM50_{tp}_data.csv')
    parser.add_argument('--output-dir', required=True,
                        help='Directory where per-subject figures will be saved')
    parser.add_argument('--structure', default='cord', choices=['cord', 'canal', 'aSCOR'])
    parser.add_argument('--timepoints', nargs='+', default=['M0', 'M6', 'M12'])
    parser.add_argument('--labels-dir',
                        help='Directory containing labels_M0.csv, labels_M6.csv, ... (time-aware labels)')
    parser.add_argument('--participants-file',
                        default='/Users/kahina/NeuroPoly/data/dcm-zurich/participants.tsv',
                        help='participants.tsv path (must contain participant_id and maximum_stenosis)')
    parser.add_argument('--metrics', nargs='+',
                        help='Metrics to plot; default = standard metrics present in data')
    parser.add_argument('--subjects', nargs='+',
                        help='Optional list of subject IDs (e.g., sub-001 sub-005). Default: all subjects')
    return parser


def main() -> None:
    args = get_parser().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labels_dir = args.labels_dir or infer_labels_dir(args.data_dir)
    if not labels_dir:
        raise ValueError('No labels directory provided or inferred. Please pass --labels-dir.')

    print(f'Loading PAM50 per-slice data from: {args.data_dir}')
    df = load_perslice_pam50_data(args.data_dir, args.structure, args.timepoints)

    print(f'Loading time-aware labels from: {labels_dir}')
    labels_df = load_timepoint_aware_labels(labels_dir, args.timepoints)
    max_stenosis_map, all_stenosis_map = load_stenosis_maps(args.participants_file)

    df = df.merge(labels_df, on=['subject', 'timepoint'], how='left')
    df['therapeutic_group'] = df['therapeutic_group'].fillna('unknown')

    default_metrics = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)',
                       'MEAN(eccentricity)', 'MEAN(solidity)']
    if args.structure == 'aSCOR':
        default_metrics = ['aSCOR']

    metrics = args.metrics if args.metrics else [m for m in default_metrics if m in df.columns]
    if not metrics:
        raise ValueError('No requested metrics found in loaded data.')

    junctions = get_disc_junctions(df)

    subjects = sorted(df['subject'].dropna().unique().tolist())
    if args.subjects:
        subjects = [s for s in args.subjects if s in subjects]

    print(f'Subjects to plot: {len(subjects)} | Metrics: {", ".join(metrics)}')

    n_ok = 0
    for subject in subjects:
        df_subj = df[df['subject'] == subject].copy()
        if df_subj.empty:
            continue

        out_file = os.path.join(args.output_dir, f'{subject}_individual_pam50_trajectories.png')
        plot_subject(
            df_subj,
            metrics,
            junctions,
            out_file,
            max_stenosis_label=max_stenosis_map.get(subject),
            other_stenosis_labels=all_stenosis_map.get(subject)
        )
        n_ok += 1

    print(f'Done. Saved {n_ok} subject figures in: {args.output_dir}')


if __name__ == '__main__':
    main()
