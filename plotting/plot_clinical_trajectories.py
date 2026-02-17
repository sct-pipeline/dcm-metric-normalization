"""
Create trajectory plots for longitudinal clinical scores.
Generates one figure per score (e.g., mJOA, Nurick), connecting each
participant across sessions and overlaying mean ± SD per session.

Example usage single stratification:
python plotting/plot_clinical_trajectories.py \
    -clinical-file data/clinical_scores.xlsx \
    -o figures/clinical_trajectories \
    --stratify-by therapeutic_decision

    --stratify-by mjoa

Example usage dual stratification:
python plotting/plot_clinical_trajectories.py \
    -clinical-file data/clinical_scores.xlsx \
    -o figures/clinical_trajectories \
    --stratify-by therapeutic_decision,normative_mean_c2

    --stratify-by mjoa,normative_mean_c2
"""

import os
import sys
import argparse

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from generate_figure_PAM50_multiple_subjects import (read_clinical_file, MYELOPATHY_COLORS, AGE_GROUP_COLORS, SEX_COLORS_PATIENTS,
                                                     THERAPEUTIC_DECISION_COLORS, MCL_COLORS, MJOA_COLORS, NORMATIVE_C2_COLORS)

# Plot fonts
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 16

# Default scores to plot: name -> list of column names in expected order
SCORES = {
    'mJOA': {
        'columns': ['total_mjoa_BL', 'total_mjoa_6mth'],
        'y_label': 'mJOA'
    },
    'Motor Dysfunction UE': {
        'columns': ['motor_dysfunction_UE_bl_BL', 'motor_dysfunction_UE_6mth_6mth'],
        'y_label': 'Motor Dysfunction UE'
    },
    'Motor Dysfunction LE': {
        'columns': ['motor_dysfunction_LE_bl_BL', 'motor_dysfunction_LE_6mth_6mth'],
        'y_label': 'Motor Dysfunction LE'
    },
    'Sensory Dysfunction UE': {
        'columns': ['sensory_dysfunction_UE_bl_BL', 'sensory_dysfunction_UE_6mth_6mth'],
        'y_label': 'Sensory Dysfunction UE'
    },
    'Sphincter Dysfunction': {
        'columns': ['sphincter_dysfunction_bl_BL', 'sphincter_dysfunction_6mth_6mth'],
        'y_label': 'Motor Dysfunction LE'
    },
    'Pinprick UE': {
        'columns': ['UEPP_C4_T1_bl', 'UEPP_C4_T1_6mth'],
        'y_label': 'Pinprick UE'
    },
    'Lightouch UE': {
        'columns': ['UELT_C4_T1_bl_BL', 'UELT_C4_T1_6mth_6mth'],
        'y_label': 'Lightouch UE'
    },
    'Total Motor Score UE': {
        'columns': ['upper_extrem_motor_total_BL', 'upper_extrem_motor_total_6mth'],
        'y_label': 'Total Motor Score UE'
    },
}

# Session labels to display (same length and order as each score's columns)
# SESSION_LABELS_DEFAULT = ['BL', '6 mth', '12 mth']
SESSION_LABELS_DEFAULT = ['Baseline', '6-month']

STRATIFICATION_TO_TITLE = {
    'myelopathy': 'myelopathy',
    'normative_mean_c2': 'normative mean C2 cord area',
    'therapeutic_decision': 'therapeutic decision',
    'mjoa': 'mJOA severity',
    'mJOA_severity_bl': 'mJOA severity'
}

# Allow per-score y-axis limits; extend/edit as needed
SCORE_TO_YLIM = {
    'mJOA': (11, 18.5),
    'Nurick': (0, 5),
    'Pinprick total': (60, 115),
    'Pinprick cervical': (18, 30),
    'Pinprick below cervical': (50, 90),
    'Lightouch total': (70, 115),
    'Lightouch cervical': (18, 30),
    'Lightouch below cervical': (50, 90),
}


def get_parser():
    p = argparse.ArgumentParser(description='Plot longitudinal clinical score trajectories (one figure per score).')
    p.add_argument('-clinical-file', required=True, type=str,
                   help='Path to Excel file with clinical scores (columns like total_mjoa_bl, nurick_bl, ...)')
    p.add_argument('-participants-to-use', required=True, type=str,
                   help='Path to text file with participant IDs to include (one ID per line)')
    p.add_argument('-o', '--outdir', required=True, type=str,
                   help='Output directory for figures')
    # Stratification option: can provide one or two columns separated by comma
    p.add_argument('--stratify-by', type=str, default=None,
                   help='Column(s) to stratify by. Provide one or two, comma-separated (e.g., "therapeutic_decision,mjoa"). ')
    return p

def _get_ylim_for_score(score_name: str):
    return SCORE_TO_YLIM.get(score_name)


def _palette_for_strata(strata_values, stratify_by):
    """Return a dict mapping each stratum value to a color using shared palettes.
    Falls back to matplotlib tab10 for any values not covered.
    """
    # Choose base palette based on stratification key
    if stratify_by in {'myelopathy'}:
        base = MYELOPATHY_COLORS
    elif stratify_by in {'age', 'age_group'}:
        base = AGE_GROUP_COLORS
    elif stratify_by in {'sex'}:
        base = SEX_COLORS_PATIENTS
    elif stratify_by in {'therapeutic_decision'}:
        base = THERAPEUTIC_DECISION_COLORS
    elif stratify_by in {'MCL', 'maximum_stenosis', 'highest_stenosis'}:
        base = MCL_COLORS
    elif stratify_by in {'mJOA_severity_bl', 'mjoa_severity', 'mjoa'}:
        base = MJOA_COLORS
    elif stratify_by in {'normative_mean_c2'}:
        base = NORMATIVE_C2_COLORS
    else:
        base = {}

    colors = {}
    # First assign known colors
    for v in strata_values:
        if v in base:
            colors[v] = base[v]
    # Assign fallback colors for unknown values
    if any(v not in colors for v in strata_values):
        cmap = plt.get_cmap('tab10')
        next_idx = 0
        for v in strata_values:
            if v not in colors:
                colors[v] = cmap(next_idx % 10)
                next_idx += 1
    return colors


def _markers_for_strata(strata_values, stratify_by):
    """Return a dict mapping each stratum value to a marker style.
    Use sensible defaults for common strata; fallback to a cycle of markers.
    """
    # Prefer explicit mapping for known keys
    if stratify_by == 'normative_mean_c2':
        base = {
            'Above normative mean C2 cord area': 'o',
            'Below normative mean C2 cord area': 's',
        }
    elif stratify_by == 'therapeutic_decision':
        base = {
            'operative': 'o',
            'conservative': 's',
        }
    elif stratify_by == 'myelopathy':
        base = {
            'yes': 'o',
            'no': 's',
        }
    else:
        base = {}

    cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    markers = {}
    idx = 0
    for v in strata_values:
        if v in base:
            markers[v] = base[v]
        else:
            markers[v] = cycle[idx % len(cycle)]
            idx += 1
    return markers


def _linestyles_for_strata(strata_values, stratify_by):
    """Return a dict mapping each stratum value to a line style (e.g., '-', '--')."""
    if stratify_by == 'normative_mean_c2':
        base = {
            'Above normative mean C2 cord area': '-',
            'Below normative mean C2 cord area': '--',
        }
    elif stratify_by == 'therapeutic_decision':
        base = {
            'operative': '-',
            'conservative': '--',
        }
    elif stratify_by == 'myelopathy':
        base = {
            'yes': '-',
            'no': '--',
        }
    else:
        base = {}
    cycle = ['-', '--', ':', '-.']
    styles = {}
    idx = 0
    for v in strata_values:
        if v in base:
            styles[v] = base[v]
        else:
            styles[v] = cycle[idx % len(cycle)]
            idx += 1
    return styles


def _parse_stratify_arg(val: str | None):
    """Parse --stratify-by argument into a list of 0, 1, or 2 keys.
    Accepts comma-separated string (e.g., 'therapeutic_decision,normative_mean_c2').
    Caps the number of keys at 2.
    """
    if not val:
        return []
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(',') if p.strip()]
    else:
        # Shouldn't happen with argparse here, but keep safe
        parts = list(val)
    # Keep max two
    return parts[:2]


def _normalize_strat_keys(keys: list[str]) -> list[str]:
    """Map user-friendly strat keys to actual dataframe columns.
    Currently maps 'mjoa' to 'mJOA_severity_bl'.
    """
    out = []
    for k in keys:
        lk = k.strip()
        if lk in {'mjoa'}:
            out.append('mJOA_severity_bl')
        else:
            out.append(lk)
    return out[:2]


def build_long_df_for_score(df: pd.DataFrame, score_name: str, columns: list[str], session_labels: list[str], stratify_by: list[str] | str | None = None) -> pd.DataFrame:
    # Require that all expected session columns exist; otherwise skip this score
    if not all(c in df.columns for c in columns):
        return pd.DataFrame(columns=['participant_id', 'session_numeric', 'session_label', 'score'])

    # Coerce to numeric and keep only subjects with non-NaN values across ALL sessions (complete cases)
    df_num = df.copy()
    for c in columns:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

    # Normalize stratify_by to a list of 0-2 keys
    keys = _parse_stratify_arg(stratify_by) if isinstance(stratify_by, str) or stratify_by is None else list(stratify_by)
    keys = keys[:2]

    drop_subset = columns.copy()
    for k in keys:
        if k in df_num.columns:
            drop_subset.append(k)
    df_complete = df_num.dropna(subset=drop_subset)

    if df_complete.empty:
        return pd.DataFrame(columns=['participant_id', 'session_numeric', 'session_label', 'score'])

    # Build long-format from complete cases only
    records = []
    for s_num, (col, lab) in enumerate(zip(columns, session_labels), start=1):
        base_cols = ['participant_id', col]
        extra_cols = [k for k in keys if k in df_complete.columns]
        use_cols = base_cols + extra_cols
        for _, row in df_complete[use_cols].iterrows():
            rec = {
                'participant_id': row['participant_id'],
                'session_numeric': s_num,
                'session_label': lab,
                'score': float(row[col]),
            }
            if len(extra_cols) >= 1:
                rec['stratum1'] = row[extra_cols[0]]
            if len(extra_cols) >= 2:
                rec['stratum2'] = row[extra_cols[1]]
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _compute_session_stats(plot_df: pd.DataFrame, value_col: str = 'score'):
    sessions = sorted(plot_df['session_numeric'].unique())
    stats = []
    for s in sessions:
        vals = plot_df.loc[plot_df['session_numeric'] == s, value_col]
        if len(vals) > 0:
            n = int(vals.count())
            mean_v = float(vals.mean())
            std_v = float(vals.std(ddof=1)) if n > 1 else 0.0
            stats.append({'session_numeric': s, 'mean': mean_v, 'std': std_v, 'n': n})
    return stats


def plot_score_trajectory(plot_df: pd.DataFrame, score_name: str, y_label: str, outdir: str, figure_suffix: str = ""):
    if plot_df.empty:
        print(f"No data available for {score_name}; skipping.")
        return

    mpl.rcParams['font.family'] = 'Arial'

    sessions = sorted(plot_df['session_numeric'].unique())
    width = max(6, int(2.5 * len(sessions)))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4))

    # # Apply custom y-limits if provided for this score
    # ylim = _get_ylim_for_score(score_name)
    # if ylim is not None:
    #     ax.set_ylim(*ylim)

    # # Plot individual trajectories (only when subject has >1 time points)
    # for pid, g in plot_df.groupby('participant_id'):
    #     g_sorted = g.sort_values('session_numeric')
    #     if len(g_sorted) > 1:
    #         ax.plot(g_sorted['session_numeric'], g_sorted['score'],
    #                 color='black', alpha=0.3, linewidth=0.5, marker='o', markersize=0, linestyle='dashed', zorder=3)

    # Mean ± SD per session
    stats = _compute_session_stats(plot_df)

    if len(stats) >= 1:
        xs = [d['session_numeric'] for d in stats]
        means = [d['mean'] for d in stats]
        stds = [d['std'] for d in stats]
        ax.plot(xs, means, color='blue', linewidth=2, marker='o', markersize=3, label='Mean ± SD', zorder=6)
        ax.errorbar(xs, means, yerr=stds, color='blue', capsize=4, capthick=2, linestyle='None', zorder=5)
        # Annotate mean ± SD values near each mean point
        for i, (x, m, s) in enumerate(zip(xs, means, stds)):
            label = f"{m:.1f} ± {s:.1f}"
            xoff = -6 if x == xs[-1] else 6
            ax.annotate(label, xy=(x, m),
                        xytext=(xoff, 8),
                        textcoords='offset points',
                        ha='right' if x == xs[-1] else 'left',
                        va='bottom', fontsize=TICK_FONT_SIZE-2, color='blue',
                        backgroundcolor='white')

    # X ticks with (n=)
    tick_labels = []
    for s in sessions:
        lab = plot_df.loc[plot_df['session_numeric'] == s, 'session_label'].iloc[0]
        n = plot_df.loc[plot_df['session_numeric'] == s, 'participant_id'].nunique()
        tick_labels.append(f"{lab}\n(n={n})")

    ax.set_xticks(sessions)
    ax.set_xticklabels(tick_labels, fontsize=TICK_FONT_SIZE)

    ax.set_xlabel('Session', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'{score_name} across sessions', fontsize=TITLE_FONT_SIZE)

    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=TICK_FONT_SIZE-2)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    suffix = ("_" + figure_suffix) if figure_suffix else ""
    out_path = os.path.join(outdir, f'clinical_score_trajectory_{score_name.replace(" ", "_")}{suffix}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Clinical trajectory figure saved to {out_path}')


def plot_score_trajectory_stratified(plot_df: pd.DataFrame, score_name: str, y_label: str, outdir: str,
                                      stratify_by: str):
    if plot_df.empty or 'stratum' not in plot_df.columns:
        print(f"No stratified data available for {score_name}; skipping.")
        return

    mpl.rcParams['font.family'] = 'Arial'

    sessions = sorted(plot_df['session_numeric'].unique())
    width = max(6, int(2.8 * len(sessions)))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4))

    # Apply custom y-limits if provided for this score
    ylim = _get_ylim_for_score(score_name)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Determine strata order and labels from data
    strata_vals = list(plot_df['stratum'].dropna().unique())
    try:
        strata = sorted(strata_vals)
    except Exception:
        strata = strata_vals

    # Use shared palette when available; fallback to tab10 for unknown values
    colors = _palette_for_strata(strata, stratify_by)

    # Helper to shorten normative_mean_c2 labels for ticks/annotations
    def _to_short_label(v):
        # Shorten mJOA severity labels
        if stratify_by in {'mJOA_severity_bl', 'mjoa', 'mjoa_severity'}:
            s = str(v).lower()
            if 'mild' in s:
                return 'mild'
            if 'moderate' in s:
                return 'moderate'
            if 'severe' in s:
                return 'severe'
            return str(v)
        if stratify_by != 'normative_mean_c2':
            return str(v)
        s = str(v).lower()
        if 'above' in s:
            return 'above'
        if 'below' in s:
            return 'below'
        return str(v)

    # Prepare container for per-session annotation text lines
    session_annotation_lines = {}

    # Plot per stratum
    for idx, val in enumerate(strata):
        gdf = plot_df[plot_df['stratum'] == val]
        label_long = str(val)
        label_short = _to_short_label(val)

        # Individual trajectories per stratum
        for pid, g in gdf.groupby('participant_id'):
            g_sorted = g.sort_values('session_numeric')
            if len(g_sorted) > 1:
                ax.plot(g_sorted['session_numeric'], g_sorted['score'],
                        color=colors[val], alpha=0.3, linewidth=0.5, linestyle='solid', zorder=3)

        # Mean ± SD per session for this stratum
        stats = _compute_session_stats(gdf)
        if len(stats) >= 1:
            xs = [d['session_numeric'] for d in stats]
            means = [d['mean'] for d in stats]
            stds = [d['std'] for d in stats]
            # Keep long text in legend
            ax.plot(xs, means, color=colors[val], linewidth=2, linestyle='solid', marker='o', markersize=3, label=label_long, zorder=6)
            ax.errorbar(xs, means, yerr=stds, color=colors[val], capsize=4, capthick=2, linestyle='None', zorder=5)
            # Collect values for aggregated per-session annotation (use short label)
            for d in stats:
                s_num = d['session_numeric']
                session_annotation_lines.setdefault(s_num, []).append((label_short, d['mean'], d['std']))

    # After plotting all strata, add one multi-line annotation per session to avoid overlap
    if len(session_annotation_lines) > 0:
        ymin, ymax = ax.get_ylim()
        ypad = 0.03 * (ymax - ymin)
        last_session = max(sessions)
        for s in sessions:
            if s not in session_annotation_lines:
                continue
            lines = session_annotation_lines[s]
            # Keep strata order consistent with legend order; use short labels when comparing
            order_keys = [_to_short_label(v) for v in strata]
            ordered = [l for key in order_keys for l in lines if l[0] == key]
            text = "\n".join([f"{lab}: {m:.1f} ± {sd:.1f}" for (lab, m, sd) in ordered])
            # Place just above the highest mean at this session
            y_local_max = max(m for (_, m, _) in lines)
            xoff = -8 if s == last_session else 8
            ax.annotate(text, xy=(s, y_local_max + ypad), xytext=(xoff, 0), textcoords='offset points',
                        ha='right' if s == last_session else 'left', va='bottom', fontsize=TICK_FONT_SIZE-3,
                        color='black', backgroundcolor='white', zorder=10, clip_on=False)

    # X ticks with per-stratum n
    tick_labels = []
    for s in sessions:
        lab = plot_df.loc[plot_df['session_numeric'] == s, 'session_label'].iloc[0]
        parts = []
        for val in strata:
            n = plot_df[(plot_df['session_numeric'] == s) & (plot_df['stratum'] == val)]['participant_id'].nunique()
            parts.append(f"{_to_short_label(val)}={n}")
        parts_join = "\n".join(parts)
        tick_labels.append(f"{lab}\n{parts_join}")

    ax.set_xticks(sessions)
    ax.set_xticklabels(tick_labels, fontsize=TICK_FONT_SIZE)

    ax.set_xlabel('Session', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'{score_name} across sessions by {STRATIFICATION_TO_TITLE[stratify_by]}', fontsize=TITLE_FONT_SIZE)

    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title=stratify_by, fontsize=TICK_FONT_SIZE-2, title_fontsize=TICK_FONT_SIZE-1, loc='lower left')

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f'clinical_score_trajectory_{score_name.replace(" ", "_")}_by_{stratify_by}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Clinical trajectory figure saved to {out_path}')


def plot_score_trajectory_stratified_multi(plot_df: pd.DataFrame, score_name: str, y_label: str, outdir: str,
                                            stratify_by: list[str]):
    """Plot trajectories stratified by two factors.
    The first factor controls color; the second controls line style. Two legends are added accordingly.
    """
    if plot_df.empty or 'stratum1' not in plot_df.columns or 'stratum2' not in plot_df.columns:
        print(f"No multi-stratified data available for {score_name}; skipping.")
        return

    key1, key2 = stratify_by[0], stratify_by[1]

    mpl.rcParams['font.family'] = 'Arial'

    sessions = sorted(plot_df['session_numeric'].unique())
    width = max(7, int(3.2 * len(sessions)))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4.6))

    # # Apply custom y-limits if provided for this score
    # ylim = _get_ylim_for_score(score_name)
    # if ylim is not None:
    #     ax.set_ylim(*ylim)

    # Determine unique strata
    vals1 = list(plot_df['stratum1'].dropna().unique())
    vals2 = list(plot_df['stratum2'].dropna().unique())
    try:
        strata1 = sorted(vals1)
    except Exception:
        strata1 = vals1
    try:
        strata2 = sorted(vals2)
    except Exception:
        strata2 = vals2

    colors1 = _palette_for_strata(strata1, key1)
    linestyles2 = _linestyles_for_strata(strata2, key2)

    def _short_label(k, v):
        # Shorten mJOA severity labels for ticks/annotations
        if k in {'mJOA_severity_bl', 'mjoa', 'mjoa_severity'}:
            s = str(v).lower()
            if 'mild' in s:
                return 'mild'
            if 'moderate' in s:
                return 'moderate'
            if 'severe' in s:
                return 'severe'
            return str(v)
        if k != 'normative_mean_c2':
            return str(v)
        s = str(v).lower()
        if 'above' in s:
            return 'above'
        if 'below' in s:
            return 'below'
        return str(v)

    # Plot per combination
    for v1 in strata1:
        df1 = plot_df[plot_df['stratum1'] == v1]
        for v2 in strata2:
            gdf = df1[df1['stratum2'] == v2]
            if gdf.empty:
                continue

            # # Individual trajectories
            # for pid, g in gdf.groupby('participant_id'):
            #     g_sorted = g.sort_values('session_numeric')
            #     if len(g_sorted) > 1:
            #         ax.plot(g_sorted['session_numeric'] + offset, g_sorted['score'],
            #                 color=colors1[v1], alpha=0.3, linewidth=0.3, linestyle=linestyles2[v2], zorder=3)

            # Mean ± SD per session for this combo
            stats = _compute_session_stats(gdf)
            if len(stats) >= 1:
                xs = [d['session_numeric'] for d in stats]
                means = [d['mean'] for d in stats]
                stds = [d['std'] for d in stats]
                ax.plot(xs, means, color=colors1[v1], linewidth=2, marker='o', markersize=3,
                        linestyle=linestyles2[v2], label=f"{v1} • {_short_label(key2, v2)}", zorder=6, alpha=0.95)
                ax.errorbar(xs, means, yerr=stds, color=colors1[v1], capsize=3, capthick=1.5, linestyle='None', zorder=5)

    # X ticks with per-combination n
    tick_labels = []
    for s in sessions:
        lab = plot_df.loc[plot_df['session_numeric'] == s, 'session_label'].iloc[0]
        parts = []
        for v1 in strata1:
            for v2 in strata2:
                n = plot_df[(plot_df['session_numeric'] == s) & (plot_df['stratum1'] == v1) & (plot_df['stratum2'] == v2)]['participant_id'].nunique()
                if n > 0:
                    parts.append(f"{_short_label(key1, v1)}/{_short_label(key2, v2)}={n}")
        parts_join = "\n".join(parts)
        tick_labels.append(f"{lab}\n{parts_join}" if parts_join else lab)

    ax.set_xticks(sessions)
    ax.set_xticklabels(tick_labels, fontsize=TICK_FONT_SIZE)

    ax.set_xlabel('Session', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
    ax.set_title(f"{score_name} across sessions by {STRATIFICATION_TO_TITLE.get(key1, key1)} and {STRATIFICATION_TO_TITLE.get(key2, key2)}",
                 fontsize=TITLE_FONT_SIZE)

    ax.tick_params(axis='y', labelsize=TICK_FONT_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Build separate legends: one for colors (key1) and one for line styles (key2)
    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=colors1[v], lw=1, marker='o', markersize=3, label=str(v)) for v in strata1]
    style_handles = [Line2D([0], [0], color='black', lw=1, linestyle=linestyles2[v], label=_short_label(key2, v)) for v in strata2]

    # Place legends inside the axes to avoid cropping
    leg1 = ax.legend(handles=color_handles,
                     title=STRATIFICATION_TO_TITLE.get(key1, key1),
                     loc='lower left',
                     fontsize=TICK_FONT_SIZE-4,
                     title_fontsize=TICK_FONT_SIZE-3,
                     frameon=True)
    leg1.get_frame().set_alpha(0.85)
    leg1.get_frame().set_facecolor('white')
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=style_handles,
                     title=STRATIFICATION_TO_TITLE.get(key2, key2),
                     loc='lower right',
                     fontsize=TICK_FONT_SIZE-4,
                     title_fontsize=TICK_FONT_SIZE-3,
                     frameon=True)
    leg2.get_frame().set_alpha(0.85)
    leg2.get_frame().set_facecolor('white')

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"clinical_score_trajectory_{score_name.replace(' ', '_')}_by_{key1}_and_{key2}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Clinical trajectory figure saved to {out_path}')


def _normalize_normative_c2(value):
    """Normalize normative_mean_c2 labels from participants.tsv to expected palette keys.
    Returns pandas.NA for empty/missing to allow dropna when stratifying.
    """
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if s == '' or s.lower() in {'na', 'n/a'}:
        return pd.NA
    # If already an exact key, keep it
    if s in NORMATIVE_C2_COLORS:
        return s
    lower = s.lower()
    if 'below' in lower:
        return 'Below normative mean C2 cord area'
    if 'above' in lower:
        return 'Above normative mean C2 cord area'
    return s

def merge_stratification(df_clinical: pd.DataFrame, participants_file: str | None, stratify_by: list[str] | str | None) -> pd.DataFrame:
    """Merge one or two stratification columns from participants.tsv into clinical dataframe when needed."""
    keys = _parse_stratify_arg(stratify_by) if isinstance(stratify_by, str) or stratify_by is None else list(stratify_by)
    if len(keys) == 0:
        return df_clinical

    df_out = df_clinical.copy()

    if not participants_file or not os.path.isfile(participants_file):
        if any(k not in df_out.columns for k in keys):
            print(f"Warning: participants file not provided or not found; cannot fetch {keys}.")
        return df_out

    try:
        df_part = pd.read_csv(participants_file, sep='\t')
    except Exception as e:
        print(f"Warning: failed to read participants TSV: {e}")
        return df_out

    if 'participant_id' not in df_part.columns:
        print("Warning: 'participant_id' missing in participants.tsv; cannot merge stratification.")
        return df_out

    # Prepare subset with available keys
    available = [k for k in keys if k in df_part.columns and k not in df_out.columns]
    if len(available) == 0:
        # Nothing to merge (either not found, or already present)
        missing = [k for k in keys if (k not in df_part.columns and k not in df_out.columns)]
        if missing:
            print(f"Warning: {missing} not found in participants.tsv; cannot merge these.")
        return df_out

    df_sub = df_part[['participant_id'] + available].copy()

    # Special handling
    for k in available:
        if k == 'normative_mean_c2':
            df_sub[k] = df_sub[k].apply(_normalize_normative_c2)

    merged = df_out.merge(df_sub, on='participant_id', how='left')
    return merged


def main():
    args = get_parser().parse_args()

    df_clinical, _ = read_clinical_file(os.path.abspath(args.clinical_file))
    # Read txt file with participant IDs to include
    with open(args.participants_to_use, 'r') as f:
        participant_ids = [line.strip() for line in f if line.strip()]
    # Keep only requested participants
    df_clinical = df_clinical[df_clinical['participant_id'].isin(participant_ids)].copy()

    # Rename myelopathy values from 0 to 'myelopathy no' and 1 to 'myelopathy yes'
    df_clinical['myelopathy'] = df_clinical['myelopathy'].map({0: 'myelopathy_no', 1: 'myelopathy_yes'})

    # Parse stratification keys and normalize aliases (e.g., 'mjoa' -> 'mJOA_severity_bl')
    strat_keys_in = _parse_stratify_arg(args.stratify_by)
    strat_keys = _normalize_strat_keys(strat_keys_in)

    # For each requested score, build long-format DF and plot
    for score in SCORES.keys():
        cfg = SCORES[score]

        plot_df = build_long_df_for_score(df_clinical, score, cfg['columns'], SESSION_LABELS_DEFAULT, stratify_by=strat_keys)
        if plot_df.empty:
            print(f"No data available for {score} with complete sessions: {cfg['columns']}")
            continue

        # If no stratification requested, plot a single non-stratified figure
        if len(strat_keys) == 0 or (len(strat_keys) >= 1 and 'stratum1' not in plot_df.columns):
            plot_score_trajectory(plot_df, score, cfg['y_label'], args.outdir)
        elif len(strat_keys) == 1:
            # Create a simplified view with a single 'stratum' column for the existing function
            df_single = plot_df.copy()
            df_single['stratum'] = df_single['stratum1']
            # Use user-friendly label if alias was given
            title_key = strat_keys_in[0] if strat_keys_in else strat_keys[0]
            plot_score_trajectory_stratified(df_single, score, cfg['y_label'], args.outdir, title_key)
        else:
            # Dual stratification
            plot_score_trajectory_stratified_multi(plot_df, score, cfg['y_label'], args.outdir, strat_keys)


if __name__ == '__main__':
    main()
