"""
Create trajectory plots for longitudinal clinical scores.
Generates one figure per score (e.g., mJOA, Nurick), connecting each
participant across sessions and overlaying mean ± SD per session.
"""

import os
import sys
import argparse

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from generate_figure_PAM50_multiple_subjects import (MYELOPATHY_COLORS, AGE_GROUP_COLORS, SEX_COLORS_PATIENTS,
                                                     THERAPEUTIC_DECISION_COLORS, MCL_COLORS, MJOA_COLORS, NORMATIVE_C2_COLORS)

# Plot fonts
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 16

# Default scores to plot: name -> list of column names in expected order
SCORES = {
    'mJOA': {
        'columns': ['total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth'],
        'y_label': 'mJOA'
    },
    'Nurick': {
        'columns': ['nurick_bl', 'nurick_6mth', 'nurick_12mth'],
        'y_label': 'Nurick grade'
    },
}

# Session labels to display (same length and order as each score's columns)
SESSION_LABELS_DEFAULT = ['BL', '6 mth', '12 mth']

STRATIFICATION_TO_TITLE = {
    'myelopathy': 'myelopathy',
    'normative_mean_c2': 'normative mean C2 cord area',
    'therapeutic_decision': 'therapeutic decision'
}



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


def get_parser():
    p = argparse.ArgumentParser(description='Plot longitudinal clinical score trajectories (one figure per score).')
    p.add_argument('-clinical-file', required=True, type=str,
                   help='Path to Excel file with clinical scores (columns like total_mjoa_bl, nurick_bl, ...)')
    p.add_argument('-o', '--outdir', required=True, type=str,
                   help='Output directory for figures')
    p.add_argument('--subject-col', default='record_id', type=str, required=False,
                   help='Subject ID column in the clinical Excel (default: record_id)')
    p.add_argument('--scores', dest='scores', nargs='*', default=list(SCORES.keys()),
                   help='Subset of scores to plot (default: all known)')
    # Stratification option: if provided, one stratified overlay figure will be saved with filename including the column name
    p.add_argument('--stratify-by', type=str, default=None,
                   help='Column to stratify by (e.g., myelopathy, normative_mean_c2). If not in clinical Excel, provide --participants-file.')
    p.add_argument('--participants-file', type=str, default=None,
                   help='Path to participants.tsv to fetch stratification columns like myelopathy or normative_mean_c2 (tab-separated).')
    return p


def load_clinical_excel(path_xlsx: str, subject_col: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path_xlsx)
    except Exception as e:
        sys.exit(f'Error reading clinical Excel: {e}')

    if subject_col not in df.columns:
        sys.exit(f"Subject column '{subject_col}' not found. Available: {list(df.columns)}")

    # Keep all columns; we will check existence later per score
    df = df.copy()

    # Rename subject column to participant_id and format as sub-XXX when numeric
    df = df.rename(columns={subject_col: 'participant_id'})
    df['participant_id'] = df['participant_id'].apply(
        lambda x: f"sub-{int(x):03d}" if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
    )

    return df


def build_long_df_for_score(df: pd.DataFrame, score_name: str, columns: list[str], session_labels: list[str], stratify_by: str | None = None) -> pd.DataFrame:
    # Require that all expected session columns exist; otherwise skip this score
    if not all(c in df.columns for c in columns):
        return pd.DataFrame(columns=['participant_id', 'session_numeric', 'session_label', 'score'])

    # Coerce to numeric and keep only subjects with non-NaN values across ALL sessions (complete cases)
    df_num = df.copy()
    for c in columns:
        df_num[c] = pd.to_numeric(df_num[c], errors='coerce')

    drop_subset = columns.copy()
    if stratify_by is not None and stratify_by in df_num.columns:
        drop_subset = drop_subset + [stratify_by]
    df_complete = df_num.dropna(subset=drop_subset)

    if df_complete.empty:
        return pd.DataFrame(columns=['participant_id', 'session_numeric', 'session_label', 'score'])

    # Build long-format from complete cases only
    records = []
    for s_num, (col, lab) in enumerate(zip(columns, session_labels), start=1):
        for _, row in df_complete[['participant_id', col] + ([stratify_by] if stratify_by and stratify_by in df_complete.columns else [])].iterrows():
            rec = {
                'participant_id': row['participant_id'],
                'session_numeric': s_num,
                'session_label': lab,
                'score': float(row[col]),
            }
            if stratify_by and stratify_by in df_complete.columns:
                rec['stratum'] = row[stratify_by]
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

    # Plot individual trajectories (only when subject has >1 time points)
    for pid, g in plot_df.groupby('participant_id'):
        g_sorted = g.sort_values('session_numeric')
        if len(g_sorted) > 1:
            ax.plot(g_sorted['session_numeric'], g_sorted['score'],
                    color='black', alpha=0.5, linewidth=1, marker='o', markersize=0, linestyle='dashed', zorder=3)

    # Mean ± SD per session
    stats = _compute_session_stats(plot_df)

    if len(stats) >= 1:
        xs = [d['session_numeric'] for d in stats]
        means = [d['mean'] for d in stats]
        stds = [d['std'] for d in stats]
        ax.plot(xs, means, color='blue', linewidth=3, marker='o', markersize=6, label='Mean ± SD', zorder=6)
        ax.errorbar(xs, means, yerr=stds, color='blue', capsize=4, capthick=2, linestyle='None', zorder=5)
        # Annotate mean ± SD values near each mean point
        for i, (x, m, s) in enumerate(zip(xs, means, stds)):
            label = f"{m:.2f} ± {s:.2f}"
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
                        color=colors[val], alpha=0.35, linewidth=1, linestyle='dashed', zorder=3)

        # Mean ± SD per session for this stratum
        stats = _compute_session_stats(gdf)
        if len(stats) >= 1:
            xs = [d['session_numeric'] for d in stats]
            means = [d['mean'] for d in stats]
            stds = [d['std'] for d in stats]
            # Keep long text in legend
            ax.plot(xs, means, color=colors[val], linewidth=3, marker='o', markersize=6, label=label_long, zorder=6)
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
            text = "\n".join([f"{lab}: {m:.2f} ± {sd:.2f}" for (lab, m, sd) in ordered])
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
    ax.legend(title=stratify_by, fontsize=TICK_FONT_SIZE-2, title_fontsize=TICK_FONT_SIZE-1)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f'clinical_score_trajectory_{score_name.replace(" ", "_")}_by_{stratify_by}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Clinical trajectory figure saved to {out_path}')


def _process_myelopathy(value):
    """Map raw myelopathy to 'yes'/'no' (NA -> 'no')."""
    if pd.isna(value) or str(value).strip().lower() in {'n/a', 'na', ''}:
        return 'no'
    return 'yes'


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


def merge_stratification(df_clinical: pd.DataFrame, participants_file: str | None, stratify_by: str | None) -> pd.DataFrame:
    """Merge stratification column from participants.tsv into clinical dataframe when needed."""
    if not stratify_by:
        return df_clinical

    # If the stratify column already exists in clinical DF, nothing to merge
    if stratify_by in df_clinical.columns:
        return df_clinical

    if not participants_file or not os.path.isfile(participants_file):
        print(f"Warning: participants file not provided or not found; cannot fetch '{stratify_by}'.")
        return df_clinical

    try:
        df_part = pd.read_csv(participants_file, sep='\t')
    except Exception as e:
        print(f"Warning: failed to read participants TSV: {e}")
        return df_clinical

    if 'participant_id' not in df_part.columns:
        print("Warning: 'participant_id' missing in participants.tsv; cannot merge stratification.")
        return df_clinical

    if stratify_by not in df_part.columns:
        print(f"Warning: '{stratify_by}' not found in participants.tsv; cannot merge stratification.")
        return df_clinical

    # Prepare subset with the stratification column
    df_sub = df_part[['participant_id', stratify_by]].copy()

    # Special handling
    if stratify_by == 'myelopathy':
        df_sub[stratify_by] = df_sub[stratify_by].apply(_process_myelopathy)
    elif stratify_by == 'normative_mean_c2':
        df_sub[stratify_by] = df_sub[stratify_by].apply(_normalize_normative_c2)

    merged = df_clinical.merge(df_sub, on='participant_id', how='left')
    return merged


def main():
    args = get_parser().parse_args()

    df = load_clinical_excel(args.clinical_file, args.subject_col)
    # Merge stratification info from participants.tsv if needed
    df = merge_stratification(df, args.participants_file, args.stratify_by)

    # For each requested score, build long-format DF and plot
    for score in args.scores:
        if score not in SCORES:
            print(f"Unknown score '{score}', skipping. Known: {list(SCORES.keys())}")
            continue
        cfg = SCORES[score]

        plot_df = build_long_df_for_score(df, score, cfg['columns'], SESSION_LABELS_DEFAULT, stratify_by=args.stratify_by)
        if plot_df.empty:
            print(f"No data available for {score} with complete sessions: {cfg['columns']}")
            continue

        # If no stratification requested, plot a single non-stratified figure
        if not args.stratify_by or 'stratum' not in plot_df.columns:
            plot_score_trajectory(plot_df, score, cfg['y_label'], args.outdir)
        else:
            plot_score_trajectory_stratified(plot_df, score, cfg['y_label'], args.outdir, args.stratify_by)


if __name__ == '__main__':
    main()
