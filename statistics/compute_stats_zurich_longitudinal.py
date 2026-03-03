#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Longitudinal statistics for the dcm-zurich dataset.

Two complementary analyses are performed:

1. **Per-timepoint analysis** (mirrors the baseline pipeline)
   For each supplied timepoint (e.g. bl, 6m, 12m), the script:
   - Loads morphometric MRI metrics from the corresponding CSV
   - Merges with clinical, anatomical, electrophysiological, and motion data
   - Selects the correct clinical/electro columns for that timepoint
   - Computes descriptive statistics (mean ± SD) and Spearman correlation matrices
   - Runs a stepwise logistic regression to predict therapeutic decision
   Results are saved to <path-out>/timepoint_<tp>/

2. **Temporal evolution analysis**  (across timepoints)
   For each metric, a per-subject linear slope (units / month) is estimated using
   ordinary least squares across all available timepoints.
   A one-sample t-test (H₀: mean slope = 0) quantifies group-level progression.
   Results are saved to <path-out>/temporal_slopes.csv and
   <path-out>/temporal_slopes_significance.csv.

Usage example
-------------
python compute_stats_zurich_longitudinal.py \\
    -input-files bl=metrics_bl.csv,6m=metrics_6m.csv,12m=metrics_12m.csv \\
    -participants-file participants.tsv \\
    -clinical-file clinical_scores.xlsx \\
    -anatomical-file anatomical_data.xlsx \\
    -electro-file electrophysiological_measurements.xlsx \\
    -motion-file motion_data.xlsx \\
    -path-out results_longitudinal

Authors: Sandrine Bédard, Jan Valosek  (baseline)
         Extended for longitudinal analysis — 2026 - Kahina Baouche
"""

import os
import sys
import logging
import argparse
from textwrap import dedent

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from utils.utils import SmartFormatter, format_pvalue, fit_reg
from utils.read_files import (
    read_metric_file, read_participants_file, read_clinical_file,
    read_electrophysiology_file, read_anatomical_file,
    read_motion_file, read_motion_file_maximum_stenosis,
    merge_anatomical_morphological_final_for_pred,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
FNAME_LOG = 'log_longitudinal.txt'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METRICS = [
    'area_ratio',
    'diameter_AP_ratio',
    'diameter_RL_ratio',
    'eccentricity_ratio',
    'solidity_ratio',
]
METRICS_NORM = [m + '_PAM50_normalized' for m in METRICS]

DICT_DISC_LABELS = {
    'C1/C2': 2, 'C2/C3': 3, 'C3/C4': 4,
    'C4/C5': 5, 'C5/C6': 6, 'C6/C7': 7,
}

# Mapping from timepoint label to months (extend as needed)
MONTHS_MAP = {'bl': 0, 'baseline': 0, '': 0, '6m': 6, '12m': 12}

# Columns that carry no analysis value and must be dropped before regression
COLS_DROP = [
    'pathology', 'record_id', 'record_id_y', 'record_id_x',
    'compression_level', 'date_previous_surgery', 'surgery_date',
    'date_of_scan', 'manufacturers_model_name', 'manufacturer',
    'stenosis', 'maximum_stenosis', 'maximum_stenosis_y', 'maximum_stenosis_x',
    'slice(I->S)',
    'eccentricity_ratio_PAM50', 'diameter_RL_ratio_PAM50',
    'diameter_AP_ratio_PAM50', 'area_ratio_PAM50', 'solidity_ratio_PAM50',
    # baseline-only electro (absent in non-baseline timepoints after column selection)
    'dSEP_C6_both_patho_bl', 'dSEP_C8_both_patho_bl',
    'CHEPS_C6_patho_bl', 'CHEPS_C8_patho_bl', 'CHEPS_T4_grading_patho_bl',
    'amp_max_sten_sag_or_ax1_or_ax2_bl', 'disp_max_sten_sag_or_ax1_or_ax2_mm_bl',
    'dSEP_both_patho_bl', 'CHEPS_patho_bl',
]


# ===========================================================================
# CLI
# ===========================================================================
def get_parser():
    parser = argparse.ArgumentParser(
        description=dedent(__doc__),
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        '-input-files', required=True, metavar='<tp=path,...>',
        help="R|Per-timepoint morphometric CSV files.\n"
             "Format: 'bl=path_bl.csv,6m=path_6m.csv,12m=path_12m.csv'\n"
             "At least one timepoint is required (use 'bl' for baseline only).")
    parser.add_argument(
        '-participants-file', required=True, metavar='<file>',
        help="Path to the dcm-zurich participants.tsv file.")
    parser.add_argument(
        '-clinical-file', required=True, metavar='<file>',
        help="Excel file with clinical scores (mJOA, ASIA, GRASSP). Example: clinical_scores.xlsx")
    parser.add_argument(
        '-anatomical-file', required=True, metavar='<file>',
        help="Excel file with aSCOR and aMSCC per cervical level. Example: anatomical_data.xlsx")
    parser.add_argument(
        '-electro-file', required=True, metavar='<file>',
        help="Excel file with electrophysiology (SEP, CHEPS). Example: electrophysiological_measurements.xlsx")
    parser.add_argument(
        '-motion-file', required=True, metavar='<file>',
        help="Excel file with segmental motion data. Example: motion_data.xlsx")
    parser.add_argument(
        '-path-out', required=True, metavar='<dir>',
        help="Output directory. Subdirectories are created per timepoint.")
    parser.add_argument(
        '-exclude', required=False, metavar='<yaml>',
        help="YAML file listing subject filenames to exclude. Example:\n"
             "  - sub-001_T2w.nii.gz\n  - sub-002_T2w.nii.gz")
    parser.add_argument(
        '-no-prediction', action='store_true',
        help="Skip the stepwise logistic regression (faster; useful for quick summaries).")
    return parser


# ===========================================================================
# Data helpers
# ===========================================================================
def parse_input_files(s):
    """
    Parse 'bl=path.csv,6m=path.csv,...' into {'bl': 'path.csv', '6m': 'path.csv', ...}.
    """
    result = {}
    for pair in s.split(','):
        pair = pair.strip()
        if '=' not in pair:
            raise ValueError(f"Bad -input-files entry '{pair}'. Expected format: tp=path.csv")
        tp, path = pair.split('=', 1)
        result[tp.strip()] = path.strip()
    return result


def select_timepoint_columns(df, tp):
    """
    Given a merged DataFrame that contains columns for all timepoints
    (baseline, 6m, 12m), return a copy filtered to the requested timepoint.

    Column selection logic
    ----------------------
    - 'bl' / '' / 'baseline':
        Keep columns that do NOT end with '_6m' or '_12m'.
    - '6m':
        Rename *_6m columns to their base name.
        Drop *_12m, *_bl (baseline-specific clinical/electro), and any
        base-named column that already has a 6m counterpart.
    - '12m':  same logic for '_12m'.

    MRI metric columns (area_ratio, ..., _PAM50_normalized) carry no
    timepoint suffix and are always kept.
    """
    df = df.copy()
    known_suffixes = ['_6m', '_12m']

    if tp in ('bl', '', 'baseline'):
        cols = [c for c in df.columns if not any(c.endswith(s) for s in known_suffixes)]
        return df[cols]

    suffix = f'_{tp}'                                           # e.g. '_6m'
    other_suffixes = [s for s in known_suffixes if s != suffix]
    tp_base_names = {c[: -len(suffix)] for c in df.columns if c.endswith(suffix)}

    final_cols, rename_map = [], {}
    for c in df.columns:
        if c.endswith(suffix):                           # our timepoint → rename to base
            rename_map[c] = c[: -len(suffix)]
            final_cols.append(c)
        elif any(c.endswith(s) for s in other_suffixes): # different follow-up tp → skip
            pass
        elif c.endswith('_bl'):                          # baseline-specific column → skip
            pass
        elif c in tp_base_names:                         # superseded by tp version → skip
            pass
        else:                                            # static / MRI metric → keep
            final_cols.append(c)

    return df[final_cols].rename(columns=rename_map)


def aggregate_ascore_for_timepoint(df, anatomical_df, tp):
    """
    Update the 'aSCOR' column of df with values from the requested timepoint.

    The merged df contains aSCOR aggregated at baseline by
    merge_anatomical_morphological_final_for_pred(). For follow-up
    timepoints this function overwrites aSCOR with 'aSCOR_C{n}_{tp}'.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'participant_id' (column) and 'level' (numeric, disc labels).
    anatomical_df : pd.DataFrame
        Indexed by participant_id; contains 'aSCOR_C{n}' (bl) and
        'aSCOR_C{n}_6m' / 'aSCOR_C{n}_12m' (follow-up).
    tp : str
        Timepoint label ('bl', '6m', '12m', …).

    Returns a modified copy of df.
    """
    col_suffix = '' if tp in ('bl', '', 'baseline') else f'_{tp}'
    df = df.copy()

    for _, row in df[['participant_id', 'level']].dropna().iterrows():
        subj, level = row['participant_id'], row['level']
        if level == 2 or subj not in anatomical_df.index:
            continue
        level_str = 'C' + str(int(level) - 1)          # C4/C5 (level=5) → C4
        ascore_col = f'aSCOR_{level_str}{col_suffix}'
        if ascore_col in anatomical_df.columns:
            df.loc[df['participant_id'] == subj, 'aSCOR'] = anatomical_df.loc[subj, ascore_col]

    return df


# ===========================================================================
# Statistics helpers (same logic as baseline, extracted as functions)
# ===========================================================================
def compute_descriptive_stats(df, path_out):
    """
    Compute and save mean ± SD tables broken down by therapeutic decision and
    level of maximum stenosis. Mirrors compute_mean_std() in the baseline script.
    """
    logger.info(f'  n subjects: {df.shape[0]}')

    # Overall mean / SD
    mean_std_all = df.agg([np.mean, np.std])
    mean_std_all.to_csv(os.path.join(path_out, 'mean_std_all.csv'))
    logger.info(f'  Overall:\n{mean_std_all}')

    # By therapeutic decision
    if 'therapeutic_decision' in df.columns:
        mean_std_by_td = df.groupby('therapeutic_decision', as_index=False).agg([np.mean, np.std])
        mean_std_by_td.to_csv(os.path.join(path_out, 'mean_std_by_therapeutic.csv'))
        logger.info(f'  By therapeutic decision:\n{mean_std_by_td}')

        count_cons = (df['therapeutic_decision'] == 0).sum()
        count_oper = (df['therapeutic_decision'] == 1).sum()
        n_total = count_cons + count_oper
        if n_total > 0:
            logger.info(f'  Conservative: {100*count_cons/n_total:.1f}%  Operative: {100*count_oper/n_total:.1f}%')

    # By level
    if 'level' in df.columns:
        mean_std_by_level = df.groupby('level', as_index=False).agg([np.mean, np.std])
        mean_std_by_level.to_csv(os.path.join(path_out, 'mean_std_by_level.csv'))
        ratio = df['level'].value_counts(normalize=True) * 100
        logger.info(f'  Level distribution (%):\n{ratio}')

    # Categorical ratios
    for col, label0, label1 in [
        ('sex', 'F', 'M'), ('myelopathy', 'No myelopathy', 'Myelopathy'),
    ]:
        if col in df.columns:
            c0 = (df[col] == 0).sum()
            c1 = (df[col] == 1).sum()
            n = c0 + c1
            if n > 0:
                logger.info(f'  {label0}: {100*c0/n:.1f}%  {label1}: {100*c1/n:.1f}%')


def compute_correlation_matrices(df, path_out):
    """
    Compute and save Spearman correlation matrix, p-values, and combined
    correlation + significance table (*, **, ***).
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # Spearman correlation matrix
    corr = numeric_df.corr(method='spearman')
    corr.to_csv(os.path.join(path_out, 'corr_table.csv'))

    # P-values
    pvalues = numeric_df.corr(
        method=lambda x, y: stats.spearmanr(x, y)[1]
    ) - np.eye(len(numeric_df.columns))
    # Replace exact zeros (floating-point underflow) with a very small value
    for col in pvalues.columns:
        for idx in pvalues.index:
            if col != idx and pvalues.loc[idx, col] == 0:
                pvalues.loc[idx, col] = 1e-30
    pvalues.to_csv(os.path.join(path_out, 'corr_table_pvalue.csv'))

    # Combined: coefficient + significance stars
    stars = pvalues.applymap(
        lambda x: ''.join(['*' for t in [0.001, 0.01, 0.05] if 0 < x <= t])
    )
    corr_with_stars = corr.round(2).astype(str) + stars
    corr_with_stars.to_csv(os.path.join(path_out, 'corr_table_and_pvalue.csv'))

    # Point-biserial correlations for binary variables
    for bin_col in ['sex', 'previous_surgery', 'myelopathy', 'therapeutic_decision']:
        if bin_col in numeric_df.columns:
            r = numeric_df.drop(columns=[bin_col]).corrwith(
                numeric_df[bin_col].astype(float), method=stats.pointbiserialr
            )
            r.to_csv(os.path.join(path_out, f'corr_table_{bin_col}.csv'))

    logger.info(f'  Correlation matrices saved to {path_out}')
    return corr, pvalues


def compute_stepwise(y, X, threshold_in=0.05, threshold_out=0.05, method='logistic'):
    """
    Bidirectional stepwise predictor selection based on p-values.
    Identical to the baseline version; reproduced here for self-containment.

    Parameters
    ----------
    y      : pd.Series  — dependent variable
    X      : pd.DataFrame — candidate predictors
    method : 'logistic' or 'linear'

    Returns
    -------
    included : list of selected predictor names
    """
    import random
    cols = list(X.columns)
    random.shuffle(cols)
    X = X[cols]
    included = []

    for iteration in range(1, 101):          # hard cap at 100 iterations
        changed = False
        excluded = [c for c in X.columns if c not in included]
        new_pval = pd.Series(index=excluded, dtype=float)

        for col in excluded:
            try:
                if method == 'logistic':
                    model = sm.Logit(y, X[included + [col]]).fit(disp=0)
                else:
                    model = sm.OLS(y, X[included + [col]]).fit()
                new_pval[col] = model.pvalues[col]
            except Exception:
                new_pval[col] = 1.0

        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_pred = excluded[new_pval.argmin()]
            included.append(best_pred)
            changed = True
            logger.info(f'  [step {iteration}] Add  {best_pred:40s} p={best_pval:.4f}')

        if included:
            try:
                if method == 'logistic':
                    model = sm.Logit(y, X[included]).fit(disp=0)
                    pvalues = model.pvalues
                else:
                    model = sm.OLS(y, X[included]).fit()
                    pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max()
                if worst_pval > threshold_out:
                    worst_pred = included[pvalues.argmax()]
                    included.remove(worst_pred)
                    changed = True
                    logger.info(f'  [step {iteration}] Drop {worst_pred:40s} p={worst_pval:.4f}')
            except Exception:
                pass

        if not changed:
            break

    return included


def run_logistic_regression(df_no_norm, df_norm, df_all, path_out):
    """
    Stepwise logistic regression to predict therapeutic decision.

    Parameters
    ----------
    df_no_norm : DataFrame without normalised metrics, indexed by participant_id
    df_norm    : DataFrame with only normalised metrics, indexed by participant_id
    df_all     : DataFrame with all metrics (used for myelopathy test), indexed by participant_id
    path_out   : output directory
    """
    # Keep only columns without follow-up suffixes (for timepoints that still have them)
    def _drop_followup_cols(d):
        return d[[c for c in d.columns if '6m' not in c and '12m' not in c]]

    df_no_norm = _drop_followup_cols(df_no_norm.copy()).dropna()
    df_norm = _drop_followup_cols(df_norm.copy()).dropna()
    df_all = _drop_followup_cols(df_all.copy()).dropna()

    if 'therapeutic_decision' not in df_no_norm.columns:
        logger.warning('  therapeutic_decision column not found – skipping logistic regression.')
        return

    y = df_no_norm['therapeutic_decision'].astype(int)

    # Model without normalisation
    logger.info('\n  -- Logistic regression (non-normalised metrics) --')
    X = df_no_norm.drop(columns=['therapeutic_decision']).astype(float)
    included = compute_stepwise(y, X, method='logistic')
    logger.info(f'  Selected predictors: {included}')
    if included:
        fit_reg(X[included], y, 'logistic', logger)

    # Model with normalisation
    logger.info('\n  -- Logistic regression (normalised metrics) --')
    X_norm = df_norm.drop(columns=['therapeutic_decision']).astype(float)
    included_norm = compute_stepwise(y, X_norm, method='logistic')
    logger.info(f'  Selected predictors: {included_norm}')
    if included_norm:
        fit_reg(X_norm[included_norm], y, 'logistic', logger)


# ===========================================================================
# Temporal evolution
# ===========================================================================
def compute_temporal_slopes(dfs_by_tp, metrics, months_map=None, path_out=None):
    """
    Estimate a per-subject linear slope (units / month) for each metric
    across all supplied timepoints using ordinary least squares.

    Statistical test
    ----------------
    A one-sample t-test (H₀: μ_slope = 0) is applied to the population
    of slopes for each metric to detect group-level temporal change.

    Parameters
    ----------
    dfs_by_tp : dict  {timepoint_label: pd.DataFrame indexed by participant_id}
    metrics   : list  column names to analyse
    months_map : dict  {timepoint_label: months} — default uses MONTHS_MAP
    path_out   : str  directory to save CSVs (optional)

    Returns
    -------
    df_slopes : pd.DataFrame  per-subject slopes (columns: '<metric>_slope_per_month')
    df_tests  : pd.DataFrame  group-level significance tests
    """
    if months_map is None:
        months_map = MONTHS_MAP

    # Only use timepoints that have a known numeric month value, sorted chronologically
    tps_sorted = sorted(
        [(tp, months_map[tp]) for tp in dfs_by_tp if tp in months_map],
        key=lambda x: x[1],
    )
    if len(tps_sorted) < 2:
        logger.warning('compute_temporal_slopes: fewer than 2 mapped timepoints – skipping.')
        return None, None

    logger.info(f'  Timepoints used for slope: {[(tp, f"{mo}m") for tp, mo in tps_sorted]}')
    all_subjects = sorted(set.union(*[set(df.index) for df in dfs_by_tp.values()]))

    rows = []
    for subj in all_subjects:
        row = {'participant_id': subj}
        for metric in metrics:
            xs, ys = [], []
            for tp, months in tps_sorted:
                df_tp = dfs_by_tp[tp]
                if subj not in df_tp.index or metric not in df_tp.columns:
                    continue
                val = df_tp.loc[subj, metric]
                if not pd.isna(val):
                    xs.append(months)
                    ys.append(float(val))
            row[f'{metric}_slope_per_month'] = (
                np.polyfit(xs, ys, 1)[0] if len(xs) >= 2 else np.nan
            )
        rows.append(row)

    df_slopes = pd.DataFrame(rows).set_index('participant_id')

    # Group-level test: one-sample t-test against 0
    test_rows = []
    for sc in df_slopes.columns:
        vals = df_slopes[sc].dropna().values
        t_stat, p_val = stats.ttest_1samp(vals, 0) if len(vals) >= 3 else (np.nan, np.nan)
        test_rows.append({
            'metric_slope': sc,
            'n_subjects': len(vals),
            'mean_slope': np.nanmean(vals),
            'std_slope':  np.nanstd(vals),
            'sem_slope':  np.nanstd(vals) / np.sqrt(len(vals)) if len(vals) > 0 else np.nan,
            't_stat': t_stat,
            'p_value': p_val,
            'significant_p005': p_val < 0.05 if not np.isnan(p_val) else False,
        })
    df_tests = pd.DataFrame(test_rows)

    if path_out is not None:
        df_slopes.to_csv(os.path.join(path_out, 'temporal_slopes.csv'))
        df_tests.to_csv(os.path.join(path_out, 'temporal_slopes_significance.csv'), index=False)
        logger.info(f'  Saved: temporal_slopes.csv')
        logger.info(f'  Saved: temporal_slopes_significance.csv')
        logger.info('\n' + df_tests.to_string(index=False))

    return df_slopes, df_tests


def plot_temporal_trajectories(dfs_by_tp, metrics, months_map=None, path_out=None,
                                group_col='therapeutic_decision',
                                group_labels={0: 'Conservative', 1: 'Operative'}):
    """
    Line plots showing group-mean ± 95% CI metric trajectories over time,
    colour-coded by therapeutic decision (or any binary grouping variable).

    One figure per metric; saved to <path_out>/trajectory_<metric>.png.
    """
    if months_map is None:
        months_map = MONTHS_MAP

    tps_sorted = sorted(
        [(tp, months_map[tp]) for tp in dfs_by_tp if tp in months_map],
        key=lambda x: x[1],
    )
    if len(tps_sorted) < 2:
        logger.warning('plot_temporal_trajectories: fewer than 2 timepoints – skipping plots.')
        return

    palette = {0: '#2196F3', 1: '#F44336'}   # blue = conservative, red = operative

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.set_style('ticks', {'axes.grid': True})

        for grp, label in group_labels.items():
            means, cis, months = [], [], []
            for tp, mo in tps_sorted:
                df_tp = dfs_by_tp[tp]
                if metric not in df_tp.columns or group_col not in df_tp.columns:
                    continue
                vals = df_tp.loc[df_tp[group_col] == grp, metric].dropna()
                if vals.empty:
                    continue
                n = len(vals)
                sem = vals.std() / np.sqrt(n)
                ci = 1.96 * sem
                means.append(vals.mean())
                cis.append(ci)
                months.append(mo)

            if len(months) >= 2:
                ax.errorbar(
                    months, means, yerr=cis,
                    marker='o', label=label,
                    color=palette.get(grp, None), capsize=4, lw=2,
                )

        ax.set_xlabel('Months post-baseline', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'Temporal trajectory — {metric}', fontsize=12)
        ax.legend(fontsize=10)
        sns.despine()
        plt.tight_layout()

        if path_out is not None:
            fname = os.path.join(path_out, f'trajectory_{metric}.png')
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            logger.info(f'  Saved: {os.path.basename(fname)}')
        plt.close()


# ===========================================================================
# Per-timepoint pipeline
# ===========================================================================
def build_merged_df(df_participants, df_morphometrics_tp,
                    anatomical_df, motion_df, electrophysiology_df, motion_file):
    """
    Merge all data sources into a single DataFrame for one timepoint's
    morphometric CSV.  Mirrors the merge logic in the baseline main().
    """
    if 'maximum_stenosis' in motion_file:
        df_clinical_all = merge_anatomical_morphological_final_for_pred(
            anatomical_df, motion_df, df_morphometrics_tp, add_motion=False)
        final_df = pd.merge(df_participants, df_clinical_all,
                            on='participant_id', how='outer', sort=True)
        final_df = pd.merge(final_df, motion_df,
                            on='participant_id', how='outer', sort=True)
    else:
        df_clinical_all = merge_anatomical_morphological_final_for_pred(
            anatomical_df, motion_df, df_morphometrics_tp, add_motion=False)
        final_df = pd.merge(df_participants, df_clinical_all,
                            on='participant_id', how='outer', sort=True)

    final_df = pd.merge(final_df, electrophysiology_df,
                        on='participant_id', how='outer', sort=True)
    return final_df


def encode_categoricals(df):
    """Encode categorical columns to integers (same mapping as baseline)."""
    df = df.replace({'sex': {'F': 0, 'M': 1}})
    df = df.replace({'level': DICT_DISC_LABELS})
    df = df.replace({'therapeutic_decision': {'conservative': 0, 'operative': 1}})
    df = df.replace({'previous_surgery': {'no': 0, 'yes': 1}})
    return df


def prepare_regression_dfs(final_df_tp):
    """
    Clean the merged DataFrame and return three views used in regression:
    - df_all   : all metrics (norm + non-norm), used for descriptive stats
    - df_no_norm: only non-normalised MRI metrics
    - df_norm  : only normalised MRI metrics

    Mirrors the column-drop and DataFrame-construction logic of baseline main().
    """
    df = final_df_tp.copy()

    # Binary-encode myelopathy
    df['myelopathy'] = df['myelopathy'].fillna(0.0)
    df.loc[df['myelopathy'] != 0, 'myelopathy'] = 1.0
    df['myelopathy'] = df['myelopathy'].astype(float)

    # Drop metadata / redundant columns
    cols_to_drop = [c for c in COLS_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.set_index('participant_id')

    df_all = df.copy()
    df_no_norm = df.drop(columns=[c for c in METRICS_NORM if c in df.columns])
    df_norm    = df.drop(columns=[c for c in METRICS if c in df.columns])

    return df_all, df_no_norm, df_norm


def run_timepoint_analysis(tp, tp_file, df_participants, clinical_df,
                           anatomical_df, motion_df, electrophysiology_df,
                           dict_exclude_subj, args, path_out, run_prediction=True):
    """
    Run the complete per-timepoint pipeline (same steps as the baseline script)
    for a single timepoint.

    Steps
    -----
    1. Load morphometric CSV
    2. Merge all data sources
    3. Update aSCOR to the correct timepoint
    4. Encode categoricals
    5. Select timepoint-appropriate columns
    6. Build regression DataFrames
    7. Descriptive statistics (mean ± SD)
    8. Correlation matrices
    9. Stepwise logistic regression (optional)

    Returns df_reg_all indexed by participant_id (used for temporal analysis).
    """
    logger.info(f'\n{"="*64}\nTimepoint: {tp}\n{"="*64}')

    # 1. Load morphometric metrics
    df_morphometrics_tp = read_metric_file(tp_file, list(dict_exclude_subj), df_participants)

    # 2. Merge
    final_df_tp = build_merged_df(
        df_participants, df_morphometrics_tp,
        anatomical_df, motion_df, electrophysiology_df, args.motion_file,
    )

    # 3. Update aSCOR to the right timepoint
    final_df_tp = aggregate_ascore_for_timepoint(final_df_tp, anatomical_df, tp)

    # 4. Encode categoricals
    final_df_tp = encode_categoricals(final_df_tp)

    # 5. Filter to timepoint-appropriate columns
    final_df_tp = select_timepoint_columns(final_df_tp, tp)

    # Drop subjects missing the core MRI metric
    if 'area_ratio_PAM50_normalized' in final_df_tp.columns:
        final_df_tp.dropna(axis=0, subset=['area_ratio_PAM50_normalized'], inplace=True)
        final_df_tp.dropna(axis=0,
                           subset=['area_ratio_PAM50_normalized', 'total_mjoa',
                                   'therapeutic_decision', 'age', 'height'],
                           inplace=True)

    (final_df_tp.isna()).to_csv(os.path.join(path_out, 'missing_data.csv'))

    logger.info(f'  n subjects (after QC): {final_df_tp["participant_id"].nunique()}')

    # 6. Build regression DataFrames
    df_all, df_no_norm, df_norm = prepare_regression_dfs(final_df_tp)
    df_all.dropna(inplace=True)
    logger.info(f'  n subjects (after dropna): {df_all.shape[0]}')

    # 7. Descriptive statistics
    compute_descriptive_stats(df_all, path_out)

    # 8. Correlation matrices
    compute_correlation_matrices(df_all, path_out)

    # 9. Predictive model
    if run_prediction:
        try:
            run_logistic_regression(df_no_norm.dropna(), df_norm.dropna(), df_all, path_out)
        except Exception as e:
            logger.warning(f'  Logistic regression skipped for {tp}: {e}')

    return df_all


# ===========================================================================
# main
# ===========================================================================
def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup output directory and logging
    os.makedirs(args.path_out, exist_ok=True)
    path_out = args.path_out
    fh = logging.FileHandler(os.path.join(path_out, FNAME_LOG))
    logging.root.addHandler(fh)

    # Load exclusion list
    import yaml
    dict_exclude_subj = []
    if args.exclude:
        if not os.path.isfile(args.exclude):
            sys.exit(f'ERROR: exclude file not found: {args.exclude}')
        with open(args.exclude, 'r') as f:
            dict_exclude_subj = yaml.safe_load(f) or []
    logger.info(f'Excluded subjects: {dict_exclude_subj}')

    # -------------------------------------------------------------------
    # Load shared data (loaded once; reused for every timepoint)
    # -------------------------------------------------------------------
    logger.info('\nLoading shared data...')
    df_participants = read_participants_file(args.participants_file)
    clinical_df     = read_clinical_file(args.clinical_file)
    # Merge clinical scores into participants
    df_participants  = pd.merge(df_participants, clinical_df,
                                on='record_id', how='outer', sort=True)

    electrophysiology_df = read_electrophysiology_file(args.electro_file, df_participants)
    anatomical_df        = read_anatomical_file(args.anatomical_file, df_participants)

    if 'maximum_stenosis' in args.motion_file:
        motion_df = read_motion_file_maximum_stenosis(args.motion_file, df_participants)
    else:
        motion_df = read_motion_file(args.motion_file, df_participants)

    # -------------------------------------------------------------------
    # Per-timepoint analysis
    # -------------------------------------------------------------------
    tps_files = parse_input_files(args.input_files)
    logger.info(f'Timepoints to analyse: {list(tps_files.keys())}')

    dfs_by_tp = {}   # {tp: df_reg_all indexed by participant_id} — for temporal analysis

    for tp, tp_file in tps_files.items():
        tp_out = os.path.join(path_out, f'timepoint_{tp}')
        os.makedirs(tp_out, exist_ok=True)

        # Per-timepoint log
        fh_tp = logging.FileHandler(os.path.join(tp_out, f'log_stats_{tp}.txt'))
        logging.root.addHandler(fh_tp)

        df_reg_all = run_timepoint_analysis(
            tp=tp, tp_file=tp_file,
            df_participants=df_participants,
            clinical_df=clinical_df,
            anatomical_df=anatomical_df,
            motion_df=motion_df,
            electrophysiology_df=electrophysiology_df,
            dict_exclude_subj=dict_exclude_subj,
            args=args,
            path_out=tp_out,
            run_prediction=not args.no_prediction,
        )
        dfs_by_tp[tp] = df_reg_all

        logging.root.removeHandler(fh_tp)
        fh_tp.close()

    # -------------------------------------------------------------------
    # Temporal evolution (across all timepoints)
    # -------------------------------------------------------------------
    if len(dfs_by_tp) >= 2:
        logger.info(f'\n{"="*64}\nTemporal evolution analysis\n{"="*64}')
        temporal_out = os.path.join(path_out, 'temporal_evolution')
        os.makedirs(temporal_out, exist_ok=True)

        df_slopes, df_tests = compute_temporal_slopes(
            dfs_by_tp, METRICS + METRICS_NORM, path_out=temporal_out
        )

        plot_temporal_trajectories(
            dfs_by_tp, METRICS + METRICS_NORM, path_out=temporal_out
        )
    else:
        logger.info('\nOnly one timepoint provided – temporal evolution analysis skipped.')

    logger.info('\nDone.')


if __name__ == '__main__':
    main()
