"""
LME trajectory analysis for degenerative cervical myelopathy with logarithmic time axis.

Motivation for logarithmic time
--------------------------------
Schading et al. (2023, NeuroImage Clinical) — "Dynamics of progressive degeneration of
major spinal pathways following spinal cord injury: A longitudinal study" — modelled MRI
biomarker trajectories after tSCI using log-transformed time since injury. The rationale
is that neurological change (whether recovery or degeneration) is rapid early on and
decelerates over time, producing a concave-down curve that is linearised in log-time.
The same biological argument applies to DCM: most functional change after surgery (or
during natural history) occurs within the first weeks-to-months, with diminishing returns
thereafter. Using time_log = log(days_from_BL + 1) compresses the x-axis, making early
dynamics visually prominent while still capturing the 12-month window.
NOTE: At baseline day 0 → log(0+1) = log(1) = 0, preserving the zero-origin.

Three LME models are fitted:
  A. Stratified by MRI hyperintensity alone:
       score ~ C(myelopathy) + time_log + C(myelopathy):time_log + covariates
  B. Stratified by therapeutic decision alone:
       score ~ C(therapeutic_decision) + time_log + C(therapeutic_decision):time_log + covariates
  C. Combined (both factors + 3-way interaction):
       score ~ C(myelopathy) * C(therapeutic_decision) * time_log + covariates

Surgery handling (exact dates, NOT categorical timepoints):
  - surg_date_before_BL     : if a date is present → surgery before baseline → exclude subject
  - surg_date_before_6mth   : if a date is present → surgery between BL and 6-month visit;
                               this defines the 'operative' group and provides the surgery event day
  - surg_date_before_12mth  : if a date is present → surgery between 6-month and 12-month visit

JOA dates (exact):
  - orthopedics_assessment_date_BL   → day 0 (all time is relative to this)
  - orthopedics_assessment_date_6mth  → actual days elapsed since BL
  - orthopedics_assessment_date_12mth → actual days elapsed since BL

Example usage:
    python lme_trajectories_log_time.py \
        -clinical-file data/clinical_scores.xlsx \
        -morphometrics-file data/morphometrics.csv \
        -participants-to-use data/participants.txt \
        -o figures/lme_log_time
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils import read_clinical_file, read_morphometrics_file

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────
LABEL_FONT_SIZE = 13
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 11

# Colors: T2w- green, T2w+ red
COLOR_T2W_MINUS = '#2ca02c'
COLOR_T2W_PLUS  = '#d62728'
# Colors: conservative blue, operative orange
COLOR_CONSERVATIVE = '#1f77b4'
COLOR_OPERATIVE    = '#ff7f0e'

ALPHA_SPAGHETTI = 0.25
LW_SPAGHETTI    = 0.5
LW_LME          = 2.2
MARKER_SIZE     = 4

# ──────────────────────────────────────────────────────────────────────────────
# Clinical scores to analyse
# ──────────────────────────────────────────────────────────────────────────────
CLINICAL_SCORES = {
    'mJOA': {
        'col_BL':   'total_mjoa_BL',
        'col_6mth': 'total_mjoa_6mth',
        'col_12mth':'total_mjoa_12mth',
        'y_label':  'mJOA score',
        'ylim':     (11.5, 18.5),
    },
}

# Surgery date column names (exact-date columns; presence = surgery happened)
SURG_COL_BEFORE_BL   = 'surg_date_before_BL'    # → exclude subject
SURG_COL_BEFORE_6MTH = 'surg_date_before_6mth'  # → operative group; event marker
SURG_COL_BEFORE_12MTH= 'surg_date_before_12mth' # → handled separately

# Assessment date columns
DATE_COL_BL    = 'orthopedics_assessment_date_BL'
DATE_COL_6MTH  = 'orthopedics_assessment_date_6mth'
DATE_COL_12MTH = 'orthopedics_assessment_date_12mth'


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def get_parser():
    p = argparse.ArgumentParser(
        description='LME trajectory analysis with log-transformed time axis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('-clinical-file', required=True,
                   help='Path to Excel file with clinical scores.')
    p.add_argument('-morphometrics-file', required=True,
                   help='Path to CSV file with morphometric data (spinal cord area).')
    p.add_argument('-participants-to-use', required=True,
                   help='Path to text file with one participant ID per line.')
    p.add_argument('-sessions', type=int, choices=[2, 3], default=3,
                   help='Number of sessions: 2 (BL + 6mth) or 3 (BL + 6mth + 12mth).')
    p.add_argument('-o', '--outdir', required=True,
                   help='Output directory for figures and CSVs.')
    p.add_argument('--level', type=int, default=3, choices=[2, 3],
                   help='Vertebral level for baseline spinal cord area (default: C3).')
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def log_time(days):
    """
    Logarithmic time transform: log(days + 1).
    Baseline (day 0) maps to 0; preserves zero-origin.
    """
    return np.log(np.asarray(days, dtype=float) + 1.0)


def days_from_log(t_log):
    """Inverse of log_time: recover days from log-time value."""
    return np.exp(t_log) - 1.0


def log_print(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare_data(df_clinical, df_morphometrics, level=3, sessions=3):
    """
    Reshape clinical data to long format with exact dates and log-time.

    Columns added per row (= one patient × timepoint):
      time_days  – actual days elapsed since BL assessment
      time_log   – log(time_days + 1)          ← model predictor
      surg_days  – days from BL to surgery (NaN if no surgery in this window)
      surg_log   – log(surg_days + 1)           ← for surgery event marker

    Returns
    -------
    dict: score_name → pd.DataFrame (long format)
    """

    # ── morphometrics: baseline area at requested level ───────────────────────
    df_morph_bl = df_morphometrics[
        (df_morphometrics['session_id'] == 'ses-M0') &
        (df_morphometrics['VertLevel'] == level)
    ].copy()
    df_area = (df_morph_bl.groupby('participant_id')['MEAN(area)'].mean()
               .reset_index().rename(columns={'MEAN(area)': f'area_C{level}'}))

    df = df_clinical.merge(df_area, on='participant_id', how='left')

    # ── myelopathy (T2w hyperintensity) ───────────────────────────────────────
    if 'myelopathy' not in df.columns and 'Myelopathy' in df.columns:
        df['myelopathy'] = df['Myelopathy']
    df['myelopathy'] = df['myelopathy'].map(
        {0: 'no', 1: 'yes', '0': 'no', '1': 'yes', 'no': 'no', 'yes': 'yes'}
    )

    # ── exclude: surgery before baseline ─────────────────────────────────────
    if SURG_COL_BEFORE_BL in df.columns:
        df[SURG_COL_BEFORE_BL] = pd.to_datetime(df[SURG_COL_BEFORE_BL], errors='coerce')
        n_before = df['participant_id'].nunique()
        df = df[df[SURG_COL_BEFORE_BL].isna()].copy()
        n_after = df['participant_id'].nunique()
        print(f"Excluded {n_before - n_after} subjects with surgery before baseline "
              f"(non-null {SURG_COL_BEFORE_BL}). Remaining: {n_after}")
    else:
        # Fallback: use the old integer flag column
        if 'surg_timepoint___1_12mth' in df.columns:
            n_before = df['participant_id'].nunique()
            df = df[df['surg_timepoint___1_12mth'] != 1].copy()
            n_after = df['participant_id'].nunique()
            print(f"Excluded {n_before - n_after} subjects with surgery before baseline "
                  f"(surg_timepoint___1_12mth == 1). Remaining: {n_after}")

    # ── therapeutic decision from exact surgery dates ─────────────────────────
    for col in [SURG_COL_BEFORE_6MTH, SURG_COL_BEFORE_12MTH]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if SURG_COL_BEFORE_6MTH in df.columns:
        df['therapeutic_decision'] = df[SURG_COL_BEFORE_6MTH].apply(
            lambda x: 'operative' if pd.notna(x) else 'conservative'
        )
    else:
        print(f"Warning: {SURG_COL_BEFORE_6MTH} not found; "
              "using existing 'therapeutic_decision' column.")

    # ── parse assessment dates ────────────────────────────────────────────────
    for col in [DATE_COL_BL, DATE_COL_6MTH, DATE_COL_12MTH]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ── build long-format per score ───────────────────────────────────────────
    long_data_dict = {}

    for score_name, cfg in CLINICAL_SCORES.items():
        col_bl    = cfg['col_BL']
        col_6mth  = cfg['col_6mth']
        col_12mth = cfg.get('col_12mth', None)

        if col_bl not in df.columns:
            print(f"Warning: {col_bl} not found — skipping {score_name}")
            continue

        rows = []

        for _, row in df.iterrows():
            pid              = row['participant_id']
            myelopathy       = row.get('myelopathy', np.nan)
            therapeutic_dec  = row.get('therapeutic_decision', np.nan)
            age              = pd.to_numeric(row.get('age', np.nan), errors='coerce')
            sex              = row.get('sex', 'unknown')
            max_stenosis     = row.get('maximum_stenosis', 'unknown')
            stenosis_type    = row.get('single_vs_multi_stenosis', 'unknown')
            area             = row.get(f'area_C{level}', np.nan)
            date_bl          = row.get(DATE_COL_BL, pd.NaT)

            # Surgery day (from BL) — used for event marker
            surg_date_6mth = row.get(SURG_COL_BEFORE_6MTH, pd.NaT)
            if pd.notna(date_bl) and pd.notna(surg_date_6mth):
                surg_days = (surg_date_6mth - date_bl).days
            else:
                surg_days = np.nan

            common = dict(
                participant_id      = pid,
                myelopathy          = myelopathy,
                therapeutic_decision= therapeutic_dec,
                age                 = age,
                sex                 = sex,
                maximum_stenosis    = max_stenosis,
                stenosis            = stenosis_type,
                baseline_area       = area,
                surg_days           = surg_days,
                surg_log            = log_time(surg_days) if not np.isnan(surg_days) else np.nan,
            )

            # ── Baseline (day 0) ─────────────────────────────────────────
            score_bl = pd.to_numeric(row.get(col_bl, np.nan), errors='coerce')
            if pd.notna(score_bl):
                rows.append({**common,
                             'time_days':  0.0,
                             'time_log':   0.0,
                             'time_label': 'Baseline',
                             'date':       date_bl,
                             'score':      float(score_bl)})

            # ── 6-month follow-up ────────────────────────────────────────
            if col_6mth in df.columns:
                score_6m = pd.to_numeric(row.get(col_6mth, np.nan), errors='coerce')
                date_6m  = row.get(DATE_COL_6MTH, pd.NaT)
                if pd.notna(score_6m):
                    days_6m = (date_6m - date_bl).days if (pd.notna(date_bl) and pd.notna(date_6m)) else np.nan
                    rows.append({**common,
                                 'time_days':  days_6m,
                                 'time_log':   log_time(days_6m) if not np.isnan(days_6m) else np.nan,
                                 'time_label': '6-month',
                                 'date':       date_6m,
                                 'score':      float(score_6m)})

            # ── 12-month follow-up ───────────────────────────────────────
            if sessions == 3 and col_12mth and col_12mth in df.columns:
                score_12m = pd.to_numeric(row.get(col_12mth, np.nan), errors='coerce')
                date_12m  = row.get(DATE_COL_12MTH, pd.NaT)
                if pd.notna(score_12m):
                    days_12m = (date_12m - date_bl).days if (pd.notna(date_bl) and pd.notna(date_12m)) else np.nan
                    rows.append({**common,
                                 'time_days':  days_12m,
                                 'time_log':   log_time(days_12m) if not np.isnan(days_12m) else np.nan,
                                 'time_label': '12-month',
                                 'date':       date_12m,
                                 'score':      float(score_12m)})

        if not rows:
            print(f"Warning: No data rows for {score_name}")
            continue

        df_long = pd.DataFrame(rows)
        df_long = df_long.dropna(subset=['time_log', 'score'])

        # keep only participants with ≥2 timepoints
        vc = df_long['participant_id'].value_counts()
        df_long = df_long[df_long['participant_id'].isin(vc[vc >= 2].index)].copy()

        # descriptive summary
        print(f"\n{score_name}:")
        print(f"  Participants with ≥2 timepoints : {df_long['participant_id'].nunique()}")
        print(f"  Total observations              : {len(df_long)}")
        print(f"  Operative (surgery BL→6mth)     : "
              f"{(df_long['therapeutic_decision'] == 'operative').sum()} obs")
        print(f"  T2w+ (myelopathy)               : "
              f"{(df_long['myelopathy'] == 'yes').sum()} obs")
        print(f"  Days range (BL→last visit)      : "
              f"{df_long['time_days'].min():.0f}–{df_long['time_days'].max():.0f}")

        long_data_dict[score_name] = df_long

    return long_data_dict


# ──────────────────────────────────────────────────────────────────────────────
# Model fitting
# ──────────────────────────────────────────────────────────────────────────────
def _build_and_fit(df_model, formula, score_name, log_file=None):
    """Internal helper: fit MixedLM and return result or None."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            model = MixedLM.from_formula(
                formula,
                data=df_model,
                groups=df_model['participant_id'],
                re_formula='1 + time_log',      # random intercept + random slope in log-time
            )
            result = model.fit(method='lbfgs', maxiter=2000)
    except Exception as e:
        log_print(f"  ERROR fitting {score_name}: {e}", log_file)
        return None

    log_print(f"  Converged : {result.converged}", log_file)
    log_print(f"  AIC       : {result.aic:.2f}", log_file)
    log_print(f"  BIC       : {result.bic:.2f}", log_file)
    log_print(f"  LogLik    : {result.llf:.2f}", log_file)
    log_print("  Fixed effects:", log_file)
    for param in result.params.index:
        coef = result.params[param]
        se   = result.bse[param]
        pval = result.pvalues[param]
        ci   = result.conf_int().loc[param]
        sig  = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        log_print(
            f"    {param:55s}: β={coef:8.4f} ±{se:6.4f}  p={pval:7.4f}{sig:3s}  "
            f"95%CI=[{ci[0]:8.4f}, {ci[1]:8.4f}]", log_file
        )
    return result


def _prep_df(df_long, extra_terms, log_file=None):
    """
    Center continuous covariates and build model-ready DataFrame.
    Returns (df_model, covariate_terms_list).
    """
    df_model = df_long.copy()

    # Categorical reference levels
    df_model['myelopathy'] = pd.Categorical(
        df_model['myelopathy'], categories=['no', 'yes'])
    df_model['therapeutic_decision'] = pd.Categorical(
        df_model['therapeutic_decision'], categories=['conservative', 'operative'])

    # Center age
    age_num = pd.to_numeric(df_model['age'], errors='coerce')
    df_model['age_c'] = age_num - age_num.mean() if age_num.notna().any() else np.nan

    # Center spinal cord area
    if df_model['baseline_area'].notna().any():
        df_model['area_c'] = df_model['baseline_area'] - df_model['baseline_area'].mean()

    cov_terms = []
    if 'age_c' in df_model.columns and df_model['age_c'].notna().any():
        cov_terms.append('age_c')
    if df_model['sex'].nunique() > 1 and 'unknown' not in df_model['sex'].values:
        cov_terms.append('C(sex)')
    if (df_model['maximum_stenosis'].nunique() > 1 and
            'unknown' not in df_model['maximum_stenosis'].values):
        cov_terms.append('C(maximum_stenosis, Treatment(reference="C3/C4"))')
    if (df_model['stenosis'].nunique() > 1 and
            'unknown' not in df_model['stenosis'].values):
        cov_terms.append('C(stenosis)')

    all_terms  = list(extra_terms) + cov_terms
    drop_cols  = ['score', 'time_log', 'myelopathy', 'therapeutic_decision']
    df_model   = df_model.dropna(subset=drop_cols)

    return df_model, all_terms


def fit_model_A(df_long, score_name, log_file=None):
    """Model A: stratified by myelopathy (T2w+/T2w-) only."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model A – {score_name}  (stratification: myelopathy / T2w)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(myelopathy)',
        'time_log',
        'C(myelopathy):time_log',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


def fit_model_B(df_long, score_name, log_file=None):
    """Model B: stratified by therapeutic decision (surgery vs conservative) only."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model B – {score_name}  (stratification: therapeutic decision)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(therapeutic_decision)',
        'time_log',
        'C(therapeutic_decision):time_log',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


def fit_model_C(df_long, score_name, log_file=None):
    """Model C: combined model – both factors + 3-way interaction."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model C – {score_name}  (combined: myelopathy × therapeutic decision × time)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(myelopathy)',
        'C(therapeutic_decision)',
        'time_log',
        'C(myelopathy):time_log',
        'C(therapeutic_decision):time_log',
        'C(myelopathy):C(therapeutic_decision)',
        'C(myelopathy):C(therapeutic_decision):time_log',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


# ──────────────────────────────────────────────────────────────────────────────
# X-axis helpers (log-time ↔ display labels)
# ──────────────────────────────────────────────────────────────────────────────
def build_xticks(df_long, present_labels):
    """
    Compute x-tick positions (in log-time) and labels for canonical timepoints.
    Tick position = median log-time per label (robust to date variability).
    """
    med_log  = df_long.groupby('time_label')['time_log'].median()
    med_days = df_long.groupby('time_label')['time_days'].median()
    positions = []
    labels    = []
    for lbl in present_labels:
        if lbl not in med_log:
            continue
        t_log  = med_log[lbl]
        t_days = med_days[lbl]
        t_mon  = t_days / 30.4
        positions.append(t_log)
        if lbl == 'Baseline':
            labels.append('Baseline\n(day 0)')
        else:
            labels.append(f'{lbl}\n(~{t_mon:.0f} mo / day {t_days:.0f})')
    return positions, labels, med_log, med_days


def add_log_time_xaxis(ax, df_long, present_labels):
    """Apply log-time x-ticks and a secondary real-time axis annotation."""
    tick_pos, tick_lbl, med_log, _ = build_xticks(df_long, present_labels)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=TICK_FONT_SIZE)
    ax.set_xlabel('Time since baseline  [log(days + 1) scale]', fontsize=LABEL_FONT_SIZE)


# ──────────────────────────────────────────────────────────────────────────────
# Predicted trajectory from LME parameters
# ──────────────────────────────────────────────────────────────────────────────
def _predict(params, intercept_adj, slope_adj, t_log_arr):
    """
    Linear prediction from LME fixed-effect parameters.
    intercept_adj / slope_adj are group-specific adjustments on top of
    the reference-group baseline + time_log coefficient.
    """
    β0 = params.get('Intercept', 0) + intercept_adj
    β1 = params.get('time_log', 0)  + slope_adj
    return β0 + β1 * t_log_arr


def _get(params, key, default=0.0):
    return params.get(key, default)


# ──────────────────────────────────────────────────────────────────────────────
# Surgery event markers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_surgery_markers(ax, df_long, color='#555555', alpha=0.5):
    """
    Draw a thin vertical rug tick at each surgical patient's surgery log-time.
    Only drawn for the 'operative' group and when surg_log is available.
    """
    surgical = df_long[
        (df_long['therapeutic_decision'] == 'operative') &
        df_long['surg_log'].notna()
    ]
    plotted_pids = set()
    for _, row in surgical.iterrows():
        pid = row['participant_id']
        if pid in plotted_pids:
            continue
        plotted_pids.add(pid)
        ax.axvline(
            x=row['surg_log'],
            ymin=0.0, ymax=0.04,    # short tick at bottom of axes
            color=color,
            alpha=alpha,
            linewidth=0.8,
            zorder=2,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plotting – Model A (myelopathy)
# ──────────────────────────────────────────────────────────────────────────────
def plot_model_A(df_long, result, score_name, cfg, outdir, log_file=None):
    """
    Spaghetti + LME-fitted trajectories, stratified by T2w hyperintensity.
    Colors: T2w- green, T2w+ red.
    """
    mpl.rcParams['font.family'] = 'Arial'

    present_labels = [l for l in ['Baseline', '6-month', '12-month']
                      if l in df_long['time_label'].unique()]
    tick_pos, tick_lbl, med_log, _ = build_xticks(df_long, present_labels)

    t_smooth = np.linspace(0, df_long['time_log'].max() * 1.05, 300)

    fig, ax = plt.subplots(figsize=(6, 4))

    groups = {
        'no':  {'color': COLOR_T2W_MINUS, 'label': 'T2w−'},
        'yes': {'color': COLOR_T2W_PLUS,  'label': 'T2w+'},
    }

    for myelo, ginfo in groups.items():
        gdf   = df_long[df_long['myelopathy'] == myelo]
        color = ginfo['color']
        n     = gdf['participant_id'].nunique()

        # individual spaghetti
        for pid, pdata in gdf.groupby('participant_id'):
            pdata_s = pdata.sort_values('time_log')
            if len(pdata_s) >= 2:
                ax.plot(pdata_s['time_log'], pdata_s['score'],
                        color=color, alpha=ALPHA_SPAGHETTI,
                        linewidth=LW_SPAGHETTI, zorder=3)

        # surgery markers (only where relevant)
        _plot_surgery_markers(ax, gdf, color=color, alpha=0.6)

        # LME predicted trajectory
        if result is not None:
            p = result.params
            i_adj = _get(p, 'C(myelopathy)[T.yes]') if myelo == 'yes' else 0.0
            s_adj = _get(p, 'C(myelopathy)[T.yes]:time_log') if myelo == 'yes' else 0.0
            y_hat = _predict(p, i_adj, s_adj, t_smooth)
            ax.plot(t_smooth, y_hat,
                    color=color, linewidth=LW_LME,
                    label=f'{ginfo["label"]} (n={n})', zorder=6)

        # observed mean ± SD at each canonical timepoint
        for t_pos in tick_pos:
            obs = gdf[np.isclose(gdf['time_log'], t_pos, atol=0.3)]['score']
            if len(obs) > 0:
                ax.errorbar(t_pos, obs.mean(), yerr=obs.std(),
                            fmt='o', color=color, markersize=MARKER_SIZE,
                            capsize=3, capthick=1.2, linewidth=1.2, zorder=7)

    add_log_time_xaxis(ax, df_long, present_labels)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    ax.legend(fontsize=TICK_FONT_SIZE, frameon=True, framealpha=0.85)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} – stratified by T2w hyperintensity', fontsize=TITLE_FONT_SIZE)

    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_log_time_A_myelopathy_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting – Model B (therapeutic decision)
# ──────────────────────────────────────────────────────────────────────────────
def plot_model_B(df_long, result, score_name, cfg, outdir, log_file=None):
    """
    Spaghetti + LME trajectories, stratified by therapeutic decision.
    Colors: conservative blue, operative orange.
    """
    mpl.rcParams['font.family'] = 'Arial'

    present_labels = [l for l in ['Baseline', '6-month', '12-month']
                      if l in df_long['time_label'].unique()]
    tick_pos, tick_lbl, med_log, _ = build_xticks(df_long, present_labels)

    t_smooth = np.linspace(0, df_long['time_log'].max() * 1.05, 300)

    fig, ax = plt.subplots(figsize=(6, 4))

    groups = {
        'conservative': {'color': COLOR_CONSERVATIVE, 'label': 'Conservative'},
        'operative':    {'color': COLOR_OPERATIVE,    'label': 'Operative (surgery)'},
    }

    for td, ginfo in groups.items():
        gdf   = df_long[df_long['therapeutic_decision'] == td]
        color = ginfo['color']
        n     = gdf['participant_id'].nunique()

        # spaghetti
        for pid, pdata in gdf.groupby('participant_id'):
            pdata_s = pdata.sort_values('time_log')
            if len(pdata_s) >= 2:
                ax.plot(pdata_s['time_log'], pdata_s['score'],
                        color=color, alpha=ALPHA_SPAGHETTI,
                        linewidth=LW_SPAGHETTI, zorder=3)

        # surgery markers (operative group only)
        if td == 'operative':
            _plot_surgery_markers(ax, gdf, color='#333333', alpha=0.5)

        # LME predicted
        if result is not None:
            p     = result.params
            i_adj = _get(p, 'C(therapeutic_decision)[T.operative]') if td == 'operative' else 0.0
            s_adj = _get(p, 'C(therapeutic_decision)[T.operative]:time_log') if td == 'operative' else 0.0
            y_hat = _predict(p, i_adj, s_adj, t_smooth)
            ax.plot(t_smooth, y_hat,
                    color=color, linewidth=LW_LME,
                    label=f'{ginfo["label"]} (n={n})', zorder=6)

        # observed mean ± SD
        for t_pos in tick_pos:
            obs = gdf[np.isclose(gdf['time_log'], t_pos, atol=0.3)]['score']
            if len(obs) > 0:
                ax.errorbar(t_pos, obs.mean(), yerr=obs.std(),
                            fmt='o', color=color, markersize=MARKER_SIZE,
                            capsize=3, capthick=1.2, linewidth=1.2, zorder=7)

    add_log_time_xaxis(ax, df_long, present_labels)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    ax.legend(fontsize=TICK_FONT_SIZE, frameon=True, framealpha=0.85)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} – stratified by therapeutic decision', fontsize=TITLE_FONT_SIZE)

    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_log_time_B_therapeutic_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting – Model C (combined, 4-group)
# ──────────────────────────────────────────────────────────────────────────────
def plot_model_C(df_long, result, score_name, cfg, outdir, log_file=None):
    """
    Combined model: 4 groups (myelopathy × treatment).
    Color encodes T2w status; linestyle encodes treatment.
    """
    mpl.rcParams['font.family'] = 'Arial'

    present_labels = [l for l in ['Baseline', '6-month', '12-month']
                      if l in df_long['time_label'].unique()]
    tick_pos, tick_lbl, med_log, _ = build_xticks(df_long, present_labels)

    t_smooth = np.linspace(0, df_long['time_log'].max() * 1.05, 300)

    # group spec: (myelopathy, treatment) → color, linestyle
    group_spec = {
        ('no',  'conservative'): (COLOR_T2W_MINUS, '--', 'T2w− / Conservative'),
        ('no',  'operative'):    (COLOR_T2W_MINUS, '-',  'T2w− / Operative'),
        ('yes', 'conservative'): (COLOR_T2W_PLUS,  '--', 'T2w+ / Conservative'),
        ('yes', 'operative'):    (COLOR_T2W_PLUS,  '-',  'T2w+ / Operative'),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (myelo, td), (color, ls, label) in group_spec.items():
        gdf = df_long[
            (df_long['myelopathy'] == myelo) &
            (df_long['therapeutic_decision'] == td)
        ]
        n = gdf['participant_id'].nunique()
        if n == 0:
            continue

        # spaghetti
        for pid, pdata in gdf.groupby('participant_id'):
            pdata_s = pdata.sort_values('time_log')
            if len(pdata_s) >= 2:
                ax.plot(pdata_s['time_log'], pdata_s['score'],
                        color=color, linestyle=ls,
                        alpha=ALPHA_SPAGHETTI, linewidth=LW_SPAGHETTI, zorder=3)

        # surgery event markers
        if td == 'operative':
            _plot_surgery_markers(ax, gdf, color=color, alpha=0.7)

        # LME prediction (3-way model)
        if result is not None:
            p = result.params
            # intercept adjustments
            i_adj = 0.0
            if myelo == 'yes':
                i_adj += _get(p, 'C(myelopathy)[T.yes]')
            if td == 'operative':
                i_adj += _get(p, 'C(therapeutic_decision)[T.operative]')
            if myelo == 'yes' and td == 'operative':
                i_adj += _get(p, 'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]')

            # slope adjustments
            s_adj = 0.0
            if myelo == 'yes':
                s_adj += _get(p, 'C(myelopathy)[T.yes]:time_log')
            if td == 'operative':
                s_adj += _get(p, 'C(therapeutic_decision)[T.operative]:time_log')
            if myelo == 'yes' and td == 'operative':
                s_adj += _get(p, 'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]:time_log')

            y_hat = _predict(p, i_adj, s_adj, t_smooth)
            ax.plot(t_smooth, y_hat,
                    color=color, linestyle=ls, linewidth=LW_LME,
                    label=f'{label} (n={n})', zorder=6)

        # observed mean ± SD at canonical timepoints
        for t_pos in tick_pos:
            obs = gdf[np.isclose(gdf['time_log'], t_pos, atol=0.3)]['score']
            if len(obs) > 0:
                ax.errorbar(t_pos, obs.mean(), yerr=obs.std(),
                            fmt='o', color=color, markersize=MARKER_SIZE,
                            capsize=3, capthick=1.2, linewidth=1.2, zorder=7)

    add_log_time_xaxis(ax, df_long, present_labels)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    # Two-part legend: color legend (T2w) + linestyle legend (treatment)
    c_handles = [
        Line2D([0], [0], color=COLOR_T2W_MINUS, lw=2, label='T2w−'),
        Line2D([0], [0], color=COLOR_T2W_PLUS,  lw=2, label='T2w+'),
    ]
    l_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Conservative'),
        Line2D([0], [0], color='black', lw=2, linestyle='-',  label='Operative'),
    ]
    leg1 = ax.legend(handles=c_handles, loc='lower left',
                     bbox_to_anchor=(0.01, 0.01), fontsize=TICK_FONT_SIZE,
                     frameon=True, framealpha=0.85, title='T2w hyperintensity')
    ax.add_artist(leg1)
    ax.legend(handles=l_handles, loc='lower right',
              bbox_to_anchor=(0.99, 0.01), fontsize=TICK_FONT_SIZE,
              frameon=True, framealpha=0.85, title='Treatment')

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} – combined model (T2w × therapeutic decision)',
                 fontsize=TITLE_FONT_SIZE)

    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_log_time_C_combined_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────────────────────────────────────
def save_results_csv(results_dict, outdir):
    """
    Save fixed-effect coefficients from all models to a single CSV.
    results_dict: {(score, model_label) : result_object}
    """
    rows = []
    for (score, model_lbl), res in results_dict.items():
        if res is None:
            continue
        for param in res.params.index:
            rows.append({
                'score':       score,
                'model':       model_lbl,
                'parameter':   param,
                'coefficient': res.params[param],
                'std_error':   res.bse[param],
                'z_value':     res.tvalues[param],
                'p_value':     res.pvalues[param],
                'ci_lower':    res.conf_int().loc[param, 0],
                'ci_upper':    res.conf_int().loc[param, 1],
                'aic':         res.aic,
                'bic':         res.bic,
                'log_lik':     res.llf,
                'converged':   res.converged,
            })
    if rows:
        df_out = pd.DataFrame(rows)
        fname  = os.path.join(outdir, 'lme_log_time_results.csv')
        df_out.to_csv(fname, index=False)
        print(f"\nAll model results saved to: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ── Read data ─────────────────────────────────────────────────────────────
    print("Reading clinical data...")
    # NOTE: read_clinical_file from utils.py reads the old column names.
    # We pass sessions=args.sessions so it reads the 12-month columns if needed.
    # The new surgery date columns (surg_date_before_BL, etc.) are expected to
    # already be present in the Excel file and will be parsed below.
    df_clinical, _ = read_clinical_file(
        os.path.abspath(args.clinical_file), sessions=args.sessions
    )

    print("Reading morphometric data...")
    df_morphometrics = read_morphometrics_file(
        os.path.abspath(args.morphometrics_file)
    )

    print("Reading participant list...")
    with open(args.participants_to_use) as fh:
        participant_ids = [l.strip() for l in fh if l.strip()]

    df_clinical      = df_clinical[df_clinical['participant_id'].isin(participant_ids)].copy()
    df_morphometrics = df_morphometrics[df_morphometrics['participant_id'].isin(participant_ids)].copy()

    print(f"\nTotal participants in clinical file : {df_clinical['participant_id'].nunique()}")
    print(f"Sex   : {df_clinical['sex'].value_counts().to_dict()}")
    print(f"Age   : {pd.to_numeric(df_clinical['age'], errors='coerce').describe().loc[['mean','std','min','max']].to_dict()}")

    # ── Prepare long-format data ──────────────────────────────────────────────
    print("\nPreparing longitudinal data with log-time axis...")
    long_data_dict = prepare_data(
        df_clinical, df_morphometrics,
        level=args.level, sessions=args.sessions
    )

    if not long_data_dict:
        print("ERROR: No data available for analysis.")
        return

    # ── Fit models and plot ───────────────────────────────────────────────────
    all_results = {}
    log_path    = os.path.join(args.outdir, 'lme_log_time_analysis.log')

    with open(log_path, 'w') as log_file:
        log_print("LME Log-Time Analysis Log", log_file)
        log_print(f"Sessions : {args.sessions}", log_file)
        log_print(f"SC level : C{args.level}", log_file)
        log_print(f"Output   : {args.outdir}\n", log_file)

        for score_name, df_long in long_data_dict.items():
            cfg = CLINICAL_SCORES[score_name]

            # ── Model A: myelopathy ──────────────────────────────────────
            result_A, df_A = fit_model_A(df_long, score_name, log_file)
            all_results[(score_name, 'A_myelopathy')] = result_A
            if result_A is not None:
                plot_model_A(df_long, result_A, score_name, cfg, args.outdir, log_file)
            else:
                log_print(f"  Model A failed for {score_name}", log_file)

            # ── Model B: therapeutic decision ────────────────────────────
            result_B, df_B = fit_model_B(df_long, score_name, log_file)
            all_results[(score_name, 'B_therapeutic')] = result_B
            if result_B is not None:
                plot_model_B(df_long, result_B, score_name, cfg, args.outdir, log_file)
            else:
                log_print(f"  Model B failed for {score_name}", log_file)

            # ── Model C: combined (3-way) ────────────────────────────────
            result_C, df_C = fit_model_C(df_long, score_name, log_file)
            all_results[(score_name, 'C_combined')] = result_C
            if result_C is not None:
                plot_model_C(df_long, result_C, score_name, cfg, args.outdir, log_file)
            else:
                log_print(f"  Model C failed for {score_name}", log_file)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    save_results_csv(all_results, args.outdir)

    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"Log    : {log_path}")
    print(f"Output : {args.outdir}")
    print('='*60)


if __name__ == '__main__':
    main()