"""
LME trajectory analysis for degenerative cervical myelopathy with categorical time.

Time representation
-------------------
Time is treated as a categorical factor with three levels: Baseline (reference),
6-month, and 12-month. Two binary dummy variables encode this:
  time_6m  = 1 at the 6-month visit, 0 otherwise
  time_12m = 1 at the 12-month visit, 0 otherwise
  Baseline : time_6m = 0, time_12m = 0  (implicit reference)

This allows non-monotonic trajectories (e.g. improvement at 6m followed by
decline at 12m), unlike the log-time approach which forces a concave curve.
The trade-off is that the model estimates separate fixed effects for each
timepoint rather than a single slope parameter.

Random effects: random intercept per subject only (re_formula='1').
Adding random slopes on binary dummies (re_formula='1 + time_6m + time_12m')
is feasible but substantially increases the number of variance components to
estimate; with n≈100 subjects this can cause convergence issues.

Three LME models are fitted:
  A. Stratified by T2w hyperintensity alone:
       score ~ C(t2w_hyperintensity) + time_6m + time_12m
               + C(t2w_hyperintensity):time_6m + C(t2w_hyperintensity):time_12m
               + covariates
  B. Stratified by therapeutic decision alone:
       score ~ C(therapeutic_decision) + time_6m + time_12m
               + C(therapeutic_decision):time_6m + C(therapeutic_decision):time_12m
               + covariates
  C. Combined (T2w × treatment interaction on intercept; shared time effect):
       score ~ C(t2w_hyperintensity) + C(therapeutic_decision) + time_6m + time_12m
               + C(t2w_hyperintensity):C(therapeutic_decision)
               + covariates

Surgery handling (exact dates, NOT categorical timepoints):
  - surg_date_before_BL     : if a date is present → surgery before baseline → exclude subject
  - surg_date_before_6mth   : if a date is present → surgery between BL and 6-month visit;
                               this defines the 'operative' group and provides the surgery event day
  - surg_date_before_12mth  : if a date is present → surgery between 6-month and 12-month visit

Example usage:
    python lme_trajectories_categorical_time.py \\
        -clinical-file data/clinical_scores.xlsx \\
        -morphometrics-file data/morphometrics.csv \\
        -participants-to-use data/participants.txt \\
        -o figures/lme_categorical_time
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from utils import read_clinical_file, read_morphometrics_file

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────
LABEL_FONT_SIZE = 13
TICK_FONT_SIZE  = 10
TITLE_FONT_SIZE = 11

COLOR_T2W_MINUS    = '#2ca02c'
COLOR_T2W_PLUS     = '#d62728'
COLOR_CONSERVATIVE = '#1f77b4'
COLOR_OPERATIVE    = '#ff7f0e'

LW_LME      = 2.2
MARKER_SIZE = 4

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

SURG_COL_BEFORE_BL    = 'surg_date_before_BL'
SURG_COL_BEFORE_6MTH  = 'surg_date_before_6mth'
SURG_COL_BEFORE_12MTH = 'surg_date_before_12mth'

DATE_COL_BL    = 'orthopedics_assessment_date_BL'
DATE_COL_6MTH  = 'orthopedics_assessment_date_6mth'
DATE_COL_12MTH = 'orthopedics_assessment_date_12mth'

XLIM           = (0, 365)
TIME_LABELS    = ['Baseline', '6-month', '12-month']

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def get_parser():
    p = argparse.ArgumentParser(
        description='LME trajectory analysis with categorical time (binary dummies).',
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
    p.add_argument('--anchor-col',
                   choices=['orthopedics_assessment_date_BL', 'date_inclusion'],
                   default='orthopedics_assessment_date_BL',
                   help='Column to use as day 0 anchor for time calculations '
                        '(default: orthopedics_assessment_date_BL).')
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def log_time(days):
    arr = np.asarray(days, dtype=float)
    out = np.full_like(arr, np.nan)
    valid = ~np.isnan(arr)
    out[valid] = np.log(arr[valid] + 1.0)
    return out.item() if out.ndim == 0 else out


def log_print(msg, log_file=None):
    print(msg)
    if log_file is not None:
        log_file.write(msg + '\n')
        log_file.flush()


def _get(params, key, default=0.0):
    return params.get(key, default)


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────
def prepare_data(df_clinical, df_morphometrics, level=3, sessions=3,
                 anchor_col='orthopedics_assessment_date_BL'):
    """
    Reshape clinical data to long format.

    Columns per row (= one patient × timepoint):
      time_days  – actual days elapsed since anchor date
      time_6m    – 1 if 6-month visit, else 0  (binary dummy)
      time_12m   – 1 if 12-month visit, else 0 (binary dummy)
      surg_days  – days from anchor to surgery (NaN if no surgery in this window)
    """
    df_morph_bl = df_morphometrics[
        (df_morphometrics['session_id'] == 'ses-M0') &
        (df_morphometrics['VertLevel'] == level)
    ].copy()
    df_area = (df_morph_bl.groupby('participant_id')['MEAN(area)'].mean()
               .reset_index().rename(columns={'MEAN(area)': f'area_C{level}'}))

    df = df_clinical.merge(df_area, on='participant_id', how='left')

    # ── T2w hyperintensity ────────────────────────────────────────────────────
    if 't2w_hyperintensity' not in df.columns and 'Myelopathy' in df.columns:
        df['t2w_hyperintensity'] = df['Myelopathy']
    df['t2w_hyperintensity'] = df['t2w_hyperintensity'].map(
        {0: 'no', 1: 'yes', '0': 'no', '1': 'yes', 'no': 'no', 'yes': 'yes'}
    )

    # ── exclude: surgery before baseline ─────────────────────────────────────
    if SURG_COL_BEFORE_BL in df.columns:
        df[SURG_COL_BEFORE_BL] = pd.to_datetime(df[SURG_COL_BEFORE_BL], errors='coerce')
        n_before = df['participant_id'].nunique()
        df = df[df[SURG_COL_BEFORE_BL].isna()].copy()
        n_after = df['participant_id'].nunique()
        print(f"Excluded {n_before - n_after} subjects with surgery before baseline. "
              f"Remaining: {n_after}")

    # ── therapeutic decision from exact surgery dates ─────────────────────────
    for col in [SURG_COL_BEFORE_6MTH, SURG_COL_BEFORE_12MTH]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if SURG_COL_BEFORE_6MTH in df.columns:
        n_with_surg_date = df[SURG_COL_BEFORE_6MTH].notna().sum()
        print(f"\n── Surgery diagnostic ──────────────────────────────────")
        print(f"  Subjects entering prepare_data           : {df['participant_id'].nunique()}")
        print(f"  Non-null {SURG_COL_BEFORE_6MTH}  : {n_with_surg_date}")
        df['therapeutic_decision'] = df[SURG_COL_BEFORE_6MTH].apply(
            lambda x: 'operative' if pd.notna(x) else 'conservative'
        )
        print(f"  therapeutic_decision=='operative' (re-derived): "
              f"{(df['therapeutic_decision'] == 'operative').sum()}")
        print(f"────────────────────────────────────────────────────────\n")
    else:
        print(f"Warning: {SURG_COL_BEFORE_6MTH} not found; "
              "using existing 'therapeutic_decision' column.")

    # ── parse assessment dates ────────────────────────────────────────────────
    for col in [DATE_COL_BL, DATE_COL_6MTH, DATE_COL_12MTH, anchor_col]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ── report subjects where anchor differs from BL assessment date ──────────
    if anchor_col in df.columns and DATE_COL_BL in df.columns:
        _diff = (df[DATE_COL_BL] - df[anchor_col]).dt.days
        _nonzero = df.loc[_diff != 0, ['participant_id']].copy()
        _nonzero[anchor_col]  = df.loc[_diff != 0, anchor_col].dt.date
        _nonzero[DATE_COL_BL] = df.loc[_diff != 0, DATE_COL_BL].dt.date
        _nonzero['diff_days'] = _diff[_diff != 0].values
        print(f"\nSubjects where {anchor_col} ≠ {DATE_COL_BL} (n={len(_nonzero)}):")
        print("  None" if _nonzero.empty else _nonzero.to_string(index=False))
        print()

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
            pid                = row['participant_id']
            t2w_hyperintensity = row.get('t2w_hyperintensity', np.nan)
            therapeutic_dec    = row.get('therapeutic_decision', np.nan)
            age                = pd.to_numeric(row.get('age', np.nan), errors='coerce')
            sex                = row.get('sex', 'unknown')
            max_stenosis       = row.get('maximum_stenosis', 'unknown')
            stenosis_type      = row.get('single_vs_multi_stenosis', 'unknown')
            area               = row.get(f'area_C{level}', np.nan)
            date_bl            = row.get(DATE_COL_BL, pd.NaT)
            date_anchor        = row.get(anchor_col, pd.NaT)

            surg_date_6mth = row.get(SURG_COL_BEFORE_6MTH, pd.NaT)
            surg_days = (surg_date_6mth - date_anchor).days if (
                pd.notna(date_anchor) and pd.notna(surg_date_6mth)) else np.nan

            common = dict(
                participant_id       = pid,
                t2w_hyperintensity   = t2w_hyperintensity,
                therapeutic_decision = therapeutic_dec,
                age                  = age,
                sex                  = sex,
                maximum_stenosis     = max_stenosis,
                stenosis             = stenosis_type,
                baseline_area        = area,
                surg_days            = surg_days,
            )

            # Baseline (time_6m=0, time_12m=0)
            score_bl = pd.to_numeric(row.get(col_bl, np.nan), errors='coerce')
            if pd.notna(score_bl):
                days_bl = (date_bl - date_anchor).days if (
                    pd.notna(date_bl) and pd.notna(date_anchor)) else np.nan
                rows.append({**common,
                             'time_days':  days_bl,
                             'time_6m':    0,
                             'time_12m':   0,
                             'time_label': 'Baseline',
                             'score':      float(score_bl)})

            # 6-month (time_6m=1, time_12m=0)
            if col_6mth in df.columns:
                score_6m = pd.to_numeric(row.get(col_6mth, np.nan), errors='coerce')
                date_6m  = row.get(DATE_COL_6MTH, pd.NaT)
                if pd.notna(score_6m):
                    days_6m = (date_6m - date_anchor).days if (
                        pd.notna(date_anchor) and pd.notna(date_6m)) else np.nan
                    rows.append({**common,
                                 'time_days':  days_6m,
                                 'time_6m':    1,
                                 'time_12m':   0,
                                 'time_label': '6-month',
                                 'score':      float(score_6m)})

            # 12-month (time_6m=0, time_12m=1)
            if sessions == 3 and col_12mth and col_12mth in df.columns:
                score_12m = pd.to_numeric(row.get(col_12mth, np.nan), errors='coerce')
                date_12m  = row.get(DATE_COL_12MTH, pd.NaT)
                if pd.notna(score_12m):
                    days_12m = (date_12m - date_anchor).days if (
                        pd.notna(date_anchor) and pd.notna(date_12m)) else np.nan
                    rows.append({**common,
                                 'time_days':  days_12m,
                                 'time_6m':    0,
                                 'time_12m':   1,
                                 'time_label': '12-month',
                                 'score':      float(score_12m)})

        if not rows:
            print(f"Warning: No data rows for {score_name}")
            continue

        df_long = pd.DataFrame(rows)
        df_long = df_long.dropna(subset=['time_6m', 'time_12m', 'score'])

        vc = df_long['participant_id'].value_counts()
        df_long = df_long[df_long['participant_id'].isin(vc[vc >= 2].index)].copy()

        n_operative    = df_long[df_long['therapeutic_decision'] == 'operative']['participant_id'].nunique()
        n_surg_days_ok = df_long[df_long['surg_days'].notna()]['participant_id'].nunique()
        print(f"\n{score_name}:")
        print(f"  Participants with ≥2 timepoints  : {df_long['participant_id'].nunique()}")
        print(f"  Total observations               : {len(df_long)}")
        print(f"  Operative subjects               : {n_operative}  "
              f"({(df_long['therapeutic_decision'] == 'operative').sum()} obs)")
        print(f"  Of which surg_days non-null      : {n_surg_days_ok}  "
              f"(these get surgery markers)")
        print(f"  T2w+ (t2w_hyperintensity)        : "
              f"{df_long[df_long['t2w_hyperintensity'] == 'yes']['participant_id'].nunique()} subjects")
        print(f"  Days range (BL→last visit)       : "
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
                re_formula='1',   # random intercept only; time is categorical
            )
            result = model.fit(method='lbfgs', maxiter=2000)
    except Exception as e:
        log_print(f"  ERROR fitting {score_name}: {e}", log_file)
        return None

    llf = result.llf
    k   = len(result.params)
    n   = len(df_model)
    aic = -2 * llf + 2 * k
    bic = -2 * llf + k * np.log(n)
    aic_str = f"{aic:.2f}" if not np.isnan(result.aic) else f"{aic:.2f} (manual)"
    bic_str = f"{bic:.2f}" if not np.isnan(result.bic) else f"{bic:.2f} (manual)"

    log_print(f"  Converged : {result.converged}", log_file)
    log_print(f"  AIC       : {aic_str}", log_file)
    log_print(f"  BIC       : {bic_str}", log_file)
    log_print(f"  LogLik    : {llf:.2f}", log_file)
    log_print(f"  k (params): {k}   n (obs): {n}", log_file)
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

    result._aic_manual = aic
    result._bic_manual = bic
    return result


def _prep_df(df_long, extra_terms, log_file=None):
    """Center continuous covariates and build model-ready DataFrame."""
    df_model = df_long.copy()

    df_model['t2w_hyperintensity'] = pd.Categorical(
        df_model['t2w_hyperintensity'], categories=['no', 'yes'])
    df_model['therapeutic_decision'] = pd.Categorical(
        df_model['therapeutic_decision'], categories=['conservative', 'operative'])

    age_num = pd.to_numeric(df_model['age'], errors='coerce')
    df_model['age_c'] = age_num - age_num.mean() if age_num.notna().any() else np.nan

    if df_model['baseline_area'].notna().any():
        df_model['area_c'] = df_model['baseline_area'] - df_model['baseline_area'].mean()

    cov_terms = []
    if 'age_c' in df_model.columns and df_model['age_c'].notna().any():
        cov_terms.append('age_c')
    if df_model['sex'].nunique() > 1 and 'unknown' not in df_model['sex'].values:
        cov_terms.append('C(sex, Treatment(reference="M"))')

    drop_cols = ['score', 'time_6m', 'time_12m', 't2w_hyperintensity', 'therapeutic_decision']
    df_model  = df_model.dropna(subset=drop_cols)
    return df_model, cov_terms


def fit_model_A(df_long, score_name, log_file=None):
    """Model A: stratified by T2w hyperintensity, separate effects at 6m and 12m."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model A – {score_name}  (stratification: t2w_hyperintensity)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(t2w_hyperintensity)',
        'time_6m',
        'time_12m',
        'C(t2w_hyperintensity):time_6m',
        'C(t2w_hyperintensity):time_12m',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


def fit_model_B(df_long, score_name, log_file=None):
    """Model B: stratified by therapeutic decision, separate effects at 6m and 12m."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model B – {score_name}  (stratification: therapeutic decision)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(therapeutic_decision)',
        'time_6m',
        'time_12m',
        'C(therapeutic_decision):time_6m',
        'C(therapeutic_decision):time_12m',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


def fit_model_C(df_long, score_name, log_file=None):
    """Model C: T2w × treatment interaction on intercept; shared time effects."""
    log_print(f"\n{'='*70}", log_file)
    log_print(f"Model C – {score_name}  "
              f"(t2w_hyperintensity × therapeutic decision; shared time)", log_file)
    log_print(f"{'='*70}", log_file)

    interaction_terms = [
        'C(t2w_hyperintensity)',
        'C(therapeutic_decision)',
        'time_6m',
        'time_12m',
        'C(t2w_hyperintensity):C(therapeutic_decision)',
    ]
    df_model, cov_terms = _prep_df(df_long, interaction_terms, log_file)
    formula = f"score ~ {' + '.join(interaction_terms + cov_terms)}"
    log_print(f"  Formula: {formula}", log_file)
    return _build_and_fit(df_model, formula, score_name, log_file), df_model


# ──────────────────────────────────────────────────────────────────────────────
# Prediction and CI
# ──────────────────────────────────────────────────────────────────────────────
def _compute_ci_at_timepoints(result, i_param_names, s6m_param_names, s12m_param_names,
                               present_labels):
    """
    95% CI for predicted values at BL, 6m, 12m using binary time dummies.

    Contrast matrix rows (one per timepoint):
      BL  (time_6m=0, time_12m=0): Intercept + i_params
      6m  (time_6m=1, time_12m=0): Intercept + time_6m + i_params + s6m_params
      12m (time_6m=0, time_12m=1): Intercept + time_12m + i_params + s12m_params

    Returns
    -------
    y_hat, ci_lo, ci_hi : np.ndarray of shape (len(present_labels),)
    """
    label_row   = {'Baseline': 0, '6-month': 1, '12-month': 2}
    param_names = result.params.index.tolist()
    param_idx   = {p: i for i, p in enumerate(param_names)}
    cov         = result.cov_params().values
    n_params    = len(param_names)

    C_full = np.zeros((3, n_params))
    C_full[:, param_idx['Intercept']] = 1.0
    if 'time_6m' in param_idx:
        C_full[1, param_idx['time_6m']] = 1.0
    if 'time_12m' in param_idx:
        C_full[2, param_idx['time_12m']] = 1.0
    for pname in i_param_names:
        if pname in param_idx:
            C_full[:, param_idx[pname]] = 1.0
    for pname in s6m_param_names:
        if pname in param_idx:
            C_full[1, param_idx[pname]] = 1.0
    for pname in s12m_param_names:
        if pname in param_idx:
            C_full[2, param_idx[pname]] = 1.0

    row_indices = [label_row[l] for l in present_labels if l in label_row]
    C     = C_full[row_indices]
    y_hat = C @ result.params.values
    var   = np.einsum('ij,jk,ik->i', C, cov, C)
    se    = np.sqrt(np.maximum(var, 0.0))
    return y_hat, y_hat - 1.96 * se, y_hat + 1.96 * se


# ──────────────────────────────────────────────────────────────────────────────
# Surgery markers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_surgery_on_line(ax, df_surg, y_bl, y_6m, x_6m_median,
                          color, alpha=0.7, markersize=7):
    """
    Place an 'x' for each surgical patient on the BL→6m line segment,
    at their actual surgery day (linear interpolation between BL and 6m values).
    """
    surg_rows = df_surg[df_surg['surg_days'].notna()].drop_duplicates('participant_id')
    if len(surg_rows) == 0 or x_6m_median <= 0:
        return
    for _, row in surg_rows.iterrows():
        sd = row['surg_days']
        if sd <= 0 or sd > 365:
            continue
        frac  = np.clip(sd / x_6m_median, 0.0, 1.0)
        y_pos = y_bl + frac * (y_6m - y_bl)
        ax.plot(sd, y_pos, 'x', color=color, alpha=alpha,
                markersize=markersize, markeredgewidth=1.5, zorder=7, linestyle='none')


# ──────────────────────────────────────────────────────────────────────────────
# Plotting – Models A, B, C
#
# Key design:
#   X-axis    : actual days (median days per timepoint label)
#   Points    : 3 predicted values (BL, 6m, 12m) connected by lines
#   CI        : 95% fixed-effect CI shown as error bars at each timepoint
#   Surgery   : 'x' interpolated on BL→6m line segment at each patient's surgery day
# ──────────────────────────────────────────────────────────────────────────────
def _get_timepoint_x(df_long):
    """Return (present_labels, x_pts, med_days) from df_long."""
    med_days       = df_long.groupby('time_label')['time_days'].median()
    present_labels = [l for l in TIME_LABELS if l in med_days.index]
    x_pts          = [med_days[l] for l in present_labels]
    pad            = max(x_pts) * 0.06 if x_pts else 30   # 6% padding on each side
    xlim           = (min(x_pts) - pad, max(x_pts) + pad)
    return present_labels, x_pts, med_days, xlim


def plot_session_day_distributions(df_long, score_name, outdir, anchor_col,
                                   log_file=None):
    """
    Diagnostic: distribution of actual days per categorical timepoint.

    Each column = one session label (Baseline, 6-month, 12-month).
    Points are jittered individual observations; box shows median/IQR.
    A well-behaved categorical time variable should show tight, non-overlapping
    distributions. Overlapping or very wide distributions suggest that treating
    time as categorical may poorly capture the actual timing variation.
    """
    mpl.rcParams['font.family'] = 'Arial'

    present_labels = [l for l in TIME_LABELS if l in df_long['time_label'].unique()]
    if not present_labels:
        return

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(6, 4))

    x_positions = list(range(len(present_labels)))

    for xi, lbl in zip(x_positions, present_labels):
        days = df_long.loc[df_long['time_label'] == lbl, 'time_days'].dropna().values

        # Box (25th–75th percentile)
        q25, q50, q75 = np.percentile(days, [25, 50, 75])
        iqr = q75 - q25
        w   = 0.25
        ax.add_patch(plt.Rectangle((xi - w, q25), 2*w, iqr,
                                   facecolor='lightgrey', edgecolor='black',
                                   linewidth=1.2, zorder=3))
        ax.hlines(q50, xi - w, xi + w, colors='black', linewidth=2, zorder=4)

        # Whiskers (1.5×IQR)
        lo = max(days.min(), q25 - 1.5*iqr)
        hi = min(days.max(), q75 + 1.5*iqr)
        ax.vlines(xi, lo, q25, colors='black', linewidth=1, zorder=3)
        ax.vlines(xi, q75, hi, colors='black', linewidth=1, zorder=3)

        # Jittered individual points
        jitter = rng.uniform(-0.18, 0.18, size=len(days))
        ax.scatter(xi + jitter, days, s=12, color='steelblue', alpha=0.5, zorder=5,
                   linewidths=0)

        # Annotate n, median, mean ± std
        mean = days.mean()
        std  = days.std()
        ax.text(xi, hi + 8,
                f'n={len(days)}\nmedian={q50:.0f}d\nmean={mean:.0f}±{std:.0f}d',
                ha='center', va='bottom', fontsize=8)

    _ylabel = 'Days from inclusion' if anchor_col == 'date_inclusion' else 'Days from baseline'
    ax.set_xticks(x_positions)
    ax.set_xticklabels(present_labels, fontsize=TICK_FONT_SIZE)
    ax.set_ylabel(_ylabel, fontsize=LABEL_FONT_SIZE)
    ax.set_xlim(-0.6, len(present_labels) - 0.4)
    ax.set_ylim(bottom=-5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} – actual days per session label', fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_cattime_session_days_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


def plot_model_A(df_long, result, score_name, cfg, outdir, log_file=None,
                 anchor_col='orthopedics_assessment_date_BL'):
    """LME trajectories stratified by T2w hyperintensity (categorical time)."""
    mpl.rcParams['font.family'] = 'Arial'

    groups = {
        'no':  {'color': COLOR_T2W_MINUS, 'label': 'T2w−',
                'i_params': [], 's6m_params': [], 's12m_params': []},
        'yes': {'color': COLOR_T2W_PLUS,  'label': 'T2w+',
                'i_params':   ['C(t2w_hyperintensity)[T.yes]'],
                's6m_params': ['C(t2w_hyperintensity)[T.yes]:time_6m'],
                's12m_params':['C(t2w_hyperintensity)[T.yes]:time_12m']},
    }

    present_labels, x_pts, med_days, xlim = _get_timepoint_x(df_long)
    fig, ax = plt.subplots(figsize=(6, 4))

    log_print(f"\n{score_name} – Model A  (T2w hyperintensity):", log_file)
    for myelo, ginfo in groups.items():
        gdf   = df_long[df_long['t2w_hyperintensity'] == myelo]
        color = ginfo['color']
        n     = gdf['participant_id'].nunique()
        log_print(f"  {ginfo['label']:<10s}: n = {n}", log_file)
        if n == 0 or result is None:
            continue

        y_hat, ci_lo, ci_hi = _compute_ci_at_timepoints(
            result, ginfo['i_params'], ginfo['s6m_params'], ginfo['s12m_params'],
            present_labels)
        ax.plot(x_pts, y_hat, color=color, linestyle='-', linewidth=LW_LME,
                label=f'{ginfo["label"]} (n={n})', zorder=6)
        ax.errorbar(x_pts, y_hat,
                    yerr=np.vstack([y_hat - ci_lo, ci_hi - y_hat]),
                    fmt='none', color=color, capsize=4, capthick=1.2,
                    linewidth=1.2, zorder=7)

        surg_df  = gdf[gdf['surg_days'].notna()].drop_duplicates('participant_id')
        x_6m_med = med_days.get('6-month', np.nan)
        if not np.isnan(x_6m_med) and len(y_hat) >= 2:
            _plot_surgery_on_line(ax, surg_df, y_hat[0], y_hat[1], x_6m_med, color)

    _xlabel = 'Days from inclusion' if anchor_col == 'date_inclusion' else 'Days from baseline'
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(x_pts)
    ax.set_xticklabels([f'{int(round(x))}' for x in x_pts], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(_xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    surg_handle = Line2D([0], [0], color='gray', marker='x', linestyle='none',
                         markersize=6, markeredgewidth=1.5, label='Surgery date')
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles + [surg_handle], labels_leg + ['Surgery date'],
              fontsize=TICK_FONT_SIZE, frameon=True, framealpha=0.85)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} stratified by T2w hyperintensity', fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_cattime_A_t2w_hyperintensity_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


def plot_model_B(df_long, result, score_name, cfg, outdir, log_file=None,
                 anchor_col='orthopedics_assessment_date_BL'):
    """LME trajectories stratified by therapeutic decision (categorical time)."""
    mpl.rcParams['font.family'] = 'Arial'

    groups = {
        'conservative': {'color': COLOR_CONSERVATIVE, 'label': 'Conservative',
                         'i_params': [], 's6m_params': [], 's12m_params': []},
        'operative':    {'color': COLOR_OPERATIVE,    'label': 'Operative (surgery)',
                         'i_params':   ['C(therapeutic_decision)[T.operative]'],
                         's6m_params': ['C(therapeutic_decision)[T.operative]:time_6m'],
                         's12m_params':['C(therapeutic_decision)[T.operative]:time_12m']},
    }

    present_labels, x_pts, med_days, xlim = _get_timepoint_x(df_long)
    fig, ax = plt.subplots(figsize=(6, 4))

    log_print(f"\n{score_name} – Model B  (therapeutic decision):", log_file)
    for td, ginfo in groups.items():
        gdf   = df_long[df_long['therapeutic_decision'] == td]
        color = ginfo['color']
        n     = gdf['participant_id'].nunique()
        log_print(f"  {ginfo['label']:<25s}: n = {n}", log_file)
        if n == 0 or result is None:
            continue

        y_hat, ci_lo, ci_hi = _compute_ci_at_timepoints(
            result, ginfo['i_params'], ginfo['s6m_params'], ginfo['s12m_params'],
            present_labels)
        ax.plot(x_pts, y_hat, color=color, linestyle='-', linewidth=LW_LME,
                label=f'{ginfo["label"]} (n={n})', zorder=6)
        ax.errorbar(x_pts, y_hat,
                    yerr=np.vstack([y_hat - ci_lo, ci_hi - y_hat]),
                    fmt='none', color=color, capsize=4, capthick=1.2,
                    linewidth=1.2, zorder=7)

        if td == 'operative':
            surg_df  = gdf[gdf['surg_days'].notna()].drop_duplicates('participant_id')
            x_6m_med = med_days.get('6-month', np.nan)
            if not np.isnan(x_6m_med) and len(y_hat) >= 2:
                _plot_surgery_on_line(ax, surg_df, y_hat[0], y_hat[1], x_6m_med, color)

    _xlabel = 'Days from inclusion' if anchor_col == 'date_inclusion' else 'Days from baseline'
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(x_pts)
    ax.set_xticklabels([f'{int(round(x))}' for x in x_pts], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(_xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    surg_handle = Line2D([0], [0], color=COLOR_OPERATIVE, marker='x', linestyle='none',
                         markersize=6, markeredgewidth=1.5, label='Surgery date')
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles + [surg_handle], labels_leg + ['Surgery date'],
              fontsize=TICK_FONT_SIZE, frameon=True, framealpha=0.85)

    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    ax.set_title(f'{score_name} stratified by therapeutic decision', fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_cattime_B_therapeutic_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)


def plot_model_C(df_long, result, score_name, cfg, outdir, log_file=None,
                 anchor_col='orthopedics_assessment_date_BL'):
    """
    Combined model: 4 groups (t2w_hyperintensity × treatment).
    Color = T2w status; linestyle = treatment. Shared time effects.
    """
    mpl.rcParams['font.family'] = 'Arial'

    # Model C has no time interactions — s6m_params and s12m_params are empty for all groups.
    group_spec = {
        ('no',  'conservative'): (COLOR_T2W_MINUS, '--', 'T2w− / Conservative',
                                  [], [], []),
        ('no',  'operative'):    (COLOR_T2W_MINUS, '-',  'T2w− / Operative',
                                  ['C(therapeutic_decision)[T.operative]'], [], []),
        ('yes', 'conservative'): (COLOR_T2W_PLUS,  '--', 'T2w+ / Conservative',
                                  ['C(t2w_hyperintensity)[T.yes]'], [], []),
        ('yes', 'operative'):    (COLOR_T2W_PLUS,  '-',  'T2w+ / Operative',
                                  ['C(t2w_hyperintensity)[T.yes]',
                                   'C(therapeutic_decision)[T.operative]',
                                   'C(t2w_hyperintensity)[T.yes]:C(therapeutic_decision)[T.operative]'],
                                  [], []),
    }

    present_labels, x_pts, med_days, xlim = _get_timepoint_x(df_long)
    fig, ax = plt.subplots(figsize=(6, 4))

    log_print(f"\n{score_name} – Model C  (T2w hyperintensity × therapeutic decision):", log_file)
    for (myelo, td), (color, ls, label, i_params, s6m_params, s12m_params) in group_spec.items():
        gdf = df_long[
            (df_long['t2w_hyperintensity'] == myelo) &
            (df_long['therapeutic_decision'] == td)
        ]
        n = gdf['participant_id'].nunique()
        log_print(f"  {label:<30s}: n = {n}", log_file)
        if n == 0 or result is None:
            continue

        y_hat, ci_lo, ci_hi = _compute_ci_at_timepoints(
            result, i_params, s6m_params, s12m_params, present_labels)
        ax.plot(x_pts, y_hat, color=color, linestyle=ls, linewidth=LW_LME,
                label=f'{label} (n={n})', zorder=6)
        ax.errorbar(x_pts, y_hat,
                    yerr=np.vstack([y_hat - ci_lo, ci_hi - y_hat]),
                    fmt='none', color=color, capsize=4, capthick=1.2,
                    linewidth=1.2, alpha=0.6, zorder=7)

        if td == 'operative':
            surg_df  = gdf[gdf['surg_days'].notna()].drop_duplicates('participant_id')
            x_6m_med = med_days.get('6-month', np.nan)
            if not np.isnan(x_6m_med) and len(y_hat) >= 2:
                _plot_surgery_on_line(ax, surg_df, y_hat[0], y_hat[1], x_6m_med, color)

    _xlabel = 'Days from inclusion' if anchor_col == 'date_inclusion' else 'Days from baseline'
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks(x_pts)
    ax.set_xticklabels([f'{int(round(x))}' for x in x_pts], fontsize=TICK_FONT_SIZE)
    ax.set_xlabel(_xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(cfg['y_label'], fontsize=LABEL_FONT_SIZE)
    if cfg['ylim']:
        ax.set_ylim(cfg['ylim'])

    c_handles = [
        Line2D([0], [0], color=COLOR_T2W_MINUS, lw=2, label='T2w−'),
        Line2D([0], [0], color=COLOR_T2W_PLUS,  lw=2, label='T2w+'),
    ]
    l_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Conservative'),
        Line2D([0], [0], color='black', lw=2, linestyle='-',  label='Operative'),
        Line2D([0], [0], color='gray',  marker='x', linestyle='none',
               markersize=6, markeredgewidth=1.5, label='Surgery date'),
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
    ax.set_title(f'{score_name} – T2w hyperintensity × therapeutic decision',
                 fontsize=TITLE_FONT_SIZE)
    plt.tight_layout()
    fname = os.path.join(outdir, f'lme_cattime_C_combined_{score_name}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log_print(f"Saved: {fname}", log_file)

    # ── Save data CSVs ────────────────────────────────────────────────────────
    obs_cols = ['participant_id', 'sex', 'age', 't2w_hyperintensity', 'therapeutic_decision',
                'surg_days', 'time_label', 'time_days', 'time_6m', 'time_12m', 'score']
    obs_csv = os.path.join(outdir, f'lme_cattime_C_combined_{score_name}_observations.csv')
    df_long[[c for c in obs_cols if c in df_long.columns]].rename(
        columns={'score': f'{score_name.lower()}_score'}
    ).to_csv(obs_csv, index=False)
    log_print(f"Saved: {obs_csv}", log_file)

    if result is not None:
        fitted_rows = []
        for (myelo, td), (_, ls, label, i_params, s6m_params, s12m_params) in group_spec.items():
            y_hat, ci_lo, ci_hi = _compute_ci_at_timepoints(
                result, i_params, s6m_params, s12m_params, present_labels)
            for lbl, x, yh, yl, yu in zip(present_labels, x_pts, y_hat, ci_lo, ci_hi):
                fitted_rows.append({
                    'group':                label,
                    't2w_hyperintensity':   myelo,
                    'therapeutic_decision': td,
                    'time_label':           lbl,
                    'days':                 round(x, 2),
                    'y_fitted':             round(yh, 4),
                    'y_ci_lower':           round(yl, 4),
                    'y_ci_upper':           round(yu, 4),
                })
        fitted_csv = os.path.join(outdir, f'lme_cattime_C_combined_{score_name}_fitted.csv')
        pd.DataFrame(fitted_rows).to_csv(fitted_csv, index=False)
        log_print(f"Saved: {fitted_csv}", log_file)


# ──────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ──────────────────────────────────────────────────────────────────────────────
def save_results_csv(results_dict, outdir):
    rows = []
    for (score, model_lbl), res in results_dict.items():
        if res is None:
            continue
        aic = getattr(res, '_aic_manual', res.aic)
        bic = getattr(res, '_bic_manual', res.bic)
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
                'aic':         aic,
                'bic':         bic,
                'log_lik':     res.llf,
                'converged':   res.converged,
            })
    if rows:
        df_out = pd.DataFrame(rows)
        fname  = os.path.join(outdir, 'lme_cattime_results.csv')
        df_out.to_csv(fname, index=False)
        print(f"\nAll model results saved to: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = get_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Reading clinical data...")
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

    print("\nPreparing longitudinal data...")
    long_data_dict = prepare_data(
        df_clinical, df_morphometrics,
        level=args.level, sessions=args.sessions,
        anchor_col=args.anchor_col,
    )

    if not long_data_dict:
        print("ERROR: No data available for analysis.")
        return

    all_results = {}
    log_path    = os.path.join(args.outdir, 'lme_cattime_analysis.log')

    with open(log_path, 'w') as log_file:
        log_print("LME Categorical-Time Analysis Log", log_file)
        log_print(f"Sessions   : {args.sessions}", log_file)
        log_print(f"SC level   : C{args.level}", log_file)
        log_print(f"Anchor col : {args.anchor_col}", log_file)
        log_print(f"Output     : {args.outdir}\n", log_file)

        for score_name, df_long in long_data_dict.items():
            cfg = CLINICAL_SCORES[score_name]

            # ── Diagnostic: day distributions per session ────────────────
            plot_session_day_distributions(df_long, score_name, args.outdir,
                                           args.anchor_col, log_file)

            result_A, df_A = fit_model_A(df_long, score_name, log_file)
            all_results[(score_name, 'A_t2w_hyperintensity')] = result_A
            if result_A is not None:
                plot_model_A(df_long, result_A, score_name, cfg, args.outdir, log_file,
                             anchor_col=args.anchor_col)
            else:
                log_print(f"  Model A failed for {score_name}", log_file)

            result_B, df_B = fit_model_B(df_long, score_name, log_file)
            all_results[(score_name, 'B_therapeutic')] = result_B
            if result_B is not None:
                plot_model_B(df_long, result_B, score_name, cfg, args.outdir, log_file,
                             anchor_col=args.anchor_col)
            else:
                log_print(f"  Model B failed for {score_name}", log_file)

            result_C, df_C = fit_model_C(df_long, score_name, log_file)
            all_results[(score_name, 'C_combined')] = result_C
            if result_C is not None:
                plot_model_C(df_long, result_C, score_name, cfg, args.outdir, log_file,
                             anchor_col=args.anchor_col)
            else:
                log_print(f"  Model C failed for {score_name}", log_file)

    save_results_csv(all_results, args.outdir)

    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"Log    : {log_path}")
    print(f"Output : {args.outdir}")
    print('='*60)


if __name__ == '__main__':
    main()
