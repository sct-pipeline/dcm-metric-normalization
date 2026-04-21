#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Assessing the Contribution of Age and Sex to Variance in Spinal Cord
Morphometric Measures — A Linear Mixed Model Approach for Longitudinal Data.

Methodological protocol (see associated documentation):
  Step 1 - Null model (REML): Intraclass Correlation Coefficient (ICC)
  Step 2 - Covariate models (ML): Likelihood Ratio Tests for sex and age
  Step 3 - Time trajectory models with random slopes
  Step 4 - Extract ICC, LRT p-values, AIC/BIC, coefficients ± SE, R²m / R²c
  Step 5 - Decision rule: retain covariate if LRT p < 0.05 AND ΔAIC > 2

Metrics analysed (native space, averaged across C2–C7 per subject per timepoint):
  MEAN(area), MEAN(diameter_AP), MEAN(diameter_RL), MEAN(eccentricity), MEAN(solidity)
  + aSCOR if available.

Usage:
    python lmm_age_sex_variance.py \\
        --data-dir /path/to/timepoint_data \\
        --participants-file /path/to/participants.tsv \\
        --output-dir /path/to/output \\
        --structure cord \\
        --timepoints M0 M6 M12 M24 M36 M48 M60

Authors: Kahina Baouche, 2026
"""

import os
import re
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

# ============================================================================
# Constants
# ============================================================================

TIMEPOINT_MONTHS = {
    'M0': 0, 'M3': 3, 'M6': 6, 'M12': 12,
    'M24': 24, 'M36': 36, 'M48': 48, 'M60': 60,
}

METRICS_NATIVE = [
    'MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)',
    'MEAN(eccentricity)', 'MEAN(solidity)',
]

METRIC_LABELS = {
    'MEAN(area)':         'CSA (mm²)',
    'MEAN(diameter_AP)':  'AP Diameter (mm)',
    'MEAN(diameter_RL)':  'RL Diameter (mm)',
    'MEAN(eccentricity)': 'Eccentricity (a.u.)',
    'MEAN(solidity)':     'Solidity (%)',
    'aSCOR':              'aSCOR (ratio)',
}


def safe_col(metric: str) -> str:
    """Convert metric name to a statsmodels-safe column name."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', metric).strip('_')


# ============================================================================
# Data loading and preparation
# ============================================================================

def _extract_subject(filename) -> str | None:
    m = re.search(r'sub-\d+', str(filename))
    return m.group(0) if m else None


def load_longitudinal_perlevel(data_dir: str, structure: str,
                                timepoints: list[str]) -> pd.DataFrame:
    """
    Load and concatenate per-level CSV files for all timepoints.
    Filters to cervical levels C2–C7 (VertLevel 2–7).
    """
    frames = []
    for tp in timepoints:
        f = Path(data_dir) / f'T2w_ax_{structure}_metrics_perlevel_{tp}_data.csv'
        if not f.exists():
            print(f"  Warning: {f.name} not found, skipping")
            continue
        df = pd.read_csv(f)
        if 'subject' not in df.columns:
            col = 'Filename_sc' if 'Filename_sc' in df.columns else 'Filename'
            df['subject'] = df[col].apply(_extract_subject)
        df['timepoint'] = tp
        df = df[(df['VertLevel'] >= 2) & (df['VertLevel'] <= 7)]
        frames.append(df)
    if not frames:
        raise ValueError(f"No perlevel CSV files found for structure '{structure}'")
    return pd.concat(frames, ignore_index=True)


def prepare_lmm_data(df_perlevel: pd.DataFrame, participants_file: str,
                     timepoints: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Aggregate per-level data to per-subject-per-timepoint (mean across C2–C7),
    merge with participants.tsv for age/sex, and add derived columns.

    Returns
    -------
    df_lmm : pd.DataFrame
        One row per subject × timepoint, with metric columns and demographics.
    metric_cols : list[str]
        Original metric column names present in the data.
    """
    # Identify metric columns
    metric_cols = [c for c in METRICS_NATIVE if c in df_perlevel.columns]
    if 'aSCOR' in df_perlevel.columns:
        metric_cols.append('aSCOR')

    # Average across vertebral levels
    df_agg = (
        df_perlevel
        .groupby(['subject', 'timepoint'])[metric_cols]
        .mean()
        .reset_index()
    )

    # Add time in months + centered time
    df_agg['time_months'] = df_agg['timepoint'].map(TIMEPOINT_MONTHS)
    time_mean = df_agg['time_months'].mean()
    df_agg['time_c'] = df_agg['time_months'] - time_mean

    # Load participants
    df_part = pd.read_csv(participants_file, sep='\t')
    id_col = 'participant_id' if 'participant_id' in df_part.columns else 'subject'
    df_part = df_part.rename(columns={id_col: 'subject'})
    keep = ['subject'] + [c for c in ['age', 'sex', 'height', 'weight']
                          if c in df_part.columns]
    df_lmm = df_agg.merge(df_part[keep], on='subject', how='left')

    # Encode sex (M=1, F=0)
    if 'sex' in df_lmm.columns:
        df_lmm['sex_bin'] = df_lmm['sex'].map({'M': 1, 'F': 0, 'm': 1, 'f': 0})

    # Center age
    if 'age' in df_lmm.columns:
        age_mean = df_lmm['age'].mean()
        df_lmm['age_c'] = df_lmm['age'] - age_mean
        print(f"  Age mean used for centering: {age_mean:.1f} years")

    # Add safe formula-compatible column names
    for m in metric_cols:
        df_lmm[safe_col(m)] = df_lmm[m]

    n_subj = df_lmm['subject'].nunique()
    n_obs = len(df_lmm)
    print(f"  LMM dataset: {n_obs} observations, {n_subj} subjects, "
          f"{len(metric_cols)} metrics, {len(timepoints)} timepoints")
    return df_lmm, metric_cols


# ============================================================================
# LMM fitting helpers
# ============================================================================

def _fit(formula: str, data: pd.DataFrame, safe_metric: str,
         reml: bool, re_formula: str | None = None):
    """
    Fit a MixedLM via statsmodels formula API.
    Returns the fitted result, or None on convergence failure.
    """
    df = data.dropna(subset=[safe_metric, 'subject']).copy()
    if df['subject'].nunique() < 5:
        return None
    kwargs: dict = {'groups': df['subject']}
    if re_formula:
        kwargs['re_formula'] = re_formula
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = smf.mixedlm(formula, df, **kwargs)
            result = model.fit(reml=reml, method='lbfgs', maxiter=800)
        return result
    except Exception:
        return None


def compute_icc(result) -> tuple[float, float, float]:
    """
    ICC = σ²_between / (σ²_between + σ²_residual)
    from a REML null model.
    """
    try:
        sigma2_b = float(result.cov_re.iloc[0, 0])
        sigma2_w = float(result.scale)
        icc = sigma2_b / (sigma2_b + sigma2_w)
        return icc, sigma2_b, sigma2_w
    except Exception:
        return np.nan, np.nan, np.nan


def lrt(null_result, full_result) -> tuple[float, int, float, float, float]:
    """
    Likelihood Ratio Test: 2*(loglik_full - loglik_null) ~ χ²(df_diff).

    Returns
    -------
    lr_stat, df_diff, p_value, delta_aic, delta_bic
        where delta_aic = AIC_null - AIC_full  (positive → full model is better)
    """
    if full_result is None or null_result is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    if np.isnan(full_result.llf) or np.isnan(null_result.llf):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    lr_stat = 2.0 * (full_result.llf - null_result.llf)
    df_diff = max(len(full_result.fe_params) - len(null_result.fe_params), 1)
    p_value = float(stats.chi2.sf(lr_stat, df=df_diff))
    delta_aic = null_result.aic - full_result.aic   # positive = improvement
    delta_bic = null_result.bic - full_result.bic
    return lr_stat, df_diff, p_value, delta_aic, delta_bic


def compute_r2(result) -> tuple[float, float]:
    """
    Marginal R² (fixed effects only) and conditional R² (fixed + random) following Nakagawa & Schielzeth (2013).
    """
    try:
        X = result.model.exog
        beta = result.fe_params.values
        fixed_pred = X @ beta
        sigma2_f = float(np.var(fixed_pred, ddof=1))
        sigma2_r = float(result.cov_re.iloc[0, 0])
        sigma2_e = float(result.scale)
        total = sigma2_f + sigma2_r + sigma2_e
        return sigma2_f / total, (sigma2_f + sigma2_r) / total
    except Exception:
        return np.nan, np.nan


def _coef(result, param: str) -> tuple[float, float, float]:
    """Return (coefficient, SE, p-value) for a named parameter."""
    if result is None:
        return np.nan, np.nan, np.nan
    return (
        float(result.params.get(param, np.nan)),
        float(result.bse.get(param, np.nan)),
        float(result.pvalues.get(param, np.nan)),
    )


# ============================================================================
# Per-metric analysis
# ============================================================================

def analyze_metric(metric: str, safe_metric: str,
                   data: pd.DataFrame, output_dir: str) -> dict:
    """
    Run the full 5-step LMM protocol for a single metric.
    Returns a flat dict of all key quantities for the summary table.
    """
    label = METRIC_LABELS.get(metric, metric)
    sep = '─' * 64
    print(f"\n{sep}\n  {label}\n{sep}")

    # Determine which covariates are available so that all models (null and full) are fitted on the exact same dataset.
    _has_sex = 'sex_bin' in data.columns and data['sex_bin'].notna().sum() > 0
    _has_age = 'age_c' in data.columns and data['age_c'].notna().sum() > 0
    drop_cols = [safe_metric]
    if _has_sex:
        drop_cols.append('sex_bin')
    if _has_age:
        drop_cols.append('age_c')
    df = data.dropna(subset=drop_cols).copy()

    res: dict = {
        'metric': metric,
        'label': label,
        'n_subjects': df['subject'].nunique(),
        'n_observations': len(df),
    }

    has_sex = _has_sex and df['sex_bin'].notna().sum() > 0
    has_age = _has_age and df['age_c'].notna().sum() > 0
    has_time = 'time_c' in df.columns and df['time_c'].nunique() > 1

    # ── Step 1 ─ Null model (REML) → ICC ─────────────────────────────────
    r0_reml = _fit(f"{safe_metric} ~ 1", df, safe_metric, reml=True)
    if r0_reml is None:
        print("  ERROR: null REML model failed — skipping metric")
        return res
    icc, sigma2_b, sigma2_w = compute_icc(r0_reml)
    res.update({'icc': round(icc, 4),
                'sigma2_between': round(sigma2_b, 6),
                'sigma2_within': round(sigma2_w, 6)})
    print(f"  ICC = {icc:.3f}  (σ²_between={sigma2_b:.4f}, σ²_within={sigma2_w:.4f})")

    # ── Step 2 ─ Covariate models (ML) → LRT ─────────────────────────────
    r0_ml = _fit(f"{safe_metric} ~ 1", df, safe_metric, reml=False)
    if r0_ml is None:
        print("  ERROR: null ML model failed — skipping LRT")
        return res

    r1 = r2 = r3 = None

    # Sex only
    if has_sex:
        r1 = _fit(f"{safe_metric} ~ sex_bin", df, safe_metric, reml=False)
        stat, dfd, p, daic, dbic = lrt(r0_ml, r1)
        coef, se, pv = _coef(r1, 'sex_bin')
        res.update({'lrt_sex_chi2': round(stat, 3), 'lrt_sex_df': dfd,
                    'lrt_sex_p': round(p, 4), 'delta_aic_sex': round(daic, 3),
                    'delta_bic_sex': round(dbic, 3),
                    'sex_coef': round(coef, 4), 'sex_se': round(se, 4),
                    'sex_p': round(pv, 4)})
        print(f"  LRT sex:     χ²({dfd})={stat:.2f}, p={p:.4f}, "
              f"ΔAIC={daic:.2f}, β={coef:.4f}±{se:.4f}")

    # Age only
    if has_age:
        r2 = _fit(f"{safe_metric} ~ age_c", df, safe_metric, reml=False)
        stat, dfd, p, daic, dbic = lrt(r0_ml, r2)
        coef, se, pv = _coef(r2, 'age_c')
        res.update({'lrt_age_chi2': round(stat, 3), 'lrt_age_df': dfd,
                    'lrt_age_p': round(p, 4), 'delta_aic_age': round(daic, 3),
                    'delta_bic_age': round(dbic, 3),
                    'age_coef': round(coef, 4), 'age_se': round(se, 4),
                    'age_p': round(pv, 4)})
        print(f"  LRT age:     χ²({dfd})={stat:.2f}, p={p:.4f}, "
              f"ΔAIC={daic:.2f}, β={coef:.4f}±{se:.4f}")

    # Sex + Age
    if has_sex and has_age:
        r3 = _fit(f"{safe_metric} ~ sex_bin + age_c", df, safe_metric, reml=False)
        stat, dfd, p, daic, dbic = lrt(r0_ml, r3)
        res.update({'lrt_both_chi2': round(stat, 3), 'lrt_both_p': round(p, 4),
                    'delta_aic_both': round(daic, 3)})
        print(f"  LRT sex+age: χ²({dfd})={stat:.2f}, p={p:.4f}, ΔAIC={daic:.2f}")

        # Incremental tests
        if r2 is not None and r3 is not None:
            _, dfd_s, p_s, _, _ = lrt(r2, r3)
            res['lrt_sex_over_age_p'] = round(p_s, 4)
            print(f"    sex | age: χ²({dfd_s}), p={p_s:.4f}")
        if r1 is not None and r3 is not None:
            _, dfd_a, p_a, _, _ = lrt(r1, r3)
            res['lrt_age_over_sex_p'] = round(p_a, 4)
            print(f"    age | sex: χ²({dfd_a}), p={p_a:.4f}")

    # R² from sex+age model
    if has_sex and has_age:
        best_formula = f"{safe_metric} ~ sex_bin + age_c"
    elif has_age:
        best_formula = f"{safe_metric} ~ age_c"
    elif has_sex:
        best_formula = f"{safe_metric} ~ sex_bin"
    else:
        best_formula = f"{safe_metric} ~ 1"

    r_best_reml = _fit(best_formula, df, safe_metric, reml=True)
    if r_best_reml is not None:
        r2m, r2c = compute_r2(r_best_reml)
        res.update({'r2_marginal': round(r2m, 4), 'r2_conditional': round(r2c, 4)})
        print(f"  R²m = {r2m:.3f}  |  R²c = {r2c:.3f}")

    # ── Step 3 ─ Time trajectory models ──────────────────────────────────
    if has_time:
        print("  Step 3: Time trajectory models")

        base_parts = []
        if has_age:
            base_parts.append('age_c')
        if has_sex:
            base_parts.append('sex_bin')
        base_str = ' + '.join(base_parts) if base_parts else '1'

        # M4: random slope for time
        f4 = f"{safe_metric} ~ {base_str} + time_c"
        r4 = _fit(f4, df, safe_metric, reml=False, re_formula='~time_c')
        if r4 is None:
            print("    (random slope failed → falling back to random intercept)")
            r4 = _fit(f4, df, safe_metric, reml=False)
            res['time_model_random_slope'] = False
        else:
            res['time_model_random_slope'] = True

        if r4 is not None:
            t_coef, t_se, t_p = _coef(r4, 'time_c')
            res.update({'time_coef': round(t_coef, 6), 'time_se': round(t_se, 6),
                        'time_p': round(t_p, 4)})
            print(f"    time: β={t_coef:.4f} ± {t_se:.4f}, p={t_p:.4f}")

            # Confidence intervals for time coefficient
            try:
                ci = r4.conf_int()
                if 'time_c' in ci.index:
                    res['time_ci_low'] = round(float(ci.loc['time_c', 0]), 6)
                    res['time_ci_high'] = round(float(ci.loc['time_c', 1]), 6)
            except Exception:
                pass

        # M5: sex × time interaction
        if has_sex and r4 is not None:
            f5_base = 'age_c + ' if has_age else ''
            f5 = f"{safe_metric} ~ {f5_base}sex_bin * time_c"
            r5 = _fit(f5, df, safe_metric, reml=False, re_formula='~time_c')
            if r5 is None:
                r5 = _fit(f5, df, safe_metric, reml=False)
            stat, dfd, p, daic, _ = lrt(r4, r5)
            res.update({'lrt_sex_x_time_chi2': round(stat, 3),
                        'lrt_sex_x_time_p': round(p, 4),
                        'delta_aic_sex_x_time': round(daic, 3)})
            print(f"    LRT sex×time: χ²({dfd})={stat:.2f}, p={p:.4f}, ΔAIC={daic:.2f}")

        # M6: age × time interaction
        if has_age and r4 is not None:
            f6_base = 'sex_bin + ' if has_sex else ''
            f6 = f"{safe_metric} ~ {f6_base}age_c * time_c"
            r6 = _fit(f6, df, safe_metric, reml=False, re_formula='~time_c')
            if r6 is None:
                r6 = _fit(f6, df, safe_metric, reml=False)
            stat, dfd, p, daic, _ = lrt(r4, r6)
            res.update({'lrt_age_x_time_chi2': round(stat, 3),
                        'lrt_age_x_time_p': round(p, 4),
                        'delta_aic_age_x_time': round(daic, 3)})
            print(f"    LRT age×time: χ²({dfd})={stat:.2f}, p={p:.4f}, ΔAIC={daic:.2f}")

    return res


# ============================================================================
# Output: summary table + plots
# ============================================================================

def save_summary_table(all_results: list[dict], output_dir: str) -> pd.DataFrame:
    """Save CSV summary and print decision-rule table to console."""
    df = pd.DataFrame(all_results)

    out_path = Path(output_dir) / 'lmm_age_sex_summary.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSummary CSV saved: {out_path}")

    # Console report
    print("\n" + "=" * 80)
    print("SUMMARY — Age & Sex Contribution to Spinal Cord Morphometric Variance")
    print("=" * 80)
    display_cols = [
        'label', 'n_subjects', 'n_observations',
        'icc',
        'lrt_sex_p', 'delta_aic_sex',
        'lrt_age_p', 'delta_aic_age',
        'r2_marginal', 'r2_conditional',
        'time_coef', 'time_p',
        'lrt_sex_x_time_p', 'lrt_age_x_time_p',
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    # Decision rule
    print("\n" + "=" * 80)
    print("DECISION RULE  (retain if LRT p < 0.05 AND ΔAIC > 2)")
    print("=" * 80)
    for _, row in df.iterrows():
        decisions = []
        for cov, p_col, daic_col in [
            ('sex', 'lrt_sex_p', 'delta_aic_sex'),
            ('age', 'lrt_age_p', 'delta_aic_age'),
        ]:
            p = row.get(p_col, np.nan)
            daic = row.get(daic_col, np.nan)
            if not (np.isnan(p) or np.isnan(daic)):
                retain = (p < 0.05) and (daic > 2)
                marker = '✓ RETAIN' if retain else '✗ omit  '
                decisions.append(f"{cov}: {marker} (p={p:.3f}, ΔAIC={daic:.2f})")
        print(f"  {row.get('label', '?'):30s}  {'  |  '.join(decisions)}")

    return df


def plot_icc(all_results: list[dict], output_dir: str) -> None:
    """Bar chart of ICC values across metrics."""
    df = pd.DataFrame(all_results)
    if 'icc' not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#1565C0' if v > 0.5 else '#F57C00' for v in df['icc']]
    bars = ax.bar(range(len(df)), df['icc'], color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.8)
    ax.axhline(0.5, color='#D32F2F', linestyle='--', linewidth=1.2,
               label='ICC = 0.50', alpha=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['label'], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('ICC (between-subject variance proportion)', fontsize=11)
    ax.set_title('Intraclass Correlation Coefficients\n'
                 'Proportion of variance attributable to stable individual differences',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    for bar, val in zip(bars, df['icc']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    out = Path(output_dir) / 'lmm_icc.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ICC plot saved: {out}")


def plot_lrt_pvalues(all_results: list[dict], output_dir: str) -> None:
    """Dot plot of LRT p-values for sex and age across metrics."""
    df = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    kw = dict(s=90, zorder=3)
    if 'lrt_sex_p' in df.columns:
        ax.scatter(x - 0.12, df['lrt_sex_p'], label='Sex', marker='o',
                   color='#E91E63', **kw)
    if 'lrt_age_p' in df.columns:
        ax.scatter(x + 0.12, df['lrt_age_p'], label='Age', marker='s',
                   color='#1565C0', **kw)
    ax.axhline(0.05, color='#D32F2F', linestyle='--', linewidth=1.3,
               label='p = 0.05', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('LRT p-value (log scale)', fontsize=11)
    ax.set_title('Likelihood Ratio Tests — Age and Sex as Fixed Effects',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    out = Path(output_dir) / 'lmm_lrt_pvalues.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LRT p-value plot saved: {out}")


def plot_coefficients(all_results: list[dict], output_dir: str) -> None:
    """Forest plot of sex and age coefficients with ±SE bars."""
    df = pd.DataFrame(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (param, col_coef, col_se, title, color) in zip(axes, [
        ('sex_bin', 'sex_coef', 'sex_se', 'Sex effect (M vs F)', '#E91E63'),
        ('age_c',  'age_coef', 'age_se', 'Age effect (per year, centered)', '#1565C0'),
    ]):
        if col_coef not in df.columns:
            ax.set_visible(False)
            continue
        y = np.arange(len(df))
        coef = df[col_coef].values
        se   = df.get(col_se, pd.Series(np.zeros(len(df)))).values
        ax.barh(y, coef, xerr=se, color=color, alpha=0.75,
                height=0.5, ecolor='black', capsize=4)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(df['label'], fontsize=10)
        ax.set_xlabel('Coefficient ± SE', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')

    plt.suptitle('Fixed-Effect Estimates (random-intercept model)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = Path(output_dir) / 'lmm_coefficients.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Coefficient forest plot saved: {out}")


# ============================================================================
# CLI
# ============================================================================

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing T2w_ax_{structure}_metrics_perlevel_{tp}_data.csv files')
    parser.add_argument('--participants-file', required=True,
                        help='Path to participants.tsv (must contain participant_id, age, sex)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save results')
    parser.add_argument('--structure', default='cord',
                        choices=['cord', 'canal', 'aSCOR'],
                        help='Structure to analyze (default: cord)')
    parser.add_argument('--timepoints', nargs='+',
                        default=['M0', 'M6', 'M12', 'M24', 'M36', 'M48', 'M60'],
                        help='Timepoints to include (default: all)')
    parser.add_argument('--metrics', nargs='+',
                        help='Specific metrics to analyze (default: all available)')
    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("LMM Analysis: Age & Sex Contribution to Morphometric Variance")
    print("=" * 70)
    print(f"Structure:         {args.structure}")
    print(f"Timepoints:        {', '.join(args.timepoints)}")
    print(f"Data directory:    {args.data_dir}")
    print(f"Participants file: {args.participants_file}")
    print(f"Output directory:  {args.output_dir}")

    # Load data
    print("\nLoading data...")
    df_perlevel = load_longitudinal_perlevel(
        args.data_dir, args.structure, args.timepoints)
    df, metric_cols = prepare_lmm_data(
        df_perlevel, args.participants_file, args.timepoints)

    if args.metrics:
        metric_cols = [m for m in args.metrics if m in metric_cols]
    if args.structure == 'aSCOR':
        metric_cols = [m for m in metric_cols if m == 'aSCOR']

    if not metric_cols:
        print("ERROR: No matching metrics found in the data. Exiting.")
        sys.exit(1)

    print(f"\nMetrics: {', '.join(metric_cols)}")

    # Run per-metric analysis
    all_results = []
    for metric in metric_cols:
        s_col = safe_col(metric)
        if s_col not in df.columns:
            print(f"Warning: {metric} not in prepared data, skipping")
            continue
        result = analyze_metric(metric, s_col, df, args.output_dir)
        all_results.append(result)

    # Save outputs
    if all_results:
        save_summary_table(all_results, args.output_dir)
        plot_icc(all_results, args.output_dir)
        plot_lrt_pvalues(all_results, args.output_dir)
        plot_coefficients(all_results, args.output_dir)

    print(f"\n{'='*70}")
    print(f"Done. Results saved in: {args.output_dir}")
    print('=' * 70)


if __name__ == '__main__':
    main()
