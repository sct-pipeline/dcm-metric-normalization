#!/usr/bin/env python
#
# Plot correlation between a delta metric and ΔmJOA after surgery
#
# Hypothesis: subjects with greater change in a spinal cord metric after
# decompression surgery also show greater neurological recovery (ΔmJOA).
#
# Inputs:
#   - metric_delta CSV: delta metric per subject per timepoint
#     (delta column = followup - baseline; positive = improvement)
#   - mjoa_by_timepoint.csv: absolute mJOA score per subject per timepoint
#     (ΔmJOA computed here as score - BL score; positive = improvement)
#   - participants.tsv: with therapeutic_decision column
#
# Outputs (prefixed with metric slug, e.g. "delta_csa_" or "delta_ap_diam_"):
#   - <prefix>mjoa_merged.csv
#   - <prefix>correlation_summary.csv
#   - <prefix>mjoa_per_timepoint.png
#   - <prefix>mjoa_pooled.png
#   - <prefix>lmm_results.txt
#   - <prefix>lmm_fixed_effects.png
#   - <prefix>lmm_slopes_by_group.png
#
# Example usage (CSA):
#   python plot_delta_csa_mjoa_correlation.py \
#       -metric_delta  results/csa_delta_from_baseline.csv \
#       -metric_col    delta_area \
#       -metric_label  "ΔCSA (mm²)" \
#       -mjoa          results/mjoa_by_timepoint.csv \
#       -participants  data/dcm-zurich/participants.tsv \
#       -o             results/2026-04-07_analysis
#
# Example usage (AP diameter):
#   python plot_delta_csa_mjoa_correlation.py \
#       -metric_delta  results/ap_diam_delta_from_baseline.csv \
#       -metric_col    delta_ap_diam \
#       -metric_label  "ΔAP diameter (mm)" \
#       -mjoa          results/mjoa_by_timepoint.csv \
#       -participants  data/dcm-zurich/participants.tsv \
#       -o             results/2026-04-07_analysis
#
# Author: Kahina Baouche

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

LABELS_FONT_SIZE = 13
TICKS_FONT_SIZE = 11
TITLE_FONT_SIZE = 14

POST_SURGERY_MONTHS = [6, 12, 24, 60]
TIMEPOINT_LABELS = {6: 'M6', 12: 'M12', 24: 'M24', 60: 'M60'}
TIMEPOINT_COLORS = {6: '#e6194b', 12: '#3cb44b', 24: '#4363d8', 60: '#f58231'}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Correlate a delta imaging metric and ΔmJOA longitudinal trajectories")
    parser.add_argument('-metric_delta', required=True,
                        help="CSV with delta metric per subject per timepoint "
                             "(must contain columns: subject, months, <metric_col>)")
    parser.add_argument('-metric_col', required=True,
                        help="Column name in metric_delta CSV that contains the delta value "
                             "(e.g. delta_area, delta_ap_diam)")
    parser.add_argument('-metric_label', required=True,
                        help="Human-readable label for plots (e.g. 'ΔCSA (mm²)', 'ΔAP diameter (mm)')")
    parser.add_argument('-mjoa', required=True,
                        help="mjoa_by_timepoint.csv produced by plot_mjoa_longitudinal.py")
    parser.add_argument('-participants', required=True,
                        help="participants.tsv with therapeutic_decision column")
    parser.add_argument('-o', required=True,
                        help="Output directory")
    return parser


def _slug(label):
    """Convert a metric label to a safe filename prefix (e.g. 'ΔCSA (mm²)' → 'delta_csa_mm2_')."""
    s = label.replace('Δ', 'delta_').replace('δ', 'delta_').replace('²', '2').replace('³', '3')
    s = re.sub(r'[^\w]+', '_', s).strip('_').lower()
    return s + '_'


def compute_delta_mjoa(df_mjoa):
    """Return long-format dataframe with delta_mjoa = score - BL score per subject."""
    baselines = (
        df_mjoa[df_mjoa['months'] == 0]
        .rename(columns={'mjoa_score': 'baseline_mjoa'})
        [['subject', 'baseline_mjoa']]
    )
    df = df_mjoa.merge(baselines, on='subject', how='inner')
    df['delta_mjoa'] = df['mjoa_score'] - df['baseline_mjoa']
    return df


def merge_deltas(df_metric, metric_col, df_mjoa_delta, participants_df, slug, output_dir):
    """Merge delta metric and ΔmJOA on subject × months, add surgical status."""
    if metric_col not in df_metric.columns:
        raise ValueError(f"Column '{metric_col}' not found in metric CSV. "
                         f"Available columns: {list(df_metric.columns)}")

    metric_post = df_metric[df_metric['months'].isin(POST_SURGERY_MONTHS)][
        ['subject', 'months', metric_col]
    ].rename(columns={metric_col: 'delta_metric'})

    mjoa_post = df_mjoa_delta[df_mjoa_delta['months'].isin(POST_SURGERY_MONTHS)][
        ['subject', 'months', 'delta_mjoa', 'baseline_mjoa', 'mjoa_score']
    ]

    merged = metric_post.merge(mjoa_post, on=['subject', 'months'], how='inner')
    merged['timepoint_label'] = merged['months'].map(TIMEPOINT_LABELS)

    merged = merged.merge(
        participants_df[['participant_id', 'therapeutic_decision']],
        left_on='subject', right_on='participant_id', how='left'
    ).drop(columns='participant_id')

    out = os.path.join(output_dir, f'{slug}mjoa_merged.csv')
    merged.to_csv(out, index=False)

    n_total = merged['subject'].nunique()
    n_op   = merged[merged['therapeutic_decision'] == 'operative']['subject'].nunique()
    n_cons = merged[merged['therapeutic_decision'] == 'conservative']['subject'].nunique()
    print(f"Merged data: {len(merged)} rows, {n_total} subjects "
          f"(operative={n_op}, conservative={n_cons}) → {out}")
    return merged


def compute_correlations(merged, metric_label, slug, output_dir):
    """Compute Pearson and Spearman correlations per timepoint, operative subjects only."""
    operative = merged[merged['therapeutic_decision'] == 'operative']
    rows = []
    for months in POST_SURGERY_MONTHS:
        sub = operative[operative['months'] == months].dropna(
            subset=['delta_metric', 'delta_mjoa'])
        n = len(sub)
        if n < 5:
            continue
        pr, pp = stats.pearsonr(sub['delta_metric'], sub['delta_mjoa'])
        sr, sp = stats.spearmanr(sub['delta_metric'], sub['delta_mjoa'])
        rows.append({
            'timepoint': TIMEPOINT_LABELS[months],
            'months': months,
            'n': n,
            'pearson_r': round(pr, 3),
            'pearson_p': round(pp, 4),
            'spearman_rho': round(sr, 3),
            'spearman_p': round(sp, 4),
        })
    df_corr = pd.DataFrame(rows)
    print(f"\nCorrelation summary ({metric_label}, operative only):")
    print(df_corr.to_string(index=False))

    out = os.path.join(output_dir, f'{slug}correlation_summary.csv')
    df_corr.to_csv(out, index=False)
    print(f"Correlation summary saved → {out}")
    return df_corr


def _annotate_correlation(ax, x, y):
    """Add Pearson r and Spearman ρ annotation to an axis."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    def fmt_p(p):
        return f'p={p:.3f}' if p >= 0.001 else 'p<0.001'

    txt = f'r = {pr:.2f} ({fmt_p(pp)})\nρ = {sr:.2f} ({fmt_p(sp)})\nn = {len(x)}'
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))


def plot_per_timepoint(merged, metric_label, slug, output_dir):
    """One scatter + regression panel per post-surgery timepoint (operative only)."""
    operative = merged[merged['therapeutic_decision'] == 'operative']
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for i, months in enumerate(POST_SURGERY_MONTHS):
        ax = axes[i]
        sub = operative[operative['months'] == months].dropna(
            subset=['delta_metric', 'delta_mjoa'])
        color = TIMEPOINT_COLORS[months]

        sns.regplot(
            data=sub, x='delta_metric', y='delta_mjoa',
            ax=ax, color=color,
            scatter_kws=dict(s=40, alpha=0.6, edgecolors='white', linewidths=0.4),
            line_kws=dict(linewidth=2),
            ci=95
        )
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        _annotate_correlation(ax, sub['delta_metric'].values, sub['delta_mjoa'].values)

        ax.set_xlabel(f'{metric_label}  [followup − baseline]', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('ΔmJOA  [followup − baseline]', fontsize=LABELS_FONT_SIZE)
        ax.set_title(f'{TIMEPOINT_LABELS[months]} post-surgery',
                     fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.tick_params(labelsize=TICKS_FONT_SIZE)

    fig.suptitle(f'Association between {metric_label} and ΔmJOA after Surgery '
                 f'(operative subjects only)\n'
                 f'(positive = improvement in both axes)',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(output_dir, f'{slug}mjoa_per_timepoint.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Per-timepoint plot saved → {out}")
    plt.close()


def plot_pooled(merged, metric_label, slug, output_dir):
    """All timepoints pooled, colored by timepoint, operative subjects only."""
    operative = merged[merged['therapeutic_decision'] == 'operative']
    fig, ax = plt.subplots(figsize=(9, 7))

    for months in POST_SURGERY_MONTHS:
        sub = operative[operative['months'] == months].dropna(
            subset=['delta_metric', 'delta_mjoa'])
        ax.scatter(sub['delta_metric'], sub['delta_mjoa'],
                   label=TIMEPOINT_LABELS[months],
                   color=TIMEPOINT_COLORS[months],
                   s=45, alpha=0.65, edgecolors='white', linewidths=0.4)

    all_data = operative[operative['months'].isin(POST_SURGERY_MONTHS)].dropna(
        subset=['delta_metric', 'delta_mjoa'])
    slope, intercept, *_ = stats.linregress(all_data['delta_metric'], all_data['delta_mjoa'])
    x_range = np.linspace(all_data['delta_metric'].min(), all_data['delta_metric'].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, color='black', linewidth=2.5,
            linestyle='-', label='Overall fit', zorder=10)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    _annotate_correlation(ax, all_data['delta_metric'].values, all_data['delta_mjoa'].values)

    ax.set_xlabel(f'{metric_label}  [followup − baseline]', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('ΔmJOA  [followup − baseline]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(f'{metric_label} vs ΔmJOA — All post-surgery timepoints pooled (operative only)',
                 fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.legend(title='Timepoint', fontsize=10, title_fontsize=10)

    plt.tight_layout()
    out = os.path.join(output_dir, f'{slug}mjoa_pooled.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Pooled plot saved → {out}")
    plt.close()


def run_mixed_effects_model(merged, metric_label, slug, output_dir):
    """
    Fit two linear mixed-effects models:

    Model 1 (operative only):
        ΔmJOA ~ Δmetric + months + (1 | subject)

    Model 2 (both groups, interaction):
        ΔmJOA ~ Δmetric * group + months + (1 | subject)

    The interaction term tests whether the slope is steeper in operative patients.
    """
    metric_mean  = merged['delta_metric'].mean()
    months_mean  = merged[merged['months'].isin(POST_SURGERY_MONTHS)]['months'].mean()

    print(f"\n=== Model 1: operative only ({metric_label}) ===")
    data_op = merged[
        (merged['therapeutic_decision'] == 'operative') &
        (merged['months'].isin(POST_SURGERY_MONTHS))
    ].dropna(subset=['delta_metric', 'delta_mjoa']).copy()
    data_op['delta_metric_c'] = data_op['delta_metric'] - metric_mean
    data_op['months_c']       = data_op['months'] - months_mean

    res_op = smf.mixedlm(
        'delta_mjoa ~ delta_metric_c + months_c', data=data_op, groups=data_op['subject']
    ).fit(reml=True)
    print(res_op.summary())

    print(f"\n=== Model 2: interaction Δmetric × group ({metric_label}) ===")
    data_all = merged[merged['months'].isin(POST_SURGERY_MONTHS)].dropna(
        subset=['delta_metric', 'delta_mjoa']
    ).copy()
    data_all['delta_metric_c'] = data_all['delta_metric'] - metric_mean
    data_all['months_c']       = data_all['months'] - months_mean
    data_all['group']          = (data_all['therapeutic_decision'] == 'operative').astype(int)

    res_int = smf.mixedlm(
        'delta_mjoa ~ delta_metric_c * group + months_c',
        data=data_all, groups=data_all['subject']
    ).fit(reml=True)
    print(res_int.summary())

    # Save text summaries
    txt_out = os.path.join(output_dir, f'{slug}lmm_results.txt')
    with open(txt_out, 'w') as f:
        f.write(f"Metric: {metric_label}\n\n")
        f.write("MODEL 1: operative subjects only\n")
        f.write(f"ΔmJOA ~ Δmetric_c + months_c + (1 | subject)\n")
        f.write(f"N obs: {int(res_op.nobs)}, N subjects: {data_op['subject'].nunique()}\n\n")
        f.write(str(res_op.summary()))
        f.write("\n\n" + "="*60 + "\n\n")
        f.write("MODEL 2: interaction Δmetric × group (operative vs conservative)\n")
        f.write(f"ΔmJOA ~ Δmetric_c * group + months_c + (1 | subject)\n")
        f.write(f"N obs: {int(res_int.nobs)}, N subjects: {data_all['subject'].nunique()}\n\n")
        f.write(str(res_int.summary()))
        f.write("\n\nInterpretation of interaction term (delta_metric_c:group):\n")
        coef = res_int.fe_params.get('delta_metric_c:group', float('nan'))
        p    = res_int.pvalues.get('delta_metric_c:group', float('nan'))
        f.write(f"  coef = {coef:.4f}, p = {p:.4f}\n")
        f.write(f"  A significant positive value means the {metric_label}–ΔmJOA slope is\n")
        f.write("  steeper in operative than conservative patients.\n")
    print(f"\nModel summaries saved → {txt_out}")

    # Forest plot
    param_labels = {
        'delta_metric_c':       f'{metric_label} (centred)',
        'group':                'Group (operative=1)',
        'months_c':             'Months (centred)',
        'delta_metric_c:group': f'{metric_label} × Group\n(interaction)',
    }
    fe     = res_int.fe_params.drop('Intercept')
    ci     = res_int.conf_int().drop('Intercept')
    colors = {
        'delta_metric_c':       '#e6194b',
        'group':                '#aaaaaa',
        'months_c':             '#4363d8',
        'delta_metric_c:group': '#f58231',
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (param, coef) in enumerate(fe.items()):
        lo, hi = ci.loc[param]
        c = colors.get(param, 'gray')
        ax.errorbar(coef, i, xerr=[[coef - lo], [hi - coef]],
                    fmt='o', color=c, markersize=9, capsize=5, linewidth=2)
        p = res_int.pvalues[param]
        p_txt = f'p={p:.3f}' if p >= 0.001 else 'p<0.001'
        weight = 'bold' if param == 'delta_metric_c:group' else 'normal'
        ax.text(max(hi, coef) + abs(hi - lo) * 0.08, i, p_txt,
                va='center', fontsize=10, fontweight=weight)

    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_yticks(range(len(fe)))
    ax.set_yticklabels([param_labels.get(p, p) for p in fe.index], fontsize=11)
    ax.set_xlabel('Fixed-effect coefficient (ΔmJOA units)', fontsize=LABELS_FONT_SIZE)
    ax.set_title(f'Interaction Model: Fixed Effects ± 95% CI\n'
                 f'ΔmJOA ~ {metric_label} × group + months + (1 | subject)',
                 fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    plt.tight_layout()

    fig_out = os.path.join(output_dir, f'{slug}lmm_fixed_effects.png')
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    print(f"Forest plot saved → {fig_out}")
    plt.close()

    _plot_slopes_by_group(data_all, metric_label, slug, output_dir)

    return res_op, res_int


def _plot_slopes_by_group(data_all, metric_label, slug, output_dir):
    """Scatter + per-group regression lines to visualise the interaction."""
    fig, ax = plt.subplots(figsize=(9, 7))

    group_styles = {
        'operative':    {'color': '#e6194b', 'label': 'Operative'},
        'conservative': {'color': '#4363d8', 'label': 'Conservative'},
    }
    for grp, style in group_styles.items():
        sub = data_all[data_all['therapeutic_decision'] == grp].dropna(
            subset=['delta_metric_c', 'delta_mjoa'])
        sns.regplot(data=sub, x='delta_metric_c', y='delta_mjoa', ax=ax,
                    color=style['color'], label=style['label'],
                    scatter_kws=dict(s=35, alpha=0.55, edgecolors='white', linewidths=0.3),
                    line_kws=dict(linewidth=2.5), ci=95)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel(f'{metric_label} centred', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('ΔmJOA [followup − baseline]', fontsize=LABELS_FONT_SIZE)
    ax.set_title(f'{metric_label} vs ΔmJOA by treatment group\n'
                 '(parallel slopes = no interaction; diverging = interaction)',
                 fontsize=TITLE_FONT_SIZE)
    ax.tick_params(labelsize=TICKS_FONT_SIZE)
    ax.legend(fontsize=11)
    plt.tight_layout()

    fig_out = os.path.join(output_dir, f'{slug}lmm_slopes_by_group.png')
    plt.savefig(fig_out, dpi=300, bbox_inches='tight')
    print(f"Slope comparison plot saved → {fig_out}")
    plt.close()


def main():
    args = get_parser().parse_args()
    os.makedirs(args.o, exist_ok=True)

    slug = _slug(args.metric_label)

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Arial'

    print(f"Metric: {args.metric_label} (column: '{args.metric_col}')")
    print(f"Output prefix: {slug}\n")

    df_metric = pd.read_csv(args.metric_delta)
    df_mjoa   = pd.read_csv(args.mjoa)
    df_mjoa_delta   = compute_delta_mjoa(df_mjoa)
    participants_df = pd.read_csv(args.participants, sep='\t')

    merged = merge_deltas(df_metric, args.metric_col, df_mjoa_delta,
                          participants_df, slug, args.o)

    compute_correlations(merged, args.metric_label, slug, args.o)

    plot_per_timepoint(merged, args.metric_label, slug, args.o)
    plot_pooled(merged, args.metric_label, slug, args.o)
    run_mixed_effects_model(merged, args.metric_label, slug, args.o)

    print("\nDone. Results in:", args.o)


if __name__ == '__main__':
    main()
