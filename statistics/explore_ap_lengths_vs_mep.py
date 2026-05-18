#
# Explore the relationship between new morphometrics at the maximally compressed slice
# (anterior length, posterior length, asymmetry, AP diameter) and MEP outcomes in dcm-zurich.
#
# Binary outcome: r_mep_rating_phys (0 = healthy, 1 = pathological/abnormal)
# Continuous outcomes: ZML latencies (left/right) and Cortex->AH amplitudes (left/right)
#
# Outputs (in -path-out):
#   raincloud_morphometrics_vs_mep_pathology.png
#   scatter_morphometrics_vs_mep_continuous.png
#   heatmap_spearman_correlations_mep.png
#   stats_results_mep.csv
#
# Example usage:
#   python statistics/explore_ap_lengths_vs_mep.py \
#       -morphometrics-file ~/results/dcm-zurich/dcm-zurich_part01_ap_lengths_2026-04-15/max_compression_metrics.csv \
#       -mep-file ~/data/data.neuro.polymtl.ca/dcm-zurich/phenotype/CHEPS-MEPS_cohort/from_Aleks_MEP.xlsx \
#       -path-out ~/results/dcm-zurich/mep_explore
#
# Author: Jan Valosek
#

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.utils import SmartFormatter, format_pvalue

FNAME_LOG = 'log_explore_ap_mep.txt'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

MORPHOMETRICS_PLOT = ['length_anterior', 'length_posterior', 'asymmetry', 'diameter_AP']
MORPHOMETRICS_ALL = MORPHOMETRICS_PLOT

METRIC_LABEL = {
    'length_anterior': 'Anterior Length [mm]',
    'length_posterior': 'Posterior Length [mm]',
    'asymmetry': 'Asymmetry [a.u.]',
    'diameter_AP': 'AP Diameter [mm]',
}

CONTINUOUS_OUTCOMES = [
    'r_mep_ZML_lum_R',
    'r_mep_ZML_lum_L',
    'r_mep_ampl_Cortex_AH_R',
    'r_mep_ampl_Cortex_AH_L',
]
CONTINUOUS_LABEL = {
    'r_mep_ZML_lum_R': 'ZML Lumbar R [ms]',
    'r_mep_ZML_lum_L': 'ZML Lumbar L [ms]',
    'r_mep_ampl_Cortex_AH_R': 'Cortex→AH Ampl. R [mV]',
    'r_mep_ampl_Cortex_AH_L': 'Cortex→AH Ampl. L [mV]',
}

BINARY_OUTCOMES = [
    'r_mep_rating_phys',
]
BINARY_LABEL = {
    'r_mep_rating_phys': 'MEP Pathological',
}

BINARY_GROUP_LABELS = {
    'r_mep_rating_phys': ['Healthy', 'Pathological'],
}

THUMBNAILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbnails_for_figure')
METRIC_THUMBNAIL = {
    'length_anterior': 'a_length.png',
    'length_posterior': 'p_length.png',
    'asymmetry': 'assymetry.png',
    'diameter_AP': 'ap_diam.png',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Explore AP lengths / asymmetry vs. MEP (binary pathology rating + continuous MEP metrics).",
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        '-morphometrics-file',
        required=True,
        metavar='<file>',
        help="Path to max_compression_metrics.csv",
    )
    parser.add_argument(
        '-mep-file',
        required=True,
        metavar='<file>',
        help="Path to from_Aleks_MEP.xlsx",
    )
    parser.add_argument(
        '-path-out',
        required=True,
        metavar='<dir>',
        help="Output directory for figures and stats CSV",
    )
    return parser


def load_and_merge(morphometrics_file, mep_file):
    morph = pd.read_csv(morphometrics_file)
    morph['record_id'] = morph['participant_id'].str.replace('sub-', '').astype(int)

    mep = pd.read_excel(mep_file, sheet_name='Sheet 1')

    df = morph.merge(mep, left_on='record_id', right_on='r_record_id', how='inner')
    logger.info(f"Merged dataset: {len(df)} subjects")
    return df


def run_statistics(df):
    rows = []

    for metric in MORPHOMETRICS_ALL:
        p_vals = []
        metric_rows = []

        for outcome in CONTINUOUS_OUTCOMES:
            sub = df[[metric, outcome]].dropna()
            rho, p = stats.spearmanr(sub[metric], sub[outcome])
            metric_rows.append({
                'morphometric': metric,
                'outcome': outcome,
                'test': 'spearman',
                'n': len(sub),
                'statistic (rho or rank-biserial r)': round(rho, 4),
                'p_value': p,
            })
            p_vals.append(p)

        for outcome in BINARY_OUTCOMES:
            sub = df[[metric, outcome]].dropna()
            g0 = sub.loc[sub[outcome] == 0, metric]
            g1 = sub.loc[sub[outcome] == 1, metric]
            u, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            r_rb = 1 - 2 * u / (len(g0) * len(g1))   # rank-biserial r
            metric_rows.append({
                'morphometric': metric,
                'outcome': outcome,
                'test': 'mann_whitney',
                'n': len(sub),
                'statistic (rho or rank-biserial r)': round(r_rb, 4),
                'p_value': p,
            })
            p_vals.append(p)

        reject, p_fdr, _, _ = multipletests(p_vals, method='fdr_bh')
        for i, row in enumerate(metric_rows):
            row['p_fdr'] = round(p_fdr[i], 4)
            row['significant_fdr'] = reject[i]
            rows.append(row)

    results = pd.DataFrame(rows)
    logger.info("\nStatistical results:")
    logger.info(results.to_string(index=False))
    return results


def plot_scatter_continuous(df, results, path_out):
    n_rows = len(MORPHOMETRICS_PLOT)
    n_cols = len(CONTINUOUS_OUTCOMES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)

    for r, metric in enumerate(MORPHOMETRICS_PLOT):
        for c, outcome in enumerate(CONTINUOUS_OUTCOMES):
            ax = axes[r, c]
            sub = df[[metric, outcome]].dropna()
            row = results[(results['morphometric'] == metric) & (results['outcome'] == outcome)].iloc[0]
            rho = row['statistic (rho or rank-biserial r)']
            p = row['p_value']
            p_fdr = row['p_fdr']

            ax.scatter(sub[outcome].values, sub[metric].values, alpha=0.65, s=32,
                       color='steelblue', edgecolors='white', linewidths=0.4, zorder=2)

            # linear trend line (visual aid only — significance comes from Spearman)
            if len(sub) >= 2:
                x = sub[outcome].values.astype(float)
                y = sub[metric].values.astype(float)
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + intercept, color='tomato', linewidth=1.5, zorder=3)

            ax.set_xlabel(CONTINUOUS_LABEL[outcome], fontsize=10)
            if c == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=10)

            sig = '*' if p_fdr < 0.05 else ''
            ax.set_title(f'ρ = {rho:.2f}, p {format_pvalue(p)}{sig}  (n={len(sub)})', fontsize=9)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    fig.suptitle(
        'Morphometrics vs. Continuous MEP Metrics\n'
        '(red = linear fit; * = significant after FDR correction)',
        fontsize=12,
    )
    fname = os.path.join(path_out, 'scatter_morphometrics_vs_mep_continuous.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def _raincloud_panel(ax, vals_list, colors, labels, rng):
    """Draw a raincloud for each group: half-violin left, thin box centre, jitter right."""
    from scipy.stats import gaussian_kde

    viol_width = 0.20   # max half-violin half-width in data-x units
    box_hw = 0.03       # half-width of the IQR box
    jitter_lo, jitter_hi = 0.05, 0.16

    for i, (vals, color, label) in enumerate(zip(vals_list, colors, labels)):
        vals = np.asarray(vals, dtype=float)
        pos = float(i)

        # --- half violin (KDE, left of pos) ---
        if len(vals) > 1:
            kde = gaussian_kde(vals, bw_method='scott')
            y_grid = np.linspace(vals.min(), vals.max(), 200)
            density = kde(y_grid)
            density = density / density.max() * viol_width
            ax.fill_betweenx(y_grid, pos - density, pos, alpha=0.65, color=color, linewidth=0)

        # --- box summary (IQR box + whiskers + median, centred at pos) ---
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        w_lo = max(vals.min(), q1 - 1.5 * iqr)
        w_hi = min(vals.max(), q3 + 1.5 * iqr)
        ax.plot([pos, pos], [w_lo, w_hi], color='k', linewidth=1.0, zorder=3)
        ax.add_patch(plt.Rectangle(
            (pos - box_hw, q1), 2 * box_hw, iqr,
            facecolor='white', edgecolor='k', linewidth=1.2, zorder=4,
        ))
        ax.hlines(med, pos - box_hw, pos + box_hw, color='k', linewidth=2.0, zorder=5)

        # --- jittered raw points (right of pos) ---
        jitter = rng.uniform(jitter_lo, jitter_hi, size=len(vals))
        ax.scatter(pos + jitter, vals, s=18, color=color, alpha=0.75,
                   edgecolors='white', linewidths=0.3, zorder=6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-viol_width - 0.15, len(labels) - 1 + jitter_hi + 0.15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return jitter_hi


def _add_significance_bar(ax, x0, x1, p_value, p_fdr, all_vals):
    """Draw a significance bracket with p-value annotation above the data."""
    y_max = max(np.concatenate(all_vals))
    y_range = y_max - min(np.concatenate(all_vals))
    bar_y = y_max + 0.06 * y_range
    tick_h = 0.02 * y_range
    text_y = bar_y + 0.02 * y_range

    ax.plot([x0, x0, x1, x1], [bar_y - tick_h, bar_y, bar_y, bar_y - tick_h],
            color='k', linewidth=1.0)
    sig = '*' if p_value < 0.05 else ''
    label = f'{sig} p {format_pvalue(p_value)}'
    ax.text((x0 + x1) / 2, text_y, label, ha='center', va='bottom', fontsize=10)

    # expand y-axis to make room
    ax.set_ylim(top=text_y + 0.08 * y_range)


OUTCOME_TITLE = {
    'r_mep_rating_phys': 'MEP (Mann-Whitney U)',
}
OUTCOME_FNAME = {
    'r_mep_rating_phys': 'raincloud_morphometrics_vs_mep_pathology.png',
}


def plot_raincloud_binary(df, results, path_out):
    n_cols = len(MORPHOMETRICS_PLOT)
    rng = np.random.default_rng(42)
    palette = sns.color_palette('pastel', 2)

    for outcome in BINARY_OUTCOMES:
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4), constrained_layout=True)

        for c, metric in enumerate(MORPHOMETRICS_PLOT):
            ax = axes[c]
            sub = df[[metric, outcome]].dropna().copy()
            sub[outcome] = sub[outcome].astype(int)

            g0 = sub.loc[sub[outcome] == 0, metric].values
            g1 = sub.loc[sub[outcome] == 1, metric].values

            base_labels = BINARY_GROUP_LABELS[outcome]
            tick_labels = [f'{base_labels[0]}\n(n={len(g0)})', f'{base_labels[1]}\n(n={len(g1)})']
            _raincloud_panel(ax, [g0, g1], palette, tick_labels, rng)

            if metric == 'diameter_AP':
                ax.set_ylim(bottom=3)

            row = results[(results['morphometric'] == metric) & (results['outcome'] == outcome)].iloc[0]
            p = row['p_value']
            p_fdr = row['p_fdr']

            _add_significance_bar(ax, 0, 1, p, p_fdr, [g0, g1])

            ax.set_ylabel(METRIC_LABEL[metric], fontsize=10)

            # thumbnail inset — bottom-left corner of each panel
            thumb_path = os.path.join(THUMBNAILS_DIR, METRIC_THUMBNAIL.get(metric, ''))
            thub_size = [0.01, -0.08, 0.5, 0.5] if metric == 'asymmetry' else [0.01, -0.08, 0.4, 0.4]
            if os.path.isfile(thumb_path):
                img = mpimg.imread(thumb_path)
                axins = ax.inset_axes(thub_size)
                axins.imshow(img)
                axins.axis('off')

        fig.suptitle(OUTCOME_TITLE[outcome], fontsize=12)
        fname = os.path.join(path_out, OUTCOME_FNAME[outcome])
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {fname}")


def plot_heatmap(df, path_out):
    all_outcomes = CONTINUOUS_OUTCOMES + BINARY_OUTCOMES

    rho_matrix = pd.DataFrame(index=MORPHOMETRICS_ALL, columns=all_outcomes, dtype=float)
    p_matrix = pd.DataFrame(index=MORPHOMETRICS_ALL, columns=all_outcomes, dtype=float)

    for metric in MORPHOMETRICS_ALL:
        for outcome in all_outcomes:
            sub = df[[metric, outcome]].dropna()
            rho, p = stats.spearmanr(sub[metric], sub[outcome])
            rho_matrix.loc[metric, outcome] = round(rho, 3)
            p_matrix.loc[metric, outcome] = p

    annot = pd.DataFrame(index=MORPHOMETRICS_ALL, columns=all_outcomes, dtype=str)
    for metric in MORPHOMETRICS_ALL:
        for outcome in all_outcomes:
            rho = rho_matrix.loc[metric, outcome]
            p = p_matrix.loc[metric, outcome]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            annot.loc[metric, outcome] = f'{rho:.2f}{stars}'

    row_labels = [METRIC_LABEL[m] for m in MORPHOMETRICS_ALL]
    all_labels = {**CONTINUOUS_LABEL, **BINARY_LABEL}
    col_labels = [all_labels[o].replace('\n', ' ') for o in all_outcomes]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        rho_matrix.astype(float),
        annot=annot,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={'label': 'Spearman ρ'},
    )
    ax.set_title(
        'Spearman Correlations: Morphometrics vs. MEP Metrics\n'
        '(*, **, *** = p<0.05, 0.01, 0.001; uncorrected)',
        fontsize=11,
    )
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    fname = os.path.join(path_out, 'heatmap_spearman_correlations_mep.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {fname}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    fh = logging.FileHandler(os.path.join(args.path_out, FNAME_LOG))
    logging.root.addHandler(fh)

    df = load_and_merge(args.morphometrics_file, args.mep_file)

    results = run_statistics(df)
    csv_path = os.path.join(args.path_out, 'stats_results_mep.csv')
    results.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    plot_scatter_continuous(df, results, args.path_out)
    plot_raincloud_binary(df, results, args.path_out)
    plot_heatmap(df, args.path_out)
    logger.info("Done.")


if __name__ == '__main__':
    main()
