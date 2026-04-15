#
# Plot morphometric metrics (including anterior and posterior cord lengths) computed from the
# spine-generic multi-subject dataset in PAM50 space, per slice and vertebral level.
# Supports overlaying multiple datasets (e.g., HC and DCM patients) in a single figure.
#
# The script reads per-subject *_PAM50.csv files and plots 7 metrics:
#   - MEAN(area)            : Cross-Sectional Area [mm²]
#   - MEAN(diameter_AP)     : AP Diameter [mm]
#   - MEAN(diameter_RL)     : Transverse Diameter [mm]
#   - compression_ratio     : Compression Ratio [a.u.]  (= diameter_AP / diameter_RL)
#   - MEAN(length_anterior) : Anterior Length [mm]
#   - MEAN(length_posterior): Posterior Length [mm]
#   - asymmetry             : Asymmetry [a.u.]  (= (length_anterior - length_posterior) / diameter_AP)
#
# Example usage (single dataset):
#   python statistics/generate_figures_spine-generic_ap_lengths.py \
#       -path-SC ~/results/spine-generic/spine-generic_ap_lengths_2026-04-14/results/PAM50 \
#       -participant-file ~/data/data.neuro.polymtl.ca/data-multi-subject/participants.tsv \
#       -path-out ~/results/spine-generic/spine-generic_ap_lengths_2026-04-14/figures
#
# Example usage (HC + DCM patients):
#   python statistics/generate_figures_spine-generic_ap_lengths.py \
#       -path-SC ~/results/spine-generic/spine-generic_ap_lengths_2026-04-14/results/PAM50 \
#                ~/results/dcm-zurich/dcm-zurich_ap_lengths_2026-04-15/results/PAM50 \
#       -dataset-labels HC DCM \
#       -path-out ~/results/combined_ap_lengths/figures
#
# Authors: Jan Valosek
# Adapted from https://github.com/spinalcordtoolbox/PAM50-normalized-metrics/blob/main/statistics/generate_figures.py
#

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

# Flat list used for statistics and normative values
METRICS = [
    'MEAN(area)',
    'MEAN(diameter_AP)',
    'MEAN(diameter_RL)',
    'MEAN(compression_ratio)',
    'MEAN(length_anterior)',
    'MEAN(length_posterior)',
    'asymmetry',
]

# Layout for the 2x4 figure (last cell is empty)
# Row 1: CSA, AP diam, anterior length, posterior length
# Row 2: RL diam, CR, asymmetry, empty
# A vertical separator line is drawn between columns 2 and 3
METRICS_LAYOUT = [
    'MEAN(area)',
    'MEAN(diameter_AP)',
    'MEAN(length_anterior)',
    'MEAN(length_posterior)',
    'MEAN(diameter_RL)',
    'MEAN(compression_ratio)',
    'asymmetry',
    None,   # empty cell
]

METRICS_DTYPE = {
    'MEAN(area)': 'float64',
    'MEAN(diameter_AP)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(length_anterior)': 'float64',
    'MEAN(length_posterior)': 'float64',
}

METRIC_TO_TITLE = {
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(diameter_RL)': 'Transverse Diameter',
    'MEAN(compression_ratio)': 'Compression Ratio',
    'MEAN(length_anterior)': 'Anterior Length',
    'MEAN(length_posterior)': 'Posterior Length',
    'asymmetry': 'Asymmetry',
}

METRIC_TO_AXIS = {
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
    'MEAN(length_anterior)': 'Anterior Length [mm]',
    'MEAN(length_posterior)': 'Posterior Length [mm]',
    'asymmetry': 'Asymmetry [a.u.]',
}

# Set ylim to avoid overlap of horizontal grid with vertebral labels
METRICS_TO_YLIM = {
    'MEAN(area)': (30, 105),
    'MEAN(diameter_AP)': (4.5, 9.5),
    'MEAN(diameter_RL)': (8.0, 15.5),
    'MEAN(compression_ratio)': (0.41, 0.84),
    'MEAN(length_anterior)': (2.0, 5.0),
    'MEAN(length_posterior)': (2.0, 5.0),
    'asymmetry': (-0.20, 0.15),
}

# ylim max offset used for showing CoV text labels
METRICS_TO_YLIM_OFFSET = {
    'MEAN(area)': 6,
    'MEAN(diameter_AP)': 0.4,
    'MEAN(diameter_RL)': 0.7,
    'MEAN(compression_ratio)': 0.03,
    'MEAN(length_anterior)': 0.2,
    'MEAN(length_posterior)': 0.2,
    'asymmetry': 0.01,
}

DISCS_DICT = {
    7: 'C7-T1',
    6: 'C6-C7',
    5: 'C5-C6',
    4: 'C4-C5',
    3: 'C3-C4',
    2: 'C2-C3',
    1: 'C1-C2',
}

MID_VERT_DICT = {
    8: 'T1',
    7: 'C7',
    6: 'C6',
    5: 'C5',
    4: 'C4',
    3: 'C3',
    2: 'C2',
    1: 'C1',
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

PALETTE = {
    'sex': {'M': 'blue', 'F': 'red'},
    'manufacturer': {'Siemens': 'green', 'Philips': 'dodgerblue', 'GE': 'black'},
}

# Default colors for dataset overlay (cycled when more than 2 datasets)
DATASET_COLORS = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'purple']


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot morphometrics including anterior and posterior lengths from one or more datasets "
                    "in PAM50 space. Multiple datasets are overlaid in the same figure using distinct colors.")
    parser.add_argument('-path-SC', required=True, type=str, nargs='+',
                        help="Path(s) to folder(s) containing per-subject *_PAM50.csv files. "
                             "Multiple paths can be provided to overlay datasets.")
    parser.add_argument('-dataset-labels', required=False, type=str, nargs='+', default=None,
                        help="Optional labels for each dataset (same order as -path-SC). "
                             "If not provided, the folder name is used as the label.")
    parser.add_argument('-participant-file', required=False, type=str, default=None,
                        help="Path to participants.tsv. Applies to all datasets. If not provided, "
                             "the script looks for participants.tsv inside each dataset folder.")
    parser.add_argument('-path-out', required=False, type=str, default='stats',
                        help="Output directory (default: 'stats').")
    return parser


def get_vert_indices(df):
    """
    Get indices of slices corresponding to mid-vertebrae and disc boundaries.
    """
    first_subject = df['participant_id'].iloc[0]
    mask = df['participant_id'] == first_subject
    vert = df[mask]['VertLevel']
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = [int(ind_vert[i:i+2].mean()) for i in range(len(ind_vert) - 1)]
    return vert, ind_vert, ind_vert_mid


def create_lineplot(df, hue, path_out, show_cv=False):
    """
    Create a 2x4 lineplot for 7 metrics defined by METRICS_LAYOUT (last cell is empty).
    Row 1: CSA, AP diameter, RL diameter, Compression Ratio
    Row 2: Anterior length, Posterior length, Asymmetry, (empty)
    """
    mpl.rcParams['font.family'] = 'Arial'

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axs = axes.ravel()

    ref_metric = 'MEAN(area)' if 'MEAN(area)' in df.columns else METRICS[0]
    n_subjects_per_level = df.dropna(subset=[ref_metric]).groupby('VertLevel')['participant_id'].nunique()

    vert, ind_vert, ind_vert_mid = get_vert_indices(df)

    print('\nNumber of subjects per vertebral level:')
    for x in ind_vert_mid:
        level = ('T' + str(vert[x] - 7)) if vert[x] > 7 else ('C' + str(vert[x]))
        print(f'  {level}: {n_subjects_per_level.get(vert[x], 0)}')

    # Pre-compute palette and legend labels (with subject counts) once, outside the metric loop
    if hue in PALETTE:
        palette = PALETTE[hue]
        legend_labels = None  # seaborn uses raw hue values
    elif hue == 'dataset' and 'dataset' in df.columns:
        datasets = list(df['dataset'].unique())
        palette = {d: DATASET_COLORS[i % len(DATASET_COLORS)] for i, d in enumerate(datasets)}
        n_per_dataset = df.groupby('dataset')['participant_id'].nunique()
        legend_labels = {d: f'{d} (n={n_per_dataset[d]})' for d in datasets}
    else:
        palette = None
        legend_labels = None

    for index, entry in enumerate(METRICS_LAYOUT):
        ax = axs[index]

        # Hide the empty cell
        if entry is None:
            ax.set_visible(False)
            continue

        metric = entry
        if hue is not None:
            sns.lineplot(ax=ax, x='Slice (I->S)', y=metric, data=df, errorbar='sd',
                         hue=hue, linewidth=2, palette=palette)
        else:
            sns.lineplot(ax=ax, x='Slice (I->S)', y=metric, data=df, errorbar='sd', linewidth=2)

        if hue is not None:
            if index == 0:
                legend = ax.legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
                # Replace legend labels with enriched versions (e.g. "HC (n=201)")
                if legend_labels is not None:
                    for text in legend.get_texts():
                        text.set_text(legend_labels.get(text.get_text(), text.get_text()))
            else:
                ax.get_legend().remove()

        ax.set_ylim(METRICS_TO_YLIM[metric])
        ymin, ymax = ax.get_ylim()

        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        for spine in ['right', 'left', 'top']:
            ax.spines[spine].set_visible(False)

        for x in ind_vert[1:-1]:
            ax.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        for x in ind_vert_mid:
            level = ('T' + str(vert[x] - 7)) if vert[x] > 7 else ('C' + str(vert[x]))
            n = n_subjects_per_level.get(vert[x], 0)
            ax.text(df.loc[x, 'Slice (I->S)'], ymin, f'{level}\nn={n}',
                    ha='center', va='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        ax.invert_xaxis()
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    # Draw a vertical separator line between columns 2 and 3 (in figure coordinates)
    # The line spans the full figure height
    fig.add_artist(
        plt.Line2D([0.496, 0.496], [0.01, 0.92],
                   transform=fig.transFigure,
                   color='black', linewidth=1, linestyle='-')
    )

    hue_suffix = f'_per{hue}' if hue else ''
    filename = f'lineplot{hue_suffix}.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {path_filename}')


def compute_cv(df, metric):
    """Compute coefficient of variation (%) for a given metric."""
    return df[metric].std() / df[metric].mean() * 100


def format_pvalue(p_value, alpha=0.001, decimal_places=3, include_space=True, include_equal=True):
    """Format p-value as string."""
    space = ' ' if include_space else ''
    if p_value < alpha:
        return space + '<' + space + str(alpha)
    eq = '=' if include_equal else ''
    return space + eq + space + str(round(p_value, decimal_places))


def compare_metrics_across_sex(df):
    """Wilcoxon rank-sum test between males and females for each metric."""
    print('\n--- Sex comparison (Wilcoxon rank-sum) ---')
    for metric in METRICS:
        if metric not in df.columns:
            continue
        slices_M = df[df['sex'] == 'M'].groupby('Slice (I->S)')[metric].mean()
        slices_F = df[df['sex'] == 'F'].groupby('Slice (I->S)')[metric].mean()
        stat, pval = stats.shapiro(slices_M)
        print(f'{metric} — Normality M: p{format_pvalue(pval)}')
        stat, pval = stats.shapiro(slices_F)
        print(f'{metric} — Normality F: p{format_pvalue(pval)}')
        stat, pval = stats.ranksums(x=slices_M, y=slices_F)
        print(f'{metric} — Wilcoxon rank-sum M vs F: p{format_pvalue(pval)}')


def compute_normative_values(df, path_out):
    """Compute mean ± SD at disc and mid-vertebral levels and save to CSV."""
    for metric in METRICS:
        if metric not in df.columns:
            continue
        print(f'\n{metric}')
        slices_mean = df.groupby('Slice (I->S)')[metric].mean()
        slices_std = df.groupby('Slice (I->S)')[metric].std()
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)

        # Disc-level normative values
        d = []
        for x in reversed(ind_vert[1:-1]):
            slice_number = df.loc[x, 'Slice (I->S)']
            disc = DISCS_DICT.get(vert[x], str(vert[x]))
            m, s = slices_mean.loc[slice_number], slices_std.loc[slice_number]
            print(f'  Disc {disc}, slice {slice_number}: {round(m, 2)} ± {round(s, 2)}')
            d.append({'Disc': disc, 'Slice': slice_number, 'Mean ± STD': f'{round(m, 2)} ± {round(s, 2)}'})
        fname = os.path.join(path_out, metric + '_disc_normative_values.csv')
        pd.DataFrame(d).to_csv(fname, index=False)
        print(f'  Saved: {fname}')

        # Mid-vertebral normative values
        d = []
        for x in reversed(ind_vert_mid):
            slice_number = df.loc[x, 'Slice (I->S)']
            level = MID_VERT_DICT.get(vert[x], str(vert[x]))
            m, s = slices_mean.loc[slice_number], slices_std.loc[slice_number]
            print(f'  Level {level}, slice {slice_number}: {round(m, 2)} ± {round(s, 2)}')
            d.append({'Level': level, 'Slice': slice_number, 'Mean ± STD': f'{round(m, 2)} ± {round(s, 2)}'})
        fname = os.path.join(path_out, metric + '_mid_level_normative_values.csv')
        pd.DataFrame(d).to_csv(fname, index=False)
        print(f'  Saved: {fname}')


def read_csv_files(path_SC, participant_file=None, dataset_name=None):
    """
    Read all *_PAM50.csv files from path_SC, compute derived metrics, and optionally
    merge demographic data from participants.tsv.
    """
    path_SC = os.path.expanduser(path_SC)
    print(f'Reading CSV files from: {path_SC}')
    df = pd.DataFrame()
    for file in sorted(os.listdir(path_SC)):
        if not file.endswith('PAM50.csv'):
            continue
        df_subject = pd.read_csv(os.path.join(path_SC, file), dtype=METRICS_DTYPE)
        df_subject['source_file'] = file
        df = pd.concat([df, df_subject], axis=0, ignore_index=True)

    # Extract participant_id from the CSV filename (e.g. sub-amu01_T2w_PAM50.csv -> sub-amu01)
    df.insert(0, 'participant_id', df['source_file'].str.split('_').str[0])

    # Compute derived metrics
    df['MEAN(compression_ratio)'] = df['MEAN(diameter_AP)'] / df['MEAN(diameter_RL)']
    df['asymmetry'] = (df['MEAN(length_anterior)'] - df['MEAN(length_posterior)']) / df['MEAN(diameter_AP)']

    # Tag each row with dataset name (used for multi-dataset coloring)
    if dataset_name is not None:
        df['dataset'] = dataset_name

    subjects = df['participant_id'].unique()
    print(f'Number of subjects: {len(subjects)}')

    # Discover participants.tsv if not provided explicitly
    if participant_file is None:
        auto_tsv = os.path.join(path_SC, 'participants.tsv')
        if os.path.isfile(auto_tsv):
            participant_file = auto_tsv

    df_participants = None
    if participant_file:
        df_participants = pd.read_csv(os.path.expanduser(participant_file), sep='\t')
        possible_cols = ['participant_id', 'age', 'sex', 'height', 'weight', 'manufacturer']
        cols_to_merge = [c for c in possible_cols if c in df_participants.columns]
        df = df.merge(df_participants[cols_to_merge], on='participant_id', how='left')

    return df, df_participants, subjects


def main():
    parser = get_parser()
    args = parser.parse_args()

    path_out_figures = os.path.join(os.path.expanduser(args.path_out), 'figures')
    path_out_csv = os.path.join(os.path.expanduser(args.path_out), 'csv')
    os.makedirs(path_out_figures, exist_ok=True)
    os.makedirs(path_out_csv, exist_ok=True)

    # Validate dataset labels length
    if args.dataset_labels is not None and len(args.dataset_labels) != len(args.path_SC):
        parser.error(f'-dataset-labels must have the same number of entries as -path-SC '
                     f'({len(args.path_SC)} paths, {len(args.dataset_labels)} labels).')

    # Load one or more datasets and concatenate
    multiple_datasets = len(args.path_SC) > 1
    dfs = []
    all_subjects = []
    for i, path in enumerate(args.path_SC):
        if args.dataset_labels is not None:
            dataset_name = args.dataset_labels[i]
        else:
            dataset_name = os.path.basename(path.rstrip('/')) if multiple_datasets else None
        df_single, _, subs = read_csv_files(path, args.participant_file, dataset_name=dataset_name)
        dfs.append(df_single)
        all_subjects.extend(subs)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    subjects = np.array(all_subjects)

    # Drop all-NaN columns and keep C2–C7 (VertLevel 2–7)
    df = df.dropna(axis=1, how='all')
    df = df[df['VertLevel'].between(2, 7)]

    subjects_after = df['participant_id'].unique()
    print(f'Subjects after filtering to C2–C7: {len(subjects_after)}')
    dropped = set(subjects) - set(subjects_after)
    if dropped:
        print(f'Dropped subjects: {sorted(dropped)}')

    # Figures — coloured by dataset if multiple datasets, otherwise ungrouped
    dataset_hue = 'dataset' if multiple_datasets else None
    create_lineplot(df, hue=dataset_hue, path_out=path_out_figures)

    # Figures and stats stratified by sex (single dataset only, to avoid confounds)
    if not multiple_datasets and 'sex' in df.columns:
        create_lineplot(df, hue='sex', path_out=path_out_figures)
        compare_metrics_across_sex(df)

    # Figures by manufacturer (single dataset only)
    if not multiple_datasets and 'manufacturer' in df.columns:
        create_lineplot(df, hue='manufacturer', path_out=path_out_figures)

    # Normative values
    compute_normative_values(df, path_out_csv)


if __name__ == '__main__':
    main()
