#
# Plot morphometrics across subjects, separately for multiple sessions
# For each slice in the PAM50 space, mean and std of morphometric metrics are plotted across subjects
# The script is compatible with spinal cord, canal, or aSCOR metrics. The normative data to plot is determined based on
# the input CSV filename (cord, canal, or aSCOR); see example usages below.
#
# Example usages (cord, canal, aSCOR):
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_cord_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_canal_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_aSCOR_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
# Author: Jan Valosek, Sandrine Bédard
#

import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

from utils import (read_clinical_file, read_morphometrics_file, read_exclude_file_and_exclude_subjects,
                   drop_highest_stenosis,
                   METRICS_DTYPE, load_normative_df_c2, _categorize_c2_area, _get_highest_stenosis)

TICKS_FONT_SIZE = 18
LABELS_FONT_SIZE = TICKS_FONT_SIZE+2

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
    'aSCOR': 'aSCOR [a.u.]'
}

AGE_DECADES = ['10-20', '21-30', '31-40', '41-50', '51-60']

SESSION_COLORS = {
    'ses-M0': '#1f77b4',  # blue
    'ses-M3': '#ff7f0e',  # orange
    'ses-M6': '#2ca02c',  # green
    'ses-M12': '#ff9999'  # light red
}

# Color mapping for Maximum Compression Level (MCL) stratification
MCL_COLORS = {
    'C2/C3': '#d62728',    # red
    'C3/C4': '#ff7f0e',    # orange
    'C4/C5': '#2ca02c',    # green
    'C5/C6': '#1f77b4',    # blue
    'C6/C7': '#9467bd',    # purple
}

AGE_GROUP_COLORS = {
    '<50': '#2ca02c',       # green
    '50-65': '#ff7f0e',     # orange
    '>65': '#d62728'        # red
}

SEX_COLORS_NORMATIVE = {
    'M': 'blue',
    'F': 'red',
}

SEX_COLORS_PATIENTS = {
    'M': '#1f77b4',     # light blue
    'F': '#ff7f0e',     # orange
}

SEX_TO_LEGEND = {
    'M': 'Males',
    'F': 'Females',
}

# Color mapping for Myelopathy stratification
MYELOPATHY_COLORS = {
    'yes': '#d62728',      # red - has myelopathy
    'no': '#2ca02c',       # green - no myelopathy
}

NORMATIVE_C2_COLORS = {
    'Above normative mean C2 cord area': '#2ca02c',  # green
    'Below normative mean C2 cord area': '#d62728',      # red
}

# Color mapping for three-group comparison: DCM T2w+, DCM T2w-, HC
HC_COMPARISON_COLORS = {
    'DCM T2w+': '#d62728',   # red
    'DCM T2w-': '#2ca02c',   # green
    'HC': '#7f7f7f',        # gray
}

THERAPEUTIC_DECISION_COLORS = {
    'operative': '#d62728',         # red
    'conservative': '#2ca02c',      # green
}

MJOA_COLORS = {
    'mJOA=18': '#2ca02c',    # green
    'mild (15 ≤ mJOA ≤ 18)': '#2ca02c',     # green (#2ca02c) or yellow (#ffdb4d) for mild
    'moderate (12 ≤ mJOA ≤ 14)': '#ff7f0e',     # orange for moderate
    'severe (mJOA ≤ 11)': '#d62728',        # red for severe
    'unknown': '#7f7f7f'        # gray for unknown
}

METRICS_YLIMITS = {
    'spinal_cord': {
        'MEAN(diameter_AP)': (5, 9),
        'MEAN(area)': (40, 90),
        'MEAN(diameter_RL)': (9, 15.5),
        'MEAN(eccentricity)': (0.53, 0.91),
        'MEAN(solidity)': (89, 100),
        'MEAN(compression_ratio)': (0.40, 0.86)
    },
    'canal': {
        'MEAN(diameter_AP)': (8, 15.5),
        'MEAN(area)': (110, 250),
        'MEAN(diameter_RL)': (16, 26),
        'MEAN(eccentricity)': (0.4, 0.8),
        'MEAN(solidity)': (89, 100),
        'MEAN(compression_ratio)': (0.4, 0.75)
    },
    'aSCOR': {
        'aSCOR': (0.15, 0.5)
    }
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot mean and std of morphometric metrics across subjects for multiple sessions")
    parser.add_argument('-i', required=True, type=str,
                        help="CSV file with patients' morphometric metrics in the PAM50 space across multiple subjects")
    parser.add_argument('-o', required=True, type=str, default='figures',
                        help="Output directory name. The figure name will be based on the input CSV file name. "
                             "Default output directory: figures.")
    parser.add_argument('-s', required=False, type=str, nargs='+',
                        default=['ses-M0'],
                        help="Session to process (e.g., 'ses-M0', 'ses-M3', etc.)",
                        )
    parser.add_argument('-exclude-file', required=False, type=str,
                        default='$HOME/code/dcm-metric-normalization/scripts/exclude_dcm-zurich.yml',
                        help="YAML file with subjects to exclude")
    parser.add_argument('-c2c3-file', required=False, type=str,
                        default='$HOME/code/dcm-metric-normalization/scripts/dcm-zurich_T2w_ax_ses-M0_canal_analysis.txt',
                        help="File with list of subjects to use C2 or C3 vert level.")
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('-clinical-file', required=False, type=str,
                        help="Excel file with clinical scores (must contain 'total_mjoa_BL' column) and demographic data")
    parser.add_argument('-ascor-file', required=False, type=str,
                        help="CSV file with aSCOR metrics (for longitudinal analysis as covariate)")
    parser.add_argument('-path-HC-spinegeneric', required=False, type=str,
                        default='$HOME/code/PAM50-normalized-metrics',
                        help="Base path to spine-generic multi-subject HC data "
                             "(expects <base>/<structure>/spine-generic_multi-subject/*.csv). "
                             "Used for myelopathy_with_hc stratification.")
    parser.add_argument('-stratify', required=False, type=str, default=None,
                        choices=['mcl', 'highest_stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'num_of_stenosis_including_C2C3', 'myelopathy',
                                 'myelopathy_with_hc', 'mjoa', 'therapeutic_decision', 'age', 'sex', 'normative_mean_c2', 'None'],
                        help="Stratification method:"
                             "'mcl' for Maximum Compression Level; "
                             "'highest_stenosis' for the highest stenosis level; "
                             "'num_of_stenosis' for number of stenosis levels; "
                             "'single_vs_multi_stenosis' for single vs. multi-level stenosis; "
                             "'num_of_stenosis_including_C2C3' for number of stenosis levels including stratification of subjects with 4 compressions to see if they have compression at C2/C3 level; "
                             "'myelopathy' for myelopathy status; "
                             "'therapeutic_decision' (operative/conservative); "
                             "'age' for age group stratification; "
                             "'sex' for sex-based stratification; "
                             "'mjoa' mJOA (mild: 15 ≤ mJOA ≤ 18; moderate 14 ≤ mJOA) "
                             "'normative_mean_c2' for stratification based on normative mean C2 cord area; "
                             "'None' for no stratification."
                             "Default: None.",
                             )
    parser.add_argument('--run-ctree', action='store_true',
                        help="Run URP-CTREE in R (partykit::ctree) to find the compression ratio cutoff "
                             "at C3 that best predicts T2w signal change. Requires R >= 4.0 and the "
                             "partykit package (auto-installed on first run). "
                             "See plotting/run_ctree_compression_ratio.R for the R code.")

    return parser


def get_vert_indices(df):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get unique participant IDs
    subjects = df['participant_id'].unique()
    # Get vert levels for one certain subject
    vert = df[df['participant_id'] == subjects[0]]['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def _read_pam50_df(path_HC):
    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file), dtype=METRICS_DTYPE)
            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)

    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])

    return df


def load_normative_data(path_HC, path_participants_pam50, structure, vert_min, vert_max):
    """
    Load normative data from spine-generic dataset in PAM50 space
    :param path_HC:
    :param path_participants_pam50:
    :param structure: 'spinal_cord' or 'canal' or 'aSCOR'
    :param vert_min: minimum vertebral level to keep (e.g., 2 for C2)
    :param vert_max: maximum vertebral level to keep (e.g., 6 for C6)
    :return:
    """

    if structure == 'aSCOR':
        # For aSCOR, we need both spinal_cord and canal metrics to compute aSCOR
        df_cord = _read_pam50_df(os.path.join(path_HC, 'spinal_cord'))
        df_canal = _read_pam50_df(os.path.join(path_HC, 'canal'))
        df = pd.merge(df_cord, df_canal, on=['participant_id', 'Slice (I->S)', 'VertLevel'],
                      suffixes=('_sc', '_canal'))
        df['aSCOR'] = df['MEAN(area)_sc'].div(df['MEAN(area)_canal'], fill_value=0)
        df['aSCOR'].replace(np.inf, np.nan)  # output nan instead of inf for /0
    # Add spinal_cord or canal subfolder to the path
    else:
        df = _read_pam50_df(os.path.join(path_HC, structure))

    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if path_participants_pam50:
        df_participants = pd.read_csv(path_participants_pam50, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id"]],
                      on='participant_id')
        # Recode age into age bins by 10 years (decades)
        df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=AGE_DECADES)

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    # Keep only specified VertLevels (C2-C6 for spinal cord; C2-C3 for canal and aSCOR)
    df = df[df['VertLevel'] >= vert_min]
    df = df[df['VertLevel'] <= vert_max]

    df_spine_generic_min, df_spine_generic_max = df['Slice (I->S)'].min(), df['Slice (I->S)'].max()

    if structure == 'spinal_cord' or structure == 'canal':
        # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
        df['MEAN(compression_ratio)'] = df['MEAN(diameter_AP)'] / df['MEAN(diameter_RL)']
        # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
        df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    # Uncomment to save aggregated dataframe with metrics across all subjects as .csv file
    #df.to_csv(os.path.join(path_out_csv, 'HC_metrics.csv'), index=False)

    return df, df_spine_generic_min, df_spine_generic_max


def fetch_participant_and_session(filename_path):
    """
    Get participant_id, session_ide and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: participant_id, session_id (e.g., sub-001, ses-01)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    participant_tmp = re.search('sub-(.*?)[_/]', filename_path)
    participant_id = participant_tmp.group(0)[:-1] if participant_tmp else ""    # [:-1] removes the last underscore or slash

    session_tmp = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    session_id = session_tmp.group(0)[:-1] if session_tmp else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return participant_id, session_id


def merge_morphometrics_and_clinical_data(subjects_df, df_clinical, clinical_columns):
    """
    Merge clinical data into dataframe with morphometrics
    """
    subjects_df = subjects_df.merge(
        df_clinical[['participant_id', 'maximum_stenosis', 'stenosis_levels', 'highest_stenosis', 'num_of_stenosis',
                     'single_vs_multi_stenosis', 'Myelopathy', 'therapeutic_decision', 'surgery_before_baseline',
                     'age', 'age_group', 'sex', 'mJOA_severity_bl'] + clinical_columns],
        on='participant_id', how='inner'
    )

    # Rename maximum_stenosis to MCL
    subjects_df = subjects_df.rename(columns={'maximum_stenosis': 'MCL'})
    print(f'Number of subjects after merging MRI and clinical data: {len(subjects_df['participant_id'].unique())}')

    return subjects_df


def read_c2c3_file_and_apply_exclusions(subjects_df, c2c3_file):
    """
    Read C2C3 file and apply exclusions to subjects_df
    """
    c2c3_ids = pd.read_csv(c2c3_file, sep=r"\s+", header=None, names=["participant_id", "level_to_use"])
    # Merge with subjects_df
    subjects_df = subjects_df.merge(c2c3_ids, on='participant_id', how='left')
    # Drop rows with level_to_use == 'exclude'
    subjects_df = subjects_df[subjects_df['level_to_use'] != 'exclude']
    print(f"Number of unique subjects after applying C2,C3 level exclusions: {len(subjects_df['participant_id'].unique())}")
    # Drop C2 for subjects with level_to_use == 'C3'
    print(f"C2 level: Number of unique subjects before dropping subjects with missing C2 level: {len(subjects_df[subjects_df['VertLevel'] == 2]['participant_id'].unique())}")
    subjects_df = subjects_df[~((subjects_df['level_to_use'] == 'C3') & (subjects_df['VertLevel'] == 2))]
    print(f"C2 level: Number of unique subjects after dropping subjects with missing C2 level: {len(subjects_df[subjects_df['VertLevel'] == 2]['participant_id'].unique())}")

    return subjects_df


def _build_age_group_compression_table(subjects_df, output_csv_path):
    """Create publication-ready table summarizing compression stats per age group and save to CSV.
    Columns include:
        age_group, n_subjects_total, n_subjects_with_stenosis_data,
        single_stenosis_count, Multi-level stenosis count:,
        count_num_of_stenosis_1..4, Highest stenosis C2_C3 .. C6_C7
    If stenosis data missing for all subjects in an age group, counts are 0 and a note column is added.
    """
    age_groups = ['<50', '50-65', '>65']
    stenosis_levels_order = ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7']
    rows = []
    # Work at subject level (collapse duplicates)
    subj_df = subjects_df[['participant_id', 'age_group']].drop_duplicates()
    # Bring stenosis-related columns if they exist (merge unique per subject)
    for col in ['stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'highest_stenosis', 'stenosis_levels']:
        if col in subjects_df.columns:
            subj_df = subj_df.merge(subjects_df[['participant_id', col]].drop_duplicates('participant_id'), on='participant_id', how='left')
    # Derive missing columns if possible
    if 'stenosis_levels' not in subj_df.columns and 'stenosis' in subj_df.columns:
        subj_df['stenosis_levels'] = subj_df['stenosis'].apply(lambda x: [lvl.strip() for lvl in str(x).split(',')] if pd.notna(x) and x != 'NA' else [])
    if 'num_of_stenosis' not in subj_df.columns and 'stenosis_levels' in subj_df.columns:
        subj_df['num_of_stenosis'] = subj_df['stenosis_levels'].apply(len)
    if 'single_vs_multi_stenosis' not in subj_df.columns and 'num_of_stenosis' in subj_df.columns:
        subj_df['single_vs_multi_stenosis'] = subj_df['num_of_stenosis'].apply(lambda x: 'Single stenosis' if x == 1 else ('Multi-level stenosis' if x > 1 else 'Unknown'))
    if 'highest_stenosis' not in subj_df.columns and 'stenosis_levels' in subj_df.columns and '_get_highest_stenosis' in globals():
        subj_df['highest_stenosis'] = subj_df['stenosis_levels'].apply(_get_highest_stenosis)

    for age in age_groups:
        age_subj = subj_df[subj_df['age_group'] == age]
        total_n = len(age_subj)
        # Determine which subjects have valid stenosis data
        if 'num_of_stenosis' in age_subj.columns:
            valid_mask = age_subj['num_of_stenosis'].notna() & (age_subj['num_of_stenosis'] > 0)
        else:
            valid_mask = pd.Series([False]*len(age_subj), index=age_subj.index)
        with_data = int(valid_mask.sum())
        row = {
            'Age group': age,
            'Total number of subjects': total_n,
            'n_subjects_with_stenosis_data': with_data,
        }
        if with_data == 0:
            # Populate zeros
            row.update({
                'Single stenosis count': 0,
                'Multi-level stenosis count': 0
            })
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = 0
            for lvl in stenosis_levels_order:
                row[f'Highest stenosis:  {lvl}'] = 0
            row['note'] = 'No stenosis data'
        else:
            valid_df = age_subj.loc[valid_mask]
            # Distribution of number of stenosis levels (1..4)
            num_counts = valid_df['num_of_stenosis'].value_counts().to_dict()
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = int(num_counts.get(k, 0))
            # Single vs multi
            if 'single_vs_multi_stenosis' in valid_df.columns:
                sm_counts = valid_df['single_vs_multi_stenosis'].value_counts().to_dict()
                row['Single stenosis count'] = int(sm_counts.get('Single stenosis', 0))
                row['Multi-level stenosis count'] = int(sm_counts.get('Multi-level stenosis', 0))
            else:
                row['Single stenosis count'] = 0
                row['Multi-level stenosis count'] = 0
            # Highest stenosis distribution
            if 'highest_stenosis' in valid_df.columns:
                highest_counts = valid_df['highest_stenosis'].value_counts().to_dict()
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = int(highest_counts.get(lvl, 0))
            else:
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = 0
        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(output_csv_path, index=False)
    print(f"Age group compression summary table saved: {output_csv_path}")
    return table_df


def _save_age_group_compression_table_formatted(table_df, output_csv_path):
    """Save the age-group compression summary into a publication-ready multi-block CSV
    resembling the provided screenshot (separate header for each block and blank lines between blocks).
    Blocks:
      1) age_group | n_subjects_total
      2) age_group | Num of stenosis: 1..4 (+ corresponding _pct columns)
      3) age_group | Single stenosis count: | multi_level_stenosis_count (+ corresponding _pct columns)
      4) age_group | Highest stenosis C2/C3 .. Highest stenosis C6/C7 (+ corresponding _pct columns)
    Percentages are computed out of n_subjects_with_stenosis_data per age group.
    """
    age_groups = ['<50', '50-65', '>65']

    # Ensure age_group ordering and fill missing rows
    def _order(df, cols):
        out = df.copy()
        out = out[cols]
        out['Age group'] = pd.Categorical(out['Age group'], categories=age_groups, ordered=True)
        out = out.sort_values('Age group')
        # Reindex to guarantee all groups exist
        idx = pd.Index(age_groups, name='Age group')
        out = out.set_index('Age group').reindex(idx).reset_index()
        return out

    # Helper to compute percentage columns relative to n_subjects_with_stenosis_data
    denom = _order(table_df, ['Age group', 'n_subjects_with_stenosis_data'])
    denom = denom.rename(columns={'n_subjects_with_stenosis_data': '_den'})

    def _add_pct(block, count_cols):
        b = block.merge(denom, on='Age group', how='left')
        for c in count_cols:
            d = b['_den'].replace({0: np.nan})
            pct = (b[c] / d) * 100.0
            pct = pct.fillna(0).round(1)
            b[c + '_pct'] = pct
        b = b.drop(columns=['_den'])
        # Interleave counts and pct columns
        cols = ['Age group']
        for c in count_cols:
            cols += [c, c + '_pct']
        return b[cols]

    # Helper to combine count and pct into single cell like "14 (38.9%)"
    def _combine_counts_with_pct(block_df, count_cols):
        out = block_df.copy()
        for c in count_cols:
            out[c] = out.apply(lambda r: f"{int(r[c])} ({r[c + '_pct']:.1f}%)", axis=1)
            out = out.drop(columns=[c + '_pct'])
        # Keep columns ordered
        return out[['Age group'] + count_cols]

    # Block 1: total subjects (keep simple to match screenshot)
    cols_block1 = [c for c in ['Age group', 'Total number of subjects'] if c in table_df.columns]
    block1 = _order(table_df, cols_block1)

    # Block 2: distribution by number of stenosis levels (1..4)
    needed2 = ['Num of stenosis: 1', 'Num of stenosis: 2', 'Num of stenosis: 3', 'Num of stenosis: 4']
    for c in needed2:
        if c not in table_df.columns:
            table_df[c] = 0
    block2_counts = _order(table_df, ['Age group'] + needed2)
    block2 = _add_pct(block2_counts, needed2)
    block2 = _combine_counts_with_pct(block2, needed2)

    # Block 3: single vs multi
    needed3 = ['Single stenosis count', 'Multi-level stenosis count']
    for c in needed3:
        if c not in table_df.columns:
            table_df[c] = 0
    block3_counts = _order(table_df, ['Age group'] + needed3)
    block3 = _add_pct(block3_counts, needed3)
    block3 = _combine_counts_with_pct(block3, needed3)

    # Block 4: highest stenosis level distribution (add slashes and pct)
    internal4 = ['Highest stenosis: C2/C3', 'Highest stenosis: C3/C4', 'Highest stenosis: C4/C5', 'Highest stenosis: C5/C6', 'Highest stenosis: C6/C7']
    for c in internal4:
        if c not in table_df.columns:
            table_df[c] = 0
    block4_counts = _order(table_df, ['Age group'] + internal4)
    block4 = _add_pct(block4_counts, internal4)
    block4 = _combine_counts_with_pct(block4, internal4)

    # Rename to add slashes for readability (only combined columns now)
    rename_map = {}
    for c in internal4:
        with_slash = c.replace('_C', '_C').replace('C2_C3', 'C2/C3').replace('C3_C4', 'C3/C4').replace('C4_C5', 'C4/C5').replace('C5_C6', 'C5/C6').replace('C6_C7', 'C6/C7')
        rename_map[c] = with_slash
    block4 = block4.rename(columns=rename_map)

    # Write multi-block CSV with blank lines between blocks
    with open(output_csv_path, 'w', newline='') as f:
        block1.to_csv(f, index=False)
        f.write('\n\n')
        block2.to_csv(f, index=False)
        f.write('\n\n')
        block3.to_csv(f, index=False)
        f.write('\n\n')
        block4.to_csv(f, index=False)
    print(f"Publication-ready age group table (with percentages) saved: {output_csv_path}")

def _build_myelopathy_compression_table(subjects_df, output_csv_path):
    """Create publication-ready table summarizing compression stats per myelopathy group (yes/no) and save to CSV.
    Columns mirror age-group table but grouped by Myelopathy ('yes','no')."""
    groups = ['yes', 'no']  # keep order
    stenosis_levels_order = ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7']
    rows = []
    # Collapse to unique subject rows with relevant columns
    subj_df = subjects_df[['participant_id', 'Myelopathy']].drop_duplicates()
    for col in ['stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'highest_stenosis', 'stenosis_levels', 'MCL']:
        if col in subjects_df.columns:
            subj_df = subj_df.merge(subjects_df[['participant_id', col]].drop_duplicates('participant_id'), on='participant_id', how='left')
    if 'stenosis_levels' not in subj_df.columns and 'stenosis' in subj_df.columns:
        subj_df['stenosis_levels'] = subj_df['stenosis'].apply(lambda x: [lvl.strip() for lvl in str(x).split(',')] if pd.notna(x) and x != 'NA' else [])
    if 'num_of_stenosis' not in subj_df.columns and 'stenosis_levels' in subj_df.columns:
        subj_df['num_of_stenosis'] = subj_df['stenosis_levels'].apply(len)
    if 'single_vs_multi_stenosis' not in subj_df.columns and 'num_of_stenosis' in subj_df.columns:
        subj_df['single_vs_multi_stenosis'] = subj_df['num_of_stenosis'].apply(lambda x: 'Single stenosis' if x == 1 else ('Multi-level stenosis' if x > 1 else 'Unknown'))

    for grp in groups:
        g_df = subj_df[subj_df['Myelopathy'] == grp]
        total_n = len(g_df)
        if 'num_of_stenosis' in g_df.columns:
            valid_mask = g_df['num_of_stenosis'].notna() & (g_df['num_of_stenosis'] > 0)
        else:
            valid_mask = pd.Series([False]*len(g_df), index=g_df.index)
        with_data = int(valid_mask.sum())
        row = {
            'Myelopathy': grp,
            'Total number of subjects': total_n,
            'n_subjects_with_stenosis_data': with_data,
        }
        if with_data == 0:
            row.update({'Single stenosis count': 0, 'Multi-level stenosis count': 0})
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = 0
            for lvl in stenosis_levels_order:
                row[f'Highest stenosis: {lvl}'] = 0
            row['note'] = 'No stenosis data'
        else:
            valid_df = g_df.loc[valid_mask]
            num_counts = valid_df['num_of_stenosis'].value_counts().to_dict()
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = int(num_counts.get(k, 0))
            if 'single_vs_multi_stenosis' in valid_df.columns:
                sm_counts = valid_df['single_vs_multi_stenosis'].value_counts().to_dict()
                row['Single stenosis count'] = int(sm_counts.get('Single stenosis', 0))
                row['Multi-level stenosis count'] = int(sm_counts.get('Multi-level stenosis', 0))
            else:
                row['Single stenosis count'] = 0
                row['Multi-level stenosis count'] = 0
            if 'highest_stenosis' in valid_df.columns:
                highest_counts = valid_df['highest_stenosis'].value_counts().to_dict()
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = int(highest_counts.get(lvl, 0))
            else:
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = 0
            if 'MCL' in valid_df.columns:
                mcl_counts = valid_df['MCL'].value_counts().to_dict()
                for mcl in stenosis_levels_order:
                    row[f'MCL: {mcl}'] = int(mcl_counts.get(mcl, 0))
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Myelopathy compression summary table saved: {output_csv_path}")
    return out_df


def _save_myelopathy_compression_table_formatted(table_df, output_csv_path):
    """Formatted multi-block CSV for myelopathy compression summary (counts with percentages in single cells)."""
    # Rename 'Myelopathy' column to 'T2w hyperintensity' for display
    table_df = table_df.rename(columns={'Myelopathy': 'T2w hyperintensity'})
    groups = ['yes', 'no']
    def _order(df, cols):
        out = df.copy()[cols]
        out['T2w hyperintensity'] = pd.Categorical(out['T2w hyperintensity'], categories=groups, ordered=True)
        out = out.sort_values('T2w hyperintensity')
        idx = pd.Index(groups, name='T2w hyperintensity')
        out = out.set_index('T2w hyperintensity').reindex(idx).reset_index()
        return out
    denom = _order(table_df, ['T2w hyperintensity', 'n_subjects_with_stenosis_data']).rename(columns={'n_subjects_with_stenosis_data': '_den'})
    def _add_pct(block, count_cols):
        b = block.merge(denom, on='T2w hyperintensity', how='left')
        # Calculate total across all columns and all myelopathy categories
        grand_total = sum(b[c].sum() for c in count_cols)
        for c in count_cols:
            if grand_total == 0:
                pct = 0.0
            else:
                pct = (b[c] / grand_total) * 100.0
            pct = pct.round(1)
            b[c + '_pct'] = pct
        b = b.drop(columns=['_den'])
        cols = ['T2w hyperintensity']
        for c in count_cols:
            cols += [c, c + '_pct']
        return b[cols]
    def _combine(block_df, count_cols):
        out = block_df.copy()
        for c in count_cols:
            out[c] = out.apply(lambda r: f"{int(r[c])} ({r[c + '_pct']:.1f}%)", axis=1)
            out = out.drop(columns=[c + '_pct'])
        # Create row labels from T2w hyperintensity values and remove the column
        out['T2w hyperintensity'] = out['T2w hyperintensity'].map({'yes': 'Yes', 'no': 'No'})
        out = out.set_index('T2w hyperintensity')
        return out[count_cols]
    # Block1
    block1 = _combine(_add_pct(_order(table_df, ['T2w hyperintensity', 'Total number of subjects']), ['Total number of subjects']), ['Total number of subjects'])
    # Block2 num of stenosis
    needed2 = ['Num of stenosis: 1', 'Num of stenosis: 2', 'Num of stenosis: 3']
    for c in needed2:
        if c not in table_df.columns: table_df[c] = 0
    block2 = _combine(_add_pct(_order(table_df, ['T2w hyperintensity'] + needed2), needed2), needed2)
    # Block3 single vs multi
    needed3 = ['Single stenosis count', 'Multi-level stenosis count']
    for c in needed3:
        if c not in table_df.columns: table_df[c] = 0
    block3 = _combine(_add_pct(_order(table_df, ['T2w hyperintensity'] + needed3), needed3), needed3)
    # Block4 highest stenosis
    internal4 = ['Highest stenosis: C4/C5', 'Highest stenosis: C5/C6', 'Highest stenosis: C6/C7']
    for c in internal4:
        if c not in table_df.columns: table_df[c] = 0
    block4 = _combine(_add_pct(_order(table_df, ['T2w hyperintensity'] + internal4), internal4), internal4)
    # Block5 MCL
    internal5 = ['MCL: C4/C5', 'MCL: C5/C6', 'MCL: C6/C7']
    for c in internal5:
        if c not in table_df.columns: table_df[c] = 0
    block5 = _combine(_add_pct(_order(table_df, ['T2w hyperintensity'] + internal5), internal5), internal5)
    # Write multi-block CSV
    with open(output_csv_path, 'w', newline='') as f:
        # Write global header with T2w hyperintensity only once at the top
        f.write('T2w hyperintensity,Total number of subjects\n')

        # Write Block1 data without header
        for idx, row in block1.iterrows():
            f.write(f'{idx},' + ','.join(map(str, row.values)) + '\n')

        f.write('\n\n')
        # Write Block3 data without header - just column names and data
        f.write(',' + ','.join(block3.columns) + '\n')
        for idx, row in block3.iterrows():
            f.write(f'{idx},' + ','.join(map(str, row.values)) + '\n')

        f.write('\n\n')
        # Write Block2 data without header
        f.write(',' + ','.join(block2.columns) + '\n')
        for idx, row in block2.iterrows():
            f.write(f'{idx},' + ','.join(map(str, row.values)) + '\n')

        f.write('\n\n')
        # Write Block4 data without header
        f.write(',' + ','.join(block4.columns) + '\n')
        for idx, row in block4.iterrows():
            f.write(f'{idx},' + ','.join(map(str, row.values)) + '\n')

        f.write('\n\n')
        # Write Block5 data without header
        f.write(',' + ','.join(block5.columns) + '\n')
        for idx, row in block5.iterrows():
            f.write(f'{idx},' + ','.join(map(str, row.values)) + '\n')
    print(f"Publication-ready myelopathy table (with percentages) saved: {output_csv_path}")

def _build_mjoa_compression_table(subjects_df, output_csv_path):
    """
    Create a table summarizing compression stats per mJOA severity group and save to CSV.
    Columns mirror age/myelopathy tables but grouped by mJOA severity.
    Groups included: mild, moderate, severe (exclude 'unknown').
    """
    groups = ['mild (15 ≤ mJOA ≤ 18)', 'moderate (12 ≤ mJOA ≤ 14)', 'severe (mJOA ≤ 11)']
    stenosis_levels_order = ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7']
    rows = []
    subj_df = subjects_df[['participant_id', 'mJOA_severity_bl']].drop_duplicates()
    for col in ['stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'highest_stenosis', 'stenosis_levels']:
        if col in subjects_df.columns:
            subj_df = subj_df.merge(subjects_df[['participant_id', col]].drop_duplicates('participant_id'), on='participant_id', how='left')
    if 'stenosis_levels' not in subj_df.columns and 'stenosis' in subj_df.columns:
        subj_df['stenosis_levels'] = subj_df['stenosis'].apply(lambda x: [lvl.strip() for lvl in str(x).split(',')] if pd.notna(x) and x != 'NA' else [])
    if 'num_of_stenosis' not in subj_df.columns and 'stenosis_levels' in subj_df.columns:
        subj_df['num_of_stenosis'] = subj_df['stenosis_levels'].apply(len)
    if 'single_vs_multi_stenosis' not in subj_df.columns and 'num_of_stenosis' in subj_df.columns:
        subj_df['single_vs_multi_stenosis'] = subj_df['num_of_stenosis'].apply(lambda x: 'Single stenosis' if x == 1 else ('Multi-level stenosis' if x > 1 else 'Unknown'))
    if 'highest_stenosis' not in subj_df.columns and 'stenosis_levels' in subj_df.columns:
        subj_df['highest_stenosis'] = subj_df['stenosis_levels'].apply(_get_highest_stenosis)

    for grp in groups:
        g_df = subj_df[subj_df['mJOA_severity_bl'] == grp]
        total_n = len(g_df)
        if 'num_of_stenosis' in g_df.columns:
            valid_mask = g_df['num_of_stenosis'].notna() & (g_df['num_of_stenosis'] > 0)
        else:
            valid_mask = pd.Series([False]*len(g_df), index=g_df.index)
        with_data = int(valid_mask.sum())
        row = {
            'mJOA severity': grp,
            'Total number of subjects': total_n,
            'n_subjects_with_stenosis_data': with_data,
        }
        if with_data == 0:
            row.update({'Single stenosis count': 0, 'Multi-level stenosis count': 0})
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = 0
            for lvl in stenosis_levels_order:
                row[f'Highest stenosis: {lvl}'] = 0
            row['note'] = 'No stenosis data'
        else:
            valid_df = g_df.loc[valid_mask]
            num_counts = valid_df['num_of_stenosis'].value_counts().to_dict()
            for k in range(1,5):
                row[f'Num of stenosis: {k}'] = int(num_counts.get(k, 0))
            if 'single_vs_multi_stenosis' in valid_df.columns:
                sm_counts = valid_df['single_vs_multi_stenosis'].value_counts().to_dict()
                row['Single stenosis count'] = int(sm_counts.get('Single stenosis', 0))
                row['Multi-level stenosis count'] = int(sm_counts.get('Multi-level stenosis', 0))
            else:
                row['Single stenosis count'] = 0
                row['Multi-level stenosis count'] = 0
            if 'highest_stenosis' in valid_df.columns:
                highest_counts = valid_df['highest_stenosis'].value_counts().to_dict()
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = int(highest_counts.get(lvl, 0))
            else:
                for lvl in stenosis_levels_order:
                    row[f'Highest stenosis: {lvl}'] = 0
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False)
    print(f"mJOA compression summary table saved: {output_csv_path}")
    return out_df

def _save_mjoa_compression_table_formatted(table_df, output_csv_path):
    """Formatted multi-block CSV for mJOA compression summary (counts with percentages in single cells)."""
    groups = ['mild (15 ≤ mJOA ≤ 18)', 'moderate (12 ≤ mJOA ≤ 14)', 'severe (mJOA ≤ 11)']
    def _order(df, cols):
        out = df.copy()[cols]
        out['mJOA severity'] = pd.Categorical(out['mJOA severity'], categories=groups, ordered=True)
        out = out.sort_values('mJOA severity')
        idx = pd.Index(groups, name='mJOA severity')
        out = out.set_index('mJOA severity').reindex(idx).reset_index()
        return out
    denom = _order(table_df, ['mJOA severity', 'n_subjects_with_stenosis_data']).rename(columns={'n_subjects_with_stenosis_data': '_den'})
    def _add_pct(block, count_cols):
        b = block.merge(denom, on='mJOA severity', how='left')
        for c in count_cols:
            d = b['_den'].replace({0: np.nan})
            pct = (b[c] / d) * 100.0
            pct = pct.fillna(0).round(1)
            b[c + '_pct'] = pct
        b = b.drop(columns=['_den'])
        cols = ['mJOA severity']
        for c in count_cols:
            cols += [c, c + '_pct']
        return b[cols]
    def _combine(block_df, count_cols):
        out = block_df.copy()
        for c in count_cols:
            out[c] = out.apply(lambda r: f"{int(r[c])} ({r[c + '_pct']:.1f}%)", axis=1)
            out = out.drop(columns=[c + '_pct'])
        return out[['mJOA severity'] + count_cols]
    # Block1 total subjects
    block1 = _order(table_df, ['mJOA severity', 'Total number of subjects'])
    # Block2 num of stenosis
    needed2 = ['Num of stenosis: 1', 'Num of stenosis: 2', 'Num of stenosis: 3', 'Num of stenosis: 4']
    for c in needed2:
        if c not in table_df.columns: table_df[c] = 0
    block2 = _combine(_add_pct(_order(table_df, ['mJOA severity'] + needed2), needed2), needed2)
    # Block3 single vs multi
    needed3 = ['Single stenosis count', 'Multi-level stenosis count']
    for c in needed3:
        if c not in table_df.columns: table_df[c] = 0
    block3 = _combine(_add_pct(_order(table_df, ['mJOA severity'] + needed3), needed3), needed3)
    # Block4 highest stenosis
    internal4 = ['Highest stenosis: C2/C3', 'Highest stenosis: C3/C4', 'Highest stenosis: C4/C5', 'Highest stenosis: C5/C6', 'Highest stenosis: C6/C7']
    for c in internal4:
        if c not in table_df.columns: table_df[c] = 0
    block4 = _combine(_add_pct(_order(table_df, ['mJOA severity'] + internal4), internal4), internal4)
    with open(output_csv_path, 'w', newline='') as f:
        block1.to_csv(f, index=False)
        f.write('\n\n'); block2.to_csv(f, index=False)
        f.write('\n\n'); block3.to_csv(f, index=False)
        f.write('\n\n'); block4.to_csv(f, index=False)
    print(f"Publication-ready mJOA table (with percentages) saved: {output_csv_path}")

def load_hc_spinegeneric_perlevel(base_path, structure, vert_levels):
    """
    Load HC normative data from spine-generic multi-subject directory and aggregate per vertebral level.
    :param base_path: base directory (e.g., ~/code/PAM50-normalized-metrics)
    :param structure: 'spinal_cord' or 'canal'
    :param vert_levels: list of vertebral levels to keep (e.g., [3])
    :return: dataframe with per-participant per-level mean metrics, with 'group' = 'HC'
    """
    sg_dir = os.path.join(os.path.expanduser(base_path), structure, 'spine-generic_multi-subject')
    df = pd.DataFrame()
    for file in os.listdir(sg_dir):
        if 'PAM50.csv' in file:
            df_subj = pd.read_csv(os.path.join(sg_dir, file), dtype=METRICS_DTYPE)
            df = pd.concat([df, df_subj], axis=0, ignore_index=True)
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])
    df = df[df['VertLevel'].isin(vert_levels)]
    df = df.dropna(axis=1, how='all')
    # Compute derived metrics
    if structure in ['spinal_cord', 'canal']:
        df['MEAN(compression_ratio)'] = df['MEAN(diameter_AP)'] / df['MEAN(diameter_RL)']
        df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100
    # Aggregate per subject per level (mean across slices)
    metrics_to_agg = [m for m in ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)',
                                   'MEAN(compression_ratio)', 'MEAN(eccentricity)', 'MEAN(solidity)']
                      if m in df.columns]
    grouped = df.groupby(['participant_id', 'VertLevel'])[metrics_to_agg].mean().reset_index()
    grouped['group'] = 'HC'
    # Merge sex and age from participants.tsv
    participants_tsv = os.path.join(sg_dir, 'participants.tsv')
    if os.path.isfile(participants_tsv):
        df_participants = pd.read_csv(participants_tsv, sep='\t')
        grouped = grouped.merge(df_participants[['participant_id', 'sex', 'age']], on='participant_id', how='left')
    print(f"Loaded HC spine-generic data: {grouped['participant_id'].nunique()} subjects, levels: {sorted(grouped['VertLevel'].unique())}")
    return grouped


def create_figure(subjects_df, df_normative_data, sessions_to_process, figure_path, stratify_type=None, df_hc_perlevel=None):
    """
    Create figure with mean and std of morphometric metrics across subjects, separately for multiple sessions
    :param subjects_df: pandas dataframe with morphometric metrics across multiple subjects
    :param df_normative_data: pandas dataframe with normative data from spine-generic dataset
    :param sessions_to_process: list of sessions to process (e.g., ['ses-M0', 'ses-M3'])
    :param figure_path: path to save figure
    :param stratify_type: type of stratification ('mcl', 'myelopathy', 'therapeutic_decision', 'mjoa') or None
    :param df_hc_perlevel: per-level HC data for myelopathy_with_hc stratification (output of load_hc_spinegeneric_perlevel)
    """
    mpl.rcParams['font.family'] = 'Arial'

    # Fetch cord, canal, or aSCOR from the input filename to include in the figure title
    if 'cord' in figure_path:
        structure = 'spinal_cord'
    elif 'canal' in figure_path:
        structure = 'canal'
    elif 'aSCOR' in figure_path:
        structure = 'aSCOR'

    if structure == 'aSCOR':
        # 2x1 grid for 1 metric; 6x10
        fig, axs = plt.subplots(2, 1, figsize=(4, 10))
        top_axes = [axs[0]]
        bottom_axes = [axs[1]]
        METRICS = ['aSCOR']
    else:
        METRICS = [
            'MEAN(area)',
            'MEAN(diameter_AP)',
            'MEAN(diameter_RL)',
            'MEAN(compression_ratio)',
            # 'MEAN(eccentricity)',
            # 'MEAN(solidity)'
        ]
        # 2xN grid (N=len(METRICS))
        fig, axs = plt.subplots(2, int(len(METRICS)), figsize=(int(len(METRICS)) * 4.5, 10))
        if len(METRICS) == 1:
            top_axes = [axs[0]]
            bottom_axes = [axs[1]]
        else:
            top_axes = axs[0, :]
            bottom_axes = axs[1, :]

    for metric_idx, metric in enumerate(METRICS):
        ax = top_axes[metric_idx]

        # # Plot normative data
        # if stratify_type == 'sex':
        #     sex_groups = ['M', 'F']  # Ensure legend order
        #     for sex in sex_groups:
        #         normative_sex_data = df_normative_data[df_normative_data['sex'] == sex]
        #         sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=normative_sex_data, errorbar='sd',
        #                      linewidth=4, color=SEX_COLORS_NORMATIVE[sex], linestyle='--',
        #                      label=f'Normative Data {SEX_TO_LEGEND[sex]} (n={len(normative_sex_data["participant_id"].unique())})')
        # else:
        #     sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df_normative_data, errorbar='sd',
        #                  linewidth=4, color='black',
        #                  label=f'Normative Data (n={len(df_normative_data["participant_id"].unique())})')

        if stratify_type == 'mcl':
            # Plot by MCL groups instead of sessions
            mcl_groups = subjects_df['MCL'].unique()
            mcl_groups = sorted([mcl for mcl in mcl_groups if mcl in MCL_COLORS])

            for mcl in mcl_groups:
                mcl_data = subjects_df[subjects_df['MCL'] == mcl]
                if len(mcl_data) > 0:
                    mcl_n_subjects = len(mcl_data['participant_id'].unique())
                    print(f"MCL group '{mcl}': {mcl_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=mcl_data, errorbar='sd',
                                linewidth=4, color=MCL_COLORS[mcl],
                                label=f"MCL {mcl} (n={mcl_n_subjects})")
        elif stratify_type == 'highest_stenosis':
            # Plot by Stenosis groups instead of sessions
            stenosis_levels = ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7']     # Ensure legend order
            for level in stenosis_levels:
                stenosis_data = subjects_df[subjects_df['highest_stenosis'] == level]
                if len(stenosis_data) > 0:
                    stenosis_n_subjects = len(stenosis_data['participant_id'].unique())
                    print(f"Stenosis group '{level}': {stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=stenosis_data, errorbar='sd',
                                linewidth=4, color=MCL_COLORS[level],
                                label=f"Highest Stenosis at {level} (n={stenosis_n_subjects})")
        elif stratify_type == 'num_of_stenosis':
            # Plot by Number of Stenosis groups instead of sessions
            num_stenosis_groups = sorted(subjects_df['num_of_stenosis'].unique())
            for num_stenosis in num_stenosis_groups:
                num_stenosis_data = subjects_df[subjects_df['num_of_stenosis'] == num_stenosis]
                if len(num_stenosis_data) > 0:
                    num_stenosis_n_subjects = len(num_stenosis_data['participant_id'].unique())
                    print(f"Number of Stenosis '{num_stenosis}': {num_stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=num_stenosis_data, errorbar='sd',
                                linewidth=4, label=f"Number of Stenosis = {num_stenosis} (n={num_stenosis_n_subjects})")
        elif stratify_type == 'single_vs_multi_stenosis':
            # Plot by Number of Stenosis groups instead of sessions
            num_stenosis_groups = ['Single stenosis', 'Multi-level stenosis']
            for num_stenosis in num_stenosis_groups:
                num_stenosis_data = subjects_df[subjects_df['single_vs_multi_stenosis'] == num_stenosis]
                if len(num_stenosis_data) > 0:
                    num_stenosis_n_subjects = len(num_stenosis_data['participant_id'].unique())
                    print(f"Number of Stenosis '{num_stenosis}': {num_stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=num_stenosis_data, errorbar='sd',
                                linewidth=4, label=f"{num_stenosis} (n={num_stenosis_n_subjects})")
        elif stratify_type == 'num_of_stenosis_including_C2C3':
            num_stenosis_groups = ['1', '2', '3', '4', '4 including C2/C3 or C3/C4']     # ensure order
            for num_stenosis in num_stenosis_groups:
                num_stenosis_data = subjects_df[subjects_df['num_of_stenosis_including_C2C3'] == num_stenosis]
                if len(num_stenosis_data) > 0:
                    num_stenosis_n_subjects = len(num_stenosis_data['participant_id'].unique())
                    print(f"Number of Stenosis '{num_stenosis}': {num_stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=num_stenosis_data, errorbar='sd',
                                linewidth=4, label=f"Number of Stenosis = {num_stenosis} (n={num_stenosis_n_subjects})")
        elif stratify_type == 'myelopathy':
            # Plot by Myelopathy groups instead of sessions
            myelopathy_groups = subjects_df['Myelopathy'].unique()
            myelopathy_groups = sorted([myelopathy for myelopathy in myelopathy_groups if myelopathy in MYELOPATHY_COLORS])

            for myelopathy in myelopathy_groups:
                myelopathy_data = subjects_df[subjects_df['Myelopathy'] == myelopathy]
                if len(myelopathy_data) > 0:
                    myelopathy_n_subjects = len(myelopathy_data['participant_id'].unique())
                    print(f"Myelopathy group '{myelopathy}': {myelopathy_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=myelopathy_data, errorbar='sd',
                                linewidth=4, color=MYELOPATHY_COLORS[myelopathy],
                                label=f"T2w- (n={myelopathy_n_subjects})" if myelopathy == 'no' else f"T2w+ (n={myelopathy_n_subjects})")
        elif stratify_type == 'myelopathy_with_hc':
            # Plot HC normative line first
            if metric in df_normative_data.columns:
                n_hc = len(df_normative_data['participant_id'].unique())
                sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df_normative_data, errorbar='sd',
                            linewidth=4, color=HC_COMPARISON_COLORS['HC'],
                            label=f'HC (n={n_hc})')
            # Then DCM T2w- and T2+
            for myelopathy, group_label in [('no', 'DCM T2w-'), ('yes', 'DCM T2w+')]:
                myelopathy_data = subjects_df[subjects_df['Myelopathy'] == myelopathy]
                if len(myelopathy_data) > 0:
                    n_subj = len(myelopathy_data['participant_id'].unique())
                    print(f"Myelopathy group '{myelopathy}': {n_subj} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=myelopathy_data, errorbar='sd',
                                linewidth=4, color=HC_COMPARISON_COLORS[group_label],
                                label=f'{group_label} (n={n_subj})')
        elif stratify_type == 'therapeutic_decision':
            # Plot by Therapeutic Decision groups instead of sessions
            decision_groups = subjects_df['therapeutic_decision'].unique()
            decision_groups = sorted([decision for decision in decision_groups if decision in THERAPEUTIC_DECISION_COLORS])

            for decision in decision_groups:
                decision_data = subjects_df[subjects_df['therapeutic_decision'] == decision]
                if len(decision_data) > 0:
                    decision_n_subjects = len(decision_data['participant_id'].unique())
                    print(f"Therapeutic Decision group '{decision}': {decision_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=decision_data, errorbar='sd',
                                linewidth=4, color=THERAPEUTIC_DECISION_COLORS[decision],
                                label=f"{decision} (n={decision_n_subjects})")

        elif stratify_type == 'mjoa':
            # Plot by mJOA severity groups instead of sessions
            mjoa_groups = subjects_df['mJOA_severity_bl'].unique()
            # Filter out 'unknown' and 'severe' groups, and only keep those in MJOA_COLORS
            mjoa_groups = sorted([mjoa for mjoa in mjoa_groups
                                  if mjoa in MJOA_COLORS
                                  and mjoa not in ['unknown', 'severe (mJOA ≤ 11)']])

            for mjoa in mjoa_groups:
                mjoa_data = subjects_df[subjects_df['mJOA_severity_bl'] == mjoa]
                if len(mjoa_data) > 0:
                    mjoa_n_subjects = len(mjoa_data['participant_id'].unique()) if metric == 'MEAN(area)' else None
                    print(f"mJOA severity group '{mjoa}': {mjoa_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=mjoa_data, errorbar='sd',
                                linewidth=4, color=MJOA_COLORS[mjoa],
                                label=f"{mjoa} (n={mjoa_n_subjects})")
        elif stratify_type == 'age':
            # Plot by age groups
            age_groups = ['<50', '50-65', '>65']  # Ensure legend order
            for age in age_groups:
                age_data = subjects_df[subjects_df['age_group'] == age]
                if len(age_data) > 0:
                    age_n_subjects = len(age_data['participant_id'].unique())
                    print(f"Age group '{age}': {age_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=age_data, errorbar='sd',
                                linewidth=4, color=AGE_GROUP_COLORS[age],
                                label=f"Age {age} (n={age_n_subjects})")
        elif stratify_type == 'sex':
            sex_groups = ['M', 'F']  # Ensure legend order
            for sex in sex_groups:
                sex_data = subjects_df[subjects_df['sex'] == sex]
                if len(sex_data) > 0:
                    sex_n_subjects = len(sex_data['participant_id'].unique())
                    print(f"Sex group '{sex}': {sex_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=sex_data, errorbar='sd',
                                linewidth=4, color=SEX_COLORS_PATIENTS[sex],
                                 label=f"{SEX_TO_LEGEND[sex]} (n={sex_n_subjects})")
        elif stratify_type == 'normative_mean_c2':
            c2_groups = subjects_df['normative_mean_c2'].unique()
            for c2_group in c2_groups:
                c2_data = subjects_df[subjects_df['normative_mean_c2'] == c2_group]
                if len(c2_data) > 0:
                    c2_n_subjects = len(c2_data['participant_id'].unique())
                    print(f"Normative mean C2 group '{c2_group}': {c2_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=c2_data, errorbar='sd',
                                linewidth=4, color=NORMATIVE_C2_COLORS[c2_group],
                                label=f"{c2_group} (n={c2_n_subjects})")
        else:
            # Plot each session's mean and std (original behavior)
            for ses in sessions_to_process:
                session_data = subjects_df[subjects_df['session_id'] == ses]
                if len(session_data) > 0:
                    ses_n_subjects = len(session_data['participant_id'].unique())
                    print(f"Session '{ses}': {ses_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=session_data, errorbar='sd',
                                linewidth=4, color=SESSION_COLORS[ses],
                                label=f"{ses} (n={ses_n_subjects})")

        # Keep the legend only for one plot to avoid duplication
        plot_to_keep_legend = 0 if structure == 'aSCOR'else (2 if (stratify_type and 'stenosis' in stratify_type or 'mcl' in stratify_type) else 0)
        if metric_idx == plot_to_keep_legend:
            # top_axes[metric_idx].legend(fontsize=TICKS_FONT_SIZE, title="mean ± std across subjects", title_fontsize=TICKS_FONT_SIZE)
            top_axes[metric_idx].legend(fontsize=TICKS_FONT_SIZE)
            # Change legend transparency
            leg = top_axes[metric_idx].get_legend()
            leg.get_frame().set_alpha(1.0)
        else:
            leg = top_axes[metric_idx].get_legend()
            if leg is not None:
                leg.remove()

        # Tweak y-axis limits
        ax.set_ylim(METRICS_YLIMITS[structure][metric][0], METRICS_YLIMITS[structure][metric][1])
        # # Remove first and last 4 slices from the x-axis to match single subject figure (to remove smoothing artifacts)
        # ax.set_xlim(df_normative_data['Slice (I->S)'].iloc[4], df_normative_data['Slice (I->S)'].iloc[-4])

        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        # Remove xticks to hide PAM50 Axial Slice numbers (top row)
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # # Add vertebral level indicators (top row)
        # ymin, ymax = ax.get_ylim()
        # vert, ind_vert, ind_vert_mid = get_vert_indices(df_normative_data)
        # for idx, x in enumerate(ind_vert[1:-1]):
        #     ax.axvline(df_normative_data.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        # for idx, x in enumerate(ind_vert_mid, 0):
        #     x_pos = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
        #     y_pos = ymin - (ymax - ymin) * 0.1  # to move below x-axis
        #     ax.text(df_normative_data.loc[ind_vert_mid[idx], 'Slice (I->S)'],
        #             y_pos, x_pos, horizontalalignment='center',
        #             verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.invert_xaxis()
        # Remove xlabel (top row)
        ax.set_xlabel('')

        # ----------------------
        # Bottom row: stratified violin plots per level
        # ----------------------
        ax_violin = bottom_axes[metric_idx]
        # Choose session for per-level mean (use first requested session if available)
        violin_df = subjects_df.copy()
        if 'session_id' in violin_df.columns and sessions_to_process:
            sel_ses = sessions_to_process[0]
            violin_df = violin_df[violin_df['session_id'] == sel_ses]
        # Compute per-participant per-level mean for the metric
        grouped = violin_df[['participant_id', 'VertLevel', metric]].dropna().groupby(['participant_id', 'VertLevel'], as_index=False).mean()
        # Attach stratification columns (per participant) if needed
        # Merge available grouping columns from the original df (drop duplicates per participant)
        merge_cols = ['participant_id']
        possible_group_cols = ['MCL', 'highest_stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis',
                               'num_of_stenosis_including_C2C3', 'Myelopathy', 'therapeutic_decision',
                               'mJOA_severity_bl', 'age_group', 'sex', 'normative_mean_c2', 'age']
        available_cols = [c for c in possible_group_cols if c in subjects_df.columns]
        merge_cols += available_cols
        per_participant = subjects_df[merge_cols].drop_duplicates('participant_id')
        grouped = grouped.merge(per_participant, on='participant_id', how='left')
        # Keep only specified vertebral levels (C2-C6 for spinal cord, C2-C3 for canal/aSCOR)
        # level_order_nums = [2, 3, 4, 5, 6] if structure == 'spinal_cord' else [2, 3]        # [2, 3, 4, 5, 6, 7]
        level_order_nums = [3]
        level_order_labels = [f'C{v}' for v in level_order_nums]
        grouped = grouped[grouped['VertLevel'].isin(level_order_nums)]
        grouped['Level'] = pd.Categorical([f'C{int(v)}' for v in grouped['VertLevel']], categories=level_order_labels, ordered=True)

        hue = None
        palette = None
        hue_order = None

        if stratify_type == 'mcl' and 'MCL' in grouped.columns:
            hue = 'MCL'
            hue_order = [lvl for lvl in ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7'] if lvl in grouped['MCL'].unique()]
            palette = {k: v for k, v in MCL_COLORS.items() if k in hue_order}
        elif stratify_type == 'highest_stenosis' and 'highest_stenosis' in grouped.columns:
            hue = 'highest_stenosis'
            hue_order = [lvl for lvl in ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7'] if lvl in grouped['highest_stenosis'].unique()]
            palette = {k: v for k, v in MCL_COLORS.items() if k in hue_order}
        elif stratify_type == 'num_of_stenosis' and 'num_of_stenosis' in grouped.columns:
            hue = 'num_of_stenosis'
            hue_order = sorted(grouped['num_of_stenosis'].dropna().unique().tolist())
            palette = None  # use seaborn default
        elif stratify_type == 'single_vs_multi_stenosis' and 'single_vs_multi_stenosis' in grouped.columns:
            hue = 'single_vs_multi_stenosis'
            hue_order = ['Single stenosis', 'Multi-level stenosis']
            hue_order = [h for h in hue_order if h in grouped[hue].unique()]
            palette = None
        elif stratify_type == 'num_of_stenosis_including_C2C3' and 'num_of_stenosis_including_C2C3' in grouped.columns:
            hue = 'num_of_stenosis_including_C2C3'
            hue_order = ['1', '2', '3', '4', '4 including C2/C3 or C3/C4']
            hue_order = [h for h in hue_order if h in grouped[hue].unique()]
            palette = None
        elif stratify_type == 'myelopathy' and 'Myelopathy' in grouped.columns:
            hue = 'Myelopathy'
            hue_order = [h for h in ['no', 'yes'] if h in grouped[hue].unique()]
            palette = {k: v for k, v in MYELOPATHY_COLORS.items() if k in hue_order}
        elif stratify_type == 'myelopathy_with_hc' and 'Myelopathy' in grouped.columns:
            # Add group column to DCM data (T2w+ / T2w-)
            grouped['group'] = grouped['Myelopathy'].map({'yes': 'DCM T2w+', 'no': 'DCM T2w-'})
            # Append HC per-level data; HC rows get NaN for clinical columns not present in hc_subset
            if df_hc_perlevel is not None and metric in df_hc_perlevel.columns:
                hc_subset = df_hc_perlevel[df_hc_perlevel['VertLevel'].isin(level_order_nums)].copy()
                hc_subset['Level'] = pd.Categorical(
                    [f'C{int(v)}' for v in hc_subset['VertLevel']],
                    categories=level_order_labels, ordered=True)
                # Retain all DCM columns so covariates (sex, age) remain available for DCM-only OLS
                grouped = pd.concat([grouped, hc_subset], ignore_index=True)
            hue = 'group'
            hue_order = [h for h in ['HC', 'DCM T2w-', 'DCM T2w+'] if h in grouped['group'].unique()]
            palette = {k: v for k, v in HC_COMPARISON_COLORS.items() if k in hue_order}
        elif stratify_type == 'therapeutic_decision' and 'therapeutic_decision' in grouped.columns:
            hue = 'therapeutic_decision'
            hue_order = [h for h in ['conservative', 'operative'] if h in grouped[hue].unique()]
            palette = {k: v for k, v in THERAPEUTIC_DECISION_COLORS.items() if k in hue_order}
        elif stratify_type == 'mjoa' and 'mJOA_severity_bl' in grouped.columns:
            hue = 'mJOA_severity_bl'
            allowed = [k for k in MJOA_COLORS.keys() if k not in ['unknown', 'severe (mJOA ≤ 11)']]
            hue_order = [h for h in allowed if h in grouped[hue].unique()]
            palette = {k: v for k, v in MJOA_COLORS.items() if k in hue_order}
        elif stratify_type == 'age' and 'age_group' in grouped.columns:
            hue = 'age_group'
            hue_order = [h for h in ['<50', '50-65', '>65'] if h in grouped[hue].unique()]
            palette = {k: v for k, v in AGE_GROUP_COLORS.items() if k in hue_order}
        elif stratify_type == 'sex' and 'sex' in grouped.columns:
            hue = 'sex'
            hue_order = [h for h in ['M', 'F'] if h in grouped[hue].unique()]
            palette = {k: v for k, v in SEX_COLORS_PATIENTS.items() if k in hue_order}
        elif stratify_type == 'normative_mean_c2' and 'normative_mean_c2' in grouped.columns:
            hue = 'normative_mean_c2'
            hue_order = [h for h in NORMATIVE_C2_COLORS.keys() if h in grouped[hue].unique()]
            palette = {k: v for k, v in NORMATIVE_C2_COLORS.items() if k in hue_order}

        if hue is None:
            sns.violinplot(ax=ax_violin, data=grouped, x='Level', y=metric, order=level_order_labels,
                           inner='box', cut=0, linewidth=2, fill=False)
        else:
            sns.violinplot(ax=ax_violin, data=grouped, x='Level', y=metric, hue=hue, order=level_order_labels,
                           hue_order=hue_order, palette=palette, inner='box', cut=0, linewidth=2, dodge=True, fill=False)

        # # Legend handling for bottom row
        # if metric_idx == plot_to_keep_legend and hue is not None:
        #     ax_violin.legend(fontsize=TICKS_FONT_SIZE)#, title=hue.replace('_', ' '), title_fontsize=TICKS_FONT_SIZE)
        # else:
        #     leg2 = ax_violin.get_legend()
        #     if leg2 is not None:
        #         leg2.remove()
        if hue is not None:
            leg2 = ax_violin.get_legend()
            leg2.remove()

        # Statistical comparison (per level)
        # Marks '*' if p < 0.05.
        if hue is not None:
            unique_groups = (hue_order if hue_order is not None else grouped[hue].dropna().unique().tolist())
            unique_groups = [g for g in unique_groups if g in grouped[hue].dropna().unique().tolist()]
            tick_labels = [t.get_text() for t in ax_violin.get_xticklabels()]
            tick_pos_map = dict(zip(tick_labels, ax_violin.get_xticks()))

            if stratify_type == 'myelopathy_with_hc' and hue == 'group':
                # Three pairwise OLS tests, all uncorrected (p < 0.05):
                #   DCM T2w- vs DCM T2w+: OLS controlling for sex, MCL, mJOA, age (consistent with -stratify myelopathy)
                #   HC vs DCM T2w-      : OLS controlling for sex, age
                #   HC vs DCM T2w+      : OLS controlling for sex, age
                n_groups = len(hue_order)
                group_width = 0.8 / n_groups
                group_pos_idx = {g: i for i, g in enumerate(hue_order)}
                ymin, ymax = ax_violin.get_ylim()
                yrange = ymax - ymin if ymax > ymin else 1.0

                for lvl_label in level_order_labels:
                    tick_x = tick_pos_map.get(lvl_label, None)
                    if tick_x is None:
                        continue
                    pvals = {}       # (g1, g2) -> raw p
                    comp_results = []  # list of dicts for formatted summary

                    # --- DCM T2w- vs DCM T2w+: OLS with covariates (mirrors binary myelopathy case) ---
                    dcm_df = grouped[grouped['group'].isin(['DCM T2w-', 'DCM T2w+']) &
                                     (grouped['Level'] == lvl_label)].copy()
                    dcm_df['grp'] = (dcm_df['group'] == 'DCM T2w+').astype(int)
                    cov_terms = []
                    if 'sex' in dcm_df.columns:
                        cov_terms.append('C(sex)')
                    if 'MCL' in dcm_df.columns:
                        cov_terms.append('C(MCL)')
                    if 'mJOA_severity_bl' in dcm_df.columns:
                        cov_terms.append('C(mJOA_severity_bl)')
                    if 'age' in dcm_df.columns:
                        dcm_df['age_c'] = pd.to_numeric(dcm_df['age'], errors='coerce')
                        dcm_df['age_c'] -= dcm_df['age_c'].mean()
                        cov_terms.append('age_c')
                    needed = ['grp', metric] + [ct.split('(')[-1].rstrip(')') if ct.startswith('C(') else ct
                                                for ct in cov_terms]
                    df_model = dcm_df.dropna(subset=needed).copy().rename(columns={metric: 'metric_value'})
                    formula = 'metric_value ~ grp' + (' + ' + ' + '.join(cov_terms) if cov_terms else '')
                    n_t2neg = int((df_model['grp'] == 0).sum())
                    n_t2pos = int((df_model['grp'] == 1).sum())
                    if n_t2neg >= 3 and n_t2pos >= 3:
                        try:
                            res = smf.ols(formula, data=df_model).fit(cov_type='HC3')
                            p = float(res.pvalues.get('grp', np.nan))
                            b = res.params['grp']
                            ci_low, ci_high = res.conf_int().loc['grp']
                            sd_grp = df_model['grp'].std()
                            sd_y = df_model['metric_value'].std()
                            b_std = b * (sd_grp / sd_y)
                            ci_low_std = ci_low * (sd_grp / sd_y)
                            ci_high_std = ci_high * (sd_grp / sd_y)
                            pvals[('DCM T2w-', 'DCM T2w+')] = p
                            cov_readable = [c.replace('C(', '').replace(')', '').replace('_bl', '').replace('_', ' ')
                                            for c in cov_terms]
                            comp_results.append({
                                'comparison': 'DCM T2w- vs DCM T2w+',
                                'g1': f'DCM T2w- (n={n_t2neg})', 'g2': f'DCM T2w+ (n={n_t2pos})',
                                'covariates': ', '.join(cov_readable) if cov_readable else 'none',
                                'formula': formula,
                                'p': p, 'b_std': b_std,
                                'ci_low_std': ci_low_std, 'ci_high_std': ci_high_std,
                            })
                        except Exception:
                            pass

                    # --- HC vs DCM T2w- and HC vs DCM T2w+: OLS controlling for sex and age ---
                    for dcm_label in ['DCM T2w-', 'DCM T2w+']:
                        hc_dcm_df = grouped[grouped['group'].isin(['HC', dcm_label]) &
                                            (grouped['Level'] == lvl_label)].copy()
                        hc_dcm_df['grp'] = (hc_dcm_df['group'] == dcm_label).astype(int)
                        hc_cov_terms = []
                        if 'sex' in hc_dcm_df.columns:
                            hc_cov_terms.append('C(sex)')
                        if 'age' in hc_dcm_df.columns:
                            hc_dcm_df['age_c'] = pd.to_numeric(hc_dcm_df['age'], errors='coerce')
                            hc_dcm_df['age_c'] -= hc_dcm_df['age_c'].mean()
                            hc_cov_terms.append('age_c')
                        hc_needed = ['grp', metric] + [ct.split('(')[-1].rstrip(')') if ct.startswith('C(') else ct
                                                        for ct in hc_cov_terms]
                        hc_model = hc_dcm_df.dropna(subset=hc_needed).copy().rename(columns={metric: 'metric_value'})
                        hc_formula = 'metric_value ~ grp' + (' + ' + ' + '.join(hc_cov_terms) if hc_cov_terms else '')
                        n_hc = int((hc_model['grp'] == 0).sum())
                        n_dcm = int((hc_model['grp'] == 1).sum())
                        if n_hc >= 3 and n_dcm >= 3:
                            try:
                                hc_res = smf.ols(hc_formula, data=hc_model).fit(cov_type='HC3')
                                p = float(hc_res.pvalues.get('grp', np.nan))
                                b = hc_res.params['grp']
                                ci_low, ci_high = hc_res.conf_int().loc['grp']
                                sd_grp = hc_model['grp'].std()
                                sd_y = hc_model['metric_value'].std()
                                b_std = b * (sd_grp / sd_y)
                                ci_low_std = ci_low * (sd_grp / sd_y)
                                ci_high_std = ci_high * (sd_grp / sd_y)
                                pvals[('HC', dcm_label)] = p
                                hc_cov_readable = [c.replace('C(', '').replace(')', '').replace('_', ' ')
                                                   for c in hc_cov_terms]
                                comp_results.append({
                                    'comparison': f'HC vs {dcm_label}',
                                    'g1': f'HC (n={n_hc})', 'g2': f'{dcm_label} (n={n_dcm})',
                                    'covariates': ', '.join(hc_cov_readable) if hc_cov_readable else 'none',
                                    'formula': hc_formula,
                                    'p': p, 'b_std': b_std,
                                    'ci_low_std': ci_low_std, 'ci_high_std': ci_high_std,
                                })
                            except Exception:
                                pass

                    # Print formatted summary
                    if comp_results:
                        metric_label = METRIC_TO_AXIS.get(metric, metric)
                        print(f'\n{"="*80}')
                        print(f'STATISTICAL COMPARISONS — {metric_label}, Level {lvl_label}')
                        print(f'{"="*80}')
                        print(f'Test: Ordinary least squares (OLS) with HC3-robust standard errors')
                        print(f'Significance threshold: p < 0.05 (uncorrected)')
                        for i, r in enumerate(comp_results, 1):
                            sig_marker = '  * significant' if r['p'] < 0.05 else '  (n.s.)'
                            print(f'\n  Comparison {i}: {r["comparison"]}')
                            print(f'    Groups    : {r["g1"]}  vs  {r["g2"]}')
                            print(f'    Covariates: {r["covariates"]}')
                            print(f'    Formula   : {r["formula"]}')
                            print(f'    p = {r["p"]:.4f}   '
                                  f'Standardised β = {r["b_std"]:.3f} '
                                  f'(95% CI: {r["ci_low_std"]:.3f}, {r["ci_high_std"]:.3f})'
                                  f'{sig_marker}')
                        print(f'{"="*80}\n')

                    # Significance: all comparisons uncorrected at p < 0.05
                    sig_pairs = [(pair, p) for pair, p in pvals.items() if p < 0.05]
                    if not sig_pairs:
                        continue

                    all_vals = grouped[grouped['Level'] == lvl_label][metric].dropna().values
                    data_max = np.nanmax(all_vals) if len(all_vals) > 0 else ymax - 0.1 * yrange
                    bracket_step = 0.11 * yrange
                    bracket_h = 0.02 * yrange

                    for b_idx, ((g1, g2), _) in enumerate(sig_pairs):
                        x1 = tick_x + (group_pos_idx[g1] - (n_groups - 1) / 2) * group_width
                        x2 = tick_x + (group_pos_idx[g2] - (n_groups - 1) / 2) * group_width
                        y_bot = data_max + bracket_step * (b_idx + 1)
                        y_top = y_bot + bracket_h
                        ax_violin.plot([x1, x1, x2, x2], [y_bot, y_top, y_top, y_bot],
                                       lw=1.5, color='black', clip_on=False)
                        ax_violin.text((x1 + x2) / 2, y_top*0.97, '*', ha='center', va='bottom',
                                       fontsize=TICKS_FONT_SIZE+10, color='black')
                        y_needed = y_top + 0.05 * yrange
                        if y_needed > ax_violin.get_ylim()[1]:
                            ax_violin.set_ylim(ymin, y_needed)
                            ymin, ymax = ax_violin.get_ylim()
                            yrange = ymax - ymin

            elif len(unique_groups) == 2:
                g1, g2 = unique_groups[0], unique_groups[1]
                # Map x tick label to position
                ymin, ymax = ax_violin.get_ylim()
                yrange = ymax - ymin if ymax > ymin else 1.0
                for lvl_label in level_order_labels:
                    vals1 = grouped[(grouped[hue] == g1) & (grouped['Level'] == lvl_label)][metric].dropna().values
                    vals2 = grouped[(grouped[hue] == g2) & (grouped['Level'] == lvl_label)][metric].dropna().values
                    if len(vals1) >= 3 and len(vals2) >= 3:
                        df_level = grouped[(grouped['Level'] == lvl_label) & (grouped[hue].isin([g1, g2]))].copy()
                        df_level['grp'] = (df_level[hue] == g2).astype(int)
                        # Build covariates (adjust for sex and age if available and not grouping variable)
                        # C(): categorical variable
                        # _c: centered continuous variable
                        cov_terms = []
                        if 'sex' in df_level.columns and hue != 'sex':
                            cov_terms.append('C(sex)')
                        if 'Myelopathy' in df_level.columns and hue != 'Myelopathy':
                            cov_terms.append('C(Myelopathy)')
                        if 'MCL' in df_level.columns and hue != 'mcl':
                            cov_terms.append('C(MCL)')
                        if 'mJOA_severity_bl' in df_level.columns and hue != 'mJOA_severity_bl':
                            cov_terms.append('C(mJOA_severity_bl)')
                        if 'age' in df_level.columns and hue not in ['age', 'age_group']:
                            df_level['age_c'] = pd.to_numeric(df_level['age'], errors='coerce')
                            df_level['age_c'] = df_level['age_c'] - df_level['age_c'].mean()
                            cov_terms.append('age_c')
                        # elif 'age_group' in df_level.columns and hue != 'age_group':
                        #     cov_terms.append('C(age_group)')
                        # Prepare model dataframe & rename metric column to avoid Patsy issues
                        needed_cols = ['grp', metric] + [ct.split('(')[-1].rstrip(')') if ct.startswith('C(') else ct for ct in cov_terms]
                        df_model = df_level.dropna(subset=needed_cols).copy().rename(columns={metric: 'metric_value'})
                        formula = 'metric_value ~ grp' + (' + ' + ' + '.join(cov_terms) if cov_terms else '')
                        n1 = int((df_model['grp'] == 0).sum())
                        n2 = int((df_model['grp'] == 1).sum())
                        if n1 >= 3 and n2 >= 3:
                            # Using HC3 due to small sample size (<100)
                            res = smf.ols(formula, data=df_model).fit(cov_type='HC3')
                            pval = float(res.pvalues.get('grp', np.nan))    # partial effect of the grp variable, after controlling for all covariates (age, sex, etc.).

                            # Standardized regression coefficient (95% CI)
                            # Extract unstandardized coefficient for 'grp'
                            b = res.params['grp']
                            ci_low, ci_high = res.conf_int().loc['grp']
                            # Compute SDs
                            sd_grp = df_model['grp'].std()
                            sd_y = df_model['metric_value'].std()
                            # Standardized effect and CI
                            b_std = b * (sd_grp / sd_y)
                            ci_low_std = ci_low * (sd_grp / sd_y)
                            ci_high_std = ci_high * (sd_grp / sd_y)
                            # b_std, (ci_low_std, ci_high_std)

                            print(f'Metric {metric}, Level {lvl_label}, Group {g1} vs {g2}: n={n1} vs {n2}, p={pval:.4f}, Standardized regression coefficient={b_std:.4f} (95% CI: {ci_low_std:.4f}, {ci_high_std:.4f}). {formula}')

                            # Summary table
                            # res.summary()
                            # Adjusted R-squared -- how much the model explains the variation in the metric. *100 for percentage
                            # r2_adj = res.rsquared_adj # Adjusted R-squared
                            # F-test p -- whether all predictors together have a statistically detectable effect
                            # f_pvalue = res.f_pvalue
                        else:
                            pval = np.nan
                        if pval < 0.05:
                            x = tick_pos_map.get(lvl_label, None)
                            # # Save df_model as CSV for debugging
                            # df_model.to_csv(f'debug_violin_{metric}_{lvl_label}.csv', index=False)
                            if x is not None:
                                data_max = np.nanmax(np.concatenate([vals1, vals2])) if (len(vals1) + len(vals2)) > 0 else ymin + 0.8 * yrange
                                y_star = data_max + 0.15 * yrange
                                # Expand ylim if needed
                                if y_star > ymax:
                                    ax_violin.set_ylim(ymin, y_star + 0.03 * yrange)
                                    ymin, ymax = ax_violin.get_ylim()
                                    yrange = ymax - ymin
                                    y_star = data_max - 0.03 * yrange
                                ax_violin.text(x, y_star, '*', ha='center', va='center', fontsize=LABELS_FONT_SIZE+50, color='black')
                                # # Add horizontal line connecting the two groups
                                # # Get the positions of the two groups for the horizontal line
                                # group_positions = []
                                # for i, group in enumerate([g1, g2]):
                                #     n_groups = len(unique_groups)
                                #     group_idx = list(unique_groups).index(group)
                                #     offset = (group_idx - (n_groups - 1) / 2) * 0.4 / (n_groups - 1) if n_groups > 1 else 0
                                #     group_positions.append(x + offset)
                                # # Draw horizontal line between the two groups
                                # if len(group_positions) == 2 and group_positions[0] != group_positions[1]:
                                #     y_line = y_star * 0.98  # slightly below the star
                                #     ax_violin.plot([group_positions[0], group_positions[1]], [y_line, y_line],
                                #                  color='black', linewidth=1.5, zorder=10)
            elif len(unique_groups) >= 3:
                # Robust Wald F-test across all groups (>2)
                ymin, ymax = ax_violin.get_ylim()
                yrange = ymax - ymin if ymax > ymin else 1.0
                for lvl_label in level_order_labels:
                    df_level = grouped[(grouped['Level'] == lvl_label) & (grouped[hue].isin(unique_groups))].copy()
                    if df_level.empty:
                        continue
                    # Use categorical group directly
                    df_level['grp'] = df_level[hue].astype(str)
                    # Adjust for covariates (same logic as binary case)
                    cov_terms = []
                    if 'sex' in df_level.columns and hue != 'sex':
                        cov_terms.append('C(sex)')
                    if 'age' in df_level.columns and hue not in ['age', 'age_group']:
                        df_level['age_c'] = pd.to_numeric(df_level['age'], errors='coerce')
                        df_level['age_c'] = df_level['age_c'] - df_level['age_c'].mean()
                        cov_terms.append('age_c')
                    elif 'age_group' in df_level.columns and hue != 'age_group':
                        cov_terms.append('C(age_group)')
                    # Minimum sample size per group
                    counts = df_level['grp'].value_counts()
                    if (len(counts) < 2) or (counts.min() < 3):
                        continue
                    # Prepare model dataframe & rename metric column
                    needed_cols = ['grp', metric] + [ct.split('(')[-1].rstrip(')') if ct.startswith('C(') else ct for ct in cov_terms]
                    df_model = df_level.dropna(subset=needed_cols).copy().rename(columns={metric: 'metric_value'})
                    if df_model.empty:
                        continue
                    formula = 'metric_value ~ C(grp)' + (' + ' + ' + '.join(cov_terms) if cov_terms else '')
                    try:
                        res = smf.ols(formula, data=df_model).fit()
                    except Exception:
                        continue
                    # Build R matrix to test all non-baseline group coefficients == 0
                    param_names = list(res.params.index)
                    test_idx = [j for j, name in enumerate(param_names) if name.startswith('C(grp)[')]
                    if len(test_idx) == 0:
                        continue
                    R = np.zeros((len(test_idx), len(param_names)))
                    for r, j in enumerate(test_idx):
                        R[r, j] = 1.0
                    try:
                        ftest = res.f_test(R)
                        pval = float(ftest.pvalue)
                    except Exception:
                        pval = np.nan
                    if pval < 0.05:
                        x = tick_pos_map.get(lvl_label, None)
                        if x is not None:
                            vals_all = grouped[grouped['Level'] == lvl_label][metric].dropna().values
                            data_max = np.nanmax(vals_all) if len(vals_all) > 0 else ymin + 0.8 * yrange
                            y_star = data_max - 0.03 * yrange
                            if y_star > ymax:
                                ax_violin.set_ylim(ymin, y_star + 0.03 * yrange)
                                ymin, ymax = ax_violin.get_ylim()
                                yrange = ymax - ymin
                                y_star = data_max - 0.03 * yrange
                            ax_violin.text(x, y_star, '*', ha='center', va='bottom', fontsize=LABELS_FONT_SIZE+50, color='black')

        # Y-axis limits for bottom row
        if metric in METRICS_YLIMITS[structure]:
            ax_violin.set_ylim(METRICS_YLIMITS[structure][metric][0], METRICS_YLIMITS[structure][metric][1]*1.1)

        # Remove xlabel
        ax_violin.set_xlabel('', fontsize=LABELS_FONT_SIZE)     # 'Vertebral level'
        # Remove xticks
        ax_violin.set_xticks([])
        ax_violin.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax_violin.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['left'].set_visible(True)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['bottom'].set_visible(True)
        # ax_violin.yaxis.grid(True)
        ax_violin.set_axisbelow(True)

    # Update title based on stratification
    if stratify_type == 'mcl':
        plotted_subjects = subjects_df[subjects_df['MCL'].isin(MCL_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by MCL (maximum compression level)"
    elif stratify_type == 'highest_stenosis':
        plotted_subjects = subjects_df[subjects_df['highest_stenosis'].isin(MCL_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Highest Stenosis Level"
    elif stratify_type == 'num_of_stenosis':
        plotted_subjects = subjects_df[subjects_df['num_of_stenosis'].notna()]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Number of Compressions"
    elif stratify_type == 'single_vs_multi_stenosis':
        plotted_subjects = subjects_df[subjects_df['single_vs_multi_stenosis'].isin(['Single stenosis', 'Multi-level stenosis'])]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Single vs. Multi-level Stenosis"
    elif stratify_type == 'num_of_stenosis_including_C2C3':
        plotted_subjects = subjects_df[subjects_df['num_of_stenosis_including_C2C3'].notna()]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Number of Compressions"
    elif stratify_type == 'myelopathy':
        plotted_subjects = subjects_df[subjects_df['Myelopathy'].isin(MYELOPATHY_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Myelopathy"
    elif stratify_type == 'myelopathy_with_hc':
        plotted_subjects = subjects_df[subjects_df['Myelopathy'].isin(MYELOPATHY_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        n_hc = df_hc_perlevel['participant_id'].nunique() if df_hc_perlevel is not None else 0
        stratification_info = f"(DCM n={n_subjects_plot}, HC n={n_hc}) DCM T2w+/T2- vs HC"
    elif stratify_type == 'therapeutic_decision':
        plotted_subjects = subjects_df[subjects_df['therapeutic_decision'].isin(THERAPEUTIC_DECISION_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Therapeutic Decision"
    elif stratify_type == 'mjoa':
        valid_mjoa = [k for k in MJOA_COLORS.keys() if k not in ['unknown', 'severe (mJOA ≤ 11)']]
        plotted_subjects = subjects_df[subjects_df['mJOA_severity_bl'].isin(valid_mjoa)]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        # stratification_info = f"(n={n_subjects_plot} subjects) stratified by mJOA severity (dropping 'severe' and 'unknown' mJOA)"
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by mJOA severity"
    elif stratify_type == 'age':
        plotted_subjects = subjects_df[subjects_df['age_group'].isin(['<50', '50-65', '>65'])]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by age group (<50, 50-65, >65)"
    elif stratify_type == 'sex':
        plotted_subjects = subjects_df[subjects_df['sex'].isin(['M', 'F'])]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Sex"
    elif stratify_type == 'normative_mean_c2':
        plotted_subjects = subjects_df[subjects_df['normative_mean_c2'].isin(NORMATIVE_C2_COLORS.keys())]['participant_id'].unique()
        n_subjects_plot = len(plotted_subjects)
        stratification_info = f"(n={n_subjects_plot} subjects) stratified by Normative Mean C2 Cord Area"
    else:
        n_subjects_plot = len(subjects_df['participant_id'].unique())
        stratification_info = f"(n={n_subjects_plot} subjects)"

    # Fetch cord, canal, or aSCOR from the input filename to include in the figure title
    if 'cord' in figure_path:
        structure = 'Spinal cord morphometrics'
    elif 'canal' in figure_path:
        structure = 'Canal morphometrics'

    # if 'aSCOR' in figure_path:
    #     plt.suptitle(f"aSCOR in the PAM50 space\n{stratification_info}",
    #                  fontsize=LABELS_FONT_SIZE-2, fontweight='bold', y=0.97)
    # else:
    #     plt.suptitle(f"{structure} in the PAM50 space {stratification_info}",
    #                  fontsize=LABELS_FONT_SIZE, fontweight='bold', y=0.92)
    print(f"Number of unique subjects included in the figure: {n_subjects_plot}")
    # Save figure

    # Update figure filename based on stratification type
    if stratify_type:
        figure_fname = f'{figure_path}_{n_subjects_plot}subjects_{stratify_type}-stratified.png'
    else:
        figure_fname = f'{figure_path}_{n_subjects_plot}subjects_{len(sessions_to_process)}sessions.png'

    # Adjust spacing to prevent overlap of y-axis labels with adjacent plots
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.savefig(figure_fname, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {figure_fname}')




def print_mjoa_by_sex(subjects_df):
    print("mJOA SCORES BY SEX ANALYSIS")
    print("=" * 80)
    # mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']
    mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth']
    # Get unique participants only (to avoid duplicate counting)
    participants_df = subjects_df[['participant_id', 'sex'] + mjoa_columns].drop_duplicates('participant_id')

    # Check if sex data is available
    if 'sex' not in participants_df.columns:
        print("Error: Sex data not available in dataset")
        return

    # Filter out missing sex data
    participants_df = participants_df[participants_df['sex'].isin(['M', 'F'])]

    if participants_df.empty:
        print("Error: No valid sex data (M/F) found")
        return

    print(f"Total participants with sex data: {len(participants_df)}")

    # Analyze each mJOA timepoint
    for mjoa_col in mjoa_columns:
        timepoint = mjoa_col.replace('total_mjoa_', '').replace('_', ' ').upper()
        if timepoint == 'BL':
            timepoint = 'BASELINE'
        elif timepoint == '6MTH':
            timepoint = '6-MONTH'
        elif timepoint == '12MTH':
            timepoint = '12-MONTH'

        print(f"\n--- {timepoint} mJOA SCORES ---")

        # Filter for available data at this timepoint
        timepoint_data = participants_df[participants_df[mjoa_col].notna()]

        if timepoint_data.empty:
            print(f"No data available for {timepoint}")
            continue

        print(f"Participants with {timepoint} data: {len(timepoint_data)}")

        # Statistics by sex
        for sex in ['M', 'F']:
            sex_data = timepoint_data[timepoint_data['sex'] == sex][mjoa_col]
            sex_label = 'Males' if sex == 'M' else 'Females'

            if sex_data.empty:
                print(f"  {sex_label}: No data")
                continue

            print(f"  {sex_label} (n={len(sex_data)}):")
            print(f"    Mean ± SD: {sex_data.mean():.2f} ± {sex_data.std():.2f}")
            print(
                f"    Median [IQR]: {sex_data.median():.1f} [{sex_data.quantile(0.25):.1f}, {sex_data.quantile(0.75):.1f}]")
            print(f"    Range: {sex_data.min():.0f} - {sex_data.max():.0f}")

        # Statistical comparison if both sexes have data
        males_data = timepoint_data[timepoint_data['sex'] == 'M'][mjoa_col]
        females_data = timepoint_data[timepoint_data['sex'] == 'F'][mjoa_col]

        if len(males_data) >= 3 and len(females_data) >= 3:
            from scipy.stats import ttest_ind, mannwhitneyu

            # T-test
            try:
                t_stat, t_pval = ttest_ind(males_data, females_data, equal_var=False)
                print(f"  Statistical comparison:")
                print(f"    Mean difference (M-F): {males_data.mean() - females_data.mean():.2f}")
                print(f"    T-test p-value: {t_pval:.4f}{'*' if t_pval < 0.05 else ''}")
            except Exception as e:
                print(f"  T-test failed: {e}")

            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_pval = mannwhitneyu(males_data, females_data, alternative='two-sided')
                print(f"    Mann-Whitney U p-value: {u_pval:.4f}{'*' if u_pval < 0.05 else ''}")
            except Exception as e:
                print(f"  Mann-Whitney U test failed: {e}")

    # Summary table
    print(f"\n--- SUMMARY TABLE ---")
    print(f"{'Timepoint':<12} {'Sex':<6} {'N':<4} {'Mean':<6} {'SD':<6} {'Median':<7} {'Min':<4} {'Max':<4}")
    print("-" * 50)

    for mjoa_col in mjoa_columns:
        timepoint_short = mjoa_col.replace('total_mjoa_', '').replace('_', '')
        timepoint_data = participants_df[participants_df[mjoa_col].notna()]

        for sex in ['M', 'F']:
            sex_data = timepoint_data[timepoint_data['sex'] == sex][mjoa_col]
            if not sex_data.empty:
                print(f"{timepoint_short:<12} {sex:<6} {len(sex_data):<4} "
                      f"{sex_data.mean():<6.1f} {sex_data.std():<6.1f} "
                      f"{sex_data.median():<7.1f} {sex_data.min():<4.0f} {sex_data.max():<4.0f}")

    print("\n* p < 0.05")
    print("=" * 80)


def run_ctree_analysis(subjects_df, output_dir):
    """
    Run URP-CTREE in R (partykit::ctree) to identify the optimal compression ratio cutoff
    at C3 that predicts T2w hyperintensity (myelopathy) in DCM patients.

    Requires R >= 4.0 on PATH and the partykit package (auto-installed by the R script on first run).
    Install R on macOS with:  brew install r
    Install R on Ubuntu with: sudo apt-get install r-base

    :param subjects_df: DataFrame with DCM morphometrics + clinical data (already filtered/merged)
    :param output_dir: Directory where the tree figure and summary CSV are saved
    """
    import shutil
    import subprocess
    import tempfile

    # Check R is available
    if shutil.which('Rscript') is None:
        raise RuntimeError(
            "Rscript not found on PATH. Install R first:\n"
            "  macOS : brew install r\n"
            "  Ubuntu: sudo apt-get install r-base\n"
            "Then restart your terminal and retry."
        )

    # Filter to DCM patients at C3, baseline session, valid Myelopathy label
    dcm_df = subjects_df[subjects_df['Myelopathy'].isin(['yes', 'no'])].copy()
    if 'session_id' in dcm_df.columns:
        dcm_df = dcm_df[dcm_df['session_id'] == 'ses-M0']
    dcm_df = dcm_df[dcm_df['VertLevel'] == 3]

    if dcm_df.empty:
        print("run_ctree_analysis: no DCM data at VertLevel=3 with valid Myelopathy — skipping.")
        return

    # Per-participant mean compression ratio (aggregate across slices within C3)
    dcm_grouped = (
        dcm_df.groupby('participant_id')
        .agg({'MEAN(compression_ratio)': 'mean', 'Myelopathy': 'first'})
        .reset_index()
    )
    n_total  = len(dcm_grouped)
    n_t2wpos = (dcm_grouped['Myelopathy'] == 'yes').sum()
    n_t2wneg = (dcm_grouped['Myelopathy'] == 'no').sum()
    print(f"\nrun_ctree_analysis: n={n_total} DCM patients at C3  (T2w-: {n_t2wneg}, T2w+: {n_t2wpos})")

    # Save a permanent copy for reproducibility (shareable with colleagues)
    os.makedirs(output_dir, exist_ok=True)
    shareable_csv = os.path.join(output_dir, 'ctree_compression_ratio_C3_input.csv')
    dcm_grouped.to_csv(shareable_csv, index=False)
    print(f"Input data saved : {shareable_csv}")

    # Write temp CSV for R
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        dcm_grouped.to_csv(f, index=False)
        tmp_csv = f.name

    # Locate companion R script (same directory as this Python file)
    r_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'run_ctree_compression_ratio.R')
    if not os.path.isfile(r_script):
        raise FileNotFoundError(f"R script not found: {r_script}")

    os.makedirs(output_dir, exist_ok=True)

    # Call R
    print("Running R script (partykit::ctree)...")
    result = subprocess.run(
        ['Rscript', '--vanilla', r_script, tmp_csv, output_dir],
        capture_output=True, text=True
    )

    # Print R stdout (formatted summary from the R script)
    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"R stderr:\n{result.stderr}")
        raise RuntimeError(f"R script exited with code {result.returncode}")

    # Clean up temp file
    os.unlink(tmp_csv)


def main():
    args = get_parser().parse_args()
    path_HC = os.path.expandvars(args.path_HC)
    path_participants_tsv_pam50 = os.path.expandvars(args.participants_file_pam50)
    path_out = os.path.abspath(args.o)
    sessions_to_process = args.s
    exclude_file = os.path.expandvars(args.exclude_file)
    c2c3_file = os.path.expandvars(args.c2c3_file)
    path_HC_sg = os.path.expandvars(args.path_HC_spinegeneric)

    # ----
    # Read files with morphometrics (CSV) and clinical (XSLX) data
    # ----
    subjects_df = read_morphometrics_file(os.path.abspath(args.i))
    df_clinical, clinical_columns = read_clinical_file(os.path.abspath(args.clinical_file))
    subjects_df = merge_morphometrics_and_clinical_data(subjects_df, df_clinical, clinical_columns)

    # Get number of unique subjects
    n_subjects = len(subjects_df['participant_id'].unique())
    print(f"Number of unique subjects: {n_subjects}")

    # ----
    # Exclude subjects based on exclude file
    # ----
    subjects_df = read_exclude_file_and_exclude_subjects(subjects_df, exclude_file)

    # ----
    # Print number of subjects with surgery before baseline
    # ----
    # print(f"Number of unique subjects before dropping subjects with surgery before baseline: {len(subjects_df['participant_id'].unique())}")
    # subjects_df = subjects_df[subjects_df['surgery_before_baseline'] != 'yes']
    # print(f"Number of unique subjects after dropping subjects with surgery before baseline: {len(subjects_df['participant_id'].unique())}")

    # ----
    # Read text file with levels to use (C3 or C2,C3 or exclude)
    # ----
    subjects_df = read_c2c3_file_and_apply_exclusions(subjects_df, c2c3_file)

    # ----
    # Drop rows with highest_stenosis == C2/C3 or C3/C4
    # Drop rows with num_of_stenosis == 4
    # ----
    subjects_df = drop_highest_stenosis(subjects_df)

    # # ----
    # # Keep only conservatively treated subjects (therapeutic_decision == 'conservative')
    # # ----
    # print(f"Number of unique subjects before therapeutic decision filtering: {len(subjects_df['participant_id'].unique())}")
    # subjects_df = subjects_df[subjects_df['therapeutic_decision'] == 'conservative']
    # print(f"Number of unique subjects after therapeutic decision filtering: {len(subjects_df['participant_id'].unique())}")

    # Save unique participant IDs to be reused by other scripts
    unique_participants = subjects_df['participant_id'].unique()
    unique_participants_file = os.path.join(path_out, f'unique_participants_ids_{len(unique_participants)}.txt')
    os.makedirs(path_out, exist_ok=True)
    with open(unique_participants_file, 'w') as f:
        for pid in unique_participants:
            f.write(f"{pid}\n")
    print(f"Unique participant IDs saved to: {unique_participants_file}")

    if 'cord' in args.i:
        structure = 'spinal_cord'
    elif 'canal' in args.i:
        structure = 'canal'
    elif 'aSCOR' in args.i:
        structure = 'aSCOR'

    # For spinal cord, keep only VertLevel C2 to C6; for canal and aSCOR, keep only C2 to C3 (due to flow void artifacts for canal seg)
    # vert_min, vert_max = (2, 6) if structure == 'spinal_cord' else (2, 3)
    vert_min, vert_max = (3, 3)
    subjects_df = subjects_df[subjects_df['VertLevel'] >= vert_min]
    subjects_df = subjects_df[subjects_df['VertLevel'] <= vert_max]

    # ----
    # Print number of unique participants per VertLevel
    # ----
    vert_counts = subjects_df.groupby('VertLevel')['participant_id'].nunique().sort_index()
    print("Number of unique participants per VertLevel:")
    for level, count in vert_counts.items():
        print(f"  VertLevel {level}: {count}")

    # Load normative data
    df_normative_data, df_min, df_max = load_normative_data(path_HC, path_participants_tsv_pam50, structure, vert_min, vert_max)

    # Load HC per-level data for myelopathy_with_hc stratification
    df_hc_perlevel = None
    if args.stratify == 'myelopathy_with_hc' and structure in ['spinal_cord', 'canal']:
        vert_levels = list(range(vert_min, vert_max + 1))
        df_hc_perlevel = load_hc_spinegeneric_perlevel(path_HC_sg, structure, vert_levels)

    if args.stratify == 'normative_mean_c2':
        # -------------
        # Load normative cord area for VertLevel C2
        # -------------
        # Get normative data at C2 only (mean across slices for each subject)
        normative_data_c2_grouped = load_normative_df_c2(path_HC, path_participants_tsv_pam50)
        # Compute mean C2 across subjects
        mean_c2_cord_normative = normative_data_c2_grouped['MEAN(area)'].mean()

        # -------------
        # Compute mean C2 for each subject from subjects_df using df.groupby
        # -------------
        subjects_df_c2 = subjects_df[subjects_df['VertLevel'] == 2]
        grouped_subjects_c2 = subjects_df_c2.groupby('participant_id').agg({
            'MEAN(area)': 'mean'
        }).reset_index()
        # Add a new column to grouped_subjects_c2 indicating whether the subject's mean C2 cord area is below or above normative mean
        grouped_subjects_c2['normative_mean_c2'] = grouped_subjects_c2['MEAN(area)'].apply(lambda x: _categorize_c2_area(x, mean_c2_cord_normative))

        # Now, add this information back to the main subjects_df
        subjects_df = subjects_df.merge(
            grouped_subjects_c2[['participant_id', 'normative_mean_c2']],
            on='participant_id', how='left'
        )

    # -------------
    # URP-CTREE analysis (optional)
    if args.run_ctree:
        run_ctree_analysis(subjects_df, path_out)

    # Plotting
    # -------------
    os.makedirs(path_out, exist_ok=True)
    # Use basename from args.i to create figure name
    figure_basename = os.path.basename(args.i).replace('.csv', '')
    figure_path = os.path.join(path_out, figure_basename)
    create_figure(subjects_df, df_normative_data, sessions_to_process, figure_path, args.stratify, df_hc_perlevel)

    # Save age group compression table if age stratification selected
    if args.stratify == 'age':
        age_table_csv = os.path.join(path_out, f"{figure_basename}_age_group_compression_summary.csv")
        table_df = _build_age_group_compression_table(subjects_df, age_table_csv)
        formatted_csv = os.path.join(path_out, f"{figure_basename}_age_group_compression_summary_formatted.csv")
        _save_age_group_compression_table_formatted(table_df, formatted_csv)
    # Save myelopathy compression table if myelopathy stratification selected
    if args.stratify == 'myelopathy':
        myelo_table_csv = os.path.join(path_out, f"{figure_basename}_myelopathy_compression_summary.csv")
        myelo_df = _build_myelopathy_compression_table(subjects_df, myelo_table_csv)
        myelo_formatted_csv = os.path.join(path_out, f"{figure_basename}_myelopathy_compression_summary_formatted.csv")
        _save_myelopathy_compression_table_formatted(myelo_df, myelo_formatted_csv)
    # Save mJOA compression table if mJOA stratification selected
    if args.stratify == 'mjoa':
        mjoa_table_csv = os.path.join(path_out, f"{figure_basename}_mJOA_compression_summary.csv")
        mjoa_df = _build_mjoa_compression_table(subjects_df, mjoa_table_csv)
        mjoa_formatted_csv = os.path.join(path_out, f"{figure_basename}_mJOA_compression_summary_formatted.csv")
        _save_mjoa_compression_table_formatted(mjoa_df, mjoa_formatted_csv)

    # ----
    # Print mJOA scores by sex
    # ----
    print_mjoa_by_sex(subjects_df)

    # ----
    # Demog
    # ----
    demog_df = subjects_df.drop_duplicates('participant_id')
    # Print total number of unique subjects in clinical file
    print(f'Total number of unique subjects in clinical file for analysis: {len(demog_df["participant_id"].unique())}')
    # Print mean + std of age
    print(f'Mean age: {demog_df["age"].mean():.2f} ± {demog_df["age"].std():.2f}')
    # Print number of males and females
    print(f'Sex distribution: {demog_df["sex"].value_counts().to_dict()}')
    # Therapeutic decision distribution
    print(f'Therapeutic decision distribution: {demog_df["therapeutic_decision"].value_counts().to_dict()}')
    # Myelopathy distribution
    print(f'Myelopathy distribution: {demog_df["Myelopathy"].value_counts().to_dict()}')
    # Print myelopathy distribution by therapeutic decision
    print(f"Myelopathy distribution by therapeutic decision:")
    myelo_therapeutic_dist = demog_df.groupby('therapeutic_decision')['Myelopathy'].value_counts().unstack(fill_value=0)
    print(myelo_therapeutic_dist)


if __name__ == '__main__':
    main()
