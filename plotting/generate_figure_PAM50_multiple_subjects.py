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
import sys
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import METRICS_DTYPE, load_normative_df_c2, _categorize_c2_area, exclude_severe_mjoa

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

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
    'Below normative mean C2 cord area': '#d62728',      # red
    'Above normative mean C2 cord area': '#2ca02c',      # green
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
    'MEAN(diameter_AP)': (5, 9),
    'MEAN(area)': (35, 90),
    'MEAN(diameter_RL)': (8, 15.5),
    'MEAN(eccentricity)': (0.53, 0.91),
    'MEAN(solidity)': (89, 100),
    'MEAN(compression_ratio)': (0.35, 0.86),
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
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('-participants-file', required=False, type=str,
                        help="Path to the patients' participants.tsv file containing data for stratification, e.g.,:"
                             "age, sex, maximum_stenosis, myelopathy.")
    parser.add_argument('-clinical-file', required=False, type=str,
                        help="Excel file with clinical scores (must contain 'total_mjoa_bl' column)")
    parser.add_argument('-stratify', required=False, type=str, default=None,
                        choices=['mcl', 'highest_stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'num_of_stenosis_including_C2C3', 'myelopathy',
                                 'mjoa', 'therapeutic_decision', 'age', 'sex', 'normative_mean_c2', 'None'],
                        help="Stratification method:"
                             "'mcl' for Maximum Compression Level; '-participants-file' is required, "
                             "'highest_stenosis' for the highest stenosis level; '-participants-file' is required, "
                             "'num_of_stenosis' for number of stenosis levels; '-participants-file' is required, "
                             "'single_vs_multi_stenosis' for single vs. multi-level stenosis; '-participants-file' is required, "
                             "'num_of_stenosis_including_C2C3' for number of stenosis levels including stratification of subjects with 4 compressions to see if they have compression at C2/C3 level; '-participants-file' is required, "
                             "'myelopathy' for myelopathy status; '-participants-file' is required, "
                             "'therapeutic_decision' (operative/conservative); -participants-file' is required, "
                             "'age' for age group stratification; '-participants-file' is required, "
                             "'sex' for sex-based stratification; '-participants-file' is required, "
                             "'mjoa' mJOA (mild: 15 ≤ mJOA ≤ 18; moderate 14 ≤ mJOA); '-clinical-file' is required. "
                             "'normative_mean_c2' for stratification based on normative mean C2 cord area; "
                             "'None' for no stratification."
                             "Default: None.",
                             )

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


def load_normative_data(path_HC, path_participants_pam50, structure):
    """
    Load normative data from spine-generic dataset in PAM50 space
    :param path_HC:
    :param path_participants_pam50:
    :param structure: 'spinal_cord' or 'canal' or 'aSCOR'
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
    # Keep only VertLevel from C2 to C7
    df = df[df['VertLevel'] >= 2]
    df = df[df['VertLevel'] <= 7]

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


def read_csv_file(csv_file, participants_file=None, clinical_file=None, stratify_type=None):
    """
    - Read CSV file with morphometrics in the PAM50 space across multiple subjects.
        This file is generated with `sct_process_segmentation -normalize-PAM50 1 -perslice 1 -append 1`.
    - Read participants.tsv file with MCL or myelopathy data for stratification (if provided) or
        clinical Excel file with mJOA scores (if provided).
    :param csv_file: input CSV file path
    :param participants_file: path to participants.tsv file with stratification data
    :param clinical_file: path to Excel file with clinical scores (must contain 'total_mjoa_bl' column)
    :param stratify_type: type of stratification ('mcl' or 'myelopathy')
    :return: pandas dataframe with additional columns participant_id, session_id, MEAN(compression_ratio), and optionally stratification data
    """

    subjects_df = pd.read_csv(csv_file)

    # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
    if 'aSCOR' not in csv_file:
        subjects_df['MEAN(compression_ratio)'] = subjects_df['MEAN(diameter_AP)'] / \
                                                 subjects_df['MEAN(diameter_RL)']
        # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
        subjects_df['MEAN(solidity)'] = subjects_df['MEAN(solidity)'] * 100

    # Fetch participant_id and session_id from the Filename column
    if 'aSCOR' in csv_file:
        filename_column = 'Filename_sc'
    else:
        filename_column = 'Filename'
    participant_ids = []
    session_ids = []
    for file_path in subjects_df[filename_column]:
        participant_id, session_id = fetch_participant_and_session(file_path)
        participant_ids.append(participant_id)
        session_ids.append(session_id)
    subjects_df.insert(0, 'participant_id', participant_ids)
    subjects_df.insert(1, 'session_id', session_ids)

    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')

        if 'maximum_stenosis' in df_participants.columns:
            # Merge MCL data
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'maximum_stenosis']],
                on='participant_id', how='left'
            )
            # Clean up maximum_stenosis values and map to standard format
            subjects_df['MCL'] = subjects_df['maximum_stenosis'].fillna('NA')
            # Standardize MCL values
            subjects_df['MCL'] = subjects_df['MCL'].apply(lambda x: x if x in MCL_COLORS else 'NA')
            # Exclude subjects with MCL == 'NA'
            subjects_df = subjects_df[subjects_df['MCL'] != 'NA']
        else:
            sys.exit("Warning: 'maximum_stenosis' column not found in participants file")

        if 'stenosis' in df_participants.columns:
            # Merge stenosis data
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'stenosis']],
                on='participant_id', how='left'
            )
            subjects_df['stenosis'] = subjects_df['stenosis'].fillna('NA')
            # Exclude subjects with stenosis == 'NA'
            subjects_df = subjects_df[subjects_df['stenosis'] != 'NA']
            # Stenosis is a str of different stenosis levels, e.g., 'C3/C4, C5/C6', convert it to list
            subjects_df['stenosis_levels'] = subjects_df['stenosis'].apply(lambda x: [level.strip() for level in x.split(',')])
            # Add a new column, 'highest_stenosis' with the highest stenosis level per subject
            subjects_df['highest_stenosis'] = subjects_df['stenosis_levels'].apply(_get_highest_stenosis)
            # Print subjects with 'C6/C7'
            c67_subjects = subjects_df[subjects_df['highest_stenosis'] == 'C6/C7']['participant_id'].unique()
            print(f"Subjects with highest stenosis at C6/C7: {c67_subjects}") if len(c67_subjects) > 0 else None
            # Now, exclude 'C6/C7' -- only 3 subjects and C2 cord is noisy
            subjects_df = subjects_df[subjects_df['highest_stenosis'] != 'C6/C7']

            # Add a new column, 'num_of_stenosis' with the number of stenosis levels per subject
            subjects_df['num_of_stenosis'] = subjects_df['stenosis_levels'].apply(len)
            # Print 'stenosis' column for unique subjects with 4 compressions
            four_stenosis_subjects = subjects_df[subjects_df['num_of_stenosis'] == 4][['participant_id', 'stenosis']].drop_duplicates(subset=['participant_id'])
            print(f"Subjects with 4 stenosis levels:\n{four_stenosis_subjects.to_string(index=False)}")
            # Add a new column, 'single_vs_multi_stenosis' with 'single' or 'multi' values
            subjects_df['single_vs_multi_stenosis'] = subjects_df['num_of_stenosis'].apply(lambda x: 'Single stenosis' if x == 1 else 'Multi-level stenosis')
            # Add a new column to further stratify subjects with 4 compressions to see how many of them have C2/C3 compression
            subjects_df['num_of_stenosis_including_C2C3'] = subjects_df['num_of_stenosis']
            # subjects_df['num_of_stenosis_including_C2C3'] = subjects_df.apply(
            #     lambda row: '4 including C2/C3' if row['num_of_stenosis'] == 4 and 'C2/C3' in row['stenosis_levels'] else
            #                 ('4' if row['num_of_stenosis'] == 4 else str(row['num_of_stenosis'])),
            #     axis=1
            # )
            subjects_df['num_of_stenosis_including_C2C3'] = subjects_df.apply(
                lambda row: '4 including C2/C3 or C3/C4' if row['num_of_stenosis'] == 4 and any(level in row['stenosis_levels'] for level in ['C2/C3', 'C3/C4']) else
                ('4' if row['num_of_stenosis'] == 4 else str(row['num_of_stenosis'])),
                axis=1
            )
        else:
            sys.exit("Warning: 'stenosis' column not found in participants file")

        if 'myelopathy' in df_participants.columns:
            # Merge myelopathy data
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'myelopathy']],
                on='participant_id', how='left'
            )
            subjects_df['Myelopathy'] = subjects_df['myelopathy'].apply(_process_myelopathy)
        else:
            sys.exit("Warning: 'myelopathy' column not found in participants file")

        if 'therapeutic_decision' in df_participants.columns:
            # Merge therapeutic decision data
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'therapeutic_decision']],
                on='participant_id', how='left'
            )
            subjects_df['therapeutic_decision'] = subjects_df['therapeutic_decision'].fillna('NA')
            # Exclude subjects with MCL == 'NA'
            subjects_df = subjects_df[subjects_df['therapeutic_decision'] != 'NA']
        else:
            sys.exit("Warning: 'therapeutic_decision' column not found in participants file")

        if 'age' in df_participants.columns:
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'age']],
                on='participant_id', how='left'
            )
            subjects_df['age_group'] = subjects_df['age'].apply(_create_age_group)
            # Exclude unknown age
            subjects_df = subjects_df[subjects_df['age_group'] != 'unknown']
        else:
            sys.exit(f"Warning: {stratify_type} column not found in participants file")

        if 'sex' in df_participants.columns:
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'sex']],
                on='participant_id', how='left'
            )
        else:
            sys.exit(f"Warning: {stratify_type} column not found in participants file")

    else:
        sys.exit(f"Warning: Participants file not found: {participants_file}")

    if clinical_file and os.path.isfile(clinical_file):
        df_clinical = pd.read_excel(clinical_file, usecols=['record_id', 'total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth'])
        # Format record_id to match participant_id format (e.g., `1` to `sub-001`)
        df_clinical['participant_id'] = df_clinical['record_id'].apply(lambda x: f'sub-{int(x):03d}')
        # Drop record_id column
        df_clinical = df_clinical.drop(columns=['record_id'])
        # Stratify mJOA
        df_clinical['mJOA_severity_bl'] = df_clinical['total_mjoa_bl'].apply(_stratify_mjoa)
        # Exclude subjects with severe baseline mJOA
        df_clinical = df_clinical[df_clinical['mJOA_severity_bl'] != 'severe (mJOA ≤ 11)']

        # Merge mJOA data
        subjects_df = subjects_df.merge(
            df_clinical[['participant_id', 'total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth', 'mJOA_severity_bl']],
            on='participant_id', how='left'
        )
    else:
        sys.exit(f"Warning: Clinical file not found: {clinical_file}")

    return subjects_df


def _create_age_group(age):
    if pd.isna(age):
        return 'unknown'
    age = float(age)
    if age < 50:
        return '<50'
    elif 50 <= age <= 65:
        return '50-65'
    else:
        return '>65'

def _process_myelopathy(value):
    """# Process myelopathy values: if not n/a, use 'yes', if n/a, use 'no'"""
    if pd.isna(value) or str(value).lower() == 'n/a':
        return 'no'
    else:
        return 'yes'

# Stratify based on mJOA scores
def _stratify_mjoa(score):
    if pd.isna(score):
        return 'unknown'
    # elif score == 18:
    #     return 'mJOA=18'
    elif 15 <= score <= 18:
        return 'mild (15 ≤ mJOA ≤ 18)'
    elif 12 <= score <= 14:
        return 'moderate (12 ≤ mJOA ≤ 14)'
    elif score < 12:
        return 'severe (mJOA ≤ 11)'
    else:
        return 'unknown'

# Get highest stenosis level from list of levels
# For example, if levels = ['C3/C4', 'C5/C6'], return 'C3/C4'
def _get_highest_stenosis(levels):
    stenosis_order = ['C2/C3', 'C3/C4', 'C4/C5', 'C5/C6', 'C6/C7']
    for level in stenosis_order:
        if level in levels:
            return level
    return 'NA'

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
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Myelopathy compression summary table saved: {output_csv_path}")
    return out_df


def _save_myelopathy_compression_table_formatted(table_df, output_csv_path):
    """Formatted multi-block CSV for myelopathy compression summary (counts with percentages in single cells)."""
    groups = ['yes', 'no']
    def _order(df, cols):
        out = df.copy()[cols]
        out['Myelopathy'] = pd.Categorical(out['Myelopathy'], categories=groups, ordered=True)
        out = out.sort_values('Myelopathy')
        idx = pd.Index(groups, name='Myelopathy')
        out = out.set_index('Myelopathy').reindex(idx).reset_index()
        return out
    denom = _order(table_df, ['Myelopathy', 'n_subjects_with_stenosis_data']).rename(columns={'n_subjects_with_stenosis_data': '_den'})
    def _add_pct(block, count_cols):
        b = block.merge(denom, on='Myelopathy', how='left')
        for c in count_cols:
            d = b['_den'].replace({0: np.nan})
            pct = (b[c] / d) * 100.0
            pct = pct.fillna(0).round(1)
            b[c + '_pct'] = pct
        b = b.drop(columns=['_den'])
        cols = ['Myelopathy']
        for c in count_cols:
            cols += [c, c + '_pct']
        return b[cols]
    def _combine(block_df, count_cols):
        out = block_df.copy()
        for c in count_cols:
            out[c] = out.apply(lambda r: f"{int(r[c])} ({r[c + '_pct']:.1f}%)", axis=1)
            out = out.drop(columns=[c + '_pct'])
        return out[['Myelopathy'] + count_cols]
    # Block1
    block1 = _order(table_df, ['Myelopathy', 'Total number of subjects'])
    # Block2 num of stenosis
    needed2 = ['Num of stenosis: 1', 'Num of stenosis: 2', 'Num of stenosis: 3', 'Num of stenosis: 4']
    for c in needed2:
        if c not in table_df.columns: table_df[c] = 0
    block2 = _combine(_add_pct(_order(table_df, ['Myelopathy'] + needed2), needed2), needed2)
    # Block3 single vs multi
    needed3 = ['Single stenosis count', 'Multi-level stenosis count']
    for c in needed3:
        if c not in table_df.columns: table_df[c] = 0
    block3 = _combine(_add_pct(_order(table_df, ['Myelopathy'] + needed3), needed3), needed3)
    # Block4 highest stenosis
    internal4 = ['Highest stenosis: C2/C3', 'Highest stenosis: C3/C4', 'Highest stenosis: C4/C5', 'Highest stenosis: C5/C6', 'Highest stenosis: C6/C7']
    for c in internal4:
        if c not in table_df.columns: table_df[c] = 0
    block4 = _combine(_add_pct(_order(table_df, ['Myelopathy'] + internal4), internal4), internal4)
    with open(output_csv_path, 'w', newline='') as f:
        block1.to_csv(f, index=False)
        f.write('\n\n'); block2.to_csv(f, index=False)
        f.write('\n\n'); block3.to_csv(f, index=False)
        f.write('\n\n'); block4.to_csv(f, index=False)
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

def create_figure(subjects_df, df_normative_data, sessions_to_process, figure_path, stratify_type=None):
    """
    Create figure with mean and std of morphometric metrics across subjects, separately for multiple sessions
    :param subjects_df: pandas dataframe with morphometric metrics across multiple subjects
    :param df_normative_data: pandas dataframe with normative data from spine-generic dataset
    :param sessions_to_process: list of sessions to process (e.g., ['ses-M0', 'ses-M3'])
    :param figure_path: path to save figure
    :param stratify_type: type of stratification ('mcl', 'myelopathy', 'therapeutic_decision', 'mjoa') or None
    """
    mpl.rcParams['font.family'] = 'Arial'

    if 'aSCOR' in figure_path:
        # 2x1 grid for 1 metric; 6x10
        fig, axs = plt.subplots(2, 1, figsize=(6, 10))
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
        fig, axs = plt.subplots(2, int(len(METRICS)), figsize=(int(len(METRICS)) * 6, 10))
        if len(METRICS) == 1:
            top_axes = [axs[0]]
            bottom_axes = [axs[1]]
        else:
            top_axes = axs[0, :]
            bottom_axes = axs[1, :]

    # Helper for level labels and order
    level_order_nums = [2, 3, 4, 5, 6, 7]
    level_order_labels = [f'C{v}' for v in level_order_nums]

    for metric_idx, metric in enumerate(METRICS):
        ax = top_axes[metric_idx]

        # Plot normative data
        if stratify_type == 'sex':
            sex_groups = ['M', 'F']  # Ensure legend order
            for sex in sex_groups:
                normative_sex_data = df_normative_data[df_normative_data['sex'] == sex]
                sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=normative_sex_data, errorbar='sd',
                             linewidth=2, color=SEX_COLORS_NORMATIVE[sex], linestyle='--',
                             label=f'Normative Data {SEX_TO_LEGEND[sex]} (n={len(normative_sex_data["participant_id"].unique())})')
        else:
            sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df_normative_data, errorbar='sd',
                         linewidth=2, color='black',
                         label=f'Normative Data (n={len(df_normative_data["participant_id"].unique())})')

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
                                linewidth=2, color=MCL_COLORS[mcl],
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
                                linewidth=2, color=MCL_COLORS[level],
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
                                linewidth=2, label=f"Number of Stenosis = {num_stenosis} (n={num_stenosis_n_subjects})")
        elif stratify_type == 'single_vs_multi_stenosis':
            # Plot by Number of Stenosis groups instead of sessions
            num_stenosis_groups = ['Single stenosis', 'Multi-level stenosis']
            for num_stenosis in num_stenosis_groups:
                num_stenosis_data = subjects_df[subjects_df['single_vs_multi_stenosis'] == num_stenosis]
                if len(num_stenosis_data) > 0:
                    num_stenosis_n_subjects = len(num_stenosis_data['participant_id'].unique())
                    print(f"Number of Stenosis '{num_stenosis}': {num_stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=num_stenosis_data, errorbar='sd',
                                linewidth=2, label=f"{num_stenosis} (n={num_stenosis_n_subjects})")
        elif stratify_type == 'num_of_stenosis_including_C2C3':
            num_stenosis_groups = ['1', '2', '3', '4', '4 including C2/C3 or C3/C4']     # ensure order
            for num_stenosis in num_stenosis_groups:
                num_stenosis_data = subjects_df[subjects_df['num_of_stenosis_including_C2C3'] == num_stenosis]
                if len(num_stenosis_data) > 0:
                    num_stenosis_n_subjects = len(num_stenosis_data['participant_id'].unique())
                    print(f"Number of Stenosis '{num_stenosis}': {num_stenosis_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=num_stenosis_data, errorbar='sd',
                                linewidth=2, label=f"Number of Stenosis = {num_stenosis} (n={num_stenosis_n_subjects})")
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
                                linewidth=2, color=MYELOPATHY_COLORS[myelopathy],
                                label=f"Myelopathy {myelopathy} (n={myelopathy_n_subjects})")
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
                                linewidth=2, color=THERAPEUTIC_DECISION_COLORS[decision],
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
                                linewidth=2, color=MJOA_COLORS[mjoa],
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
                                linewidth=2, color=AGE_GROUP_COLORS[age],
                                label=f"Age {age} (n={age_n_subjects})")
        elif stratify_type == 'sex':
            sex_groups = ['M', 'F']  # Ensure legend order
            for sex in sex_groups:
                sex_data = subjects_df[subjects_df['sex'] == sex]
                if len(sex_data) > 0:
                    sex_n_subjects = len(sex_data['participant_id'].unique())
                    print(f"Sex group '{sex}': {sex_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=sex_data, errorbar='sd',
                                linewidth=2, color=SEX_COLORS_PATIENTS[sex],
                                 label=f"{SEX_TO_LEGEND[sex]} (n={sex_n_subjects})")
        elif stratify_type == 'normative_mean_c2':
            c2_groups = subjects_df['normative_mean_c2'].unique()
            for c2_group in c2_groups:
                c2_data = subjects_df[subjects_df['normative_mean_c2'] == c2_group]
                if len(c2_data) > 0:
                    c2_n_subjects = len(c2_data['participant_id'].unique())
                    print(f"Normative mean C2 group '{c2_group}': {c2_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=c2_data, errorbar='sd',
                                linewidth=2, color=NORMATIVE_C2_COLORS[c2_group],
                                label=f"{c2_group} (n={c2_n_subjects})")
        else:
            # Plot each session's mean and std (original behavior)
            for ses in sessions_to_process:
                session_data = subjects_df[subjects_df['session_id'] == ses]
                if len(session_data) > 0:
                    ses_n_subjects = len(session_data['participant_id'].unique())
                    print(f"Session '{ses}': {ses_n_subjects} subjects") if metric == 'MEAN(area)' else None
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=session_data, errorbar='sd',
                                linewidth=2, color=SESSION_COLORS[ses],
                                label=f"{ses} (n={ses_n_subjects})")

        # Keep the legend only for one plot to avoid duplication
        plot_to_keep_legend = 0 if 'aSCOR' in figure_path else (2 if (stratify_type and 'stenosis' in stratify_type or 'mcl' in stratify_type) else 0)
        if metric_idx == plot_to_keep_legend:
            top_axes[metric_idx].legend(fontsize=TICKS_FONT_SIZE, title="mean ± std across subjects", title_fontsize=TICKS_FONT_SIZE)
        else:
            leg = top_axes[metric_idx].get_legend()
            if leg is not None:
                leg.remove()

        # Tweak y-axis limits
        ax.set_ylim(METRICS_YLIMITS[metric][0], METRICS_YLIMITS[metric][1])
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

        # Add vertebral level indicators (top row)
        ymin, ymax = ax.get_ylim()
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_normative_data)
        for idx, x in enumerate(ind_vert[1:-1]):
            ax.axvline(df_normative_data.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        for idx, x in enumerate(ind_vert_mid, 0):
            level = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
            ax.text(df_normative_data.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                    ymin - (ymax - ymin) * 0.05, level, horizontalalignment='center',
                    verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

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
                               'mJOA_severity_bl', 'age_group', 'sex', 'normative_mean_c2']
        available_cols = [c for c in possible_group_cols if c in subjects_df.columns]
        merge_cols += available_cols
        per_participant = subjects_df[merge_cols].drop_duplicates('participant_id')
        grouped = grouped.merge(per_participant, on='participant_id', how='left')
        # Keep only C2-C7
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
                           inner='box', cut=0, linewidth=1, fill=False)
        else:
            sns.violinplot(ax=ax_violin, data=grouped, x='Level', y=metric, hue=hue, order=level_order_labels,
                           hue_order=hue_order, palette=palette, inner='box', cut=0, linewidth=1, dodge=True, fill=False)

        # Legend handling for bottom row
        if metric_idx == plot_to_keep_legend and hue is not None:
            ax_violin.legend(fontsize=TICKS_FONT_SIZE)#, title=hue.replace('_', ' '), title_fontsize=TICKS_FONT_SIZE)
        else:
            leg2 = ax_violin.get_legend()
            if leg2 is not None:
                leg2.remove()

        # Tweak y-axis limits
        ax_violin.set_ylim(METRICS_YLIMITS[metric][0]*0.9, METRICS_YLIMITS[metric][1]*1.1)

        ax_violin.set_xlabel('Vertebral level', fontsize=LABELS_FONT_SIZE)
        ax_violin.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax_violin.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['left'].set_visible(False)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['bottom'].set_visible(True)
        ax_violin.yaxis.grid(True)
        ax_violin.set_axisbelow(True)

    # Fetch cord, canal, or aSCOR from the input filename to include in the figure title
    if 'cord' in figure_path:
        structure = 'Spinal cord morphometrics'
    elif 'canal' in figure_path:
        structure = 'Canal morphometrics'
    elif 'aSCOR' in figure_path:
        structure = 'aSCOR'

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

    # No title for aSCOR
    if 'aSCOR' in figure_path:
        plt.suptitle(f"aSCOR in the PAM50 space\n{stratification_info}",
                     fontsize=LABELS_FONT_SIZE-2, fontweight='bold', y=0.97)
    else:
        plt.suptitle(f"{structure} in the PAM50 space {stratification_info}",
                     fontsize=LABELS_FONT_SIZE, fontweight='bold', y=0.92)
    print(f"Number of unique subjects included in the figure: {n_subjects_plot}")
    # Save figure

    # Update figure filename based on stratification type
    if stratify_type:
        figure_fname = f'{figure_path}_{n_subjects_plot}subjects_{stratify_type}-stratified.png'
    else:
        figure_fname = f'{figure_path}_{n_subjects_plot}subjects_{len(sessions_to_process)}sessions.png'

    plt.savefig(figure_fname, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {figure_fname}')


def main():
    args = get_parser().parse_args()
    path_HC = os.path.expandvars(args.path_HC)
    path_participants_tsv_pam50 = os.path.expandvars(args.participants_file_pam50)
    path_out = os.path.abspath(args.o)
    sessions_to_process = args.s

    # Read CSV file with patients' morphometrics and optional stratification data (e.g., MCL, myelopathy)
    csv_file = os.path.abspath(args.i)
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Input CSV file not found: {csv_file}")
    subjects_df = read_csv_file(csv_file, args.participants_file, args.clinical_file, args.stratify)

    # Exclude severe and unknown mJOA subjects
    subjects_df = exclude_severe_mjoa(subjects_df)

    # # Print number of subjects for each slice
    # slice_counts = subjects_df.groupby('Slice (I->S)')['participant_id'].nunique()
    # print("Number of subjects per slice:")
    # print(slice_counts)

    # Keep only VertLevel from C2 to C7
    subjects_df = subjects_df[subjects_df['VertLevel'] >= 2]
    subjects_df = subjects_df[subjects_df['VertLevel'] <= 7]

    # Exclude sub-004 --> poor canal seg due to strong flow void artifacts
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-004']

    # Get number of unique subjects
    n_subjects = len(subjects_df['participant_id'].unique())
    print(f"Number of unique subjects in the input CSV: {n_subjects}")

    # Load normative data
    if 'cord' in args.i:
        structure = 'spinal_cord'
    elif 'canal' in args.i:
        structure = 'canal'
    elif 'aSCOR' in args.i:
        structure = 'aSCOR'
    df_normative_data, df_min, df_max = load_normative_data(path_HC, path_participants_tsv_pam50, structure)

    if args.stratify == 'normative_mean_c2':
        # Check if 'participants.tsv' contains the 'normative_mean_c2' column
        # If so, read it directly; otherwise, compute it
        if args.participants_file and os.path.isfile(args.participants_file):
            df_participants = pd.read_csv(args.participants_file, sep='\t')
        else:
            sys.exit(f"Warning: Participants file not found: {args.participants_file}")

        if 'normative_mean_c2' in df_participants.columns:
            print("Using existing 'normative_mean_c2' column from participants.tsv")
            # Merge normative_mean_c2 data
            subjects_df = subjects_df.merge(
                df_participants[['participant_id', 'normative_mean_c2']],
                on='participant_id', how='left'
            )
        else:
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

            # Save the 'normative_mean_c2' column to the participants.tsv file for future use
            df_participants = df_participants.merge(
                grouped_subjects_c2[['participant_id', 'normative_mean_c2']],
                on='participant_id', how='left'
            )
            df_participants.to_csv(args.participants_file, sep='\t', index=False)
            print(f"'normative_mean_c2' column added to participants file: {args.participants_file}")

            # Now, add this information back to the main subjects_df
            subjects_df = subjects_df.merge(
                grouped_subjects_c2[['participant_id', 'normative_mean_c2']],
                on='participant_id', how='left'
            )

    # -------------
    # Plotting
    # -------------
    os.makedirs(path_out, exist_ok=True)
    # Use basename from args.i to create figure name
    figure_basename = os.path.basename(args.i).replace('.csv', '')
    figure_path = os.path.join(path_out, figure_basename)
    create_figure(subjects_df, df_normative_data, sessions_to_process, figure_path, args.stratify)

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



if __name__ == '__main__':
    main()
