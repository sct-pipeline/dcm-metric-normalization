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
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM

from utils import METRICS_DTYPE, load_normative_df_c2, _categorize_c2_area, exclude_severe_mjoa

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

MCL_FORMAT = {
    1: 'C2/C3',
    2: 'C3/C4',
    3: 'C4/C5',
    4: 'C5/C6',
    5: 'C6/C7',
    6: 'C7/T1'
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
        'MEAN(area)': (35, 90),
        'MEAN(diameter_RL)': (8, 15.5),
        'MEAN(eccentricity)': (0.53, 0.91),
        'MEAN(solidity)': (89, 100),
        'MEAN(compression_ratio)': (0.35, 0.86)
    },
    'canal': {
        'MEAN(diameter_AP)': (7, 15.5),
        'MEAN(area)': (100, 250),
        'MEAN(diameter_RL)': (15, 26),
        'MEAN(eccentricity)': (0.4, 0.8),
        'MEAN(solidity)': (89, 100),
        'MEAN(compression_ratio)': (0.35, 0.75)
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
    parser.add_argument('-stratify', required=False, type=str, default=None,
                        choices=['mcl', 'highest_stenosis', 'num_of_stenosis', 'single_vs_multi_stenosis', 'num_of_stenosis_including_C2C3', 'myelopathy',
                                 'mjoa', 'therapeutic_decision', 'age', 'sex', 'normative_mean_c2', 'None'],
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


def read_morphometrics_file(csv_file):
    """
    Read CSV file with morphometrics in the PAM50 space across multiple subjects.
    This file is generated with `sct_process_segmentation -normalize-PAM50 1 -perslice 1 -append 1`.
    :param csv_file: input CSV file path
    :return: pandas dataframe with additional columns participant_id, session_id, MEAN(compression_ratio)
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

    return subjects_df


def read_clinical_file(clinical_file):
    """
    Read clinical Excel file with mJOA scores.
    :param clinical_file: path to Excel file with clinical scores (must contain 'total_mjoa_BL' column)
    :return: pandas dataframe with clinical data
    :returns: list of clinical columns
    """

    # specifying separately as this list is being returned by this function
    clinical_columns = [
        'total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth',
        'motor_dysfunction_UE_bl_BL', 'motor_dysfunction_UE_6mth_6mth', 'motor_dysfunction_UE_12mth_12mth',
        'motor_dysfunction_LE_bl_BL', 'motor_dysfunction_LE_6mth_6mth', 'motor_dysfunction_LE_12mth_12mth',
        'sensory_dysfunction_UE_bl_BL', 'sensory_dysfunction_UE_6mth_6mth', 'sensory_dysfunction_UE_12mth_12mth',
        'sphincter_dysfunction_bl_BL', 'sphincter_dysfunction_6mth_6mth', 'sphincter_dysfunction_12mth_12mth',
        'UEPP_C4_T1_bl', 'UEPP_C4_T1_6mth', 'UEPP_C4_T1_12mth',
        'UELT_C4_T1_bl_BL', 'UELT_C4_T1_6mth_6mth', 'UELT_C4_T1_12mth_12mth',
        'upper_extrem_motor_total_BL', 'upper_extrem_motor_total_6mth', 'upper_extrem_motor_total_12mth'
    ]

    columns_to_read = [
        'record_id_BL', 'age_BL', 'sex_BL', 'maximum_stenosis', 'myelopathy',
        'c2_stenosis_no_yes', 'c3_stenosis_no_yes',
        'c4_stenosis_no_yes', 'c5_stenosis_no_yes', 'c6_stenosis_no_yes', 'c7_stenosis_no_yes',
        'surg_timepoint___2_12mth', 'surg_timepoint___3_12mth'
        ]

    columns_to_read += clinical_columns

    # Get only baseline clinical columns (i.e., columns ending with _bl or _BL)
    baseline_clinical_columns = [col for col in clinical_columns if col.endswith('_bl') or col.endswith('_BL')]

    if clinical_file and os.path.isfile(clinical_file):
        df_clinical = pd.read_excel(clinical_file, usecols=columns_to_read)
        # Print number of missing values per column
        print("Number of missing values per column in clinical file:")
        print(df_clinical.isnull().sum())

        # Format record_id to match participant_id format (e.g., `1` to `sub-001`)
        df_clinical['participant_id'] = df_clinical['record_id_BL'].apply(lambda x: f'sub-{int(x):03d}')
        # Drop record_id column
        df_clinical = df_clinical.drop(columns=['record_id_BL'])

        # ----
        # Merge maximum_stenosis
        # ----
        # Format MCL using MCL_FORMAT
        df_clinical['maximum_stenosis'] = df_clinical['maximum_stenosis'].fillna('NA')
        df_clinical['maximum_stenosis'] = df_clinical['maximum_stenosis'].map(MCL_FORMAT)
        # Standardize MCL values --> use 'NA'
        df_clinical['maximum_stenosis'] = df_clinical['maximum_stenosis'].apply(lambda x: x if x in MCL_COLORS else 'NA')

        # ----
        # stenosis_levels
        # ----
        # C2: C2/C3, C3: C3/C4, C4: C4/C5, C5: C5/C6, C6: C6/C7, C7: C7/T1
        # 0: no stenosis, 1: stenosis
        stenosis_levels = []
        for _, row in df_clinical.iterrows():
            levels = []
            if row['c2_stenosis_no_yes'] == 1:
                levels.append('C2/C3')
            if row['c3_stenosis_no_yes'] == 1:
                levels.append('C3/C4')
            if row['c4_stenosis_no_yes'] == 1:
                levels.append('C4/C5')
            if row['c5_stenosis_no_yes'] == 1:
                levels.append('C5/C6')
            if row['c6_stenosis_no_yes'] == 1:
                levels.append('C6/C7')
            if row['c7_stenosis_no_yes'] == 1:
                levels.append('C7/T1')
            stenosis_levels.append(', '.join(levels) if levels else 'NA')
        df_clinical['stenosis_levels'] = stenosis_levels
        df_clinical['stenosis_levels'] = df_clinical['stenosis_levels'].fillna('NA')
        # Convert "C3/C4, C5/C6" to ["C3/C4", "C5/C6"]
        df_clinical['stenosis_levels'] = df_clinical['stenosis_levels'].apply(lambda x: [level.strip() for level in x.split(',')])
        # Convert [NA] to 'NA'
        df_clinical['stenosis_levels'] = df_clinical['stenosis_levels'].apply(lambda x: x if x != ['NA'] else 'NA')

        # ----
        # highest_stenosis
        # ----
        # Add a new column, 'highest_stenosis' with the highest stenosis level per subject
        df_clinical['highest_stenosis'] = df_clinical['stenosis_levels'].apply(_get_highest_stenosis)

        # Print subjects with 'C6/C7'
        c67_subjects = df_clinical[df_clinical['highest_stenosis'] == 'C6/C7']['participant_id'].unique()
        print(f"Subjects with highest stenosis at C6/C7: {c67_subjects}") if len(c67_subjects) > 0 else None
        # # Now, exclude 'C6/C7' -- only 3 subjects and C2 cord is noisy
        # df_clinical = df_clinical[df_clinical['highest_stenosis'] != 'C6/C7']

        # ----
        # num_of_stenosis
        # ----
        # Add a new column, 'num_of_stenosis' with the number of stenosis levels per subject
        df_clinical['num_of_stenosis'] = df_clinical['stenosis_levels'].apply(
            lambda x: 0 if x == ['NA'] or x == 'NA' else len(x)
        )
        # Print 'stenosis_levels' column for unique subjects with 4 compressions
        four_stenosis_subjects = df_clinical[df_clinical['num_of_stenosis'] == 4][['participant_id', 'stenosis_levels']]#.drop_duplicates(subset=['participant_id'])
        print(f"Subjects with 4 stenosis levels:\n{four_stenosis_subjects.to_string(index=False)}")

        # ----
        # single_vs_multi_stenosis
        # ----
        # Add a new column, 'single_vs_multi_stenosis' with 'single' or 'multi' values
        df_clinical['single_vs_multi_stenosis'] = df_clinical['num_of_stenosis'].apply(
            lambda x: 'No stenosis' if x == 0
            else ('Single stenosis' if x == 1
                  else 'Multi-level stenosis')
        )

        # Add a new column to further stratify subjects with 4 compressions to see how many of them have C2/C3 compression
        df_clinical['num_of_stenosis_including_C2C3'] = df_clinical['num_of_stenosis']
        # subjects_df['num_of_stenosis_including_C2C3'] = subjects_df.apply(
        #     lambda row: '4 including C2/C3' if row['num_of_stenosis'] == 4 and 'C2/C3' in row['stenosis_levels'] else
        #                 ('4' if row['num_of_stenosis'] == 4 else str(row['num_of_stenosis'])),
        #     axis=1
        # )
        df_clinical['num_of_stenosis_including_C2C3'] = df_clinical.apply(
            lambda row: '4 including C2/C3 or C3/C4' if row['num_of_stenosis'] == 4 and any(level in row['stenosis_levels'] for level in ['C2/C3', 'C3/C4']) else
            ('4' if row['num_of_stenosis'] == 4 else str(row['num_of_stenosis'])),
            axis=1
        )

        # ----
        # myelopathy
        # ----
        df_clinical['Myelopathy'] = df_clinical['myelopathy'].fillna('NA')
        df_clinical['Myelopathy'] = df_clinical['Myelopathy'].apply(
            lambda x: 'yes' if x == 1 else ('no' if x == 0 else 'NA')
        )

        # ----
        # therapeutic_decision
        # ----
        # 'surg_timepoint___2_12mth': between baseline and 6 month follow up
        # 'surg_timepoint___3_12mth': between 6 month and 12 month follow up
        # 0: conservative, 1: operative
        # Set therapeutic_decision based on surgery timepoints
        df_clinical['therapeutic_decision'] = df_clinical.apply(
            lambda row: 'operative' if (row['surg_timepoint___2_12mth'] == 1 or row['surg_timepoint___3_12mth'] == 1) else 'conservative',
            axis=1)
        # Fill missing values with 'NA'
        df_clinical['therapeutic_decision'] = df_clinical['therapeutic_decision'].fillna('NA')

        # ----
        # age
        # ----
        # rename age_BL to age
        df_clinical = df_clinical.rename(columns={'age_BL': 'age'})
        df_clinical['age_group'] = df_clinical['age'].apply(_create_age_group)
        df_clinical['age'] = df_clinical['age'].fillna('NA')

        # ----
        # sex
        # ----
        # rename sex_BL to sex
        df_clinical = df_clinical.rename(columns={'sex_BL': 'sex'})
        df_clinical['sex'] = df_clinical['sex'].fillna('NA')
        # Change 1 to F and 2 to M
        df_clinical['sex'] = df_clinical['sex'].replace({1: 'F', '1': 'F', 2: 'M', '2': 'M'})

        # ----
        # Clinical scores
        # ----
        # Print number of missing values per column for baseline_clinical_columns
        print("Number of missing values per baseline clinical column in clinical file:")
        print(df_clinical[baseline_clinical_columns].isnull().sum())

        # Stratify baseline mJOA
        df_clinical['mJOA_severity_bl'] = df_clinical['total_mjoa_BL'].apply(_stratify_mjoa)

        for col in baseline_clinical_columns:
            df_clinical[col].fillna('NA')

        # ----
        # Drop NA subjects
        # ----
        # Exclude subjects with 'NA'
        for col in ['stenosis_levels', 'highest_stenosis', 'maximum_stenosis', 'therapeutic_decision', 'age', 'sex', 'Myelopathy', 'therapeutic_decision'] + baseline_clinical_columns:
            print(f'Number of subjects with NA in {col}: {(df_clinical[col] == "NA").sum()}')
            print(f'Number of subjects before excluding NA in {col}: {df_clinical.shape[0]}')
            df_clinical = df_clinical[df_clinical[col] != 'NA']
            print(f'Number of subjects after excluding NA in {col}: {df_clinical.shape[0]}')

        # ----
        # Exclude severe and unknown mJOA subjects
        # ----
        df_clinical = exclude_severe_mjoa(df_clinical)

    else:
        sys.exit(f"Warning: Clinical file not found: {clinical_file}")

    return df_clinical, clinical_columns


def merge_morphometrics_and_clinical_data(subjects_df, df_clinical, clinical_columns):
    """
    Merge clinical data into dataframe with morphometrics
    """
    subjects_df = subjects_df.merge(
        df_clinical[['participant_id', 'maximum_stenosis', 'stenosis_levels', 'highest_stenosis', 'num_of_stenosis',
                     'single_vs_multi_stenosis', 'Myelopathy', 'therapeutic_decision', 'age', 'age_group', 'sex',
                     'mJOA_severity_bl'] + clinical_columns],
        on='participant_id', how='inner'
    )

    # Rename maximum_stenosis to MCL
    subjects_df = subjects_df.rename(columns={'maximum_stenosis': 'MCL'})
    print(f'Number of subjects after merging MRI and clinical data: {len(subjects_df['participant_id'].unique())}')

    return subjects_df

def read_exclude_file_and_exclude_subjects(subjects_df, exclude_file):
    """
    Read exclude file (YAML) and exclude subjects from subjects_df
    """
    if exclude_file and os.path.isfile(exclude_file):
        # Extract participant IDs (e.g., 'sub-004') from 'sub-XXX/ses-YYY'
        with open(exclude_file, "r") as f:
            data = yaml.safe_load(f)
        exclude_ids = []
        for section in data.values():
            for item in section:
                pid = item.split('/')[0].strip()
                exclude_ids.append(pid)
        print(f'Subjects to exclude: {exclude_ids}')  # ['sub-004', 'sub-020', 'sub-042', 'sub-045']
        print(f'Number of subjects to exclude: {len(exclude_ids)}')

        initial_count = len(subjects_df['participant_id'].unique())
        subjects_df = subjects_df[~subjects_df['participant_id'].isin(exclude_ids)]
        excluded_count = initial_count - len(subjects_df['participant_id'].unique())
        print(f"Excluded {excluded_count} subjects based on exclude file: {exclude_file}")
        print(f"Number of unique subjects after exclusion: {len(subjects_df['participant_id'].unique())}")

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

def drop_highest_stenosis(subjects_df):
    # Drop rows with highest_stenosis == C2/C3 or C3/C4
    print(f"Number of unique subjects before dropping highest_stenosis at C2/C3: {len(subjects_df['participant_id'].unique())}")
    subjects_df = subjects_df[subjects_df['highest_stenosis'] != 'C2/C3']
    print(f"Number of unique subjects after dropping highest_stenosis at C2/C3: {len(subjects_df['participant_id'].unique())}")

    print(f"Number of unique subjects before dropping highest_stenosis at C3/C4: {len(subjects_df['participant_id'].unique())}")
    subjects_df = subjects_df[subjects_df['highest_stenosis'] != 'C3/C4']
    print(f"Number of unique subjects after dropping highest_stenosis at C3/C4: {len(subjects_df['participant_id'].unique())}")

    # Drop rows with num_of_stenosis == 4
    print(f"Number of unique subjects before dropping num_of_stenosis == 4: {len(subjects_df['participant_id'].unique())}")
    subjects_df = subjects_df[subjects_df['num_of_stenosis'] != 4]
    print(f"Number of unique subjects after dropping num_of_stenosis == 4: {len(subjects_df['participant_id'].unique())}")

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
        fig, axs = plt.subplots(2, int(len(METRICS)), figsize=(int(len(METRICS)) * 4, 10))
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

        # Add vertebral level indicators (top row)
        ymin, ymax = ax.get_ylim()
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_normative_data)
        for idx, x in enumerate(ind_vert[1:-1]):
            ax.axvline(df_normative_data.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        for idx, x in enumerate(ind_vert_mid, 0):
            x_pos = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
            y_pos = ymin - (ymax - ymin) * 0.1  # to move below x-axis
            ax.text(df_normative_data.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                    y_pos, x_pos, horizontalalignment='center',
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
            if len(unique_groups) == 2:
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
                        elif 'age_group' in df_level.columns and hue != 'age_group':
                            cov_terms.append('C(age_group)')
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
                                y_star = data_max - 0.03 * yrange #+ 0.03 * yrange
                                # Expand ylim if needed
                                if y_star > ymax:
                                    ax_violin.set_ylim(ymin, y_star + 0.03 * yrange)
                                    ymin, ymax = ax_violin.get_ylim()
                                    yrange = ymax - ymin
                                    y_star = data_max - 0.03 * yrange
                                ax_violin.text(x, y_star, '*', ha='center', va='bottom', fontsize=LABELS_FONT_SIZE+10, color='black')
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
                            ax_violin.text(x, y_star, '*', ha='center', va='bottom', fontsize=LABELS_FONT_SIZE+10, color='black')

        # Y-axis limits for bottom row
        if metric in METRICS_YLIMITS[structure]:
            ax_violin.set_ylim(METRICS_YLIMITS[structure][metric][0]*0.8, METRICS_YLIMITS[structure][metric][1]*1.1)

        ax_violin.set_xlabel('Vertebral level', fontsize=LABELS_FONT_SIZE)
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


def analyze_longitudinal_mjoa_area(subjects_df, path_ascor_file, output_dir):
    """
    Analyse the relationship between baseline area ('MEAN(area)'), separately for C2 and C3, and longitudinal
    repeated measures of mJOA scores (['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']) usings
    a linear mixed-effects model with random intercepts and slopes for time. Area is used as fixed effects,
    with adjustments for age, sex, myelopathy, MCL and aSCOR (i.e., T2w_ax_aSCOR_metrics_perslice_PAM50.csv is also read).
    An interaction term (area × Time) is used to capture these dynamic effects in the model.
    In addition, to account for the correlation between mJOA measurements taken at different time points,
    an autoregressive structure of order 1 is used.

    :param subjects_df: pandas.DataFrame containing morphometric and clinical data
    :param path_ascor_file: str, path to the aSCOR metrics CSV file
    :param output_dir: str, directory to save analysis results
    :return: dict, dictionary containing model results for C2 and C3 levels
    """

    print("\n" + "="*80)
    print("LONGITUDINAL mJOA-AREA ANALYSIS")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load aSCOR data if provided
    df_ascor = None
    if path_ascor_file and os.path.exists(path_ascor_file):
        try:
            df_ascor = read_morphometrics_file(os.path.abspath(path_ascor_file))
            print(f"Loaded aSCOR data from: {path_ascor_file}")
        except Exception as e:
            print(f"Warning: Could not load aSCOR data: {e}")

    # Filter for baseline area at C2 and C3 levels only
    baseline_area_df = subjects_df[(subjects_df['VertLevel'].isin([2, 3])) & (subjects_df['session_id'] == 'ses-M0')].copy()
    baseline_ascor_df = df_ascor[(df_ascor['VertLevel'].isin([2, 3])) & (df_ascor['session_id'] == 'ses-M0')].copy()

    # Prepare longitudinal mJOA data
    mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']
    time_points = [0, 6, 12]  # months

    results = {}

    # Analyze separately for C2 and C3
    for level in [2, 3]:
        level_name = f"C{level}"
        print(f"\n--- Analysis for {level_name} ---")

        # Get baseline area for this level - aggregate per participant per level
        level_data_area = baseline_area_df[baseline_area_df['VertLevel'] == level].copy()
        level_data_acor = baseline_ascor_df[baseline_ascor_df['VertLevel'] == level].copy()

        # Compute per-participant per-level mean for the area (similar to violin plot approach)
        level_grouped_area = level_data_area[['participant_id', 'VertLevel', 'MEAN(area)']].dropna().groupby(['participant_id', 'VertLevel'], as_index=False).mean()
        level_grouped_ascor = level_data_acor[['participant_id', 'VertLevel', 'aSCOR']].dropna().groupby(['participant_id', 'VertLevel'], as_index=False).mean()

        # Merge back with clinical data (get one row per participant with baseline clinical data)
        clinical_cols = ['participant_id', 'age', 'sex', 'Myelopathy', 'MCL'] + mjoa_columns
        available_clinical_cols = [col for col in clinical_cols if col in level_data_area.columns]
        clinical_data = level_data_area[available_clinical_cols].drop_duplicates('participant_id')

        # Merge grouped area with clinical data
        level_merged = level_grouped_area.merge(clinical_data, on='participant_id', how='inner')
        # Merge aSCOR data if available
        level_merged = level_merged.merge(level_grouped_ascor, on='participant_id', how='left')

        # Create longitudinal dataset
        long_data = []

        for _, row in level_merged.iterrows():
            participant_id = row['participant_id']
            baseline_area = row['MEAN(area)']

            # Extract covariates (use baseline values)
            age = row.get('age', np.nan)
            sex = row.get('sex', 'unknown')
            myelopathy = row.get('Myelopathy', 'unknown')
            mcl = row.get('MCL', 'unknown')
            ascor_value = row.get('aSCOR', np.nan)

            # Create rows for each time point
            for time_idx, (mjoa_col, time_months) in enumerate(zip(mjoa_columns, time_points)):
                mjoa_score = row.get(mjoa_col, np.nan)

                if not pd.isna(mjoa_score):
                    long_data.append({
                        'participant_id': participant_id,
                        'time': time_months,
                        'time_categorical': f"M{time_months}",
                        'mjoa_score': mjoa_score,
                        'baseline_area': baseline_area,
                        'age': age,
                        'sex': sex,
                        'Myelopathy': myelopathy,
                        'mcl': mcl,
                        'ascor': ascor_value,
                        'level': level_name
                    })

        if not long_data:
            print(f"No longitudinal mJOA data available for {level_name}")
            continue

        # Convert to DataFrame
        long_df = pd.DataFrame(long_data)

        # Remove participants with insufficient data (need at least 2 time points)
        print(f"Number of participants before filtering: {long_df['participant_id'].nunique()}")
        participant_counts = long_df['participant_id'].value_counts()
        valid_participants = participant_counts[participant_counts >= 2].index
        long_df = long_df[long_df['participant_id'].isin(valid_participants)]
        print(f"Removed participants with less than 2 time points. Remaining participants: {len(valid_participants)}")

        print(f"Analysis dataset for {level_name}:")
        print(f"  - Participants: {long_df['participant_id'].nunique()}")
        print(f"  - Total observations: {len(long_df)}")
        print(f"  - Time points per participant: {long_df['participant_id'].value_counts().describe()}")

        # Center continuous variables
        long_df['baseline_area_c'] = long_df['baseline_area'] - long_df['baseline_area'].mean()
        # long_df['time_c'] = long_df['time'] - long_df['time'].mean()

        # Handle age centering
        age_valid = pd.to_numeric(long_df['age'], errors='coerce').notna()
        if age_valid.any():
            long_df['age_c'] = pd.to_numeric(long_df['age'], errors='coerce')
            long_df['age_c'] = long_df['age_c'] - long_df['age_c'].mean()

        # Handle aSCOR centering
        ascor_valid = pd.to_numeric(long_df['ascor'], errors='coerce').notna()
        if ascor_valid.any():
            long_df['ascor_c'] = pd.to_numeric(long_df['ascor'], errors='coerce')
            long_df['ascor_c'] = long_df['ascor_c'] - long_df['ascor_c'].mean()

        # Build model formula
        fixed_effects = ['baseline_area_c', 'time', 'baseline_area_c:time']

        # FIXED EFFECTS - in the main formula
        # Add covariates if available and have sufficient variation
        if 'age_c' in long_df.columns and pd.to_numeric(long_df['age_c'], errors='coerce').notna().any():
            fixed_effects.append('age_c')

        if long_df['sex'].nunique() > 1 and 'unknown' not in long_df['sex'].values:
            fixed_effects.append('C(sex)')

        if long_df['Myelopathy'].nunique() > 1 and 'unknown' not in long_df['Myelopathy'].values:
            fixed_effects.append('C(Myelopathy)')

        if long_df['mcl'].nunique() > 1 and 'unknown' not in long_df['mcl'].values:
            fixed_effects.append('C(mcl)')

        if 'ascor_c' in long_df.columns and pd.to_numeric(long_df['ascor_c'], errors='coerce').notna().any():
            fixed_effects.append('ascor_c')

        formula = f"mjoa_score ~ {' + '.join(fixed_effects)}"

        try:
            # Fit mixed-effects model with random intercepts and slopes
            print(f"\nFitting model: {formula}")
            print(f"Random effects: Random intercepts and slopes for time by participant")

            # Remove rows with missing values for the model
            model_df = long_df.dropna(subset=['mjoa_score'] +
                                    [col for col in ['baseline_area_c', 'time', 'age_c', 'ascor_c']
                                     if col in long_df.columns])

            if len(model_df) < 10:
                print(f"Insufficient data for modeling {level_name} (n={len(model_df)})")
                continue

            # RANDOM EFFECTS - specified separately in re_formula
            # Fit the mixed-effects model
            # Random intercepts and slopes for time
            model = MixedLM.from_formula(
                formula,
                data=model_df,
                groups=model_df["participant_id"],
                re_formula="1 + time"  # Random intercept (1) and slope (time)
                # - By allowing this intercept to vary across patients, the model lets each person start at a different disability level.
                # - Without a random intercept, the model would assume everyone starts at the same mJOA value, which is unrealistic.
                # - Allowing this slope to vary means some patients worsen quickly, some slowly, and some may barely change.
                # - Without a random slope, the model would force all patients to worsen at the same rate.
            )

            try:
                result = model.fit(method='lbfgs', maxiter=1000)
            except:
                # Fallback to simpler fitting method
                result = model.fit()

            # Extract results
            model_summary = {
                'level': level_name,
                'n_participants': model_df['participant_id'].nunique(),
                'n_observations': len(model_df),
                'formula': formula,
                'converged': result.converged,
                'log_likelihood': result.llf,
                'aic': result.aic,
                'bic': result.bic
            }

            # Extract coefficient results
            coef_results = []
            for param in result.params.index:
                coef_results.append({
                    'parameter': param,
                    'coefficient': result.params[param],
                    'std_error': result.bse[param],
                    'z_value': result.tvalues[param],
                    'p_value': result.pvalues[param],
                    'ci_lower': result.conf_int().loc[param, 0],
                    'ci_upper': result.conf_int().loc[param, 1]
                })

            model_summary['coefficients'] = pd.DataFrame(coef_results)

            # Print results
            print(f"\nModel Results for {level_name}:")
            print(f"  Converged: {result.converged}")
            print(f"  Participants: {model_summary['n_participants']}")
            print(f"  Observations: {model_summary['n_observations']}")
            print(f"  AIC: {result.aic:.2f}")
            print(f"  BIC: {result.bic:.2f}")

            print(f"\nFixed Effects:")
            for _, row in model_summary['coefficients'].iterrows():
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['parameter']}: β = {row['coefficient']:.4f} ± {row['std_error']:.4f}, "
                      f"p = {row['p_value']:.4f}{significance} "
                      f"[95% CI: {row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

            # baseline_area_c - "Does area predict mJOA on average?" -- effect of baseline CSA on mJOA
            # time - "Does mJOA improve over time?" -- effect of time on mJOA
            # baseline_area_c:time - does baseline area modify the rate of mJOA change over time -- - larger area may slow disability progression

            # Main effect (baseline_area_c): Tests if area predicts starting point (baseline mJOA)
            # Interaction (baseline__area_c:time): Tests if area predicts rate of change (slope over time)

            # Group Var - The between-participant variance in baseline mJOA scores (1.01), indicating substantial individual differences in neurological function
            # Group x time Cov - The covariance between individual baselines and slopes, showing how baseline mJOA relates to rate of change over time
            # time Var - The between-participant variance in rates of mJOA change over time (0.0035), indicating some individuals improve faster than others

            # Save detailed results
            results_file = os.path.join(output_dir, f"mixed_effects_mjoa_area_{level_name}.txt")
            with open(results_file, 'w') as f:
                f.write(f"Mixed-Effects Model Results: mJOA vs Area ({level_name})\n")
                f.write("="*60 + "\n\n")
                f.write(f"Model: {formula}\n")
                f.write(f"Random Effects: Random intercepts and slopes for time by participant\n\n")
                f.write(f"Sample Size:\n")
                f.write(f"  Participants: {model_summary['n_participants']}\n")
                f.write(f"  Observations: {model_summary['n_observations']}\n\n")
                f.write(f"Model Fit:\n")
                f.write(f"  Converged: {result.converged}\n")
                f.write(f"  Log-Likelihood: {result.llf:.4f}\n")
                f.write(f"  AIC: {result.aic:.2f}\n")
                f.write(f"  BIC: {result.bic:.2f}\n\n")
                f.write(str(result.summary()))

            # Save coefficient table
            coef_file = os.path.join(output_dir, f"mixed_effects_coefficients_{level_name}.csv")
            model_summary['coefficients'].to_csv(coef_file, index=False)

            # Save model data for further analysis
            data_file = os.path.join(output_dir, f"mixed_effects_data_{level_name}.csv")
            model_df.to_csv(data_file, index=False)

            results[level_name] = model_summary

            print(f"Results saved:")
            print(f"  - Detailed summary: {results_file}")
            print(f"  - Coefficients: {coef_file}")
            print(f"  - Model data: {data_file}")

        except Exception as e:
            print(f"Error fitting model for {level_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary report
    if results:
        summary_file = os.path.join(output_dir, "mixed_effects_summary_report.txt")
        with open(summary_file, 'w') as f:
            f.write("LONGITUDINAL mJOA-AREA ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")

            for level_name, result in results.items():
                f.write(f"{level_name} Results:\n")
                f.write(f"  Sample: {result['n_participants']} participants, {result['n_observations']} observations\n")
                f.write(f"  Model fit: AIC={result['aic']:.2f}, BIC={result['bic']:.2f}\n")

                # Key findings
                coef_df = result['coefficients']

                # Baseline area effect
                baseline_effect = coef_df[coef_df['parameter'] == 'baseline_area_c']
                if not baseline_effect.empty:
                    row = baseline_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Baseline area effect: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                # Time effect
                time_effect = coef_df[coef_df['parameter'] == 'time']
                if not time_effect.empty:
                    row = time_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Time effect: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                # Interaction effect
                interaction_effect = coef_df[coef_df['parameter'] == 'baseline_area_c:time']
                if not interaction_effect.empty:
                    row = interaction_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Area × Time interaction: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                f.write("\n")

        print(f"\nSummary report saved to: {summary_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

    return results


def print_mjoa_by_sex(subjects_df):
    print("mJOA SCORES BY SEX ANALYSIS")
    print("=" * 80)
    mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']
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
    print(f"  - Males: {len(participants_df[participants_df['sex'] == 'M'])}")
    print(f"  - Females: {len(participants_df[participants_df['sex'] == 'F'])}")

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


def main():
    args = get_parser().parse_args()
    path_HC = os.path.expandvars(args.path_HC)
    path_participants_tsv_pam50 = os.path.expandvars(args.participants_file_pam50)
    path_out = os.path.abspath(args.o)
    sessions_to_process = args.s
    exclude_file = os.path.expandvars(args.exclude_file)
    c2c3_file = os.path.expandvars(args.c2c3_file)

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
    # Read text file with levels to use (C3 or C2,C3 or exclude)
    # ----
    subjects_df = read_c2c3_file_and_apply_exclusions(subjects_df, c2c3_file)

    # ----
    # Drop rows with highest_stenosis == C2/C3 or C3/C4
    # Drop rows with num_of_stenosis == 4
    # ----
    subjects_df = drop_highest_stenosis(subjects_df)

    # Save unique participant IDs to be reused by other scripts
    unique_participants = subjects_df['participant_id'].unique()
    unique_participants_file = os.path.join(path_out, 'unique_participants_ids.txt')
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
    vert_min, vert_max = (2, 6) if structure == 'spinal_cord' else (2, 3)
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
    # Plotting
    # -------------
    os.makedirs(path_out, exist_ok=True)
    # Use basename from args.i to create figure name
    figure_basename = os.path.basename(args.i).replace('.csv', '')
    figure_path = os.path.join(path_out, figure_basename)
    # create_figure(subjects_df, df_normative_data, sessions_to_process, figure_path, args.stratify)

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
    # Longitudinal analysis of mJOA scores with baseline cord area
    # ----
    if structure == 'spinal_cord':
        structure = 'cord'

    print("\n" + "="*80)
    print("RUNNING LONGITUDINAL mJOA-AREA ANALYSIS")
    print("="*80)

    # Output directory for longitudinal analysis results
    longitudinal_output_dir = os.path.join(path_out, f"longitudinal_mjoa_{structure}_area_analysis")

    # Get aSCOR file path (from command line or auto-construct)
    ascor_file_path = getattr(args, 'ascor_file', None)
    if ascor_file_path:
        ascor_file_path = os.path.expandvars(ascor_file_path)
        if not os.path.exists(ascor_file_path):
            print(f"Warning: Specified aSCOR file not found: {ascor_file_path}")
            ascor_file_path = None
    else:
        # Try to auto-construct aSCOR file path
        try:
            input_dir = os.path.dirname(args.i)
            ascor_filename = os.path.basename(args.i).replace(f'{structure}_metrics', 'aSCOR_metrics')
            potential_ascor_path = os.path.join(input_dir, ascor_filename)
            if os.path.exists(potential_ascor_path):
                ascor_file_path = potential_ascor_path
                print(f"Auto-detected aSCOR file: {ascor_file_path}")
            else:
                print(f"No aSCOR file found at: {potential_ascor_path}")
        except Exception as e:
            print(f"Could not construct aSCOR file path: {e}")

    # Run longitudinal analysis
    try:
        results = analyze_longitudinal_mjoa_area(subjects_df, ascor_file_path, longitudinal_output_dir)

        if results:
            print(f"\nLongitudinal analysis completed successfully!")
            print(f"Results saved to: {longitudinal_output_dir}")
        else:
            print("No results from longitudinal analysis.")

    except Exception as e:
        print(f"Error running longitudinal analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
