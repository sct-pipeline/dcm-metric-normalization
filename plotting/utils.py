import os
import sys
import yaml
import re
import pandas as pd

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64',
    'aSCOR': 'float64'
}

# 'maximum_stenosis' column
MCL_FORMAT = {
    1: 'C2/C3',
    2: 'C3/C4',
    3: 'C4/C5',
    4: 'C5/C6',
    5: 'C6/C7',
    6: 'C7/T1'
}


def load_normative_df_c2(normative_dir, participants_file=None):
    """
    Load normative data from spine-generic dataset for C2 level.
    Compute mean cord and canal area across slices per subject at C2 level.
    """
    cord_dir = os.path.join(normative_dir, 'spinal_cord')
    cord_df = pd.DataFrame()
    for file in os.listdir(cord_dir):
        if 'PAM50.csv' in file:
            df = pd.read_csv(os.path.join(cord_dir, file), dtype=METRICS_DTYPE)
            cord_df = pd.concat([cord_df, df], axis=0, ignore_index=True)
    # Add participant_id
    cord_df.insert(0, 'participant_id', cord_df['Filename'].str.split('/').str[0])
    # Optionally merge participants.tsv info
    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        cord_df = cord_df.merge(df_participants[['participant_id', 'age', 'sex']], on='participant_id', how='left')
    # Keep only VertLevel C2
    cord_df = cord_df[cord_df['VertLevel'] == 2]
    cord_df = cord_df.dropna(subset=['MEAN(area)'])
    # Compute mean per level for each subject
    grouped = cord_df.groupby(['participant_id', 'VertLevel']).agg({
        'MEAN(area)': 'mean',
        'age': 'first',
        'sex': 'first'
    }).reset_index()
    return grouped

def _categorize_c2_area(area, mean_c2_cord_normative):
    """
    Categorize C2 cord area as 'Below normative mean C2' or 'Above normative mean C2'. based on normative mean cord area.
    Returns None for missing data.
    """
    if pd.isna(area) or pd.isna(mean_c2_cord_normative):
        return None
    if area < mean_c2_cord_normative:
        return 'Below normative mean C2 cord area'
    else:
        return 'Above normative mean C2 cord area'


def format_pvalue(p_value, alpha=0.001, decimal_places=3, include_space=False, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals
    :param p_value: input p-value as a float
    :param alpha: significance level
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    if p_value < alpha:
        p_value = space + "<" + space + str(alpha)
    # If the p-value is greater than alpha, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


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


def exclude_severe_mjoa(df):
    # Exclude subjects with severe mJOA (<12) or unknown mJOA
    print(f"Number of subjects: {len(df['participant_id'].unique())}")
    print("Excluding subjects with severe myelopathy (mJOA < 12)...")
    # Keep only subjects with mJOA_severity_bl 'mild (15 ≤ mJOA ≤ 18)' and 'moderate (12 ≤ mJOA ≤ 14)'
    df = df[df['mJOA_severity_bl'].isin([
        'mild (15 ≤ mJOA ≤ 18)',
        'moderate (12 ≤ mJOA ≤ 14)'
    ])]
    print(f"Number of subjects after excluding severe myelopathy: {len(df['participant_id'].unique())}")

    return df


def read_clinical_file(clinical_file, sessions=2):
    """
    Read clinical Excel file with mJOA scores.
    :param clinical_file: path to Excel file with clinical scores (must contain 'total_mjoa_BL' column)
    :param sessions: number of sessions to read (e.g., 2 for baseline and 6 month follow up; 3 for baseline, 6 month and 12 month follow up)
    :return: pandas dataframe with clinical data
    :returns: list of clinical columns
    """

    # 2 sessions: baseline and 6 month follow up
    if sessions == 2:
        clinical_columns = [
            'total_mjoa_BL', 'total_mjoa_6mth',
            'motor_dysfunction_UE_bl_BL', 'motor_dysfunction_UE_6mth_6mth',
            'motor_dysfunction_LE_bl_BL', 'motor_dysfunction_LE_6mth_6mth',
            'sensory_dysfunction_UE_bl_BL', 'sensory_dysfunction_UE_6mth_6mth',
            'sphincter_dysfunction_bl_BL', 'sphincter_dysfunction_6mth_6mth',
            'UEPP_C4_T1_bl', 'UEPP_C4_T1_6mth',
            'UELT_C4_T1_bl_BL', 'UELT_C4_T1_6mth_6mth',
            'upper_extrem_motor_total_BL', 'upper_extrem_motor_total_6mth',
        ]
        surgery_columns = [
            'surg_timepoint___1_12mth',  # surgery before baseline (used for exclusion)
            'surg_date_before_6mth',     # surgery date between baseline and 6 month follow up
        ]
        date_columns = [
            'orthopedics_assessment_date_BL',
            'orthopedics_assessment_date_6mth',
        ]
        print(f'Number of sessions: {sessions}. Reading baseline and 6 month follow up clinical data.')
    # 3 sessions: baseline, 6 month and 12 month follow up
    elif sessions == 3:
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
        surgery_columns = [
            'surg_timepoint___1_12mth',   # surgery before baseline (used for exclusion)
            'surg_date_before_6mth',      # surgery date between baseline and 6 month follow up
            'surg_date_before_12mth',     # surgery date between 6 month and 12 month follow up
        ]
        date_columns = [
            'orthopedics_assessment_date_BL',
            'orthopedics_assessment_date_6mth',
            'orthopedics_assessment_date_12mth',
        ]
        print(f'Number of sessions: {sessions}. Reading baseline, 6 month and 12 month follow up clinical data.')
    # Exit if sessions is not 2 or 3
    else:
        print("Error: sessions must be either 2 or 3.")
        sys.exit(1)

    columns_to_read = [
        'record_id_BL', 'age_BL', 'sex_BL', 'maximum_stenosis', 'myelopathy',
        'c2_stenosis_no_yes', 'c3_stenosis_no_yes',
        'c4_stenosis_no_yes', 'c5_stenosis_no_yes', 'c6_stenosis_no_yes', 'c7_stenosis_no_yes',
        ]

    columns_to_read += clinical_columns + surgery_columns + date_columns

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
        # df_clinical['maximum_stenosis'] = df_clinical['maximum_stenosis'].apply(lambda x: x if x in MCL_COLORS else 'NA')
        df_clinical['maximum_stenosis'] = df_clinical['maximum_stenosis'].apply(lambda x: x if x in MCL_FORMAT.values() else 'NA')

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
        # 'surg_date_before_6mth': surgery date between baseline and 6 month follow up (non-null = surgery happened)
        # 'surg_date_before_12mth': surgery date between 6 month and 12 month follow up (non-null = surgery happened)
        # If a date is present, surgery happened; if no date, no surgery.
        # Set therapeutic_decision based on surgery date columns
        df_clinical['therapeutic_decision'] = df_clinical.apply(
            lambda row: 'operative' if pd.notna(row['surg_date_before_6mth']) else 'conservative',
            axis=1)
        # Fill missing values with 'NA'
        df_clinical['therapeutic_decision'] = df_clinical['therapeutic_decision'].fillna('NA')

        # Exclude subjects with surgery ONLY between 6m and 12m follow up.
        # Subjects who already had surgery before 6m (surg_date_before_6mth non-null)
        # are kept regardless of surg_date_before_12mth, because their therapeutic
        # decision (operative) is already captured by the 6m column.
        if sessions == 3:
            only_6m_to_12m = (
                df_clinical['surg_date_before_12mth'].notna() &
                df_clinical['surg_date_before_6mth'].isna()
            )
            print(f"Number of subjects before excluding those with surgery only between "
                  f"6 month and 12 month follow up: {len(df_clinical['participant_id'].unique())}")
            df_clinical = df_clinical[~only_6m_to_12m]
            print(f"Number of subjects after excluding those with surgery only between "
                  f"6 month and 12 month follow up: {len(df_clinical['participant_id'].unique())}")

        # ----
        # Surgery before baseline
        # ----
        df_clinical['surgery_before_baseline'] = df_clinical['surg_timepoint___1_12mth'].apply(
            lambda x: 'yes' if x == 1 else ('no' if x == 0 else 'NA')
        )

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