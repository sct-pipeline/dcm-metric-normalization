import os
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