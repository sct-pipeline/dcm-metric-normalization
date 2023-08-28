import os
import re
import argparse

import pandas as pd

DICT_DISC_LABELS = {
                    'C1/C2': 2,
                    'C2/C3': 3,
                    'C3/C4': 4,
                    'C4/C5': 5,
                    'C5/C6': 6,
                    'C6/C7': 7
                    }


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def fetch_participant_id(filename):
    """
    Get participant ID from the input BIDS-compatible filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz)
    :return: participant_id: subject ID (e.g., sub-001)
    """
    participant_id = re.search('sub-(.*?)[_/]', filename)  # [_/] slash or underscore
    participant_id = participant_id.group(0)[:-1] if participant_id else ""  # [:-1] removes the last underscore or slash

    return participant_id


def read_metric_file(file_path, dict_exclude_subj, df_participants):
    """
    Read CSV file with computed metrics and return Pandas DataFrame
    :param file_path: path to the CSV file with computed metrics
    :param dict_exclude_subj: list with subjects to exclude
    :param df_participants: Pandas DataFrame with participants.tsv file
    :return: Pandas DataFrame
    """
    # Load CSV file
    if os.path.isfile(file_path):
        df_morphometrics = pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')

    # Fetch participant_id from filename
    list_participant_id = list()
    # Loop across rows
    for index, row in df_morphometrics.iterrows():
        participant_id = fetch_participant_id(row['filename'])
        list_participant_id.append(participant_id)
    # Insert list of subIDs into pandas DF
    df_morphometrics.insert(1, "participant_id", list_participant_id)
    # Delete column 'filename'
    df_morphometrics.drop('filename', axis=1, inplace=True)

    # Merge df_participants['maximum_stenosis'] to df_morphometrics
    df_morphometrics = pd.merge(df_morphometrics, df_participants[['participant_id', 'maximum_stenosis']],
                                on='participant_id', how='left')
    # Change coding of 'maximum_stenosis' column (e.g., from to 'C1/C2' to '2') based on DICT_DISC_LABELS
    df_morphometrics['level'] = df_morphometrics['maximum_stenosis'].map(DICT_DISC_LABELS)
    # NOTE: after this merge, df_morphometrics has the following columns:
    #   'compression_level' - obtained from sct_compute_compression function based on vert labeling, example 5
    #   'maximum_stenosis' - obtained from participants.tsv file, example 'C5/C6'
    #   'level' - obtained from participants.tsv file and recoded using DICT_DISC_LABELS, example 6

    # Subjects might have multiple levels of compression --> keep only the maximally compressed level (MCL)
    # Get unique list of subjects
    unique_subjects = df_morphometrics['participant_id'].unique().tolist()
    # Loop across subjects
    for sub in unique_subjects:
        # First, get the maximally compressed level
        max_level = df_participants.loc[df_participants['participant_id'] == sub, 'maximum_stenosis'].to_list()[0]
        # Second, get all the compressed levels
        all_compressed_levels = df_participants.loc[df_participants['participant_id'] == sub, 'stenosis'].to_list()[
            0].split(', ')
        # Third, get the index of the maximally compressed level
        idx_max = all_compressed_levels.index(max_level)

        # Get all rows of the subject from df_motphometrics
        # Note: we are resetting the index to be able to use the index as a list, also we use the drop parameter to
        # avoid the old index being added as a column
        df_sub = df_morphometrics.loc[df_morphometrics['participant_id'] == sub].reset_index(drop=True)

        # Check if the maximally compressed level is in the axial FOV (i.e., in the df_sub)
        if idx_max not in df_sub.index.to_list():
            print(f'Maximum level of compression {max_level} is not in axial FOV, excluding {sub}')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id'] == sub].index, inplace=True)
            dict_exclude_subj.append(sub)
        # If the maximally compressed level is in the axial FOV, keep only this row
        else:
            # Get 'slice(I->S)' of df_sub with the maximally compressed level
            slice_max = df_sub.loc[idx_max, 'slice(I->S)']
            df_morphometrics = df_morphometrics.drop(df_morphometrics.loc[(df_morphometrics['participant_id'] == sub) &
                                                          (df_morphometrics['slice(I->S)'] != slice_max)].index)

    return df_morphometrics


def read_participants_file(file_path):
    """
    Read participants.tsv file and return Pandas DataFrame
    :param file_path: path to participants.tsv file
    :return: Pandas DataFrame
    """
    if os.path.isfile(file_path):
        participants_pd = pd.read_csv(file_path, sep='\t')
    else:
        raise FileNotFoundError(f'{file_path} not found')

    return participants_pd


def read_clinical_file(file_path):
    """
    Read file with clinical scores (mJOA, ASIA, GRASSP) and return Pandas DataFrame
    :param file_path: path to excel file
    :return: Pandas DataFrame
    """
    if os.path.isfile(file_path):
        print('Reading: {}'.format(file_path))
        clinical_df = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')
    # mJOA
    mjoa = 'total_mjoa'         # baseline
    mjoa_6m = 'total_mjoa.1'    # 6 months
    mjoa_12m = 'total_mjoa.2'   # 12 months

    # mJOA subscores
    motor_dysfunction_UE_bl = 'motor_dysfunction_UE_bl'         # baseline
    motor_dysfunction_LE_bl = 'motor_dysfunction_LE_bl'         # baseline
    sensory_dysfunction_LE_bl = 'sensory_dysfunction_LE_bl'     # baseline
    sphincter_dysfunction_bl = 'sphincter_dysfunction_bl'       # baseline

    # ASIA/GRASSP - lt_cervical_tot
    lt_cervical_tot = 'lt_cervical_tot'         # baseline
    lt_cervical_tot_6m = 'lt_cervical_tot.1'    # 6 months
    lt_cervical_tot_12m = 'lt_cervical_tot.2'   # 12 months
    # ASIA/GRASSP - pp_cervical_tot
    pp_cervical_tot = 'pp_cervical_tot'        # baseline
    pp_cervical_tot_6m = 'pp_cervical_tot.1'   # 6 months
    pp_cervical_tot_12m = 'pp_cervical_tot.2'  # 12 months
    # ASIA/GRASSP - total_dorsal
    total_dorsal = 'total_dorsal'        # baseline
    total_dorsal_6m = 'total_dorsal.1'   # 6 months
    total_dorsal_12m = 'total_dorsal.2'  # 12 months

    # Read columns of interest from clinical file
    clinical_df = clinical_df[['record_id', mjoa, mjoa_6m, mjoa_12m,
                               motor_dysfunction_UE_bl, motor_dysfunction_LE_bl, sensory_dysfunction_LE_bl, sphincter_dysfunction_bl,
                               lt_cervical_tot, lt_cervical_tot_6m, lt_cervical_tot_12m,
                               pp_cervical_tot, pp_cervical_tot_6m, pp_cervical_tot_12m,
                               total_dorsal, total_dorsal_6m, total_dorsal_12m]]

    # Rename .1 to 6m and .2 to 12m
    clinical_df.columns = clinical_df.columns.str.replace('.1', '_6m')
    clinical_df.columns = clinical_df.columns.str.replace('.2', '_12m')

    return clinical_df


def read_electrophysiology_anatomical_and_motion_file(file_path, df_participants):
    """
    Read electrophysiology, anatomical, and motion data
    :param file_path: path to excel file
    :return anatomical_df: Pandas DataFrame with anatomical data
    :return motion_df: Pandas DataFrame with motion data
    :return electrophysiology_df: Pandas DataFrame with electrophysiology data
    """
    if os.path.isfile(file_path):
        print('Reading: {}'.format(file_path))
        df_all = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')

    # Anatomical data:
    #   - adapted spinal canal occupation ratio (aSCOR) = spinal cord area/spinal canal area
    #   - adapted maximum spinal cord compression (aMSCC) = ratio of spinal cord CSA in segment/spinal cord area at C2) - computed only at baseline so far
    anatomical_df = df_all[['record_id', 'aSCOR_C2', 'aSCOR_C3', 'aSCOR_C4', 'aSCOR_C5', 'aSCOR_C6', 'aSCOR_C7',
                            'aSCOR_C2_6mth', 'aSCOR_C3_6mth', 'aSCOR_C4_6mth', 'aSCOR_C5_6mth', 'aSCOR_C6_6mth', 'aSCOR_C7_6mth',
                            'aSCOR_C2_12mth', 'aSCOR_C3_12mth', 'aSCOR_C4_12mth', 'aSCOR_C5_12mth', 'aSCOR_C6_12mth', 'aSCOR_C7_12mth',
                            'aMSCC_C3toC2', 'aMSCC_C4toC2', 'aMSCC_C5toC2', 'aMSCC_C6toC2', 'aMSCC_C7toC2']]

    # Rename 6mnt and 12mnt columns to 6m and 12m
    anatomical_df.columns = anatomical_df.columns.str.replace('_6mth', '_6m')
    anatomical_df.columns = anatomical_df.columns.str.replace('_12mth', '_12m')

    # Insert participant_id column from df_participants to anatomical_df based on record_id
    anatomical_df = pd.merge(anatomical_df, df_participants[['record_id', 'participant_id']], on='record_id',
                             how='outer', sort=True)
    anatomical_df = anatomical_df.set_index(['participant_id'])

    # Segmental motion data:
    #   - displacement [mm] = area under the curve of the motion plot
    #   - amplitude [cm/s] = range from maximum positive to maximum negativ velocity values
    motion_df = df_all[['record_id', 'C2_amp_ax_or_sag', 'C2_disp_ax_or_sag',
                        'C3_amp_ax_or_sag', 'C3_disp_ax_or_sag',
                        'C4_amp_ax_or_sag', 'C4_disp_ax_or_sag',
                        'C5_amp_ax_or_sag', 'C5_disp_ax_or_sag',
                        'C6_amp_ax_or_sag', 'C6_disp_ax_or_sag',
                        'C7_amp_ax_or_sag']]

    # Insert participant_id column from df_participants to motion_df based on record_id
    motion_df = pd.merge(motion_df, df_participants[['record_id', 'participant_id']], on='record_id',
                         how='outer', sort=True)
    motion_df = motion_df.set_index(['participant_id'])

    # Electrophysiology data:
    # - dermatomal SEP (dSEP) with stimulation at C6 and C8 (only few pathologic results)
    # - dermatomal contact heat evoked potentials (CHEPS) with stimulation at C6, C8 and T4
    electrophysiology_df = df_all[['record_id', 'dSEP_C6_both_patho_bl', 'dSEP_C8_both_patho_bl', 'CHEPS_C6_patho_bl', 'CHEPS_C8_patho_bl', 'CHEPS_T4_grading_patho_bl',
                                   'dSEP_C6_both_patho_6mth', 'dSEP_C8_both_patho_6mth', 'CHEPS_C6_patho_6mth', 'CHEPS_C8_patho_6mth', 'CHEPS_T4_patho_6mth',
                                   'dSEP_C6_both_patho_12mth', 'dSEP_C8_both_patho_12mth', 'CHEPS_C6_patho_12mth', 'CHEPS_C8_patho_12mth', 'CHEPS_T4_patho_12mth',
                                   'CHEPS_C6_diff_6mth_bl', 'CHEPS_C6_diff_12mth_bl', 'CHEPS_C8_diff_6mth_bl', 'CHEPS_C8_diff_12mth_bl', 'CHEPS_T4_diff_6mth_bl', 'CHEPS_T4_diff_12mth_bl']]

    # Rename 6mnt and 12mnt columns to 6m and 12m
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_6mth', '_6m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_12mth', '_12m')

    # Insert participant_id column from df_participants to electrophysiology_df based on record_id
    electrophysiology_df = pd.merge(electrophysiology_df, df_participants[['record_id', 'participant_id']],
                                    on='record_id', how='outer', sort=True)
    electrophysiology_df = electrophysiology_df.set_index(['participant_id'])

    return anatomical_df, motion_df, electrophysiology_df


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