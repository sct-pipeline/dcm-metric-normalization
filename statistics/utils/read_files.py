import os
import re

import numpy as np
import pandas as pd

DICT_DISC_LABELS = {
                    'C1/C2': 2,
                    'C2/C3': 3,
                    'C3/C4': 4,
                    'C4/C5': 5,
                    'C5/C6': 6,
                    'C6/C7': 7
                    }


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
        # Handle both 'filename' and 'Filename' column names
        filename = row.get('Filename', row.get('filename', ''))
        participant_id = fetch_participant_id(filename)
        list_participant_id.append(participant_id)
    # Insert list of subIDs into pandas DF
    df_morphometrics.insert(1, "participant_id", list_participant_id)
    # Delete column 'filename' or 'Filename'
    if 'filename' in df_morphometrics.columns:
        df_morphometrics.drop('filename', axis=1, inplace=True)
    if 'Filename' in df_morphometrics.columns:
        df_morphometrics.drop('Filename', axis=1, inplace=True)

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
        # Check if subject exists in df_participants
        if sub not in df_participants['participant_id'].values:
            print(f'Subject {sub} not found in participants file, excluding')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id'] == sub].index, inplace=True)
            dict_exclude_subj.append(sub)
            continue
        
        # First, get the maximally compressed level
        max_level_list = df_participants.loc[df_participants['participant_id'] == sub, 'maximum_stenosis'].to_list()
        if not max_level_list:
            print(f'Maximum stenosis not found for {sub}, excluding')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id'] == sub].index, inplace=True)
            dict_exclude_subj.append(sub)
            continue
        max_level = max_level_list[0]
        
        # Second, get all the compressed levels
        stenosis_list = df_participants.loc[df_participants['participant_id'] == sub, 'stenosis'].to_list()
        if not stenosis_list:
            print(f'Stenosis not found for {sub}, excluding')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id'] == sub].index, inplace=True)
            dict_exclude_subj.append(sub)
            continue
        stenosis_value = stenosis_list[0]
        
        # Handle NaN or non-string values
        if pd.isna(stenosis_value) or not isinstance(stenosis_value, str):
            print(f'Stenosis value is missing or not a string for {sub}, excluding')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id'] == sub].index, inplace=True)
            dict_exclude_subj.append(sub)
            continue
        all_compressed_levels = stenosis_value.split(', ')
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
            # Get 'Slice (I->S)' of df_sub with the maximally compressed level
            slice_col = 'Slice (I->S)' if 'Slice (I->S)' in df_sub.columns else 'slice(I->S)'
            slice_max = df_sub.loc[idx_max, slice_col]
            df_morphometrics = df_morphometrics.drop(df_morphometrics.loc[(df_morphometrics['participant_id'] == sub) &
                                                          (df_morphometrics[slice_col] != slice_max)].index)

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
    
    # Convert BIDS participant_id (sub-001) to numeric record_id (1) for merging with clinical data
    if 'participant_id' in participants_pd.columns:
        participants_pd['record_id'] = participants_pd['participant_id'].str.replace('sub-', '').astype(int)

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
    
    # Map actual column names from the Excel file (_BL, _6mth, _12mth, _24mth, _36mth, _48mth, _60mth)
    # to standardized names (_6m, _12m, _24m, _36m, _48m, _60m)
    timepoint_suffixes = ['_BL', '_6mth', '_12mth', '_24mth', '_36mth', '_48mth', '_60mth']
    
    # Build column selection list dynamically for all available timepoints
    cols_to_select = []
    if 'record_id_BL' in clinical_df.columns:
        cols_to_select.append('record_id_BL')
    elif 'record_id' in clinical_df.columns:
        cols_to_select.append('record_id')
    
    # mJOA total scores across timepoints
    for suffix in timepoint_suffixes:
        col_name = f'total_mjoa{suffix}'
        if col_name in clinical_df.columns:
            cols_to_select.append(col_name)
    
    # mJOA subscores
    for subscore in ['motor_dysfunction_UE', 'motor_dysfunction_LE', 'sensory_dysfunction_UE', 'sphincter_dysfunction']:
        for suffix in timepoint_suffixes:
            col_name = f'{subscore}{suffix}'
            if col_name in clinical_df.columns:
                cols_to_select.append(col_name)
    
    # ASIA/GRASSP scores across timepoints
    for metric in ['upper_extrem_motor_total', 'lower_extrem_motor_total', 'lt_cervical_tot', 'pp_cervical_tot', 'total_dorsal']:
        for suffix in timepoint_suffixes:
            col_name = f'{metric}{suffix}'
            if col_name in clinical_df.columns:
                cols_to_select.append(col_name)
    
    # Select available columns
    clinical_df = clinical_df[[col for col in cols_to_select if col in clinical_df.columns]]
    
    # Standardize column names
    # Rename record_id_BL to record_id for merging
    if 'record_id_BL' in clinical_df.columns:
        clinical_df.rename(columns={'record_id_BL': 'record_id'}, inplace=True)
    
    # Standardize timepoint suffixes: _BL stays as is, _6mth->_6m, _12mth->_12m, _24mth->_24m, etc.
    clinical_df.columns = clinical_df.columns.str.replace('_6mth', '_6m')
    clinical_df.columns = clinical_df.columns.str.replace('_12mth', '_12m')
    clinical_df.columns = clinical_df.columns.str.replace('_24mth', '_24m')
    clinical_df.columns = clinical_df.columns.str.replace('_36mth', '_36m')
    clinical_df.columns = clinical_df.columns.str.replace('_48mth', '_48m')
    clinical_df.columns = clinical_df.columns.str.replace('_60mth', '_60m')

    return clinical_df


def read_electrophysiology_file(file_path, df_participants):
    """
    Read electrophysiology data
    :param file_path: path to excel file
    :param df_participants: Pandas DataFrame with participant data
    :return electrophysiology_df: Pandas DataFrame with electrophysiology data
    """
    if os.path.isfile(file_path):
        print('Reading: {}'.format(file_path))
        df_all = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')

    # Build column selection dynamically
    # Note: record_id also has _BL suffix
    cols_to_select = []
    if 'record_id_BL' in df_all.columns:
        cols_to_select.append('record_id_BL')
    elif 'record_id' in df_all.columns:
        cols_to_select.append('record_id')
    
    # Electrophysiology data columns with actual suffixes from the Excel file
    timepoint_suffixes = ['_BL', '_6mth', '_12mth', '_24mth', '_36mth', '_48mth', '_60mth']
    electro_metrics = ['dSEP_C6_both_patho_bl', 'dSEP_C8_both_patho_bl', 
                      'CHEPS_C6_patho', 'CHEPS_C8_patho', 'CHEPS_T4_grading_patho',
                      'CHEPS_C6_diff', 'CHEPS_C8_diff', 'CHEPS_T4_diff']
    
    for metric in electro_metrics:
        for suffix in timepoint_suffixes:
            # Handle different naming patterns
            for pattern in [f'{metric}{suffix}', f'{metric}_bl{suffix}']:
                if pattern in df_all.columns:
                    cols_to_select.append(pattern)
    
    # Select only columns that exist
    electrophysiology_df = df_all[[col for col in cols_to_select if col in df_all.columns]]
    
    # Standardize column names
    if 'record_id_BL' in electrophysiology_df.columns:
        electrophysiology_df.rename(columns={'record_id_BL': 'record_id'}, inplace=True)
    
    # Standardize timepoint suffixes
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_6mth', '_6m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_12mth', '_12m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_24mth', '_24m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_36mth', '_36m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_48mth', '_48m')
    electrophysiology_df.columns = electrophysiology_df.columns.str.replace('_60mth', '_60m')
    
    # Create combined metrics if individual components exist
    if 'dSEP_C6_both_patho_bl_BL' in df_all.columns or 'dSEP_C8_both_patho_bl_BL' in df_all.columns:
        available_cols = [c for c in ["dSEP_C6_both_patho_bl_BL", "dSEP_C8_both_patho_bl_BL"] if c in electrophysiology_df.columns]
        if available_cols:
            electrophysiology_df['dSEP_both_patho_BL'] = electrophysiology_df[available_cols].sum(axis=1, min_count=1)
    
    # Merge with participants to get participant_id
    electrophysiology_df = pd.merge(electrophysiology_df, df_participants[['record_id', 'participant_id']],
                                    on='record_id', how='outer', sort=True)
    electrophysiology_df = electrophysiology_df.set_index(['participant_id'])

    return electrophysiology_df


def read_anatomical_file(file_path, df_participants):
    """
    Read anatomical data
    :param file_path: path to excel file
    :param df_participants: Pandas DataFrame with participant data
    :return anatomical_df: Pandas DataFrame with anatomical data
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

    return anatomical_df


def read_motion_file(file_path, df_participants):
    """
    Read motion data
    :param file_path: path to excel file
    :param df_participants: Pandas DataFrame with participant data
    :return motion_df: Pandas DataFrame with motion data
    """
    if os.path.isfile(file_path):
        print('Reading: {}'.format(file_path))
        df_all = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')

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

    return motion_df


def read_motion_file_maximum_stenosis(file_path, df_participants):
    """
    Read motion data from the maximum stenosis
    :param file_path: path to excel file
    :param df_participants: Pandas DataFrame with participant data
    :return motion_df: Pandas DataFrame with motion data
    """
    if os.path.isfile(file_path):
        print('Reading: {}'.format(file_path))
        df_all = pd.read_excel(file_path)
    else:
        raise FileNotFoundError(f'{file_path} not found')

    # Segmental motion data:
    #   - displacement [mm] = area under the curve of the motion plot
    #   - amplitude [cm/s] = range from maximum positive to maximum negativ velocity values
    motion_df = df_all[['record_ID_bl', 'amp_max_sten_sag_or_ax1_or_ax2_bl', 'disp_max_sten_sag_or_ax1_or_ax2_mm_bl']]

    # Rename record_ID_bl to record_id
    motion_df = motion_df.rename(columns={'record_ID_bl': 'record_id'})

    # Insert participant_id column from df_participants to motion_df based on record_id
    motion_df = pd.merge(motion_df, df_participants[['record_id', 'participant_id']], on='record_id',
                         how='outer', sort=True)
    motion_df = motion_df.set_index(['participant_id'])

    return motion_df


def merge_anatomical_morphological_final_for_pred(anatomical_df, motion_df, df_morphometric, add_motion=False):
    """
    Aggregate anatomical and motion scores from the maximum level of compression and merge them with computed
    morphometrics
    """
    final_df = df_morphometric.copy()
    # set index of participant to have the same index
    final_df = final_df.set_index(['participant_id'])
    
    # Check for duplicate indices and reset if necessary -- debug
    if final_df.index.has_duplicates:
        print(f'Warning: Duplicate participant IDs found in morphometric data. Resetting index.')
        final_df = final_df.reset_index().drop_duplicates(subset=['participant_id']).set_index(['participant_id'])
    
    # Loop through levels to get the corresponding motion or anatomical metric or the maximum compressed level
    levels = np.unique(final_df['level'].to_list())
    for level in levels:
        # Skip C1/C2 level (2)
        if not level == 2:
            level_conversion = 'C' + str(int(level)-1)  # -1 to convert to zurich disc
            # Add scores at maximum level of compression
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'aSCOR'] = anatomical_df.loc[final_df.index[final_df['level']==level].tolist(), 'aSCOR_' + level_conversion]
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'aMSCC'] = anatomical_df.loc[final_df.index[final_df['level']==level].tolist(), 'aMSCC_' + level_conversion + 'toC2']
            if add_motion:
                final_df.loc[final_df.index[final_df['level']==level].tolist(), 'amp_ax_or_sag'] = motion_df.loc[final_df.index[final_df['level']==level].tolist(), level_conversion+ '_amp_ax_or_sag']
                final_df.loc[final_df.index[final_df['level']==level].tolist(), 'disp_ax_or_sag'] = motion_df.loc[final_df.index[final_df['level']==level].tolist(), level_conversion+ '_disp_ax_or_sag']

    return final_df
