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
