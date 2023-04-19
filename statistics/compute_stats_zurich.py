#!/usr/bin/env python
# -*- coding: utf-8
# The script compute logistic linear regression to predict the therapeutic decision in the dcm-zurich dataset
# Author: Sandrine Bédard

import os 
import argparse
import pandas as pd
import logging
import math
import sys
import yaml
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from textwrap import dedent
import seaborn as sns

FNAME_LOG = 'log_stats.txt'
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


metrics = [
    'area',
    'diameter_AP',
    'dimeter_RL',
    'eccentricity',
    'solidity'
]

DICT_DISC_LABELS = {
                    'C1/C2':2,
                    'C2/C3':3,
                    'C3/C4':4,
                    'C4/C5':5,
                    'C5/C6':6,
                    'C6/C7':7
}


metrics_norm = [metric + '_norm' for metric in metrics]


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def get_parser():
    parser = argparse.ArgumentParser(
        description="", # TODO
        formatter_class=SmartFormatter
        )
    parser.add_argument(
        '-ifolder',
        required=True,
        metavar='<file_path>',
        help="Path to folder with results")
    parser.add_argument(
        '-participants-file',
        required=True,
        metavar='<file_path>',
        help="dcm-zurich participants.tsv file")
    parser.add_argument(
        '-path-out',
        required=True,
        metavar='<file_path>',
        help="Path where results will be saved")
    parser.add_argument('-exclude',
                        metavar='<file>',
                        required=False,
                        help=
                        "R|Config yaml file listing subjects to exclude from statistical analysis.\n"
                        "Yaml file can be validated at this website: http://www.yamllint.com/.\n"
                        "Below is an example yaml file:\n"
                        + dedent(
                                 """
                                 - sub-1000032_T1w.nii.gz
                                 - sub-1000498_T1w.nii.gz
                                 """))
    return parser


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    print(filename)
    data = pd.read_csv(filename, encoding='utf-8')
    return data

def read_MSCC(path_results, exclude, df_participants):
    list_files_results = os.listdir(path_results)
    list_files_results = [file for file in list_files_results if '_norm.csv' in file]  # Only take norm (include both)
    # TODO : check to simply add inside the participants.tsv 
    columns = ['subject','level'] + metrics + metrics_norm
    mscc_df = pd.DataFrame(columns = columns) # todo add columns of metrics and 
    subject = []
    level = []
    data_metrics = {}
    for metric in (metrics_norm + metrics):
        data_metrics[metric] = []
    print(data_metrics)
    for file in list_files_results:
        # Fetch subject ID
        sub_id = file.split('_')[0]
        # Check if subject is in exlude list
        if sub_id not in exclude:
            df = csv2dataFrame(os.path.join(path_results, file))
            print(df)
            max_level = df_participants.loc[df_participants['participant_id']==sub_id, 'maximum_stenosis'].to_list()[0]
            all_compressed_levels = df_participants.loc[df_participants['participant_id']==sub_id, 'stenosis'].to_list()[0].split(', ')
            compressed_levels_metrics = df['Compression Level'].to_list()
            idx_max = all_compressed_levels.index(max_level)
            print(sub_id, max_level, all_compressed_levels)
            print((df['Compression Level']))
            max_level_id = DICT_DISC_LABELS[max_level]
            subject.append(sub_id)
          #  for metric in metrics:
                
            #idx_max = df.index[df['Compression Level']==max_level].tolist()
           # print(idx_max, max_level_id)
            # Fill list to create final df
           # level.append(df.loc[idx_max,'Compression Level'])
           # mscc.append(df.loc[idx_max,'MSCC'])
            #mscc_norm.append(df.loc[idx_max,'Normalized MSCC'])
    #mscc_df['subject'] = subject
    #mscc_df['level'] = level
    #mscc_df['MSCC'] = mscc
    #mscc_df['MSCC_norm'] = mscc_norm
    return mscc_df


def read_participants_file(file_path):
    """
    Read participants.tsv file and return Pandas DataFrame
    :param file_path:
    :return:
    """
    if os.path.isfile(file_path):
        participants_pd = pd.read_csv(file_path, sep='\t')
        return participants_pd
    else:
        raise FileNotFoundError(f'{file_path} not found')


# TODO:
# 0. Exclude subjects
# 1. Calcul statistique 2 groupes (mean ± std)
# 2. Binary logistic regression (stepwise)


def main():

    parser = get_parser()
    args = parser.parse_args()

    # If argument path-out doesn't exists, create it.
    if not os.path.exists(args.path_out):
        os.mkdir(args.path_out)

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(args.path_out), FNAME_LOG))
    logging.root.addHandler(fh)

    # Create a dict with subjects to exclude if input .yml config file is passed
    if args.exclude is not None:
        # Check if input yml file exists
        if os.path.isfile(args.exclude):
            fname_yml = args.exclude
        else:
            sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(args.exclude))
        with open(fname_yml, 'r') as stream:
            try:
                dict_exclude_subj = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
    else:
        # Initialize empty dict if n
                dict_exclude_subj = dict()

    logger.info('Exlcuded subjects: {}'.format(dict_exclude_subj))
    df_participants = read_participants_file(args.participants_file)
    print(df_participants)
    mscc_df = read_MSCC(args.ifolder, dict_exclude_subj, df_participants)

if __name__ == '__main__':
    main()