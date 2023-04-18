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
    #list_files_mscc = [file for file in list_files_mscc if '_mscc.csv' in file]
    columns = ['subject','level'] + metrics + metrics_norm
    print(columns)
    mscc_df = pd.DataFrame(columns = columns) # todo add columns of metrics and 
    subject = []
    mscc = []
    mscc_norm = []
    level = []
    for file in list_files_results:
        # Only get MSCC csv files
        if '_mscc' in file:
            # Fetch subject ID
            sub_id = file.split('_')[0]
            # Check if subject is in exlude list
            if sub_id not in exclude:
                df = csv2dataFrame(os.path.join(path_results, file))
                max_level = df_participants.loc[df_participants['participant_id']==sub_id, 'max_compression_level'].to_list()[0]
                max_level = DICT_DISC_LABELS[max_level]
                idx_max = df.index[df['Compression Level']==max_level].tolist()
                if len(idx_max)<1:
                    max_level = df['Compression Level'].tolist()[np.abs(np.array(df['Compression Level'].tolist()) - max_level).argmin()]
                    idx_max = df.index[df['Compression Level']==max_level].tolist()
                idx_max = idx_max[0]
                # Fill list to create final df
                subject.append(sub_id)
                level.append(df.loc[idx_max,'Compression Level'])
                mscc.append(df.loc[idx_max,'MSCC'])
                mscc_norm.append(df.loc[idx_max,'Normalized MSCC'])
    mscc_df['subject'] = subject
    mscc_df['level'] = level
    mscc_df['MSCC'] = mscc
    mscc_df['MSCC_norm'] = mscc_norm
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