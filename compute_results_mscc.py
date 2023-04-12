#!/usr/bin/env python
# -*- coding: utf-8
# The script compute spearmans correlation coefficient between MSCC, MSCC_norm and mJOA score.

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


DICT_DISC_LABELS = {
                    'C2/C3':3,
                    'C3/C4':4,
                    'C4/C5':5,
                    'C5/C6':6,
                    'C6/C7':7
}


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
        help="Path to folder with MSCC results")
    parser.add_argument(
        '-participants-file-inspired',
        required=True,
        metavar='<file_path>',
        help="canproco participants.tsv file (includes pathology and phenotype columns)")
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


def read_MSCC(path_mscc, exclude, df_participants):
    list_files_mscc = os.listdir(path_mscc)
    list_files_mscc = [file for file in list_files_mscc if '_mscc.csv' in file]
    mscc_df = pd.DataFrame(columns = ['subject','level', 'MSCC', 'MSCC_norm'])
    subject = []
    mscc = []
    mscc_norm = []
    level = []
    for file in list_files_mscc:
        # Only get MSCC csv files
        if '_mscc' in file:
            # Fetch subject ID
            sub_id = file.split('_')[0]
            # Check if subject is in exlude list
            if sub_id not in exclude:
                df = csv2dataFrame(os.path.join(path_mscc, file))
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


def compute_spearmans(a,b):
    a = np.array(a)
    b = np.array(b)
    return spearmanr(a,b)


def gen_chart_corr_mjoa_mscc(df, path_out=""):
    sns.set_style("ticks",{'axes.grid' : True})
    plt.figure()
    fig, ax = plt.subplots()
    #ax.set_box_aspect(1)
    # MSCC with mJOA
    x_vals = df['mJOA']
    y_vals_mscc = df['MSCC']
    y_vals_mscc_norm = df['MSCC_norm']

    r_mscc, p_mscc = compute_spearmans(x_vals, y_vals_mscc)
    r_mscc_norm, p_mscc_norm = compute_spearmans(x_vals, y_vals_mscc_norm)

    logger.info('MSCC: Spearmans r = {} and p = {}'.format(r_mscc, p_mscc))
    logger.info('MSCC norm: Spearmans r = {} and p = {}'.format(r_mscc_norm, p_mscc_norm))

    sns.regplot(x=x_vals, y=y_vals_mscc, label='MSCC') #ci=None,
    sns.regplot(x=x_vals, y=y_vals_mscc_norm, color='crimson', label='MSCC_norm') # ci=None,
    plt.ylabel('MSCC', fontsize=16)
    plt.xlabel('mJOA', fontsize=16)
    plt.tight_layout()
    plt.legend( fontsize=12, bbox_to_anchor=(0.5, 1.12), loc="upper center", ncol=2, framealpha=0.95, handletextpad=0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # save figure
    fname_fig = os.path.join(path_out, 'scatter_mscc_mjoa.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')


def add_mJOA_to_df(participant_df, mscc_df):

    for subject in mscc_df['subject'].to_list():
        mscc_df.loc[mscc_df['subject']==subject,'mJOA'] = participant_df.loc[participant_df['participant_id']==subject,'mjoa'].to_list()
    return mscc_df


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), args.path_out, FNAME_LOG))
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
    df_participants = read_participants_file(args.participants_file_inspired)
    mscc_df = read_MSCC(args.ifolder, dict_exclude_subj, df_participants)
    
    mscc_df = add_mJOA_to_df(df_participants, mscc_df)
    gen_chart_corr_mjoa_mscc(mscc_df, args.path_out)
    logger.info('MSCC data: \n {}'.format(mscc_df))
if __name__ == '__main__':
    main()