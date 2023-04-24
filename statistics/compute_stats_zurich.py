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
    'diameter_RL',
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
        '-clinical-file',
        required=True,
        metavar='<file_path>',
        help="excel file fo clinical data")
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
    #print(filename)
    data = pd.read_csv(filename, encoding='utf-8')
    return data

def read_MSCC(path_results, exclude, df_participants, file_metric):
    list_files_results = os.listdir(path_results)
    list_files_results = [file for file in list_files_results if '_norm.csv' in file]  # Only take norm (include both)
    subjects_list = np.unique([sub.split('_')[0] for sub in list_files_results])
    print(len(subjects_list))

    # TODO : check to simply add inside the participants.tsv 
    columns = ['participant_id', 'id','level'] + metrics + metrics_norm
    df_combined = pd.DataFrame(columns = columns) # todo add columns of metrics and 
    df_combined['participant_id'] = subjects_list
    for sub in subjects_list:
        # Check if subject is in exlude list
        files_subject = [file for file in list_files_results if sub in file]
        df = csv2dataFrame(os.path.join(path_results, files_subject[0]))
        max_level = df_participants.loc[df_participants['participant_id']==sub, 'maximum_stenosis'].to_list()[0]
        all_compressed_levels = df_participants.loc[df_participants['participant_id']==sub, 'stenosis'].to_list()[0].split(', ')
        idx_max = all_compressed_levels.index(max_level)
        if idx_max not in df.index.to_list():
            print(f'Maximum level of compression {max_level} is not in axial FOV, excluding {sub}')
            df_combined.drop(df_combined.loc[df_combined['participant_id']==sub].index)
            exclude.append(sub)
        else:
            df_combined.loc[df_combined['participant_id']==sub, 'level'] = max_level
            for metric in metrics:
                file = [file for file in files_subject if metric in file]#[0]
                df = csv2dataFrame(os.path.join(path_results, file[0]))
                # Fill list to create final df
                column_norm = 'normalized_'+ metric + '_ratio'
                column_no_norm = metric + '_ratio'
                df_combined.loc[df_combined['participant_id']==sub, metric] = df.loc[idx_max, column_no_norm]
                metric_norm = metric + '_norm'
                df_combined.loc[df_combined['participant_id']==sub, metric_norm] = df.loc[idx_max, column_norm]
                
            #idx_max = df.index[df['Compression Level']==max_level].tolist()
           # print(idx_max, max_level_id)
            #mscc_norm.append(df.loc[idx_max,'Normalized MSCC'])
   # df_combined = pd.DataFrame(data_metrics)
    df_combined.to_csv(file_metric, index=False)
    #df_combined['subject'] = subject
    #df_combined['level'] = level
   # df_combined[metrics_norm + metrics] = data_metrics
    return df_combined


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


def gen_chart_corr_mjoa_mscc(df, metric, mjoa, path_out=""):
    sns.set_style("ticks",{'axes.grid' : True})
    plt.figure()
    fig, ax = plt.subplots()
    #ax.set_box_aspect(1)
    # MSCC with mJOA
    x_vals = df[mjoa]
    y_vals_mscc = df[metric]
    metric_norm = metric + '_norm'
    y_vals_mscc_norm = df[metric_norm]

    r_mscc, p_mscc = compute_spearmans(x_vals, y_vals_mscc)
    r_mscc_norm, p_mscc_norm = compute_spearmans(x_vals, y_vals_mscc_norm)

    logger.info(f'{metric} ratio: Spearmans r = {r_mscc} and p = {p_mscc}')
    logger.info(f'{metric_norm} ratio: Spearmans r = {r_mscc_norm} and p = {p_mscc_norm}')

    sns.regplot(x=x_vals, y=y_vals_mscc, label=(metric+' ratio')) #ci=None,
    sns.regplot(x=x_vals, y=y_vals_mscc_norm, color='crimson', label=(metric_norm + ' ratio')) # ci=None,
    plt.ylabel((metric + ' ratio'), fontsize=16)
    plt.xlabel('mJOA', fontsize=16)
    plt.xlim([min(x_vals) -1, max(x_vals)+1])
    plt.tight_layout()
    plt.legend( fontsize=12, bbox_to_anchor=(0.5, 1.12), loc="upper center", ncol=2, framealpha=0.95, handletextpad=0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # save figure
    fname_fig = os.path.join(path_out, 'scatter_' + metric + '_mjoa.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')

# TODO:
# 0. Exclude subjects
# 1. Calcul statistique 2 groupes (mean ± std)
# 2. Binary logistic regression (stepwise)


def compute_mean_std(df):
    print(df.mean())
    print(df.std())
    mean_by_surgery = df.groupby('therapeutic_decision', as_index=False).mean()
    std_by_surgery = df.groupby('therapeutic_decision', as_index=False).std()
    print(f'Mean and std separated for therapeutic decision')
    pd.set_option('display.max_columns', None)
    print(mean_by_surgery)
    mean_by_level= df.groupby('level', as_index=False).mean()
    print(mean_by_level)


def main():

    parser = get_parser()
    args = parser.parse_args()

    # If argument path-out doesn't exists, create it.
    if not os.path.exists(args.path_out):
        os.mkdir(args.path_out)
    path_out = args.path_out

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(path_out), FNAME_LOG))
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
        # Initialize empty list
        dict_exclude_subj = []

    logger.info('Exlcuded subjects: {}'.format(dict_exclude_subj))
    df_participants = read_participants_file(args.participants_file)
   # print(df_participants)
    if os.path.isfile(args.clinical_file):
        print('Reading: {}'.format(args.clinical_file))
        clinical_df = pd.read_excel(args.clinical_file)
    else:
        raise FileNotFoundError(f'{args.clinical_file} not found')
    mjoa = 'total_mjoa.0'  # change .1 or .2 for different time points
    clinical_df_mjoa = clinical_df[['record_id', mjoa]]  
    #print(clinical_df_mjoa)
    df_participants = pd.merge(df_participants, clinical_df_mjoa, on='record_id', how='outer', sort=True)

    # Merge clinical data to participant.tsv
    # TODO remove when will be included in participant.tsv

    file_metrics = os.path.join(path_out, 'metric_ratio_combined.csv')
    if not os.path.exists(file_metrics):
        df_combined = read_MSCC(args.ifolder, dict_exclude_subj, df_participants, file_metrics)
    else:
        df_combined = pd.read_csv(file_metrics)
        df_combined = df_combined.rename(columns={'subject': 'participant_id'})  # TODO remove when rerun
    #TODO remove subjects with no values (in read MSCC)

    final_df = pd.merge(df_participants, df_combined, on='participant_id', how='outer', sort=True)
    print(final_df.columns)

    final_df.dropna(axis=0, subset=['area_norm', mjoa], inplace=True)
    final_df.reset_index()
    print(len(final_df['participant_id'].to_list()))
    for metric in metrics:
        gen_chart_corr_mjoa_mscc(final_df, metric, mjoa, path_out)

    compute_mean_std(final_df)

if __name__ == '__main__':
    main()