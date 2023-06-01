#!/usr/bin/env python
# -*- coding: utf-8
# The script computes statistics and generates figures for the dcm-zurich dataset
#   - correlation between normalized and non-normalized metrics
#   - stepwise logistic regression to predict the therapeutic decision (surgery vs. conservative)
#   - stepwise linear regression to predict mJOA score after 6m and 12m and mJOA improvement (mJOA_baseline - mJOA_6m)
#   - plot weight and height relationship per sex
#
#
# Author: Sandrine Bédard, Jan Valosek

import os 
import argparse
import pandas as pd
import logging
import math
import sys
import yaml
import numpy as np
from scipy.stats import spearmanr, pearsonr
import scipy.stats as stats
import matplotlib.pyplot as plt
from textwrap import dedent
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import auc
import statsmodels.api as sm

FNAME_LOG = 'log_stats.txt'
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


METRICS = [
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


METRICS_NORM = [metric + '_norm' for metric in METRICS]


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes statistics and generates figures for the dcm-zurich dataset.",
        formatter_class=SmartFormatter
        )
    parser.add_argument(
        '-ifolder',
        required=True,
        metavar='<file_path>',
        help="Path to results folder with CSV files containing the metrics")
    parser.add_argument(
        '-participants-file',
        required=True,
        metavar='<file_path>',
        help="dcm-zurich participants.tsv file")
    parser.add_argument(
        '-clinical-file',
        required=True,
        metavar='<file_path>',
        help="excel file with clinical data")
    parser.add_argument(
        '-electro-file',
        required=True,
        metavar='<file_path>',
        help="excel file with electrophysiology and anatomical data")
    parser.add_argument(
        '-path-out',
        required=True,
        metavar='<file_path>',
        help="Path where results will be saved. Example: ~/statistics")
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
    columns = ['participant_id', 'level'] + METRICS + METRICS_NORM
    df_morphometrics = pd.DataFrame(columns = columns) # todo add columns of metrics and
    df_morphometrics['participant_id'] = subjects_list
    for sub in subjects_list:
        # Check if subject is in exlude list
        files_subject = [file for file in list_files_results if sub in file]
        df = csv2dataFrame(os.path.join(path_results, files_subject[0]))
        max_level = df_participants.loc[df_participants['participant_id']==sub, 'maximum_stenosis'].to_list()[0]
        all_compressed_levels = df_participants.loc[df_participants['participant_id']==sub, 'stenosis'].to_list()[0].split(', ')
        idx_max = all_compressed_levels.index(max_level)
        if idx_max not in df.index.to_list():
            print(f'Maximum level of compression {max_level} is not in axial FOV, excluding {sub}')
            df_morphometrics.drop(df_morphometrics.loc[df_morphometrics['participant_id']==sub].index)
            exclude.append(sub)
        else:
            df_morphometrics.loc[df_morphometrics['participant_id']==sub, 'level'] = max_level
            for metric in METRICS:
                file = [file for file in files_subject if metric in file]#[0]
                df = csv2dataFrame(os.path.join(path_results, file[0]))
                # Fill list to create final df
                column_norm = 'normalized_'+ metric + '_ratio'
                column_no_norm = metric + '_ratio'
                df_morphometrics.loc[df_morphometrics['participant_id']==sub, metric] = df.loc[idx_max, column_no_norm]
                metric_norm = metric + '_norm'
                df_morphometrics.loc[df_morphometrics['participant_id']==sub, metric_norm] = df.loc[idx_max, column_norm]
                
            #idx_max = df.index[df['Compression Level']==max_level].tolist()
           # print(idx_max, max_level_id)
            #mscc_norm.append(df.loc[idx_max,'Normalized MSCC'])
   # df_morphometrics = pd.DataFrame(data_metrics)
    df_morphometrics.to_csv(file_metric, index=False)
    #df_morphometrics['subject'] = subject
    #df_morphometrics['level'] = level
   # df_morphometrics[METRICS_NORM + METRICS] = data_metrics
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

    # print(df_participants)
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

    #print(clinical_df)
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


def compute_spearmans(a,b):
    a = np.array(a)
    b = np.array(b)
    return spearmanr(a,b)


def gen_chart_norm_vs_no_norm(df, metric, path_out=""):
    """
    Plot data and a linear regression model fit of normalized vs non-normalized metric
    """
    sns.set_style("ticks", {'axes.grid': True})
    plt.figure()
    fig, ax = plt.subplots()
    #ax.set_box_aspect(1)
    # MSCC with mJOA
    metric_norm = metric + '_norm'
    x_vals = df[metric]
    y_vals_mscc = df[metric_norm]
    r_mscc, p_mscc = compute_spearmans(x_vals, y_vals_mscc)
    logger.info(f'{metric} ratio: Spearman r = {r_mscc} and p = {p_mscc}')
    sns.regplot(x=x_vals, y=y_vals_mscc)
    plt.ylabel((metric_norm + ' ratio'), fontsize=16)
    plt.xlabel(metric + ' ratio', fontsize=16)
    plt.xlim([min(x_vals) -1, max(x_vals)+1])
    plt.tight_layout()
    plt.text(0.03, 0.90, 'r = {}\np{}'.format(round(r_mscc, 2), format_pvalue(p_mscc, alpha=0.001, include_space=True)),
             fontsize=10, transform=ax.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='lightgrey'))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # save figure
    fname_fig = os.path.join(path_out, 'scatter_norm_no_norm_' + metric + '_mjoa.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')


def gen_chart_corr_mjoa_mscc(df, metric, mjoa, path_out=""):
    """
    Plot data and a linear regression model fit of metric vs mJOA
    """
    sns.set_style("ticks", {'axes.grid': True})
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

    logger.info(f'{metric} ratio: Spearman r = {r_mscc} and p = {p_mscc}')
    logger.info(f'{metric_norm} ratio: Spearman r = {r_mscc_norm} and p = {p_mscc_norm}')

    sns.regplot(x=x_vals, y=y_vals_mscc, label=(metric+' ratio')) #ci=None,
    sns.regplot(x=x_vals, y=y_vals_mscc_norm, color='crimson', label=(metric_norm + ' ratio')) # ci=None,
    plt.ylabel((metric + ' ratio'), fontsize=16)
    plt.xlabel('mJOA', fontsize=16)
    plt.xlim([min(x_vals)-1, max(x_vals)+1])
    plt.tight_layout()
    # Insert text with corr coef and pval
    plt.text(0.02, 0.03, 'r = {}\np{}'.format(round(r_mscc, 2), format_pvalue(p_mscc, alpha=0.001, include_space=True)),
             fontsize=10, transform=ax.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='lightgrey'))
    plt.legend(fontsize=12, bbox_to_anchor=(0.5, 1.12), loc="upper center", ncol=2, framealpha=0.95, handletextpad=0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # save figure
    fname_fig = os.path.join(path_out, 'scatter_' + metric + '_mjoa.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')

    # Create pairplot to seperate with therpeutic decision 
    plt.figure()
    g = sns.pairplot(df, x_vars=mjoa, y_vars=metric_norm, kind='reg', hue='therapeutic_decision', palette="Set1",
                     height=4, plot_kws={'scatter_kws': {'alpha': 0.6}, 'line_kws': {'lw': 4}})
    g._legend.remove()
    plt.xlim([min(x_vals)-1, max(x_vals)+1])
    plt.legend(title='Therapeutic decision', loc='lower left', labels=['conservative', 'operative'])
    plt.tight_layout()
    fname_fig = os.path.join(path_out, 'pairwise_plot_' + metric_norm + '.png')
    plt.savefig(fname_fig)
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')


def compute_mean_std(df, path_out):

    # ADD tests

    logger.info('MEAN and STD across all metrics:')
    mean_std_all = df.agg([np.mean, np.std])
    logger.info(mean_std_all)
    # Save as .csv file
    filename = os.path.join(path_out, 'mean_std_all.csv')
    mean_std_all.to_csv(filename)
    logger.info('Created: ' + filename)

    # Seperate per therapeutic_decision
    logger.info('MEAN and STD separated per therapeutic_decision:')
    mean_std_by_surgery = df.groupby('therapeutic_decision', as_index=False).agg([np.mean, np.std])
    logger.info(mean_std_by_surgery)
    # Save as .csv file
    filename = os.path.join(path_out, 'mean_std_by_therapeutic.csv')
    mean_std_by_surgery.to_csv(filename)
    logger.info('Created: ' + filename)

    # Seperate perlevel
    logger.info('MEAN and STD separated per level:')
    mean_by_level= df.groupby('level', as_index=False).agg([np.mean, np.std])
    logger.info(mean_by_level)
    # Save as .csv file
    filename = os.path.join(path_out, 'mean_std_by_level.csv')
    mean_by_level.to_csv(filename)
    logger.info('Created: ' + filename)

    # Compute ratio of categorical variables:
    # Sex
    count_F = len(df[df['sex']==0])
    count_M = len(df[df['sex']==1])
    pct_of_no_sub = np.round(100*count_F/(count_M+count_F), 2)
    logger.info(f'F:{pct_of_no_sub} %; M: {(100-pct_of_no_sub)} %')

    # Myelopathy
    count_F = len(df[df['myelopathy']==0])
    count_M = len(df[df['myelopathy']==1])
    pct_of_no_sub = np.round(100*count_F/(count_M+count_F), 2)
    logger.info(f'No myelopathy:{pct_of_no_sub} %; Myelopathy: {(100-pct_of_no_sub)} %')

    # Therepeutic decision
    count_F = len(df[df['therapeutic_decision']==0])
    count_M = len(df[df['therapeutic_decision']==1])
    pct_of_no_sub = np.round(100*count_F/(count_M+count_F), 2)
    logger.info(f'Conservative:{pct_of_no_sub} %; Operative: {(100-pct_of_no_sub)} %')

    # Maximum level of compression
    ratio =  df['level'].value_counts(normalize=True)*100
    logger.info(f'Levels (%): \n{ratio}')


    # Maximum level of compression seperated for therapeutic decision
    ratio2 =  df.groupby('therapeutic_decision')['level'].value_counts(normalize=True)*100
    logger.info(f'Levels (%): \n{ratio2}')


def fit_reg(X, y, method):
    """
    Fit either linear regression (the dependent variable is non-binary) or logistic regression (the dependent variable
    is binary) and print summary.
    Args:
        method: 'linear' or 'logistic'
    """
    if method == 'linear':
        logit_model = sm.OLS(y, X)      # Ordinary Least Squares Regression
    elif method == 'logistic':
        logit_model = sm.Logit(y, X)    # Logistic Regression
    result = logit_model.fit()
    print(result.summary2())


def compute_stepwise(y, x, threshold_in, threshold_out, method):
    """
    Perform backward and forward predictor selection based on p-values.
    Either Linear regression (the dependent variable is non-binary) or logistic regression (the dependent variable is
    binary).

    Args:
        x (panda.DataFrame): Candidate predictors
        y (panda.DataFrame): Dependent variable
        threshold_in: include a predictor if its p-value < threshold_in
        threshold_out: exclude a predictor if its p-value > threshold_out
        ** threshold_in <= threshold_out
        method: 'linear' or 'logistic'
    Returns:
        included: list of selected predictor

    """
    # Print columns names for x (print list with 2 elements per line --> better for copy-paste)
    print('Candidate predictors: ')
    for i in range(0, len(list(x.columns)), 2):
        print(list(x.columns)[i:i+2])
    included = []  # Initialize a list for included predictors in the model
    while True:
        changed = False
        # Forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)

        for new_column in excluded:
            print(new_column)
            if method == 'linear':
                model = sm.OLS(y, x[included+[new_column]]).fit()       # Computes linear regression
            elif method == 'logistic':
                model = sm.Logit(y, x[included+[new_column]]).fit()     # Computes logistic regression
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_predictor = excluded[new_pval.argmin()]  # Gets the predictor with the lowest p_value
            included.append(best_predictor)  # Adds best predictor to included predictor list
            changed = True
            logger.info('Add  {:30} with p-value {:.6}'.format(best_predictor, best_pval))

        # backward step
        if method == 'linear':
            model = sm.OLS(y, x[included]).fit()    # Computes linear regression with included predictor
        elif method == 'logistic':
            model = sm.Logit(y, x[included]).fit()  # Computes logistic regression with included predictor
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        # Gets the worst p-value of the model
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_predictor = included[pvalues.argmax()]  # gets the predictor with worst p-value
            included.remove(worst_predictor)  # Removes the worst predictor of included predictor list
            logger.info('Drop {:30} with p-value {:.6}'.format(worst_predictor, worst_pval))
            if worst_pval == best_pval and worst_predictor == best_predictor:  # If inclusion of a paremeter doesn't change p_value, end stepwise to avoid infinite loop
                break
        if not changed:
            break

    return included


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


def compute_test_myelopathy(df):
    logger.info('\nTest Myelopathy and Ratio')
    for metric in METRICS + METRICS_NORM + ['total_mjoa']: # TODO encode MJOA
        logger.info(f'\n {metric}')
        df_myelo = df[df['myelopathy'] == 1][metric]
        df_no_myelo = df[df['myelopathy'] == 0][metric]
        stat, pval = stats.normaltest(df_myelo)
        logger.info(f'Normality test {metric} ratio and  myelopathy: p-value {format_pvalue(pval, include_space=True)}')
        stat, pval2 = stats.normaltest(df_no_myelo)
        logger.info(f'Normality test {metric} ratio and no myelopathy: p-value {format_pvalue(pval2, include_space=True)}')

        if pval < 0.001 or pval2 < 0.001:
            stat, pval = stats.mannwhitneyu(df_myelo, df_no_myelo)
            logger.info(f'Mann-Whitney U test between {metric} ratio and myelopathy/no myelopathy: p-value {format_pvalue(pval, include_space=True)}')
        else:
            stat, pval = stats.ttest_ind(df_myelo, df_no_myelo)
            logger.info(f'T test for independent samples between {metric} ratio and myelopathy/no myelopathy: p-value {format_pvalue(pval, include_space=True)}')


def fit_model_metrics(X, y, regressors=None, path_out=None, filename='Log_ROC'):
    if regressors:
        X = X[regressors]

    fig, ax = plt.subplots(figsize=(6, 6))
    #kf = StratifiedKFold(n_splits=10, shuffle=True)
    kf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100)
    scores = []
    fpr_all = []
    tpr_all = []
    auc_all = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold, (train, test) in enumerate(kf.split(X, y)):
        x_train = X.iloc[train]
        x_test = X.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        scores.append(logreg.score(x_test, y_test))

        y_pred = logreg.predict(x_test)
      #  print('Accuracy of logistic regression classifier on test set: {:.6f}'.format(logreg.score(x_test, y_test)))
       # print(sklearn.metrics.classification_report(y_test, y_pred))

        # ROC and AUC
        auc_val = sklearn.metrics.roc_auc_score(y_test, y_pred)
        auc_all.append(auc_val)
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
        fpr_all.append(fpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_all.append(interp_tpr)
       # plt.plot(fpr, tpr, label=f'Logistic Regression (area = %0.2f) fold {fold}' % auc_val)
       # plt.plot(fpr, tpr)
    
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    
    mean_tpr = np.mean(np.array(tpr_all), axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_all)
    # Compute 95% confidence interval of AUC:
    auc_all.sort()
    confidence_lower = auc_all[int(0.05 * len(auc_all))]
    confidence_upper = auc_all[int(0.95 * len(auc_all))]
    logger.info("Confidence interval for the AUC: [{:0.3f} - {:0.3}]".format(
                confidence_lower, confidence_upper))
    # Compute 95% confidence interval of accuracy:
    logger.info('Mean accuracy: {:0.4f} ± {:0.4f}'.format(np.mean(scores), np.std(scores)))
    scores.sort()
    confidence_lower_acc = scores[int(0.05 * len(scores))]
    confidence_upper_acc = scores[int(0.95 * len(scores))]
    logger.info("Confidence interval for accuracy: [{:0.3f} - {:0.3}]".format(
                confidence_lower_acc, confidence_upper_acc))
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f ± %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8
    )
    
    std_tpr = np.std(tpr_all, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # Compute 95% confidence interval for ROC curve
    tpr_all = np.array(tpr_all).T  # Transpose to sort bootstrap values
    tpr_all.sort(axis=1) #Sort to get 95% interval
    confidence_lower_tpr = tpr_all.T[int(0.05 * tpr_all.shape[1])]
    confidence_upper_tpr = tpr_all.T[int(0.95 * tpr_all.shape[1])]
    ax.fill_between(
        mean_fpr,
        confidence_lower_tpr,
        confidence_upper_tpr,
        color="blue",
        alpha=0.1,
        label=r"95% CI")
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.3,
        label=r"± std.")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc="center left")
    plt.savefig(os.path.join(path_out, filename), bbox_inches="tight")
    plt.close()
    print('Saved ROC curve to {}'.format(os.path.join(path_out, filename)))

    logger.info(f'Mean accuracy: {np.mean(scores)} ± {np.std(scores)}')


def get_z_score(df):
    for col in METRICS + METRICS_NORM:
        col_zscore = col + '_zscore'
        df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
    return df    


def predict_theurapeutic_decision(df_reg, df_reg_all, df_reg_norm, path_out):
    """
    The dependent variable is therapeutic_decision.
    df_reg: dataframe with non-normalized MRI metrics
    df_reg_all: dataframe with all the variables (both non-normalized and normalized MRI metrics)
    df_reg_norm: dataframe with normalized MRI metrics
    """
    # Keep only baseline clinical scores, i.e., drop columns containing 6m and 12m
    df_reg = df_reg[[col for col in df_reg.columns if not ('6m' in col or '12m' in col)]]
    df_reg_all = df_reg_all[[col for col in df_reg_all.columns if not ('6m' in col or '12m' in col)]]
    df_reg_norm = df_reg_norm[[col for col in df_reg_norm.columns if not ('6m' in col or '12m' in col)]]

    # Drop rows with missing values
    print(f'Non-normalized MRI metrics - number of rows before dropping missing values: {df_reg.shape[0]}')
    df_reg.dropna(inplace=True)
    print(f'Non-normalized MRI metrics - number of rows after dropping missing values: {df_reg.shape[0]}')

    print(f'Normalized MRI metrics - number of rows before dropping missing values: {df_reg_norm.shape[0]}')
    df_reg_norm.dropna(inplace=True)
    print(f'Normalized MRI metrics - number of rows after dropping missing values: {df_reg_norm.shape[0]}')

    # Model without normalization
    logger.info('\nFitting Logistic regression on all variables (no normalization)')
    x = df_reg.drop(columns=['therapeutic_decision'])  # Initialize x to data of predictors
    y = df_reg['therapeutic_decision'].astype(int)
    x = x.astype(float)
    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'logistic')
    logger.info(f'Included repressors are: {included}')
    # Fit logistic regression model on included variables
    fit_reg(x[included], y, 'logistic')

    # Model with normalization
    logger.info('\n Fitting Logistic regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['therapeutic_decision'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'logistic')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit logistic regression model on included variables
    fit_reg(x_norm[included_norm], y, 'logistic')

    # 2. Compute metrics on models (precision, recall, AUC, ROC curve)
    logger.info('Testing both models and computing ROC curve and AUC')
    logger.info('No Normalization')
    fit_model_metrics(x, y, included, path_out)
    logger.info('Normalization')
    fit_model_metrics(x_norm, y, included_norm, path_out, 'Log_ROC_norm')

    # 3. Statistical test myelopathy with Ratio --> if worse compression is associated with Myelopathy
    compute_test_myelopathy(df_reg_all)

    # 4. Compute Variance of inflation to check multicolinearity
    vif_data = pd.DataFrame()
    data = x[included]
    vif_data['Feature'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    logger.info('\nVariance of inflation no norm:')
    logger.info(vif_data)
    # For normalized model
    vif_data_norm = pd.DataFrame()
    data_norm = x_norm[included_norm]
    vif_data_norm['Feature'] = data_norm.columns
    vif_data_norm['VIF'] = [variance_inflation_factor(data_norm.values, i) for i in range(len(data_norm.columns))]
    logger.info('Variance of inflation norm:')
    logger.info(vif_data_norm)

    # 5. Compute z-score
    y_z_score = df_reg_all['therapeutic_decision'].astype(int)
    df_z_score = get_z_score(df_reg_all)
    # Do composite z_score for no norm between area, diameter_AP, diameter_RL
    df_z_score['composite_zscore'] = df_z_score[['area_zscore', 'diameter_AP_zscore', 'diameter_RL_zscore']].mean(
        axis=1)
    # Do composite z_score for norm between area, diameter_AP, diameter_RL TODO: maybe remove diameter RL?
    df_z_score['composite_zscore_norm'] = df_z_score[
        ['area_norm_zscore', 'diameter_AP_norm_zscore', 'diameter_RL_norm_zscore']].mean(axis=1)
    # mean_zscore = df_z_score.groupby('therapeutic_decision').agg([np.mean])
    # print(mean_zscore['diameter_AP_zscore'])
    mean_zscore = df_z_score.groupby('therapeutic_decision').agg([np.mean])
    print(mean_zscore['composite_zscore'])
    print(mean_zscore['composite_zscore_norm'])

    # 6. Redo Logistic regression using composite z_score instead
    logger.info('Testing both models and computing ROC curve and AUC with composite z_score')
    logger.info('No Normalization')
    x = df_z_score[['total_mjoa', 'level', 'composite_zscore']]
    # Fit logistic regression model on included variables
    print(x, y)
    fit_reg(x, y_z_score, 'logistic')
    fit_model_metrics(x, y_z_score, path_out=path_out, filename='Log_ROC_zscore')
    logger.info('Normalization')
    x_norm = df_z_score[['total_mjoa', 'level', 'composite_zscore_norm']]
    fit_reg(x_norm, y_z_score, 'logistic')
    fit_model_metrics(x_norm, y_z_score, path_out=path_out, filename='Log_ROC_norm_zscore')


def predict_mjoa_m6(df_reg, df_reg_norm):
    """
    The dependent variable is mjoa_6m.
    """

    # Drop mjoa and mjoa_12m
    df_reg.drop(inplace=True, columns=['mjoa', 'mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['mjoa', 'mjoa_12m'])

    # Drop rows (subjects) with NaN values for mjoa_6m
    df_reg.dropna(axis=0, subset=['mjoa_6m'], inplace=True)
    df_reg_norm.dropna(axis=0, subset=['mjoa_6m'], inplace=True)

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['mjoa_6m'])  # Initialize x to data of predictors
    y = df_reg['mjoa_6m'].astype(int)
    x = x.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x, y, 'linear')

    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear')

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['mjoa_6m'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x_norm, y, 'linear')

    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear')


def predict_mjoa_m6_diff(df_reg, df_reg_norm):
    """
    The dependent variable is the difference between mjoa and mjoa_6m.
    """

    # Get difference between mjoa_6m and mjoa
    df_reg['mjoa_6m_diff'] = df_reg['mjoa'] - df_reg['mjoa_6m']
    df_reg_norm['mjoa_6m_diff'] = df_reg_norm['mjoa'] - df_reg_norm['mjoa_6m']

    # Drop mjoa, mjoa_6m and mjoa_12m --> keep only mjoa_6m_diff
    df_reg.drop(inplace=True, columns=['mjoa', 'mjoa_6m', 'mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['mjoa', 'mjoa_6m', 'mjoa_12m'])

    # Drop rows (subjects) with NaN values for mjoa_6m
    df_reg.dropna(axis=0, subset=['mjoa_6m_diff'], inplace=True)
    df_reg_norm.dropna(axis=0, subset=['mjoa_6m_diff'], inplace=True)

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['mjoa_6m_diff'])  # Initialize x to data of predictors
    y = df_reg['mjoa_6m_diff'].astype(int)
    x = x.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x, y, 'linear')

    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear')

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['mjoa_6m_diff'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x_norm, y, 'linear')

    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear')


def predict_mjoa_m12(df_reg, df_reg_norm):
    """
    The dependent variable is mjoa_12m.
    """

    # Drop mjoa and mjoa_6m
    df_reg.drop(inplace=True, columns=['mjoa', 'mjoa_6m'])
    df_reg_norm.drop(inplace=True, columns=['mjoa', 'mjoa_6m'])

    # Drop rows (subjects) with NaN values for mjoa_6m
    df_reg.dropna(axis=0, subset=['mjoa_12m'], inplace=True)
    df_reg_norm.dropna(axis=0, subset=['mjoa_12m'], inplace=True)

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['mjoa_12m'])  # Initialize x to data of predictors
    y = df_reg['mjoa_12m'].astype(int)
    x = x.astype(float)
    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear')

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['mjoa_12m'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear')


def predict_mjoa_m12_diff(df_reg, df_reg_norm):
    """
    The dependent variable is the difference between mjoa and mjoa_12m.
    """

    # Get difference between mjoa_6m and mjoa
    df_reg['mjoa_12m_diff'] = df_reg['mjoa'] - df_reg['mjoa_6m']
    df_reg_norm['mjoa_12m_diff'] = df_reg_norm['mjoa'] - df_reg_norm['mjoa_6m']

    # Drop mjoa, mjoa_6m and mjoa_12m --> keep only mjoa_12m_diff
    df_reg.drop(inplace=True, columns=['mjoa', 'mjoa_6m', 'mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['mjoa', 'mjoa_6m', 'mjoa_12m'])

    # Drop rows (subjects) with NaN values for mjoa_6m
    df_reg.dropna(axis=0, subset=['mjoa_12m_diff'], inplace=True)
    df_reg_norm.dropna(axis=0, subset=['mjoa_12m_diff'], inplace=True)

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['mjoa_12m_diff'])  # Initialize x to data of predictors
    y = df_reg['mjoa_12m_diff'].astype(int)
    x = x.astype(float)
    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear')

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['mjoa_12m_diff'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear')


def compare_mjoa_between_therapeutic_decision(df_reg, path_out):
    """
    Compare mJOA change between subjects with therapeutic_decision == 0 and subjects with therapeutic_decision == 1.
    """
    # Get difference between mjoa_6m and mjoa
    df_reg['mjoa_6m_diff'] = df_reg['mjoa'] - df_reg['mjoa_6m']
    df_6m = df_reg.dropna(axis=0, subset=['mjoa_6m_diff'])
    print(df_6m[['therapeutic_decision', 'mjoa_6m_diff']].groupby(['therapeutic_decision']).agg(['mean', 'std']))

    # Get difference between mjoa_12m and mjoa
    df_reg['mjoa_12m_diff'] = df_reg['mjoa'] - df_reg['mjoa_12m']
    df_12m = df_reg.dropna(axis=0, subset=['mjoa_12m_diff'])

    print(df_12m[['therapeutic_decision', 'mjoa_12m_diff']].groupby(['therapeutic_decision']).agg(['mean', 'std']))

    # Create DataFrame with three columns: therapeutic_decision, mjoa_diff, and mjoa_6m_12m (6m or 12m) for easy
    # plotting using sns hue option
    df_6m_temp = pd.DataFrame(columns=['therapeutic_decision', 'mjoa_diff', 'mjoa_6m_12m'])
    df_6m_temp['therapeutic_decision'] = df_6m['therapeutic_decision']
    df_6m_temp['mjoa_diff'] = df_6m['mjoa_6m_diff']
    df_6m_temp['mjoa_6m_12m'] = '6 months'
    df_12_temp = pd.DataFrame(columns=['therapeutic_decision', 'mjoa_diff', 'mjoa_6m_12m'])
    df_12_temp['therapeutic_decision'] = df_12m['therapeutic_decision']
    df_12_temp['mjoa_diff'] = df_12m['mjoa_12m_diff']
    df_12_temp['mjoa_6m_12m'] = '12 months'
    df_final_temp = pd.concat([df_6m_temp, df_12_temp])

    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot scatter plot and boxplot
    sns.stripplot(x='mjoa_6m_12m', y='mjoa_diff', hue='therapeutic_decision', data=df_final_temp, ax=ax, dodge=True,
                  edgecolor='black', linewidth=1, legend=False)
    sns.boxplot(x='mjoa_6m_12m', y='mjoa_diff', hue='therapeutic_decision', data=df_final_temp, ax=ax, dodge=True,
                showfliers = False)
    # Change x axis label
    ax.set_xlabel('')
    # Change y axis label
    ax.set_ylabel('mJOA difference', fontsize=14)
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=14)
    # save figure
    fname_fig = os.path.join(path_out, 'boxplot_therapeutic_decision_mjoa.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')


def merge_anatomical_morphological_final_for_pred(anatomical_df, motion_df, df_morphometric):
    """
    Merge anatomical and motion scores with computed morphometrics to get the scores at the maximum level of compression.
    """
    final_df = df_morphometric.copy()
    # set index of participant to have the same index
    final_df = final_df.set_index(['participant_id'])
    # Loop through levels to get the corresponding motion or anatomical metric or the maximum compressed level
    levels = np.unique(final_df['level'].to_list())
    levels = levels[:-1]  # nan value
    for level in levels:
        if not level == 'C1/C2':
            level_conversion = 'C' + str(int(DICT_DISC_LABELS[level])-1)  # -1 to convert to zurich disc
            # Add scores at maximum level of compression
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'aSCOR'] = anatomical_df.loc[final_df.index[final_df['level']==level].tolist(), 'aSCOR_' + level_conversion]
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'aMSCC'] = anatomical_df.loc[final_df.index[final_df['level']==level].tolist(), 'aMSCC_' + level_conversion + 'toC2']
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'amp_ax_or_sag'] = motion_df.loc[final_df.index[final_df['level']==level].tolist(), level_conversion+ '_amp_ax_or_sag']
            final_df.loc[final_df.index[final_df['level']==level].tolist(), 'disp_ax_or_sag'] = motion_df.loc[final_df.index[final_df['level']==level].tolist(), level_conversion+ '_disp_ax_or_sag']

    return final_df


def compute_correlations_anatomical_and_morphometric_metrics(final_df, path_out):
    """
    Plot and save correlation matrix and pairplot for anatomical (aSCOR and aMSCC) and morphometric metrics
    """

    # Keep only anatomical and morphometric metrics
    metrics_dict = {'all_metrics': METRICS + METRICS_NORM + ['aSCOR', 'aMSCC'],
                    'area': ['area', 'area_norm', 'aSCOR', 'aMSCC']}

    # Either all metrics or only area
    for key, value in metrics_dict.items():
        final_df = final_df[value]

        # Make 'aSCOR' and 'aMSCC' first and second columns
        cols = final_df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        final_df = final_df[cols]

        # Drop rows with nan values
        final_df = final_df.dropna(axis=0)

        # All levels together
        sns.set(font_scale=1)
        corr_matrix = final_df.corr()
        corr_matrix.to_csv(os.path.join(path_out, 'corr_matrix_anatomical_and_morphometrics_' + key + '.csv'))
        corr_matrix = corr_matrix.round(2)
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
        # Put level and number of subjects to the title
        ax.set_title('Number of subjects = {}'.format(len(final_df)))
        plt.savefig(os.path.join(path_out, 'corr_matrix_anatomical_and_morphometrics_' + key + '.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()
        print('Correlation matrix saved to: {}'.format(
            os.path.join(path_out, 'corr_matrix_anatomical_and_morphometrics_' + key + '.png')))

        # Plot pairplot
        sns.set(font_scale=1.5)
        sns.pairplot(final_df, kind="reg", diag_kws={'color': 'orange'})
        plt.savefig(os.path.join(path_out, 'pairplot_anatomical_and_morphometrics_' + key + '.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()
        print('Pairplot saved to: {}'.format(os.path.join(path_out,
                                                          'pairplot_anatomical_and_morphometrics_' + key + '.png')))


def compute_correlations_motion_and_morphometric_metrics(motion_df, df_morphometrics, path_out):
    """
    Plot and save correlation matrix for motion data (displacement and amplitude) and morphometric metrics
    """

    # Drop subjects with NaN values for participant_id
    motion_df.dropna(axis=0, subset=['participant_id'], inplace=True)

    # Change LEVELS from numbers
    df_morphometrics = df_morphometrics.replace({"level": DICT_DISC_LABELS})

    # Merge anatomical data (aSCOR and aMSCC) with morphometrics based on participant_id
    final_df = pd.merge(motion_df, df_morphometrics, on='participant_id', how='outer', sort=True)

    # Get number of nan values for each column
    print('Number of nan values for each column:')
    print(motion_df.drop(columns=['record_id', 'participant_id']).isnull().sum(axis=0))

    # Identify columns with more than 25% nan values
    cols_to_drop = final_df.columns[final_df.isnull().sum(axis=0) > 0.25 * len(final_df)]
    # Drop these columns
    print('Dropping columns with more than 25% nan values:\n {}'.format(cols_to_drop))
    final_df = final_df.drop(columns=cols_to_drop)

    # Drop rows with nan values
    final_df = final_df.dropna(axis=0)

    # Drop columns that are not needed for correlation matrix
    corr_df = final_df.drop(columns=['record_id', 'participant_id', 'level'])

    corr_matrix = corr_df.corr()
    corr_matrix.to_csv(os.path.join(path_out, 'corr_motion_and_morphometrics_matrix.csv'))
    corr_matrix = corr_matrix.round(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
    # Put level and number of subjects to the title
    ax.set_title('Number of subjects = {}'.format(len(corr_df)))
    plt.savefig(os.path.join(path_out, 'corr_motion_and_morphometrics_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Correlation matrix saved to: {}'.format(os.path.join(path_out, 'corr_motion_and_morphometrics_matrix.png')))


def plot_correlation_for_clinical_scores(clinical_df, path_out):
    """
    Plot and save correlation matrix for mJOA, mJOA subscores, and ASIA/GRASSP
    """

    # Identify columns with more than 25% nan values
    cols_to_drop = clinical_df.columns[clinical_df.isnull().sum(axis=0) > 0.25 * len(clinical_df)]
    # Drop these columns
    print('Dropping columns with more than 25% nan values:\n {}'.format(cols_to_drop))
    final_df = clinical_df.drop(columns=cols_to_drop)

    # Drop rows with nan values
    final_df = final_df.dropna(axis=0)

    corr_matrix = final_df.drop(columns=['record_id']).corr()
    corr_matrix.to_csv(os.path.join(path_out, 'corr_matrix.csv'))
    corr_matrix = corr_matrix.round(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
    # Put level and number of subjects to the title
    ax.set_title('Number of subjects = {}'.format(len(final_df)))
    plt.savefig(os.path.join(path_out, 'corr_matrix_clinical_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Correlation matrix saved to: {}'.format(os.path.join(path_out, 'corr_matrix_clinical_scores.png')))


def gen_chart_weight_height(df_reg, path_out):
    """
    Plot weight and height relationship per sex
    """

    sns.regplot(x='weight', y='height', data=df_reg[df_reg['sex'] == 1], label='Male', color='blue')
    sns.regplot(x='weight', y='height', data=df_reg[df_reg['sex'] == 0], label='Female', color='red')
    plt.legend()
    # x axis label
    plt.xlabel('Weight (kg)')
    # y axis label
    plt.ylabel('Height (m)')
    # save figure
    fname_fig = os.path.join(path_out, 'regplot_weight_height_relationship_persex.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f'Created: {fname_fig}.\n')


# TODO:
# 0. Exclude subjects
# 1. Calcul statistique 2 groupes (mean ± std) --> DONE
# 1.1. Calcul de proportion  --> DONE
# 1.2. Matrice de correlation   --> DONE
# 2. Binary logistic regression (stepwise)
# 3. Stastitical test myelopathy with Ratio --> if worse compression is associated with Myelopathy --> DONE


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

    # Read participants.tsv file
    df_participants = read_participants_file(args.participants_file)

    # Read file with clinical scores (mJOA, ASIA, GRASSP)
    clinical_df = read_clinical_file(args.clinical_file)

    # Plot correlation matrix for clinical scores
    plot_correlation_for_clinical_scores(clinical_df, path_out)

    # Merge clinical scores (mJOA, ASIA, GRASSP) to participant.tsv
    df_participants = pd.merge(df_participants, clinical_df, on='record_id', how='outer', sort=True)

    # Read electrophysiology, anatomical, and motion data
    anatomical_df, motion_df, electrophysiology_df = read_electrophysiology_anatomical_and_motion_file(
        args.electro_file, df_participants)

    file_metrics = os.path.join(path_out, 'metric_ratio_combined.csv')
    if not os.path.exists(file_metrics):
        df_morphometrics = read_MSCC(args.ifolder, dict_exclude_subj, df_participants, file_metrics)
    else:
        df_morphometrics = pd.read_csv(file_metrics)
        df_morphometrics = df_morphometrics.rename(columns={'subject': 'participant_id'})  # TODO remove when rerun
        # TODO remove subjects with no values (in read_MSCC)

    # Merge all dataframes for prediction
    df_clinical_all = merge_anatomical_morphological_final_for_pred(anatomical_df, motion_df, df_morphometrics)
    # Merge morphometrics to participant.tsv
    final_df = pd.merge(df_participants, df_clinical_all, on='participant_id', how='outer', sort=True)

    # Plot and save correlation matrix and pairplot for anatomical (aSCOR and aMSCC) and morphometric metrics
    compute_correlations_anatomical_and_morphometric_metrics(final_df, path_out)

    # Plot and save correlation matrix for motion data (displacement and amplitude) and morphometric metrics
    compute_correlations_motion_and_morphometric_metrics(motion_df, df_morphometrics, path_out)

    # Merge morphometrics to participant.tsv
    #final_df = pd.merge(df_participants, df_morphometrics, on='participant_id', how='outer', sort=True)
    #print(final_df.columns)

    # Change SEX for 0 and 1
    final_df = final_df.replace({"sex": {'F': 0, 'M': 1}})
    # Change LEVELS fro numbers
    final_df = final_df.replace({"level": DICT_DISC_LABELS})
    # Change therapeutic decision for 0 and 1
    final_df = final_df.replace({"therapeutic_decision": {'conservative': 0, 'operative': 1}})
    # Replace previous_surgery for 0 and 1
    final_df = final_df.replace({"previous_surgery": {'no': 0, 'yes': 1}})

    # Drop subjects with NaN values
    final_df.dropna(axis=0, subset=['area_norm', 'total_mjoa', 'therapeutic_decision', 'age', 'height'], inplace=True)  # added height since significant predictor
    final_df.reset_index()
    number_subjects = len(final_df['participant_id'].to_list())
    logger.info(f'Number of subjects (after dropping subjects with NaN values): {number_subjects}')
    
    # Loop across metrics
    for metric in METRICS:
        # Create charts mJOA vs individual metrics (both normalized and not normalized)
        gen_chart_corr_mjoa_mscc(final_df, metric, 'total_mjoa', path_out)
        # Plot scatter plot normalized vs not normalized
        logger.info(f'Correlation {metric} norm vs no norm')
        gen_chart_norm_vs_no_norm(final_df, metric, path_out)
    
    # Create sub-dataset to compute logistic regression
    df_reg = final_df.copy()

    # Change myelopathy for yes no column
    df_reg['myelopathy'].fillna(0, inplace=True)
    df_reg.loc[df_reg['myelopathy'] != 0, 'myelopathy'] = 1

    # Drop useless columns
    df_reg = df_reg.drop(columns=['record_id', 
                                  'pathology', 
                                  'date_previous_surgery', 
                                  'surgery_date', 
                                  'date_of_scan', 
                                  'manufacturers_model_name', 
                                  'manufacturer', 
                                  'stenosis',
                                  'maximum_stenosis'
                                  #'weight',  # missing data - TODO - try this
                                  #'height'   # missing data - TODO - try this
                                  ])
    df_reg.set_index(['participant_id'], inplace=True)
    df_reg_all = df_reg.copy()
    print(df_reg.columns.values)
    df_reg_norm = df_reg.copy()
    df_reg.drop(inplace=True, columns=METRICS_NORM)
    df_reg_norm.drop(inplace=True, columns=METRICS)

    # Create sns.regplot between sex and weight
    # Drop nan for weight and height
    df_reg.dropna(axis=0, subset=['weight', 'height'], inplace=True)
    print(f'Number of subjects after dropping nan for weight and height: {len(df_reg)}')
    gen_chart_weight_height(df_reg, path_out)

    # get mean ± std of predictors
    logger.info('Computing mean ± std')
    compute_mean_std(df_reg_all, path_out)

    # Get correlation matrix
    corr_matrix = df_reg_all.corr(method='spearman')
    corr_filename = os.path.join(path_out, 'corr_table')
    # Save a.csv file of the correlation matrix in the results folder
    corr_matrix.to_csv(corr_filename + '.csv')

    # Stepwise regressions
    # NOTE: uncomment always only one of the following lines (because we are doing inplace operations)
    predict_theurapeutic_decision(df_reg, df_reg_all, df_reg_norm, path_out)
    #predict_mjoa_m6(df_reg, df_reg_norm)
    #predict_mjoa_m12(df_reg, df_reg_norm)
   # predict_mjoa_m6_diff(df_reg, df_reg_norm)
    #predict_mjoa_m12_diff(df_reg, df_reg_norm)

    #compare_mjoa_between_therapeutic_decision(df_reg, path_out)

    # TODO - predict development of myelopathy - we do not have such data in 6m and 12m --> ask clinicians


if __name__ == '__main__':
    main()

# TODO
#Plot area_ratio and area_ratio_norm  --> DONE
#Create composite score from all metrics with variance
#Compute effect size normalized vs non-normalized
