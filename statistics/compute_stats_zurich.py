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

from utils.utils import SmartFormatter, format_pvalue, fit_reg
from utils.read_files import read_metric_file, read_participants_file, read_clinical_file, \
    read_electrophysiology_file, read_anatomical_file, read_motion_file, read_motion_file_maximum_stenosis,\
    merge_anatomical_morphological_final_for_pred
from utils.generate_figures import gen_chart_norm_vs_no_norm, gen_chart_corr_mjoa_mscc, gen_chart_weight_height, \
    plot_correlation_for_clinical_scores, plot_correlations_motion_and_morphometric_metrics, \
    plot_correlations_anatomical_and_morphometric_metrics

FNAME_LOG = 'log_stats.txt'
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


METRICS = [
    'area_ratio',
    'diameter_AP_ratio',
    'diameter_RL_ratio',
    'eccentricity_ratio',
    'solidity_ratio',
]

METRICS_NORM = [metric + '_PAM50_normalized' for metric in METRICS]

DICT_DISC_LABELS = {
                    'C1/C2': 2,
                    'C2/C3': 3,
                    'C3/C4': 4,
                    'C4/C5': 5,
                    'C5/C6': 6,
                    'C6/C7': 7
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes statistics and generates figures for the dcm-zurich dataset.",
        formatter_class=SmartFormatter
        )
    parser.add_argument(
        '-input-file',
        required=True,
        metavar='<file_path>',
        help="Path to the CSV file with computed morphometric metrics")
    parser.add_argument(
        '-participants-file',
        required=True,
        metavar='<file_path>',
        help="Path to the dcm-zurich participants.tsv file")
    parser.add_argument(
        '-clinical-file',
        required=True,
        metavar='<file_path>',
        help="Path to the Excel file with clinical scores such as mJOA, Nurick, ASIA. Example: clinical_scores.xlsx")
    parser.add_argument(
        '-anatomical-file',
        required=True,
        metavar='<file_path>',
        help="Path to the Excel file with anatomical data (aSCOR and aMSCC). Example: anatomical_data.xlsx")
    parser.add_argument(
        '-electro-file',
        required=True,
        metavar='<file_path>',
        help="Path to the Excel file with electrophysiology such as SEP, CHEPS. Example: "
             "electrophysiological_measurements.xlsx")
    parser.add_argument(
        '-motion-file',
        required=True,
        metavar='<file_path>',
        help="Path to the Excel file with motion data (amplitude and displacement). Example: motion_data.xlsx or "
             "motion_data_maximum_stenosis.xlsx")
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


def get_correlation_table(df):
    """
    Return correlation matrix of a DataFrame using Pearson's correlation coefficient, p-values of correlation coefficients and correlation matrix with level of significance *.
    Args:
        df (panda.DataFrame)
    Returns:
        corr_table (panda.DataFrame): correlation matrix of df
        corr_table_pvalue (panda.DataFrame): p-values of correlation matrix of df
        corr_table_and_p_value (panda.DataFrame): correlation matrix of df with level of significance labeled as *, ** or ***
    """
    # TODO: remove half
    corr_table = df.corr(method='spearman')
    corr_table_pvalue = df.corr(method=lambda x, y: stats.spearmanr(x, y)[1]) - np.eye(len(df.columns))
    # Overcome smallest possible 64bit floating point
    for column in corr_table_pvalue.columns:
        for index in corr_table_pvalue.index:
            if column != index and corr_table_pvalue.loc[index, column] == 0:
                corr_table_pvalue.loc[index, column] = 1e-30
    p = corr_table_pvalue.applymap(lambda x: ''.join(['*' for t in [0.001, 0.05, 0.01] if x <= t and x > 0]))
    corr_table_and_p_value = corr_table.round(2).astype(str) + p
    return corr_table, corr_table_pvalue, corr_table_and_p_value


def compute_mean_std(df, path_out):

    # ADD tests
    logger.info(f'Size df_reg_all after dropna: {df.shape[0]}')

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
    ratio = df['level'].value_counts(normalize=True)*100
    logger.info(f'Levels (%): \n{ratio}')

    # Maximum level of compression seperated for therapeutic decision
    ratio2 = df.groupby('therapeutic_decision')['level'].value_counts(normalize=True)*100
    logger.info(f'Levels (%): \n{ratio2}')


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
    # Shuffle columns of x:
    import random
    columns = list(x.columns)
    random.shuffle(columns)
    x = x.reindex(columns=columns)
    logger.info('Candidate predictors: ')
    for i in range(0, len(list(x.columns)), 2):
        logger.info(list(x.columns)[i:i+2])
    included = []  # Initialize a list for included predictors in the model
    i=0
    while True:
        i+=1
        logger.info('Iteration {}'.format(i))
        changed = False
        # Forward step
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype=np.float64)

        for new_column in excluded:
            logger.info('\n')
            logger.info(new_column)
            if method == 'linear':
                model = sm.OLS(y, x[included+[new_column]]).fit()       # Computes linear regression
            elif method == 'logistic':
                model = sm.Logit(y, x[included+[new_column]]).fit()     # Computes logistic regression
            logger.info(model.pvalues[new_column])
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        logger.info(best_pval)
        if best_pval < threshold_in:
            best_predictor = excluded[new_pval.argmin()]  # Gets the predictor with the lowest p_value
            included.append(best_predictor)  # Adds best predictor to included predictor list
            changed = True
            logger.info('Add  {:30} with p-value {:.6}'.format(best_predictor, best_pval))

        # backward step
        if method == 'linear':
            model = sm.OLS(y, x[included]).fit()    # Computes linear regression with included predictor
            pvalues = model.pvalues.iloc[1:]
        elif method == 'logistic':
            model = sm.Logit(y, x[included]).fit()  # Computes logistic regression with included predictor
            pvalues = model.pvalues
        # Use all coefs except intercept
        print('p-values', model.pvalues)
        # Gets the worst p-value of the model
        worst_pval = pvalues.max()  # null if pvalues is empty
        logger.info('worst p-value: {}'.format(worst_pval))
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
    sns.set_style("ticks", {'axes.grid': True})
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
        logreg = LogisticRegression(solver='liblinear')
        logreg.fit(x_train, y_train)
        scores.append(logreg.score(x_test, y_test))

        y_pred = logreg.predict(x_test)
        # print('Accuracy of logistic regression classifier on test set: {:.6f}'.format(logreg.score(x_test, y_test)))
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
        color="blue",
        label=r"Mean ROC (AUC = %0.2f ± %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=1,
        zorder=10
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
    plt.plot([0, 1], [0, 1], color='red', linestyle='dashed')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver operating characteristic', fontsize=14)
    plt.yticks(fontsize=11)
    plt.xticks(fontsize=11)
    plt.legend(loc="lower right")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc="center left")
    plt.savefig(os.path.join(path_out, filename), bbox_inches="tight")
    plt.close()
    logger.info('Saved ROC curve to {}'.format(os.path.join(path_out, filename)))

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
    logger.info(f'Non-normalized MRI metrics - number of rows before dropping missing values: {df_reg.shape[0]}')
    df_reg.dropna(inplace=True)
    logger.info(f'Non-normalized MRI metrics - number of rows after dropping missing values: {df_reg.shape[0]}')

    logger.info(f'Normalized MRI metrics - number of rows before dropping missing values: {df_reg_norm.shape[0]}')
    df_reg_norm.dropna(inplace=True)
    logger.info(f'Normalized MRI metrics - number of rows after dropping missing values: {df_reg_norm.shape[0]}')


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
    fit_reg(x[included], y, 'logistic', logger)

    #Fit a model on all regressors:
    #logger.info('Model on all columns:')
    #fit_reg(x, y, 'logistic', logger)


    # Model with normalization
    logger.info('\n Fitting Logistic regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['therapeutic_decision'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'logistic')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit logistic regression model on included variables
    fit_reg(x_norm[included_norm], y, 'logistic', logger)
    # Compute ROC curve
    fit_model_metrics(x, y, path_out=path_out, filename='Log_ROC_allreg')
    # Fit a model on all regressors:
    logger.info('Model on all columns:')
    #fit_reg(x_norm, y, 'logistic')
    # Compute ROC curve
    fit_model_metrics(x_norm, y, path_out=path_out, filename='Log_ROC_allreg_norm')


    # 2. Compute metrics on models (precision, recall, AUC, ROC curve)
    logger.info('Testing both models and computing ROC curve and AUC')
    logger.info('No Normalization')

    fit_model_metrics(x, y, included, path_out=path_out)
    logger.info('Normalization')
    fit_model_metrics(x_norm, y, included_norm, path_out=path_out, filename='Log_ROC_norm')

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
    df_z_score['composite_zscore'] = df_z_score[['area_ratio_zscore', 'diameter_AP_ratio_zscore', 'diameter_RL_ratio_zscore']].mean(
        axis=1)
    # Do composite z_score for norm between area, diameter_AP, diameter_RL TODO: maybe remove diameter RL?
    df_z_score['composite_zscore_norm'] = df_z_score[
        ['area_ratio_PAM50_normalized_zscore', 'diameter_AP_ratio_PAM50_normalized_zscore', 'diameter_RL_ratio_PAM50_normalized_zscore']].mean(axis=1)
    # mean_zscore = df_z_score.groupby('therapeutic_decision').agg([np.mean])
    # print(mean_zscore['diameter_AP_zscore'])
    mean_zscore = df_z_score.groupby('therapeutic_decision').agg([np.mean])
    logger.info(mean_zscore['composite_zscore'])
    logger.info(mean_zscore['composite_zscore_norm'])

    # 6. Redo Logistic regression using composite z_score instead
    logger.info('Testing both models and computing ROC curve and AUC with composite z_score')
    logger.info('No Normalization')
    x = df_z_score[['total_mjoa', 'level', 'composite_zscore']]
    # Fit logistic regression model on included variables
    fit_reg(x, y_z_score, 'logistic', logger)
    fit_model_metrics(x, y_z_score, path_out=path_out, filename='Log_ROC_zscore')
    logger.info('Normalization')
    x_norm = df_z_score[['total_mjoa', 'level', 'composite_zscore_norm']]
    fit_reg(x_norm, y_z_score, 'logistic', logger)
    fit_model_metrics(x_norm, y_z_score, path_out=path_out, filename='Log_ROC_norm_zscore')


def predict_mjoa_m6(df_reg, df_reg_norm):
    """
    The dependent variable is mjoa_6m.
    """
    # Drop mjoa and mjoa_12m
    df_reg.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_12m'])
  
    # Drop useless columns
    columns_to_drop=['motor_dysfunction_LE_bl', 'sphincter_dysfunction_bl', 'motor_dysfunction_UE_bl','dSEP_C8_both_patho_6m','CHEPS_C8_patho_6m','dSEP_C8_both_patho_12m','dSEP_C6_both_patho_12m', 'CHEPS_C8_patho_12m', 'CHEPS_T4_diff_12m_bl', 'total_dorsal_12m', 'CHEPS_T4_diff_6m_bl', 'lt_cervical_tot_6m', 'pp_cervical_tot_6m', 'pp_cervical_tot_12m','CHEPS_C6_patho_6m','CHEPS_C6_diff_12m_bl','CHEPS_T4_patho_12m','lt_cervical_tot_12m', 'total_dorsal_6m','CHEPS_C6_patho_12m','CHEPS_C8_diff_6m_bl', 'CHEPS_C8_diff_12m_bl', 'dSEP_C6_both_patho_6m', 'CHEPS_C6_diff_6m_bl', 'CHEPS_T4_patho_6m']
    df_reg.drop(inplace=True, columns=columns_to_drop)
    df_reg_norm.drop(inplace=True, columns=columns_to_drop)

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    df_reg.dropna(inplace=True)
    df_reg_norm.dropna(inplace=True)
    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['total_mjoa_6m'])  # Initialize x to data of predictors
    y = df_reg['total_mjoa_6m'].astype(int)
    x = x.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x, y, 'linear', logger)

    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear', logger)

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['total_mjoa_6m'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x_norm, y, 'linear', logger)

    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear', logger)


def predict_mjoa_m6_diff(df_reg, df_reg_norm):
    """
    The dependent variable is the difference between mjoa and mjoa_6m.
    """

    
    df_reg['total_mjoa_6m_diff'] = df_reg['total_mjoa'] - df_reg['total_mjoa_6m']
    df_reg_norm['total_mjoa_6m_diff'] = df_reg_norm['total_mjoa'] - df_reg_norm['total_mjoa_6m']

    # Drop mjoa, mjoa_6m and mjoa_12m --> keep only mjoa_6m_diff
    df_reg.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m', 'total_mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m', 'total_mjoa_12m'])

    # Drop useless columns
    df_reg.reindex()
    columns_to_drop=['motor_dysfunction_LE_bl', 'sphincter_dysfunction_bl', 'motor_dysfunction_UE_bl','dSEP_C8_both_patho_6m','CHEPS_C8_patho_6m','dSEP_C8_both_patho_12m','dSEP_C6_both_patho_12m', 'CHEPS_C8_patho_12m', 'CHEPS_T4_diff_12m_bl', 'total_dorsal_12m', 'CHEPS_T4_diff_6m_bl', 'lt_cervical_tot_6m', 'pp_cervical_tot_6m', 'pp_cervical_tot_12m','CHEPS_C6_patho_6m','CHEPS_C6_diff_12m_bl','CHEPS_T4_patho_12m','lt_cervical_tot_12m', 'total_dorsal_6m','CHEPS_C6_patho_12m','CHEPS_C8_diff_6m_bl', 'CHEPS_C8_diff_12m_bl', 'dSEP_C6_both_patho_6m', 'CHEPS_C6_diff_6m_bl', 'CHEPS_T4_patho_6m']
    df_reg.drop(inplace=True, axis=1, columns=columns_to_drop)
    df_reg_norm.drop(inplace=True, axis=1, columns=columns_to_drop)
    # Get difference between mjoa_6m and mjoa

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    df_reg.dropna(inplace=True)
    df_reg_norm.dropna(inplace=True)

    # Drop useless columns
    columns_to_drop=['motor_dysfunction_LE_bl', 'sphincter_dysfunction_bl', 'motor_dysfunction_UE_bl','dSEP_C8_both_patho_6m','CHEPS_C8_patho_6m','dSEP_C8_both_patho_12m','dSEP_C6_both_patho_12m', 'CHEPS_C8_patho_12m', 'CHEPS_T4_diff_12m_bl', 'total_dorsal_12m', 'CHEPS_T4_diff_6m_bl', 'lt_cervical_tot_6m', 'pp_cervical_tot_6m', 'pp_cervical_tot_12m','CHEPS_C6_patho_6m','CHEPS_C6_diff_12m_bl','CHEPS_T4_patho_12m','lt_cervical_tot_12m', 'total_dorsal_6m','CHEPS_C6_patho_12m','CHEPS_C8_diff_6m_bl', 'CHEPS_C8_diff_12m_bl', 'dSEP_C6_both_patho_6m', 'CHEPS_C6_diff_6m_bl', 'CHEPS_T4_patho_6m']
    df_reg.drop(inplace=True, columns=columns_to_drop)
    df_reg_norm.drop(inplace=True, columns=columns_to_drop)
    df_reg.dropna(inplace=True)
    df_reg_norm.dropna(inplace=True)

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['total_mjoa_6m_diff'])  # Initialize x to data of predictors
    y = df_reg['total_mjoa_6m_diff'].astype(int)
    x = x.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x, y, 'linear', logger)

    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear', logger)

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['total_mjoa_6m_diff'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)

    # Fit linear regression model on all variables - to get p-values for all variables
    fit_reg(x_norm, y, 'linear', logger)

    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear', logger)


def predict_mjoa_m12(df_reg, df_reg_norm):
    """
    The dependent variable is mjoa_12m.
    """

    # Drop mjoa and mjoa_6m
    df_reg.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m'])
    df_reg_norm.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m'])

    # Drop rows (subjects) with NaN values for mjoa_6m
    df_reg.dropna(axis=0, subset=['total_mjoa_12m'], inplace=True)
    df_reg_norm.dropna(axis=0, subset=['total_mjoa_12m'], inplace=True)

    # Drop useless columns
    columns_to_drop=['motor_dysfunction_LE_bl', 'sphincter_dysfunction_bl', 'motor_dysfunction_UE_bl','dSEP_C8_both_patho_6m','CHEPS_C8_patho_6m','dSEP_C8_both_patho_12m','dSEP_C6_both_patho_12m', 'CHEPS_C8_patho_12m', 'CHEPS_T4_diff_12m_bl', 'total_dorsal_12m', 'CHEPS_T4_diff_6m_bl', 'lt_cervical_tot_6m', 'pp_cervical_tot_6m', 'pp_cervical_tot_12m','CHEPS_C6_patho_6m','CHEPS_C6_diff_12m_bl','CHEPS_T4_patho_12m','lt_cervical_tot_12m', 'total_dorsal_6m','CHEPS_C6_patho_12m','CHEPS_C8_diff_6m_bl', 'CHEPS_C8_diff_12m_bl', 'dSEP_C6_both_patho_6m', 'CHEPS_C6_diff_6m_bl', 'CHEPS_T4_patho_6m']
    df_reg.drop(inplace=True, columns=columns_to_drop)
    df_reg_norm.drop(inplace=True, columns=columns_to_drop)
    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]
    df_reg.dropna(inplace=True)
    df_reg_norm.dropna(inplace=True)

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['total_mjoa_12m'])  # Initialize x to data of predictors
    y = df_reg['total_mjoa_12m'].astype(int)
    x = x.astype(float)
    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear', logger)

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['total_mjoa_12m'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear', logger)


def predict_mjoa_m12_diff(df_reg, df_reg_norm):
    """
    The dependent variable is the difference between mjoa and mjoa_12m.
    """

    # Get difference between mjoa_6m and mjoa
    df_reg['total_mjoa_12m_diff'] = df_reg['total_mjoa'] - df_reg['total_mjoa_6m']
    df_reg_norm['total_mjoa_12m_diff'] = df_reg_norm['total_mjoa'] - df_reg_norm['total_mjoa_6m']

    # Drop mjoa, mjoa_6m and mjoa_12m --> keep only mjoa_12m_diff
    df_reg.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m', 'total_mjoa_12m'])
    df_reg_norm.drop(inplace=True, columns=['total_mjoa', 'total_mjoa_6m', 'total_mjoa_12m'])

    # Drop useless columns
    columns_to_drop=['motor_dysfunction_LE_bl', 'sphincter_dysfunction_bl', 'motor_dysfunction_UE_bl','dSEP_C8_both_patho_6m','CHEPS_C8_patho_6m','dSEP_C8_both_patho_12m','dSEP_C6_both_patho_12m', 'CHEPS_C8_patho_12m', 'CHEPS_T4_diff_12m_bl', 'total_dorsal_12m', 'CHEPS_T4_diff_6m_bl', 'lt_cervical_tot_6m', 'pp_cervical_tot_6m', 'pp_cervical_tot_12m','CHEPS_C6_patho_6m','CHEPS_C6_diff_12m_bl','CHEPS_T4_patho_12m','lt_cervical_tot_12m', 'total_dorsal_6m','CHEPS_C6_patho_12m','CHEPS_C8_diff_6m_bl', 'CHEPS_C8_diff_12m_bl', 'dSEP_C6_both_patho_6m', 'CHEPS_C6_diff_6m_bl', 'CHEPS_T4_patho_6m']
    df_reg.drop(inplace=True, columns=columns_to_drop)
    df_reg_norm.drop(inplace=True, columns=columns_to_drop)

    # Keep only subject with therapeutic_decision == 0, i.e. no surgery.
    # Otherwise, the only predictor is therapeutic_decision
    df_reg = df_reg[df_reg['therapeutic_decision'] == 0]
    df_reg_norm = df_reg_norm[df_reg_norm['therapeutic_decision'] == 0]

    df_reg.dropna(inplace=True)
    df_reg_norm.dropna(inplace=True)

    # Model without normalization
    logger.info('\nFitting Linear regression on all variables (no normalization)')
    x = df_reg.drop(columns=['total_mjoa_12m_diff'])  # Initialize x to data of predictors
    y = df_reg['total_mjoa_12m_diff'].astype(int)
    x = x.astype(float)
    # P_values for forward and backward stepwise
    p_in = 0.05
    p_out = 0.05
    logger.info('Stepwise:')
    included = compute_stepwise(y, x, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included}')
    # Fit linear regression model on included variables
    fit_reg(x[included], y, 'linear', logger)

    # Model with normalization
    logger.info('\n Fitting Linear regression on all variables (normalization)')
    x_norm = df_reg_norm.drop(columns=['total_mjoa_12m_diff'])  # Initialize x to data of predictors
    x_norm = x_norm.astype(float)
    logger.info('Stepwise:')
    included_norm = compute_stepwise(y, x_norm, p_in, p_out, 'linear')
    logger.info(f'Included repressors are: {included_norm}')
    # Fit linear regression model on included variables
    fit_reg(x_norm[included_norm], y, 'linear', logger)


def compute_pca(df):
    logger.info(df.columns)
    df = df.dropna(axis=0)
    df = df.drop(columns=['therapeutic_decision'])
    from sklearn.preprocessing import StandardScaler
    # Preprocess
    std_scaler = StandardScaler()
    scaled_df = std_scaler.fit_transform(df)

    # Run PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=6)
    pca.fit_transform(scaled_df)
    logger.info(pca.components_)
    logger.info(pca.explained_variance_ratio_)


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
        # TODO: name dict_exclude_subj is confusing, it's not a dict, it's a list
        # Initialize empty list
        dict_exclude_subj = []

    logger.info('Excluded subjects: {}'.format(dict_exclude_subj))

    # Read participants.tsv file
    df_participants = read_participants_file(args.participants_file)

    # Read file with clinical scores (mJOA, ASIA, GRASSP)
    clinical_df = read_clinical_file(args.clinical_file)

    # Plot correlation matrix for clinical scores
   # plot_correlation_for_clinical_scores(clinical_df, path_out, logger)

    # Merge clinical scores (mJOA, ASIA, GRASSP) to participant.tsv
    df_participants = pd.merge(df_participants, clinical_df, on='record_id', how='outer', sort=True)

    # Read CSV file with computed metrics as Pandas DataFrame
    df_morphometrics = read_metric_file(args.input_file, dict_exclude_subj, df_participants)

    # Read electrophysiology, anatomical, and motion data
    electrophysiology_df = read_electrophysiology_file(args.electro_file, df_participants)
    anatomical_df = read_anatomical_file(args.anatomical_file, df_participants)
    if 'maximum_stenosis' in args.motion_file:
        motion_df = read_motion_file_maximum_stenosis(args.motion_file, df_participants)
        # Aggregate anatomical from the maximum level of compression and merge them with computed morphometrics
        # Note: we do not aggregate motion scores because we are reading the maximum stenosis file the command before
        df_clinical_all = merge_anatomical_morphological_final_for_pred(anatomical_df, motion_df, df_morphometrics,
                                                                        add_motion=False)
        # Merge df_clinical_all to participant.tsv
        final_df = pd.merge(df_participants, df_clinical_all, on='participant_id', how='outer', sort=True)
        final_df = pd.merge(final_df, motion_df, on='participant_id', how='outer', sort=True)
        # Add electro data
        final_df = pd.merge(final_df, electrophysiology_df, on='participant_id', how='outer', sort=True)
    else:
        motion_df = read_motion_file(args.motion_file, df_participants)
        # Aggregate anatomical and motion scores from the maximum level of compression and merge them with computed
        # morphometrics
        df_clinical_all = merge_anatomical_morphological_final_for_pred(anatomical_df, motion_df, df_morphometrics,
                                                                        add_motion=True)
        # Merge df_clinical_all to participant.tsv
        final_df = pd.merge(df_participants, df_clinical_all, on='participant_id', how='outer', sort=True)
        final_df = pd.merge(final_df, electrophysiology_df, on='participant_id', how='outer', sort=True)
    logger.info(final_df.columns)
    # Plot and save correlation matrix and pairplot for anatomical (aSCOR and aMSCC) and morphometric metrics
    #plot_correlations_anatomical_and_morphometric_metrics(final_df, path_out, logger)

    # Plot and save correlation matrix for motion data (displacement and amplitude) and morphometric metrics
  #  plot_correlations_motion_and_morphometric_metrics(final_df, path_out, logger)

    # Change SEX for 0 and 1
    final_df = final_df.replace({"sex": {'F': 0, 'M': 1}})
    # Change LEVELS for numbers
    final_df = final_df.replace({"level": DICT_DISC_LABELS})
    # Change therapeutic decision for 0 and 1
    final_df = final_df.replace({"therapeutic_decision": {'conservative': 0, 'operative': 1}})
    # Replace previous_surgery for 0 and 1
    final_df = final_df.replace({"previous_surgery": {'no': 0, 'yes': 1}})

    # Drop subjects with NaN values
    final_df.dropna(axis=0, subset=['area_ratio_PAM50_normalized'], inplace=True)
    (final_df.isna()).to_csv(os.path.join(path_out, 'missing_data') + '.csv')
    final_df.dropna(axis=0, subset=['area_ratio_PAM50_normalized', 'total_mjoa', 'therapeutic_decision', 'age', 'height'], inplace=True)  # added height since significant predictor
    final_df.reset_index()
    number_subjects = len(final_df['participant_id'].to_list())
    logger.info(f'Number of subjects (after dropping subjects with NaN values): {number_subjects}')

    # Loop across metrics
    # for metric in METRICS:
    #     # Create charts mJOA vs individual metrics (both normalized and not normalized)
    #     gen_chart_corr_mjoa_mscc(final_df, metric, 'total_mjoa', path_out, logger)
    #     # Plot scatter plot normalized vs not normalized
    #     logger.info(f'Correlation {metric} norm vs no norm')
    #     gen_chart_norm_vs_no_norm(final_df, metric, path_out, logger)

    # Create sub-dataset to compute logistic regression
    df_reg = final_df.copy()

    # Change myelopathy for yes no column
    df_reg['myelopathy'].fillna(0.0, inplace=True)
    df_reg.loc[df_reg['myelopathy'] != 0, 'myelopathy'] = 1.0
    df_reg['myelopathy'] = df_reg.myelopathy.astype(float)

    # Drop useless columns
    df_reg = df_reg.drop(columns=['pathology',
                                  'record_id',
                                  'record_id_y',
                                  'record_id_x',
                                  'compression_level',
                                  'date_previous_surgery',
                                  'surgery_date',
                                  'date_of_scan',
                                  'manufacturers_model_name',
                                  'manufacturer',
                                  'stenosis',
                                  'maximum_stenosis_y',
                                  'maximum_stenosis_x',
                                  'slice(I->S)',
                                  'eccentricity_ratio_PAM50',
                                  'diameter_RL_ratio_PAM50',
                                  'diameter_AP_ratio_PAM50',
                                  'area_ratio_PAM50',
                                  'solidity_ratio_PAM50',
                                  'eccentricity_ratio_PAM50',
                                  'dSEP_C6_both_patho_bl',  # missing data
                                  'dSEP_C8_both_patho_bl',  # missing data
                                  'CHEPS_C6_patho_bl',  # missing data
                                  'CHEPS_C8_patho_bl',
                                  'CHEPS_T4_grading_patho_bl',  # missing data
                                  'amp_max_sten_sag_or_ax1_or_ax2_bl',
                                  'disp_max_sten_sag_or_ax1_or_ax2_mm_bl',
                                  'dSEP_both_patho_bl',
                                  'CHEPS_patho_bl'  # A lot of missing value and not sign.
                                  ])

    df_reg.set_index(['participant_id'], inplace=True)
    #compute_pca(df_reg)
    df_reg_all = df_reg.copy()
    df_reg_norm = df_reg.copy()
    


    #plot_correlations_anatomical_and_morphometric_metrics(df_reg, path_out, logger)
    df_reg.drop(inplace=True, columns=METRICS_NORM)
    df_reg_norm.drop(inplace=True, columns=METRICS)

    # Create sns.regplot between sex and weight
  #  gen_chart_weight_height(df_reg, path_out, logger)

    #print('\n predicting mjoa')
    #predict_mjoa_m6(df_reg, df_reg_norm)
    #predict_mjoa_m12(df_reg, df_reg_norm)
    #predict_mjoa_m6_diff(df_reg, df_reg_norm)
    #predict_mjoa_m12_diff(df_reg, df_reg_norm)

    #compare_mjoa_between_therapeutic_decision(df_reg, path_out)



    # get mean ± std of predictors
    logger.info(f'Size df_reg_all: {df_reg_all.shape[0]}')
    df_reg_all = df_reg_all[[col for col in df_reg_all.columns if not ('6m' in col or '12m' in col)]]
    df_reg_all.dropna(inplace=True)
    logger.info(f'Size df_reg_all after dropna: {df_reg_all.shape[0]}')
    logger.info('Computing mean ± std')
    compute_mean_std(df_reg_all, path_out)
    
    
    # Get correlation matrix
    corr_matrix = df_reg_all.corr(method='spearman')
    corr_filename = os.path.join(path_out, 'corr_table')
    # Save a.csv file of the correlation matrix in the results folder
    corr_matrix.to_csv(corr_filename + '.csv')
    # Get p-value of corr matrix:
    corr, pvalues_corr, corr_and_pvalue = get_correlation_table(df_reg_all)
    pvalues_corr.to_csv(os.path.join(path_out, 'corr_table_pvalue.csv'))
    corr_and_pvalue.to_csv(os.path.join(path_out, 'corr_table_and_pvalue.csv'))

    # Get Point-biserial correlation for categorical with continuous variables:
        # Variables:  sex, previous_surgery, myelopathy, therapeutic_decision
    logger.info('\nComputing correlation matrix...')
    # sex
    corr_sex = df_reg_all.drop(columns=['sex']).corrwith(df_reg_all['sex'].astype('float'), method=stats.pointbiserialr)
    corr_sex_filename = os.path.join(path_out, 'corr_table_sex')
    corr_sex.to_csv(corr_sex_filename + '.csv')
    # previous_surgery
    corr_previous_surgery = df_reg_all.drop(columns=['previous_surgery']).corrwith(df_reg_all['previous_surgery'].astype('float'), method=stats.pointbiserialr)
    corr_previous_surgery_filename = os.path.join(path_out, 'corr_table_previous_surgery')
    corr_previous_surgery.to_csv(corr_previous_surgery_filename + '.csv')
    # myelopathy
    corr_myelopathy = df_reg_all.drop(columns=['myelopathy']).corrwith(df_reg_all['myelopathy'].astype('float'), method=stats.pointbiserialr)
    corr_myelopathy_filename = os.path.join(path_out, 'corr_table_myelopathy')
    corr_myelopathy.to_csv(corr_myelopathy_filename + '.csv')
    # therapeutic_decision
    corr_therapeutic_decision = df_reg_all.drop(columns=['therapeutic_decision']).corrwith(df_reg_all['therapeutic_decision'].astype('float'), method=stats.pointbiserialr)
    corr_therapeutic_decision_filename = os.path.join(path_out, 'corr_table_therapeutic_decision')
    corr_therapeutic_decision.to_csv(corr_therapeutic_decision_filename + '.csv')

    # Compute pearson correlation to get phi coefficient between categorical variables:
    corr_matrix_binary = df_reg_all[['sex', 'previous_surgery', 'myelopathy', 'therapeutic_decision']].corr(method='pearson')
    corr_matrix_binary_filename = os.path.join(path_out, 'corr_table_binary')
    corr_matrix_binary.to_csv(corr_matrix_binary_filename + '.csv')
    corr_table_pvalue = df_reg_all[['sex', 'previous_surgery', 'myelopathy', 'therapeutic_decision']].corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(len(df_reg_all[['sex', 'previous_surgery', 'myelopathy', 'therapeutic_decision']].columns))
    corr_table_pvalue.to_csv(corr_matrix_binary_filename + 'pvalue.csv')
    
    # Stepwise regressions
    # NOTE: uncomment always only one of the following lines (because we are doing inplace operations)
    predict_theurapeutic_decision(df_reg, df_reg_all, df_reg_norm, path_out)


    #compare_mjoa_between_therapeutic_decision(df_reg, path_out)

   

if __name__ == '__main__':
    main()

