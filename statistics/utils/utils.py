import argparse

import numpy as np
import statsmodels.api as sm

from scipy.stats import spearmanr


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


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


def compute_spearmans(a, b):
    """
    Compute Spearman's correlation coefficient and p-value for two arrays.
    """
    a = np.array(a)
    b = np.array(b)
    return spearmanr(a, b)


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