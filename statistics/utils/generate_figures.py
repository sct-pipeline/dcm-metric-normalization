import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import format_pvalue, compute_spearmans


def gen_chart_norm_vs_no_norm(df, metric, path_out="", logger=None):
    """
    Plot data and a linear regression model fit of normalized vs non-normalized metric
    """
    sns.set_style("ticks", {'axes.grid': True})
    plt.figure()
    fig, ax = plt.subplots()
    #ax.set_box_aspect(1)
    # MSCC with mJOA
    metric_norm = metric+'_PAM50_normalized'
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


def gen_chart_corr_mjoa_mscc(df, metric, mjoa, path_out="", logger=None):
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
    metric_norm = metric+'_PAM50_normalized'
    y_vals_mscc_norm = df[metric_norm]

    r_mscc, p_mscc = compute_spearmans(x_vals, y_vals_mscc)
    r_mscc_norm, p_mscc_norm = compute_spearmans(x_vals, y_vals_mscc_norm)

    logger.info(f'{metric} ratio: Spearman r = {r_mscc} and p = {p_mscc}')
    logger.info(f'{metric_norm} ratio: Spearman r = {r_mscc_norm} and p = {p_mscc_norm}')

    sns.regplot(x=x_vals, y=y_vals_mscc, label=(metric+' ratio')) #ci=None,
    sns.regplot(x=x_vals, y=y_vals_mscc_norm, color='crimson', label=metric_norm) # ci=None,
    plt.ylabel(metric, fontsize=16)
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


def compare_mjoa_between_therapeutic_decision(df_reg, path_out, logger=None):
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


def generate_correlation_matrix(df, output_pathname, logger=None):
    """
    Plot and save correlation matrix as .csv and .png
    df: dataframe to use
    output_pathname: path to save the correlation matrix
    """

    sns.set(font_scale=1)
    corr_matrix = df.corr()
    corr_matrix.to_csv(output_pathname.replace('.png', '.csv'))
    corr_matrix = corr_matrix.round(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
    # Put level and number of subjects to the title
    ax.set_title('Number of subjects = {}'.format(len(df)))
    plt.savefig(output_pathname, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('Correlation matrix saved to: {}'.format(output_pathname))


def generate_pairplot(df, output_pathname, logger=None):
    """"
    Plot and save pairplot
    df: dataframe to use
    path_out: path to save the pairplot
    """
    logger.info('Generating pairplot...(it may take a while)...')
    sns.set(font_scale=1.5)
    sns.pairplot(df, kind="reg", diag_kws={'color': 'orange'})
    plt.savefig(output_pathname, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('Pairplot saved to: {}'.format(output_pathname))


def gen_chart_weight_height(df_reg, path_out, logger=None):
    """
    Plot weight and height relationship per sex
    """

    # Drop nan for weight and height
    df_reg.dropna(axis=0, subset=['weight', 'height'], inplace=True)
    print(f'Number of subjects after dropping nan for weight and height: {len(df_reg)}')

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