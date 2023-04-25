#
# Functions to plot CSA perslice and vertebral levels
#
# Author: Sandrine Bédard, Jan Valosek
#

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

METRICS = ['MEAN(diameter_AP)', 'MEAN(area)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)', 'MEAN(solidity)']
METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64'
}

METRIC_TO_TITLE = {
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_RL)': 'RL Diameter',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity'
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'RL Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [%]',
    'MEAN(solidity)': 'Solidity [%]'
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

# # To be same as spine-generic figures (https://github.com/spine-generic/spine-generic/blob/master/spinegeneric/cli/generate_figure.py#L114)
# When the colors are overlapping, they do not look good. So we default colors.
# PALLETE = {
#     "GE": "black",
#     "Philips": "dodgerblue",
#     "Siemens": "limegreen",
# }

def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate graph of CSA perslice ")
    parser.add_argument('-path-HC', required=True, type=str,
                        help="Path to data of normative dataset computed perslice.")
    parser.add_argument('-participant-file', required=False, type=str,
                        help="Path to participants.tsv file.")
    parser.add_argument('-path-out', required=False, type=str, default='csa_perslice',
                        help="Output directory name.")

    return parser


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    data = pd.read_csv(filename)
    return data


def get_csa(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with CSA values sorted by subject eid.
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.Series): column of CSA values

    """
    sc_data = csv2dataFrame(csa_filename)
    csa = pd.DataFrame(sc_data[['Filename', 'Slice (I->S)', 'VertLevel','DistancePMJ', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32)
    
    #TODO change CSA to float!!!
    
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def create_lineplot(df, hue, path_out):
    """
    Create lineplot of CSA per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with CSA values
        hue (str): column name of the dataframe to use for hue
        path_out (str): path to output directory
    """
    for metric in METRICS:
        fig, ax = plt.subplots()
        # Note: we are ploting slices not levels to avoid averaging across levels
        sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue)
        # Get slices where array changes value
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.grid(color='lightgrey', zorder=0)
        plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=LABELS_FONT_SIZE)
        ymin, ymax = ax.get_ylim()
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax.set_xlabel('Vertebral Level (S->I)', fontsize=LABELS_FONT_SIZE)
        # Remove xticks
        ax.set_xticks([])

        # Get vert levels for one certain subject
        vert = df[df['participant_id'] == 'sub-amu01']['VertLevel']
        # Get indexes of where array changes value
        ind_vert = vert.diff()[vert.diff() != 0].index.values
        ind_vert_mid = []
        for i in range(len(ind_vert)):
            ind_vert_mid.append(int(ind_vert[i:i+2].mean()))
        ind_vert_mid.insert(0, ind_vert[0]-20)
        ind_vert_mid = ind_vert_mid
        # Insert a vertical line for each vertebral level
        for idx, x in enumerate(ind_vert[1:]):
            plt.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert, 1):
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            # Deal with C1 label position
            elif vert[x] == 1:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)']+15, ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            else:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
        # ind_vert_mid = [nb_slice - x for x in ind_vert_mid]
        ax.invert_xaxis()

        # Save figure
        if hue:
            filename = metric + '_plot_per' + hue + '.png'
        else:
            filename = metric + '_plot.png'
        path_filename = os.path.join(path_out, filename)
        plt.savefig(path_filename)
        print('Figure saved: ' + path_filename)


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


def compute_c2_c3_stats(df):
    """
    Compute mean and std from C2 and C3 levels across sex and compare females and males.
    """

    # Compute mean and std from C2 and C3 levels
    df_c2_c3 = df[(df['VertLevel'] == 2) | (df['VertLevel'] == 3)]
    c2_c3_persex = df_c2_c3.groupby('sex')['MEAN(area)'].agg([np.mean, np.std])
    print(c2_c3_persex)

    # Compare C2-C3 CSA between females and males
    c2_c3_f = df_c2_c3[df_c2_c3['sex'] == 'F']['MEAN(area)']
    c2_c3_m = df_c2_c3[df_c2_c3['sex'] == 'M']['MEAN(area)']
    # Run normality test
    stat, pval = stats.normaltest(c2_c3_f)
    print('Normality test C2-C3 females: p-value = ' + format_pvalue(pval))
    stat, pval = stats.normaltest(c2_c3_m)
    print('Normality test C2-C3 males: p-value = ' + format_pvalue(pval))
    # Compute Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(c2_c3_f, c2_c3_m)
    print('Mann-Whitney U test between females and males: p-value = ' + format_pvalue(pval))


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_HC = args.path_HC
    path_out = args.path_out
    # If the output folder directory is not present, then create it.
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file), dtype=METRICS_DTYPE)
            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)
    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])
    # Get number of unique subjects (unique strings under Filename column)
    subjects = df['Filename'].unique()
    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if args.participant_file:
        df_participants = pd.read_csv(args.participant_file, sep='\t')
        df = df.merge(df_participants[["age", "sex", "manufacturer", "participant_id"]], on='participant_id')
    # Print number of subjects
    print(f'Number of subjects: {str(len(subjects))}\n')
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True) # do we want to compute mean with missing levels for some subjects?
    # Keep only VertLevel from C1 to Th1
    df = df[df['VertLevel'] <= 8]
    # Recode age into age bins by 10 years
    df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=['10-20', '20-30', '30-40', '40-50', '50-60'])

    # Compute mean and std from C2 and C3 levels across sex and compare females and males
    compute_c2_c3_stats(df)

    # Create plots
    create_lineplot(df, None, args.path_out)        # across all subjects
    create_lineplot(df, 'age', args.path_out)       # across age
    create_lineplot(df, 'sex', args.path_out)       # across sex
    create_lineplot(df, 'manufacturer', args.path_out)  # across manufacturer (vendors)


if __name__ == '__main__':
    main()
