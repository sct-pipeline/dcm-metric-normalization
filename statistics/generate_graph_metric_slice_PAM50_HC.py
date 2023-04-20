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
    'MEAN(area)': 'Cross-Sectional Area [mm^2]',
    'MEAN(diameter_RL)': 'RL Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity'
}

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


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


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
    print('Number of subjects: ' + str(len(subjects)))
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True) # do we want to compute mean with missing levels for some subjects?
    # Keep only VertLevel from C1 to Th1
    df = df[df['VertLevel'] <= 8]
    for metric in METRICS:
        fig, ax = plt.subplots()
        # Note: we are ploting slices not levels to avoid averaging across levels
        sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue='manufacturer')
        # Get slices where array changes value
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.grid(color='lightgrey', zorder=0)
        plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=16)
        ymin, ymax = ax.get_ylim()
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=14)
        ax.set_xlabel('Vertebral Level (S->I)', fontsize=14)
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
        filename = metric + '_plot.png'
        path_filename = os.path.join(args.path_out, filename)
        plt.savefig(path_filename)
        print('Figure saved: ' + path_filename)


if __name__ == '__main__':
    main()