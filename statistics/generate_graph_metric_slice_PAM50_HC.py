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
    if not os.path.exists(path_out):
    # if the output folder directory is not present, then create it.
        os.makedirs(path_out)
    # Create empty dict to put dataframe of each healthy control
    d = {}
    # Iterator to count number of healthy subjects
    i = 0
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            d[file] = pd.read_csv(os.path.join(path_HC, file), dtype=metrics_dtype)
            i = i+1
    first_key = next(iter(d))
    # Create an empty dataframe with same columns
    df = pd.DataFrame(columns=d[first_key].columns)
    df['VertLevel'] = d[first_key]['VertLevel']
    df['Slice (I->S)'] = d[first_key]['Slice (I->S)']
    # Loop through all HC
    for key, values in d.items():
        for column in d[key].columns:
            if 'MEAN' in column:
                if df[column].isnull().values.all():
                    df[column] = d[key][column]
                else:
                    # Sum all columns that have MEAN key
                    df[column] = df[column] + d[key][column].tolist()
    # Divide by number of HC
    for column in df.columns:
        if 'MEAN' in column:
            df[column] = df[column]/i
    # Loop through metrics to create graph
    print(df)
    for metric in metrics:
        metric_std = metric + '_std'
        #print((np.array([d[k][metric] for k in d])).std(axis=0).shape)
        df[metric_std] = np.array([d[k][metric] for k in d]).std(axis=0)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True) # do we want to compute mean with missing levels for some subjects?
    print(df)
    for metric in metrics:
    #df = get_csa(args.filename)
    #df[df.rebounds != 'None']
    #df = df.replace('None', np.NaN)
    #df = df.iloc[::-1].reset_index()
        plt.figure()
        #fig, ax = plt.subplots(figsize=(5,6))
        fig, ax = plt.subplots()
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        # Get slices where array changes value
        vert = df['VertLevel'].to_numpy()
        ind_vert = np.where(vert[:-1] != vert[1:])[0]
        nb_slice = len(df['Slice (I->S)'])
        #ind_vert = nb_slice - ind_vert
        ind_vert_mid = []
        for i in range(len(ind_vert)):
            ind_vert_mid.append(int(ind_vert[i:i+2].mean()))
        ind_vert_mid.insert(0, ind_vert[0]-20)
        ind_vert_mid = ind_vert_mid[:-1]
        print(ind_vert, ind_vert_mid)
        #df = df.iloc[::-1].reset_index()

        for idx, x in enumerate(ind_vert):
            plt.axvline(df.loc[x,'Slice (I->S)'], color='darkblue', linestyle='--', alpha=0.7)
            #ax.text(0.05 , (x-10)/nb_slice, 'C'+str(vert[x]), transform=ax.transAxes, horizontalalignment='right', verticalalignment='center',color='darkblue')
        plt.plot((df['Slice (I->S)'].to_numpy())[::-1], df[metric].to_numpy()[::-1], 'r', aa=True)
        plt.fill_between(df['Slice (I->S)'].to_numpy()[::-1], 
                         df[metric].to_numpy()[::-1]-df[metric + '_std'][::-1],
                         df[metric].to_numpy()[::-1]+df[metric + '_std'][::-1],
                         alpha = 0.4,
                         color='red',
                         edgecolor=None,
                         zorder=3)
        plt.grid(color='lightgrey', zorder=0)
        plt.title('Spinal Cord ' + metric, fontsize=16)
        ymin, ymax = ax.get_ylim()
        for idx, x in enumerate(ind_vert):
            if vert[x]>7:
                level = 'T'+ str(vert[x]-7)
            else:
                level = 'C'+str(vert[x])
            ax.text(df.loc[ind_vert_mid[idx],'Slice (I->S)'], ymin, level, horizontalalignment='center', verticalalignment='bottom',color='darkblue')
        #ind_vert_mid = [nb_slice - x for x in ind_vert_mid]
        ax.invert_xaxis()
        plt.xticks(df.loc[ind_vert,'Slice (I->S)'], [])
        #plt.ylim(max(df['DistancePMJ'].to_numpy()[6:-6]), min(df['DistancePMJ'].to_numpy()[6:-6]))
        #plt.ylim(190, 35)
        #plt.xlim(30,90)
        plt.ylabel(metric +' ($mm^2$)', fontsize=14)
        plt.xlabel('Vertebral Level (S->I)', fontsize=14)
        #ax2.set_xlabel('VertLevel')
        filename = metric + '_plot.png'
        path_filename = os.path.join(args.path_out, filename)
        plt.savefig(path_filename)

if __name__ == '__main__':
    main()