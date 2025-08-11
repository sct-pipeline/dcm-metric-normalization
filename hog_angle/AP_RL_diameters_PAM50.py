#
# Plot a single subject morphometric metrics in the PAM50 space per slice and vertebral levels
# Original vs HOG-based AP and RL diameters
#
# You can use SCT's conda environment to run this script:
#       # Go to the SCT directory
#       cd $SCT_DIR
#       # Activate SCT conda environment
#       source ./python/etc/profile.d/conda.sh
#       conda activate venv_sct
#
# Example usage on a single subject:
#       python AP_RL_diameters_PAM50.py -i /path/to/morphometrics_PAM50.csv -o /path/to/output/morphometrics_PAM50.png
#
# Authors: Jan Valosek
#

import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

METRICS = [
    'MEAN(diameter_AP)',
    'MEAN(diameter_RL)',
]

# Set ylim to do not overlap horizontal grid with vertebrae labels
METRICS_TO_YLIM = {
    'MEAN(diameter_AP)': (5, 10),
    'MEAN(diameter_RL)': (8, 16),
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot AP and RL diameters measured using different methods in the PAM50 space.")
    parser.add_argument('-i', required=True, type=str,
                        help="Path to the CSV file with diameter measurements.")
    parser.add_argument('-o', required=True, type=str,
                        help="Path to save the output figure.")
    parser.add_argument('-smooth', required=False, type=int, default=0, metavar='INT',
                        help="Smooth measurements before plotting. Number of points in the moving average window."
                             "Examples: 0: no smoothing, 5: window of 5")

    return parser


# Apply smoothing to the metric data
def smooth_data(y: np.ndarray, box_pts: int) -> np.ndarray:
    """
    Smooths a 1D array using a simple moving average (box filter).
    Inspired by: https://github.com/sct-pipeline/rootlets-informed-reg2template/blob/main/csa_analysis.py#L199
    Args:
        y (np.ndarray): Input 1D array to be smoothed.
        box_pts (int): Number of points in the moving average window. Needs to be >= 1.
    Returns:
        np.ndarray: Smoothed array of the same length as the input.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def load_single_subject_data(path_single_subject, df_spine_generic_min, df_spine_generic_max):
    """
    Load single subject data
    :param path_single_subject: path to single subject CSV file (from session1 or session2)
    :param df_spine_generic_min: minimum slice number from spine-generic dataset
    :param df_spine_generic_max: maximum slice number from spine-generic dataset
    :return:
    """
    df_single_subject = pd.read_csv(path_single_subject)
    # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
    df_single_subject['MEAN(compression_ratio)'] = df_single_subject['MEAN(diameter_AP)'] / \
                                                   df_single_subject['MEAN(diameter_RL)']

    # Keep only slices from C1 to Th1 to match the slices of the spine-generic normative values
    df_single_subject = df_single_subject[(df_single_subject['Slice (I->S)'] >= df_spine_generic_min) &
                                          (df_single_subject['Slice (I->S)'] <= df_spine_generic_max)]

    return df_single_subject

def get_vert_indices(df):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get unique participant IDs
    subjects = df['Filename'].unique()
    # Get vert levels for one certain subject
    vert = df[df['Filename'] == subjects[0]]['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def create_lineplot(df, figure_path, smooth):
    """
    Create lineplot for individual metrics per vertebral levels.
    Args:
        df (pd.DataFrame): dataframe with single subject values
        figure_path (str): path to save the figure
        smooth (int): Smooth measurements before plotting. 0: no smoothing; 1: smoothing
    """
    mpl.rcParams['font.family'] = 'Arial'

    # Plot figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):

        if smooth > 0:
            # Smooth the data to improve visualization
            df[metric] = smooth_data(df[metric].values, smooth)
            df[f'{metric.replace(")", "_hog)")}'] = smooth_data(df[f'{metric.replace(")", "_hog)")}'].values, smooth)
        # Original
        sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric,
                     data=df, linewidth=2,
                     label=metric)
        # HOG-based
        sns.lineplot(ax=axs[index], x="Slice (I->S)", y=f'{metric.replace(")", "_hog)")}',
                     data=df, linewidth=2,
                     label=f'{metric.replace(")", "_hog)")}')

        # Tweak y-axis limits
        # ymin, ymax = METRICS_YLIMITS[metric]
        # axes.set_ylim(ymin, ymax)
        # Remove first and last 4 slices from the x-axis to remove smoothing artifacts
        axs[index].set_xlim(df['Slice (I->S)'].iloc[4], df['Slice (I->S)'].iloc[-4])

        # Remove xticks to hide PAM50 Axial Slice numbers
        axs[index].set_xticks([])
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get ymin and ymax for the y-axis
        axs[index].set_ylim(METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1])
        ymin, ymax = axs[index].get_ylim()

        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.3, zorder=0)
        for idx, x in enumerate(ind_vert_mid, 0):
            level = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
            axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                            ymin - (ymax-ymin)*0.05, level, horizontalalignment='center',
                            verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        axs[index].invert_xaxis()
        axs[index].yaxis.grid(True)
        axs[index].set_axisbelow(True)
        axs[index].set_ylabel(f'{metric}', fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('')

        # Decrease the font size of the legend
        axs[index].legend(loc='lower left', fontsize=TICKS_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {figure_path}')
    #plt.show()
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Get the file path from args
    csv_path = args.i
    out_path = args.o

    # Create a list of dataframes for each session file
    df = load_single_subject_data(csv_path, 700, 980)
    if df.empty:
        print('WARNING: No slices found in the range C1-Th1 in the single subject data. Exiting...')
        sys.exit(1)

    create_lineplot(df, out_path, args.smooth)


if __name__ == '__main__':
    main()