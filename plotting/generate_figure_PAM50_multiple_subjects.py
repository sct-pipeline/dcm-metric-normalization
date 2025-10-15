#
# Plot morphometrics across subjects, separately for multiple sessions
# For each slice in the PAM50 space, mean and std of morphometric metrics are plotted across subjects
# The script is compatible with spinal cord, canal, or aSCOR metrics. The normative data to plot is determined based on
# the input CSV filename (cord, canal, or aSCOR); see example usages below.
#
# Example usages (cord, canal, aSCOR):
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_cord_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_canal_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
#   python generate_figure_PAM50_multiple_subjects.py
#       -i dcm-zurich_YYYY-MM-DD/results/T2w_ax_aSCOR_metrics_perslice_PAM50.csv
#       -o dcm-zurich_YYYY-MM-DD/results/figures
#
# Author: Jan Valosek, Sandrine Bédard
#

import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64',
    'aSCOR': 'float64'
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
    'aSCOR': 'aSCOR [a.u.]'
}

AGE_DECADES = ['10-20', '21-30', '31-40', '41-50', '51-60']

SESSION_COLORS = {
    'ses-M0': '#1f77b4',  # blue
    'ses-M3': '#ff7f0e',  # orange
    'ses-M6': '#2ca02c',  # green
    'ses-M12': '#ff9999'  # light red
}

# Color mapping for Maximum Compression Level (MCL) stratification
MCL_COLORS = {
    'C2/C3': '#d62728',    # red
    'C3/C4': '#ff7f0e',    # orange
    'C4/C5': '#2ca02c',    # green
    'C5/C6': '#1f77b4',    # blue
    'C6/C7': '#9467bd',    # purple
}

# Color mapping for Myelopathy stratification
MYELOPATHY_COLORS = {
    'yes': '#d62728',      # red - has myelopathy
    'no': '#2ca02c',       # green - no myelopathy
}

METRICS_YLIMITS = {
    'MEAN(diameter_AP)': (5, 9),
    'MEAN(area)': (35, 90),
    'MEAN(diameter_RL)': (8, 15.5),
    'MEAN(eccentricity)': (0.53, 0.91),
    'MEAN(solidity)': (89, 100),
    'MEAN(compression_ratio)': (0.35, 0.86),
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot mean and std of morphometric metrics across subjects for multiple sessions")
    parser.add_argument('-i', required=True, type=str,
                        help="CSV file with morphometric metrics in the PAM50 space across multiple subjects")
    parser.add_argument('-o', required=True, type=str, default='figures',
                        help="Output directory name. The figure name will be based on the input CSV file name. "
                             "Default output directory: figures.")
    parser.add_argument('-s', required=False, type=str, nargs='+',
                        default=['ses-M0'],
                        help="Session to process (e.g., 'ses-M0', 'ses-M3', etc.)",
                        )
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('-participants-file', required=False, type=str,
                        help="Path to the participants.tsv file containing maximum_stenosis or myelopathy data for stratification.")
    parser.add_argument('-stratify', required=False, type=str, choices=['mcl', 'myelopathy'],
                        help="Stratification method: 'mcl' for Maximum Compression Level or 'myelopathy' for myelopathy status.")

    return parser


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
    subjects = df['participant_id'].unique()
    # Get vert levels for one certain subject
    vert = df[df['participant_id'] == subjects[0]]['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def _read_pam50_df(path_HC):
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

    return df


def load_normative_data(path_HC, path_participants_pam50, structure):
    """
    Load normative data from spine-generic dataset in PAM50 space
    :param path_HC:
    :param path_participants_pam50:
    :param structure: 'spinal_cord' or 'canal' or 'aSCOR'
    :return:
    """

    if structure == 'aSCOR':
        # For aSCOR, we need both spinal_cord and canal metrics to compute aSCOR
        df_cord = _read_pam50_df(os.path.join(path_HC, 'spinal_cord'))
        df_canal = _read_pam50_df(os.path.join(path_HC, 'canal'))
        df = pd.merge(df_cord, df_canal, on=['participant_id', 'Slice (I->S)', 'VertLevel'],
                      suffixes=('_sc', '_canal'))
        df['aSCOR'] = df['MEAN(area)_sc'].div(df['MEAN(area)_canal'], fill_value=0)
        df['aSCOR'].replace(np.inf, np.nan)  # output nan instead of inf for /0
    # Add spinal_cord or canal subfolder to the path
    else:
        df = _read_pam50_df(os.path.join(path_HC, structure))

    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if path_participants_pam50:
        df_participants = pd.read_csv(path_participants_pam50, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id"]],
                      on='participant_id')
        # Recode age into age bins by 10 years (decades)
        df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=AGE_DECADES)

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    # Keep only VertLevel from C2 to C7
    df = df[df['VertLevel'] >= 2]
    df = df[df['VertLevel'] <= 7]

    df_spine_generic_min, df_spine_generic_max = df['Slice (I->S)'].min(), df['Slice (I->S)'].max()

    if structure == 'spinal_cord' or structure == 'canal':
        # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
        df['MEAN(compression_ratio)'] = df['MEAN(diameter_AP)'] / df['MEAN(diameter_RL)']
        # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
        df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    # Uncomment to save aggregated dataframe with metrics across all subjects as .csv file
    #df.to_csv(os.path.join(path_out_csv, 'HC_metrics.csv'), index=False)

    return df, df_spine_generic_min, df_spine_generic_max


def fetch_participant_and_session(filename_path):
    """
    Get participant_id, session_ide and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: participant_id, session_id (e.g., sub-001, ses-01)
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    participant_tmp = re.search('sub-(.*?)[_/]', filename_path)
    participant_id = participant_tmp.group(0)[:-1] if participant_tmp else ""    # [:-1] removes the last underscore or slash

    session_tmp = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    session_id = session_tmp.group(0)[:-1] if session_tmp else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return participant_id, session_id


def read_csv_file(csv_file, participants_file=None, stratify_type=None):
    """
    Read CSV file with morphometrics in the PAM50 space across multiple subjects.
    This file is generated with `sct_process_segmentation -normalize-PAM50 1 -perslice 1 -append 1`.
    :param csv_file: input CSV file path
    :param participants_file: path to participants.tsv file with stratification data
    :param stratify_type: type of stratification ('mcl' or 'myelopathy')
    :return: pandas dataframe with additional columns participant_id, session_id, MEAN(compression_ratio), and optionally stratification data
    """

    subjects_df = pd.read_csv(csv_file)

    # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
    if 'aSCOR' not in csv_file:
        subjects_df['MEAN(compression_ratio)'] = subjects_df['MEAN(diameter_AP)'] / \
                                                 subjects_df['MEAN(diameter_RL)']
        # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
        subjects_df['MEAN(solidity)'] = subjects_df['MEAN(solidity)'] * 100

    # Fetch participant_id and session_id from the Filename column
    if 'aSCOR' in csv_file:
        filename_column = 'Filename_sc'
    else:
        filename_column = 'Filename'
    participant_ids = []
    session_ids = []
    for file_path in subjects_df[filename_column]:
        participant_id, session_id = fetch_participant_and_session(file_path)
        participant_ids.append(participant_id)
        session_ids.append(session_id)
    subjects_df.insert(0, 'participant_id', participant_ids)
    subjects_df.insert(1, 'session_id', session_ids)

    # Add stratification data if requested
    if stratify_type and participants_file:
        if os.path.isfile(participants_file):
            df_participants = pd.read_csv(participants_file, sep='\t')

            if stratify_type == 'mcl':
                if 'maximum_stenosis' in df_participants.columns:
                    # Merge MCL data
                    subjects_df = subjects_df.merge(
                        df_participants[['participant_id', 'maximum_stenosis']],
                        on='participant_id', how='left'
                    )
                    # Clean up maximum_stenosis values and map to standard format
                    subjects_df['MCL'] = subjects_df['maximum_stenosis'].fillna('NA')
                    # Standardize MCL values
                    subjects_df['MCL'] = subjects_df['MCL'].apply(lambda x: x if x in MCL_COLORS else 'NA')
                    # Exclude subjects with MCL == 'NA'
                    subjects_df = subjects_df[subjects_df['MCL'] != 'NA']
                    print(f"MCL distribution: {subjects_df['MCL'].value_counts().to_dict()}")
                else:
                    print("Warning: 'maximum_stenosis' column not found in participants file")
                    exit(1)

            elif stratify_type == 'myelopathy':
                if 'myelopathy' in df_participants.columns:
                    # Merge myelopathy data
                    subjects_df = subjects_df.merge(
                        df_participants[['participant_id', 'myelopathy']],
                        on='participant_id', how='left'
                    )
                    # Process myelopathy values: if not n/a, use 'yes', if n/a, use 'no'
                    def process_myelopathy(value):
                        if pd.isna(value) or str(value).lower() == 'n/a':
                            return 'no'
                        else:
                            return 'yes'

                    subjects_df['Myelopathy'] = subjects_df['myelopathy'].apply(process_myelopathy)
                    print(f"Myelopathy distribution: {subjects_df['Myelopathy'].value_counts().to_dict()}")
                else:
                    print("Warning: 'myelopathy' column not found in participants file")
                    exit(1)
        else:
            print(f"Warning: Participants file not found: {participants_file}")
            exit(1)

    return subjects_df


def create_figure(subjects_df, n_subjects, df_normative_data, sessions_to_process, figure_path, stratify_type=None):
    """
    Create figure with mean and std of morphometric metrics across subjects, separately for multiple sessions
    :param subjects_df: pandas dataframe with morphometric metrics across multiple subjects
    :param n_subjects: number of unique subjects in the input dataframe
    :param df_normative_data: pandas dataframe with normative data from spine-generic dataset
    :param sessions_to_process: list of sessions to process (e.g., ['ses-M0', 'ses-M3'])
    :param figure_path: path to save figure
    :param stratify_type: type of stratification ('mcl' or 'myelopathy')
    """
    mpl.rcParams['font.family'] = 'Arial'

    if 'aSCOR' in figure_path:
        # 1x1 grid for 1 metric; 6x5
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        axs = [axs]  # Make it iterable
        METRICS = ['aSCOR']
    else:
        METRICS = [
            'MEAN(area)',
            'MEAN(diameter_AP)',
            'MEAN(diameter_RL)',
            'MEAN(compression_ratio)',
            # 'MEAN(eccentricity)',
            # 'MEAN(solidity)'
        ]
        # 2x2 grid for 4 metrics; 12x10
        # 2x3 grid for 6 metrics; 18x10
        # fig, axs = plt.subplots(2, int(len(METRICS)/2), figsize=(int(len(METRICS)/2)*6, 10))
        # 1x4 grid for 4 metrics; 24x5
        fig, axs = plt.subplots(1, int(len(METRICS)), figsize=(int(len(METRICS)) * 6, 5))
        axs = axs.ravel()

    for metric_idx, metric in enumerate(METRICS):
        ax = axs[metric_idx]

        # Plot normative data
        sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df_normative_data, errorbar='sd',
                     linewidth=2, color='black',
                     label=f'normative data (n={len(df_normative_data["participant_id"].unique())})')

        if stratify_type == 'mcl':
            # Plot by MCL groups instead of sessions
            mcl_groups = subjects_df['MCL'].unique()
            mcl_groups = sorted([mcl for mcl in mcl_groups if mcl in MCL_COLORS])

            for mcl in mcl_groups:
                mcl_data = subjects_df[subjects_df['MCL'] == mcl]
                if len(mcl_data) > 0:
                    mcl_n_subjects = len(mcl_data['participant_id'].unique())
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=mcl_data, errorbar='sd',
                                linewidth=2, color=MCL_COLORS[mcl],
                                label=f"MCL {mcl} (n={mcl_n_subjects})")
        elif stratify_type == 'myelopathy':
            # Plot by Myelopathy groups instead of sessions
            myelopathy_groups = subjects_df['Myelopathy'].unique()
            myelopathy_groups = sorted([myelopathy for myelopathy in myelopathy_groups if myelopathy in MYELOPATHY_COLORS])

            for myelopathy in myelopathy_groups:
                myelopathy_data = subjects_df[subjects_df['Myelopathy'] == myelopathy]
                if len(myelopathy_data) > 0:
                    myelopathy_n_subjects = len(myelopathy_data['participant_id'].unique())
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=myelopathy_data, errorbar='sd',
                                linewidth=2, color=MYELOPATHY_COLORS[myelopathy],
                                label=f"Myelopathy {myelopathy} (n={myelopathy_n_subjects})")
        else:
            # Plot each session's mean and std (original behavior)
            for ses in sessions_to_process:
                session_data = subjects_df[subjects_df['session_id'] == ses]
                if len(session_data) > 0:
                    ses_n_subjects = len(session_data['participant_id'].unique())
                    sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=session_data, errorbar='sd',
                                linewidth=2, color=SESSION_COLORS[ses],
                                label=f"{ses} (n={ses_n_subjects})")

        # Keep the legend only for one plot to avoid duplication
        if metric_idx == 0:
            axs[metric_idx].legend(loc='lower left', fontsize=TICKS_FONT_SIZE)
        else:
            axs[metric_idx].get_legend().remove()

        # Tweak y-axis limits
        # ymin, ymax = METRICS_YLIMITS[metric]
        # ax.set_ylim(ymin, ymax)
        # Remove first and last 4 slices from the x-axis to match single subject figure (to remove smoothing artifacts)
        ax.set_xlim(df_normative_data['Slice (I->S)'].iloc[4], df_normative_data['Slice (I->S)'].iloc[-4])

        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        # ax.set_xlabel('PAM50 Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Remove xticks to hide PAM50 Axial Slice numbers
        ax.set_xticks([])
        ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        # Add vertebral level indicators
        ymin, ymax = ax.get_ylim()
        vert, ind_vert, ind_vert_mid = get_vert_indices(df_normative_data)
        for idx, x in enumerate(ind_vert[1:-1]):
            ax.axvline(df_normative_data.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        for idx, x in enumerate(ind_vert_mid, 0):
            level = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
            ax.text(df_normative_data.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                    ymin - (ymax - ymin) * 0.05, level, horizontalalignment='center',
                    verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.invert_xaxis()
        # Remove xlabel
        ax.set_xlabel('')  # Remove x-axis label ('Slice (I->S)')

    # Fetch cord, canal, or aSCOR from the input filename to include in the figure title
    if 'cord' in figure_path:
        structure = 'Spinal cord morphometrics'
    elif 'canal' in figure_path:
        structure = 'Canal morphometrics'
    elif 'aSCOR' in figure_path:
        structure = 'aSCOR'

    # Update title based on stratification
    if stratify_type == 'mcl':
        stratification_info = "stratified by MCL"
    elif stratify_type == 'myelopathy':
        stratification_info = "stratified by Myelopathy"
    else:
        stratification_info = f"across {n_subjects} subjects"

    plt.suptitle(f"{structure} in the PAM50 space: mean ± std {stratification_info}",
                 fontsize=LABELS_FONT_SIZE, fontweight='bold', y=0.92)
    # Save figure
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {figure_path}')


def main():
    args = get_parser().parse_args()
    path_HC = os.path.expandvars(args.path_HC)
    path_participants_tsv_pam50 = os.path.expandvars(args.participants_file_pam50)
    path_out = os.path.abspath(args.o)
    sessions_to_process = args.s

    csv_file = os.path.abspath(args.i)
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Input CSV file not found: {csv_file}")

    # Read CSV file with optional MCL data
    subjects_df = read_csv_file(csv_file, args.participants_file, args.stratify)

    # # Print number of subjects for each slice
    # slice_counts = subjects_df.groupby('Slice (I->S)')['participant_id'].nunique()
    # print("Number of subjects per slice:")
    # print(slice_counts)

    # Keep only VertLevel from C2 to C7
    subjects_df = subjects_df[subjects_df['VertLevel'] >= 2]
    subjects_df = subjects_df[subjects_df['VertLevel'] <= 7]

    # Exclude subjects -- canal correction needed
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-004']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-008']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-014']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-032']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-111']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-127']
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-136']

    # Get number of unique subjects
    n_subjects = len(subjects_df['participant_id'].unique())
    print(f"Number of unique subjects in the input CSV: {n_subjects}")

    # Load normative data
    if 'cord' in args.i:
        structure = 'spinal_cord'
    elif 'canal' in args.i:
        structure = 'canal'
    elif 'aSCOR' in args.i:
        structure = 'aSCOR'
    df_normative_data, df_min, df_max = load_normative_data(path_HC, path_participants_tsv_pam50, structure)

    # Plotting
    os.makedirs(path_out, exist_ok=True)
    # Use basename from args.i to create figure name
    figure_basename = os.path.basename(args.i).replace('.csv', '')

    # Update figure filename based on stratification type
    if args.stratify:
        figure_fname = f'{figure_basename}_{n_subjects}subjects_{args.stratify}-stratified.png'
    else:
        figure_fname = f'{figure_basename}_{n_subjects}subjects_{len(sessions_to_process)}sessions.png'

    figure_path = os.path.join(path_out, figure_fname)
    create_figure(subjects_df, n_subjects, df_normative_data, sessions_to_process, figure_path, args.stratify)


if __name__ == '__main__':
    main()
