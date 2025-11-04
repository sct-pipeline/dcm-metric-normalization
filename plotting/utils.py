import os
import pandas as pd

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64',
    'aSCOR': 'float64'
}

def load_normative_df_c2(normative_dir, participants_file=None):
    """
    Load normative data from spine-generic dataset for C2 level.
    Compute mean cord and canal area across slices per subject at C2 level.
    """
    cord_dir = os.path.join(normative_dir, 'spinal_cord')
    cord_df = pd.DataFrame()
    for file in os.listdir(cord_dir):
        if 'PAM50.csv' in file:
            df = pd.read_csv(os.path.join(cord_dir, file), dtype=METRICS_DTYPE)
            cord_df = pd.concat([cord_df, df], axis=0, ignore_index=True)
    # Add participant_id
    cord_df.insert(0, 'participant_id', cord_df['Filename'].str.split('/').str[0])
    # Optionally merge participants.tsv info
    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        cord_df = cord_df.merge(df_participants[['participant_id', 'age', 'sex']], on='participant_id', how='left')
    # Keep only VertLevel C2
    cord_df = cord_df[cord_df['VertLevel'] == 2]
    cord_df = cord_df.dropna(subset=['MEAN(area)'])
    # Compute mean per level for each subject
    grouped = cord_df.groupby(['participant_id', 'VertLevel']).agg({
        'MEAN(area)': 'mean',
        'age': 'first',
        'sex': 'first'
    }).reset_index()
    return grouped

def _categorize_c2_area(area, mean_c2_cord_normative):
    """
    Categorize C2 cord area as 'Below normative mean C2' or 'Above normative mean C2'. based on normative mean cord area.
    """
    if area < mean_c2_cord_normative:
        return 'Below normative mean C2 cord area'
    else:
        return 'Above normative mean C2 cord area'
