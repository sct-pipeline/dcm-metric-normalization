"""
Split the cohort into two groups based on the normative C2 cord area and create a table with group characteristics.
"""

import os
import sys
import argparse
import pandas as pd
import scipy.stats

from utils import load_normative_df_c2, _categorize_c2_area, format_pvalue
from generate_figure_PAM50_multiple_subjects import fetch_participant_and_session, MCL_COLORS, _process_myelopathy, _stratify_mjoa, _create_age_group


def get_parser():
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument('-i', required=True, type=str,
                        help="CSV file with patients' morphometric metrics in the PAM50 space across multiple subjects")
    parser.add_argument('-o', required=True, type=str, default='figures',
                        help="Output directory name. The figure name will be based on the input CSV file name. "
                             "Default output directory: figures.")
    parser.add_argument('-path-HC', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics',
                        help="Path to the folder with CSV files with normative data from spine-generic dataset")
    parser.add_argument('-participants-file-pam50', required=False, type=str,
                        default='$SCT_DIR/data/PAM50_normalized_metrics/participants.tsv',
                        help="Path to the spine-generic participants.tsv file (used to filter per sex).")
    parser.add_argument('-participants-file', required=False, type=str,
                        help="Path to the patients' participants.tsv file containing maximum_stenosis or myelopathy data for stratification.")
    parser.add_argument('-clinical-file', required=False, type=str,
                        help="Excel file with clinical scores (must contain 'total_mjoa' column)")
    parser.add_argument('-stratify', required=False, type=str,
                        choices=['mcl', 'myelopathy', 'mjoa', 'therapeutic_decision', 'age', 'None'],
                        help="Stratification method:"
                             "'mcl' for Maximum Compression Level; '-participants-file' is required, "
                             "'myelopathy' for myelopathy status; '-participants-file' is required, "
                             "'therapeutic_decision' (operative/conservative); -participants-file' is required, "
                             "'age' for age group stratification; '-participants-file' is required, "
                             "'mjoa' mJOA (mild: 15 ≤ mJOA ≤ 17; moderate 14 ≤ mJOA); '-clinical-file' is required. "
                             )

    return parser


def read_csv_file(csv_file, participants_file=None, clinical_file=None):
    """
    - Read CSV file with morphometrics in the PAM50 space across multiple subjects.
        This file is generated with `sct_process_segmentation -normalize-PAM50 1 -perslice 1 -append 1`.
    - Read participants.tsv file with MCL and myelopathy data and clinical Excel file with mJOA scores.
    :param csv_file: input CSV file path
    :param participants_file: path to participants.tsv file with stratification data
    :param clinical_file: path to Excel file with clinical scores (must contain 'total_mjoa' column)
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

    if participants_file and os.path.isfile(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')

        # Merge MCL, myelopathy, therapeutic_decision, age columns
        subjects_df = subjects_df.merge(
            df_participants[['participant_id', 'maximum_stenosis', 'stenosis', 'myelopathy', 'therapeutic_decision', 'age', 'sex']],
            on='participant_id', how='left'
        )

        # Clean up maximum_stenosis values and map to standard format
        subjects_df['MCL'] = subjects_df['maximum_stenosis'].fillna('NA')
        # Standardize MCL values
        subjects_df['MCL'] = subjects_df['MCL'].apply(lambda x: x if x in MCL_COLORS else 'NA')
        # # Exclude subjects with MCL == 'NA'
        # subjects_df = subjects_df[subjects_df['MCL'] != 'NA']

        # Clean up therapeutic_decision values
        subjects_df['therapeutic_decision'] = subjects_df['therapeutic_decision'].fillna('NA')
        # # Exclude subjects with MCL == 'NA'
        # subjects_df = subjects_df[subjects_df['therapeutic_decision'] != 'NA']

        subjects_df['stenosis'] = subjects_df['stenosis'].fillna('NA')
        # # Exclude subjects with stenosis == 'NA'
        # subjects_df = subjects_df[subjects_df['stenosis'] != 'NA']
        # Stenosis is a str of different stenosis levels, e.g., 'C3/C4, C5/C6', convert it to list
        subjects_df['stenosis_levels'] = subjects_df['stenosis'].apply(lambda x: [level.strip() for level in x.split(',')])
        # Add a new column, 'num_of_stenosis' with the number of stenosis levels per subject
        subjects_df['num_of_stenosis'] = subjects_df['stenosis_levels'].apply(len)
        # Add a new column, 'single_vs_multi_stenosis' with 'single' or 'multi' values
        subjects_df['single_vs_multi_stenosis'] = subjects_df['num_of_stenosis'].apply(lambda x: 'Single stenosis' if x == 1 else 'Multi-level stenosis')

        # Clean up myelopathy values
        # Process myelopathy values: if not n/a, use 'yes', if n/a, use 'no'
        subjects_df['Myelopathy'] = subjects_df['myelopathy'].apply(_process_myelopathy)

        # Clean up age values and create age_group column
        subjects_df['age_group'] = subjects_df['age'].apply(_create_age_group)
        # # Exclude unknown age
        # subjects_df = subjects_df[subjects_df['age_group'] != 'unknown']

    else:
        sys.exit(f"Warning: Participants file not found: {participants_file}")

    if clinical_file and os.path.isfile(clinical_file):
        df_clinical = pd.read_excel(clinical_file, usecols=['record_id', 'total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth'])
        # Format record_id to match participant_id format (e.g., `1` to `sub-001`)
        df_clinical['participant_id'] = df_clinical['record_id'].apply(lambda x: f'sub-{int(x):03d}')
        # Drop record_id column
        df_clinical = df_clinical.drop(columns=['record_id'])
        # Stratify mJOA
        df_clinical['mJOA_severity_bl'] = df_clinical['total_mjoa_bl'].apply(_stratify_mjoa)

        # Merge mJOA data
        subjects_df = subjects_df.merge(
            df_clinical[['participant_id', 'total_mjoa_bl', 'total_mjoa_6mth', 'total_mjoa_12mth', 'mJOA_severity_bl']],
            on='participant_id', how='left'
        )
    else:
        sys.exit(f"Warning: Clinical file not found: {clinical_file}")

    return subjects_df

def prepare_table(grouped_subjects_c2, path_out):
    """
    Prepare table to be saved as CSV
    Subjects are grouped based on normative_mean_c2 column
    """
    table_rows = []
    # Collect age data for statistical test
    age_data = {}
    for group in grouped_subjects_c2['normative_mean_c2'].unique():
        group_df = grouped_subjects_c2[grouped_subjects_c2['normative_mean_c2'] == group]
        age_data[group] = group_df['age'].dropna().values
        # Subject count
        table_rows.append({
            'Characteristic': 'Subject count',
            group: len(group_df)
        })
        # Sex distribution
        if 'sex' in group_df.columns:
            for sex, count in group_df['sex'].value_counts().items():
                table_rows.append({
                    'Characteristic': f'Sex: {sex}',
                    group: count
                })
        # Age
        age_mean = group_df['age'].mean()
        age_std = group_df['age'].std()
        table_rows.append({
            'Characteristic': 'Age (mean ± SD)',
            group: f"{age_mean:.1f} ± {age_std:.1f}"
        })
        # Age group
        for age_group, count in group_df['age_group'].value_counts().items():
            table_rows.append({
                'Characteristic': f'Age group: {age_group}',
                group: count
            })
        # Number of stenosis
        for num_stenosis, count in group_df['num_of_stenosis'].value_counts().items():
            table_rows.append({
                'Characteristic': f'Number of stenosis: {num_stenosis}',
                group: count
            })
        # Single vs multi-level stenosis
        for svs, count in group_df['single_vs_multi_stenosis'].value_counts().items():
            table_rows.append({
                'Characteristic': f'{svs}',
                group: count
            })
        # MCL
        for mcl, count in group_df['MCL'].value_counts().items():
            table_rows.append({
                'Characteristic': f'MCL: {mcl}',
                group: count
            })
        # Myelopathy
        for myelopathy, count in group_df['Myelopathy'].value_counts().items():
            table_rows.append({
                'Characteristic': f'Myelopathy: {myelopathy}',
                group: count
            })
        # Therapeutic decision
        for td, count in group_df['therapeutic_decision'].value_counts().items():
            table_rows.append({
                'Characteristic': f'Therapeutic decision: {td}',
                group: count
            })
        # # mJOA severity
        # if 'mJOA_severity_bl' in group_df.columns:
        #     for sev, count in group_df['mJOA_severity_bl'].value_counts().items():
        #         table_rows.append({
        #             'Characteristic': f'mJOA severity bl: {sev}',
        #             group: count
        #         })
        # mJOA score
        if 'total_mjoa_bl' in group_df.columns:
            mjoa_mean = group_df['total_mjoa_bl'].mean()
            mjoa_std = group_df['total_mjoa_bl'].std()
            table_rows.append({
                'Characteristic': 'mJOA score bl (mean ± SD)',
                group: f"{mjoa_mean:.2f} ± {mjoa_std:.2f}"
            })
        # mJOA score at 6 months
        if 'total_mjoa_6mth' in group_df.columns:
            mjoa_6mth_mean = group_df['total_mjoa_6mth'].mean()
            mjoa_6mth_std = group_df['total_mjoa_6mth'].std()
            table_rows.append({
                'Characteristic': 'mJOA score 6mth (mean ± SD)',
                group: f"{mjoa_6mth_mean:.2f} ± {mjoa_6mth_std:.2f}"
            })
        # mJOA score at 12 months
        if 'total_mjoa_12mth' in group_df.columns:
            mjoa_12mth_mean = group_df['total_mjoa_12mth'].mean()
            mjoa_12mth_std = group_df['total_mjoa_12mth'].std()
            table_rows.append({
                'Characteristic': 'mJOA score 12mth (mean ± SD)',
                group: f"{mjoa_12mth_mean:.2f} ± {mjoa_12mth_std:.2f}"
            })
    # Statistical test for age between groups
    group_names = list(age_data.keys())
    if len(group_names) == 2:
        stat, p_value = scipy.stats.ttest_ind(age_data[group_names[0]], age_data[group_names[1]], nan_policy='omit')
        # Add p-value to the age row
        for row in table_rows:
            if row['Characteristic'] == 'Age (mean ± SD)':
                row['p-value'] = format_pvalue(p_value, include_equal=False)
    # Define desired order for characteristics
    characteristic_order = [
        'Subject count',
        'Sex: F', 'Sex: M',
        'Age (mean ± SD)',
        'Age group: <50', 'Age group: 50-65', 'Age group: >65', 'Age group: unknown',
        'Number of stenosis: 1', 'Number of stenosis: 2', 'Number of stenosis: 3', 'Number of stenosis: 4',
        'Single stenosis', 'Multi-level stenosis',
        'MCL: C2/C3', 'MCL: C3/C4', 'MCL: C4/C5', 'MCL: C5/C6', 'MCL: C6/C7', 'MCL: NA',
        'Myelopathy: yes', 'Myelopathy: no',
        'Therapeutic decision: operative', 'Therapeutic decision: conservative', 'Therapeutic decision: NA',
        # 'mJOA severity bl: mJOA=18',
        # 'mJOA severity bl: mild (15 ≤ mJOA ≤ 17)',
        # 'mJOA severity bl: moderate (12 ≤ mJOA ≤ 14)',
        # 'mJOA severity bl: severe (mJOA ≤ 11)'
        # 'mJOA severity bl: unknown'
        'mJOA score bl (mean ± SD)',
        'mJOA score 6mth (mean ± SD)',
        'mJOA score 12mth (mean ± SD)',
    ]
    # Convert to DataFrame
    table_df = pd.DataFrame(table_rows)
    # Ensure p-value column exists if any row has it
    if any('p-value' in row for row in table_rows):
        if 'p-value' not in table_df.columns:
            table_df['p-value'] = [row.get('p-value', '-') for row in table_rows]
        else:
            table_df['p-value'] = table_df['p-value'].fillna('-')
    else:
        table_df['p-value'] = '-'
    # Pivot to wide format
    value_columns = list(grouped_subjects_c2['normative_mean_c2'].unique())
    if 'p-value' in table_df.columns:
        value_columns.append('p-value')
    table_pub = table_df.pivot_table(index='Characteristic', values=value_columns, aggfunc='first').reset_index()
    # Order rows
    table_pub['order'] = table_pub['Characteristic'].apply(lambda x: characteristic_order.index(x) if x in characteristic_order else 999)
    table_pub = table_pub.sort_values('order').drop('order', axis=1)
    # Save to CSV
    table_pub.to_csv(os.path.join(path_out, 'c2_area_group_characteristics_table.csv'), index=False)
    print(f"Table saved to {os.path.join(path_out, 'c2_area_group_characteristics_table.csv')}")


def main():
    args = get_parser().parse_args()
    path_HC = os.path.expandvars(args.path_HC)
    path_participants_tsv_pam50 = os.path.expandvars(args.participants_file_pam50)
    path_out = os.path.abspath(args.o)

    # -------------
    # Read CSV file with patients' morphometrics and optional stratification data (e.g., MCL, myelopathy)
    # -------------
    csv_file = os.path.abspath(args.i)
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Input CSV file not found: {csv_file}")
    subjects_df = read_csv_file(csv_file, args.participants_file, args.clinical_file)

    # Exclude sub-004 --> poor canal seg due to strong flow void artifacts
    subjects_df = subjects_df[subjects_df['participant_id'] != 'sub-004']

    # Compute mean C2 for each patient from subjects_df using df.groupby
    subjects_df_c2 = subjects_df[subjects_df['VertLevel'] == 2]
    grouped_subjects_c2 = subjects_df_c2.groupby('participant_id').agg({
        'MEAN(area)': 'mean',
        'sex': 'first',
        'age': 'first',
        'age_group': 'first',
        'MCL': 'first',
        'num_of_stenosis': 'first',
        'single_vs_multi_stenosis': 'first',
        'Myelopathy': 'first',
        'therapeutic_decision': 'first',
        'total_mjoa_bl': 'first',
        'total_mjoa_6mth': 'first',
        'total_mjoa_12mth': 'first',
        'mJOA_severity_bl': 'first'
    }).reset_index()

    # -------------
    # Load normative cord and canal data for VertLevel C2; also read age and sex
    # -------------
    grouped = load_normative_df_c2(path_HC, path_participants_tsv_pam50)
    # Compute mean C2 across subjects
    mean_c2_cord_normative = grouped['MEAN(area)'].mean()
    print(f"Mean C2 cord normative: {mean_c2_cord_normative}")

    # -------------
    # Group patients cohort based on mean_c2_cord_normative into two groups: below and above normative mean
    # -------------
    # Add a new column to grouped_subjects_c2 indicating whether the patient's mean C2 cord area is below or above normative mean
    grouped_subjects_c2['normative_mean_c2'] = grouped_subjects_c2['MEAN(area)'].apply(lambda x: _categorize_c2_area(x, mean_c2_cord_normative))

    prepare_table(grouped_subjects_c2, path_out)

if __name__ == "__main__":
    main()
