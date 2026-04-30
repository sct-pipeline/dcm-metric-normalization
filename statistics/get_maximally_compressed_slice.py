"""
Extract anterior and posterior lengths at the maximally compressed level (MCL) per DCM subject.

The MCL is read from clinical_scores.xlsx ('maximum_stenosis' column, integer-coded:
1=C2/C3, 2=C3/C4, 3=C4/C5, 4=C5/C6, 5=C6/C7). Subjects are identified via 'record_id_BL',
mapped to participant_id as sub-{record_id_BL:03d}.
In the native-space (i.e., no PAM50 normalization) per-slice CSV, each slice carries a VertLevel info.
The disk between vertebra N-1 and vertebra N sits at the transition between the two VertLevel
blocks:

    ... | VertLevel N (more inferior) | disk | VertLevel N-1 (more superior) | ...

The disk boundary is defined as the pair of adjacent slices at that transition:
    - bound_lo: last slice of VertLevel N   (most superior slice of the lower vertebra)
    - bound_hi: first slice of VertLevel N-1 (most inferior slice of the upper vertebra)

A symmetric window of DISK_HALF_WINDOW slices is added on each side, giving a total
window of 2 * (DISK_HALF_WINDOW + 1) slices:

    DISK_HALF_WINDOW = 0  →  2 slices  (boundary pair only)
    DISK_HALF_WINDOW = 1  →  4 slices  (current default)
    DISK_HALF_WINDOW = 2  →  6 slices

The slice with the minimum CSA within the window is selected, and its anterior and posterior lengths are reported.

Usage:
    python statistics/get_maximally_compressed_slice.py \
        -i             <T2w_ax_cord_metrics_perlevel.csv> \
        -clinical-file <clinical_scores.xlsx> \
        -path-out        <output_dir>
"""

import argparse
import os

import pandas as pd


# Maps clinical_scores.xlsx integer codes to disk label strings
EXCEL_STENOSIS_TO_DISC = {1: 'C2/C3', 2: 'C3/C4', 3: 'C4/C5', 4: 'C5/C6', 5: 'C6/C7'}

# Maps 'C(N-1)/CN' disk label → lower vertebral level number N
DICT_DISC_LABELS = {
    'C1/C2': 2,
    'C2/C3': 3,
    'C3/C4': 4,
    'C4/C5': 5,
    'C5/C6': 6,
    'C6/C7': 7,
}

# Slices added on each side of the boundary pair; total window = 2 * (DISK_HALF_WINDOW + 1)
DISK_HALF_WINDOW = 1  # → 4 slices


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', required=True, metavar='CSV',
                        help='Native-space per-slice CSV (e.g., T2w_ax_cord_metrics_perlevel.csv)')
    parser.add_argument('-clinical-file', required=True, metavar='XLSX',
                        help='clinical_scores.xlsx with record_id_BL and maximum_stenosis columns')
    parser.add_argument('-path-out', default='stats', metavar='DIR',
                        help='Output directory (default: stats)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    # Clinical table: record_id_BL → participant_id, integer maximum_stenosis → disk label
    pts = pd.read_excel(args.clinical_file, usecols=['record_id_BL', 'maximum_stenosis',
                                                      # 'myelopathy',
                                                      # 'cheps_assessment_date_BL',
                                                      # 'c6_grading_cheps_BL', 'c8_grading_cheps_BL',
                                                      # 't4_grading_cheps_BL'
                                                     ])
    pts = pts.dropna(subset=['maximum_stenosis'])
    pts['participant_id'] = pts['record_id_BL'].apply(lambda x: f'sub-{int(x):03d}')
    pts['maximum_stenosis_str'] = pts['maximum_stenosis'].apply(lambda x: EXCEL_STENOSIS_TO_DISC.get(int(x)))
    unmapped = pts[pts['maximum_stenosis_str'].isna()]['maximum_stenosis'].unique()
    if len(unmapped):
        print(f'Warning: unrecognised maximum_stenosis values (skipped): {unmapped}')
    pts = pts.dropna(subset=['maximum_stenosis_str'])
    pts['disk_level'] = pts['maximum_stenosis_str'].map(DICT_DISC_LABELS).astype(int)

    # Native-space per-slice metrics
    df = pd.read_csv(args.i)
    df['participant_id'] = df['Filename'].str.extract(r'(sub-\d+)')[0]
    df['VertLevel'] = pd.to_numeric(df['VertLevel'], errors='coerce')
    df = df.dropna(subset=['VertLevel'])
    df['VertLevel'] = df['VertLevel'].astype(int)

    records = []
    skipped = []

    for _, pt_row in pts.iterrows():
        pid = pt_row['participant_id']
        disk_n = pt_row['disk_level']   # e.g., 6 for C5/C6

        sub_df = df[df['participant_id'] == pid]
        if sub_df.empty:
            skipped.append((pid, 'not in metrics CSV'))
            continue

        # Disk CX/C(X+1) is between VertLevel X (more superior, higher slice index)
        # and VertLevel X+1 (more inferior, lower slice index) in I->S ordering.
        upper_vert = disk_n - 1   # e.g., 5  (more superior)
        lower_vert = disk_n       # e.g., 6  (more inferior)

        upper_slices = sub_df[sub_df['VertLevel'] == upper_vert]['Slice (I->S)']
        lower_slices = sub_df[sub_df['VertLevel'] == lower_vert]['Slice (I->S)']

        if upper_slices.empty or lower_slices.empty:
            skipped.append((pid, f'disk {pt_row["maximum_stenosis_str"]} not in axial FOV'))
            continue

        # Boundary: the two adjacent slices on either side of the level transition
        bound_lo = lower_slices.max()   # last (most superior) slice of the lower vertebra
        bound_hi = upper_slices.min()   # first (most inferior) slice of the upper vertebra

        window = sub_df[
            (sub_df['Slice (I->S)'] >= bound_lo - DISK_HALF_WINDOW) &
            (sub_df['Slice (I->S)'] <= bound_hi + DISK_HALF_WINDOW) &
            sub_df['MEAN(length_anterior)'].notna()
        ]

        if window.empty:
            skipped.append((pid, f'no valid slices in window around disk {pt_row["maximum_stenosis_str"]}'))
            continue

        idx = window['MEAN(area)'].idxmin()
        row = window.loc[idx]
        a  = row['MEAN(length_anterior)']
        p  = row['MEAN(length_posterior)']
        ap = row['MEAN(diameter_AP)']
        asymmetry = (a - p) / ap if ap != 0 else float('nan')

        records.append({
            'participant_id':        pid,
            'maximum_stenosis':      int(pt_row['maximum_stenosis']),
            'maximum_stenosis_label': pt_row['maximum_stenosis_str'],
            'compressed_slice_T2w_ax_mcl':  int(row['Slice (I->S)']),
            'diameter_AP':           round(ap, 3),
            'length_anterior':       round(a, 3),
            'length_posterior':      round(p, 3),
            'asymmetry':             round(asymmetry, 4),
            # 'myelopathy':              pt_row['myelopathy'],
            # 'cheps_assessment_date_BL': pt_row['cheps_assessment_date_BL'],
            # 'c6_grading_cheps_BL':     pt_row['c6_grading_cheps_BL'],
            # 'c8_grading_cheps_BL':   pt_row['c8_grading_cheps_BL'],
            # 't4_grading_cheps_BL':   pt_row['t4_grading_cheps_BL'],
        })

    if skipped:
        print('Skipped subjects:')
        for pid, reason in skipped:
            print(f'  {pid}: {reason}')
        print()

    out = pd.DataFrame(records).sort_values('participant_id').reset_index(drop=True)
    out_path = os.path.join(args.path_out, 'max_compression_metrics.csv')
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
