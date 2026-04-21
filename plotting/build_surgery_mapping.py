#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the surgery scan mapping table (baseline_surgery_dates_merged_clean.csv).

This script synthesises the logic that was previously spread across four scripts:
  - check_surg_dates.py          : extract surgery dates from clinical_scores.xlsx
  - merge_all_subjects_final.py  : merge surgery dates with M0 scan dates
  - add_months_diff.py           : compute months between M0 scan and surgery
  - add_timepoint_columns.py     : assign nominal before/after surgery timepoints

It then adds a verification step (not in the original scripts) that confirms
whether the nominal timepoints actually have MRI data in the per-slice CSVs,
producing the `before_surg_found` / `after_surg_found` columns of the clean file.

OUTPUT
------
A CSV with one row per subject containing:
  subject                      : BIDS ID (sub-XXX)
  date_of_scan                 : M0 scan date (from participants.tsv)
  surgery_date                 : date of surgery (from clinical_scores.xlsx)
  months_diff_scan_to_surgery  : calendar months from M0 scan to surgery
  before_surg                  : nominal last timepoint before surgery
  after_surg                   : nominal first timepoint after surgery
  before_surg_found            : last timepoint before surgery WITH actual MRI data
  after_surg_found             : first timepoint after surgery WITH actual MRI data

Usage:
    python build_surgery_mapping.py \\
        --clinical-scores data/dcm-zurich/phenotype/clinical_scores.xlsx \\
        --participants    data/dcm-zurich/participants.tsv \\
        --perslice-dir    results/.../timepoint_data \\
        --output          results/surgery_mapping_new.csv

Authors: Kahina Baouche
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

# ============================================================================
# Constants
# ============================================================================

TIMEPOINTS = ['M0', 'M6', 'M12', 'M24', 'M36', 'M48', 'M60']

# Nominal months elapsed since M0 for each timepoint
TIMEPOINT_MONTHS = {
    'M0': 0, 'M6': 6, 'M12': 12,
    'M24': 24, 'M36': 36, 'M48': 48, 'M60': 60,
}


# ============================================================================
# CLI
# ============================================================================

def get_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--clinical-scores', required=True,
                   help='Path to clinical_scores.xlsx')
    p.add_argument('--participants', required=True,
                   help='Path to participants.tsv (BIDS)')
    p.add_argument('--perslice-dir', required=True,
                   help='Directory containing T2w_ax_cord_metrics_perslice_{tp}_data.csv files')
    p.add_argument('--output', required=True,
                   help='Path for the output CSV (will not overwrite if file exists)')
    p.add_argument('--structure', default='cord',
                   help='Structure name in perslice filenames (default: cord)')
    return p


# ============================================================================
# STEP 1 — Extract surgery dates from clinical_scores.xlsx
# ============================================================================

def extract_surgery_dates(clinical_scores_path):
    """
    Read clinical_scores.xlsx and extract one surgery date per subject.

    The spreadsheet stores surgery dates redundantly across many columns
    (surg_date_BL, surg_date_6mth, surg_date_12mth, …). For a given subject
    all non-empty entries should agree (same date at every timepoint where the
    field was filled). This function:

      1. Collects all columns whose name contains 'surg_date' (but NOT
         'surg_date_before', which marks the scan date just before surgery,
         not the surgery date itself).
      2. For each subject, checks that all non-empty surg_date columns hold the
         same date → "coherent" subject.
      3. Subjects with contradictory dates across columns are flagged as
         incoherent and excluded.

    Returns
    -------
    DataFrame with columns: subject (int), surgery_date (datetime)
    """
    print('\n--- STEP 1: Extracting surgery dates from clinical_scores.xlsx ---')

    xl = pd.read_excel(clinical_scores_path)

    # Identify surgery-date columns (exclude "before" columns which are scan dates)
    surg_date_cols = [
        c for c in xl.columns
        if 'surg_date' in c.lower() and 'before' not in c.lower()
    ]
    print(f'  Found {len(surg_date_cols)} surg_date columns: {surg_date_cols}')

    coherent, incoherent = [], []

    for _, row in xl.iterrows():
        subject_id = row.get('record_id_BL')
        if pd.isna(subject_id):
            continue

        # Collect all non-null surgery-date values for this subject
        dates = {
            col: str(row[col]).strip()
            for col in surg_date_cols
            if pd.notna(row[col])
        }

        if not dates:
            continue  # No surgery date for this subject

        unique_dates = set(dates.values())

        if len(unique_dates) == 1:
            # All filled columns agree → coherent
            coherent.append({
                'subject': int(subject_id),
                'surgery_date': pd.to_datetime(list(unique_dates)[0]),
            })
        else:
            # Contradictory dates → exclude and warn
            incoherent.append({'subject': int(subject_id), 'dates': dates})

    print(f'  Coherent subjects (consistent surgery date): {len(coherent)}')
    print(f'  Incoherent subjects (contradictory dates, excluded): {len(incoherent)}')
    if incoherent:
        for s in incoherent:
            print(f'    sub-{s["subject"]:03d}: {s["dates"]}')

    return pd.DataFrame(coherent)  # columns: subject (int), surgery_date


# ============================================================================
# STEP 2 — Load M0 scan dates from participants.tsv
# ============================================================================

def load_baseline_scan_dates(participants_path):
    """
    Read participants.tsv and extract the baseline (M0) scan date for each subject.

    The `date_of_scan` column in participants.tsv holds the date of the M0 MRI
    session. This is used as the time origin for computing months_diff.

    Returns
    -------
    DataFrame with columns: subject (int), date_of_scan (datetime)
    """
    print('\n--- STEP 2: Loading M0 scan dates from participants.tsv ---')

    ptx = pd.read_csv(participants_path, sep='\t')

    if 'participant_id' not in ptx.columns or 'date_of_scan' not in ptx.columns:
        raise ValueError('participants.tsv must contain participant_id and date_of_scan')

    # Extract numeric subject ID from 'sub-XXX'
    ptx['subject'] = (
        ptx['participant_id']
        .str.extract(r'(\d+)')[0]
        .astype(int)
    )
    ptx['date_of_scan'] = pd.to_datetime(ptx['date_of_scan'], errors='coerce')

    result = ptx[['subject', 'date_of_scan']].dropna(subset=['date_of_scan'])
    print(f'  Subjects with M0 scan date: {len(result)} / {len(ptx)}')
    return result


# ============================================================================
# STEP 3 — Merge scan dates with surgery dates
# ============================================================================

def merge_scan_and_surgery(baseline_df, surgery_df):
    """
    Left-join baseline (M0) scan dates with surgery dates, keeping all subjects.

    Subjects without a surgery date will have NaN in surgery_date —
    they are retained in the merged table so the output covers the full cohort,
    but only subjects with a surgery date proceed to the later steps.

    Returns
    -------
    DataFrame with columns: subject (int), date_of_scan, surgery_date, subject_bids
    """
    print('\n--- STEP 3: Merging scan dates with surgery dates ---')

    merged = baseline_df.merge(surgery_df, on='subject', how='left')

    # Add BIDS-formatted subject ID (sub-XXX)
    merged['subject_bids'] = 'sub-' + merged['subject'].astype(str).str.zfill(3)

    n_with_surg = merged['surgery_date'].notna().sum()
    print(f'  Total subjects: {len(merged)}')
    print(f'  With surgery date: {n_with_surg}')
    print(f'  Without surgery date: {len(merged) - n_with_surg}')
    return merged


# ============================================================================
# STEP 4 — Compute months from M0 scan to surgery
# ============================================================================

def compute_months_diff(merged_df):
    """
    For each subject with both a scan date and surgery date, compute the
    number of calendar months elapsed from the M0 scan to surgery.

    Formula:
        months_diff = years * 12 + months + days / 30.0

    where (years, months, days) is the relativedelta between the two dates.
    A positive value means surgery happened AFTER the M0 scan (the expected case).
    A negative value would mean surgery was before the M0 scan (excluded later).

    Returns
    -------
    Same DataFrame with an added `months_diff_scan_to_surgery` column.
    """
    print('\n--- STEP 4: Computing months from M0 scan to surgery ---')

    def _months_diff(row):
        if pd.isna(row['surgery_date']) or pd.isna(row['date_of_scan']):
            return None
        diff = relativedelta(row['surgery_date'], row['date_of_scan'])
        return round(diff.years * 12 + diff.months + diff.days / 30.0, 2)

    merged_df = merged_df.copy()
    merged_df['months_diff_scan_to_surgery'] = merged_df.apply(_months_diff, axis=1)

    valid = merged_df['months_diff_scan_to_surgery'].notna()
    print(f'  Subjects with valid months_diff: {valid.sum()}')
    print(f'  Range: {merged_df.loc[valid, "months_diff_scan_to_surgery"].min():.1f} – '
          f'{merged_df.loc[valid, "months_diff_scan_to_surgery"].max():.1f} months')
    return merged_df


# ============================================================================
# STEP 5 — Assign nominal before/after surgery timepoints
# ============================================================================

def assign_nominal_timepoints(merged_df):
    """
    Based on `months_diff_scan_to_surgery`, assign the nominal study timepoints
    that bracket surgery:

      - `before_surg` : the last nominal timepoint whose scheduled scan date
                        falls BEFORE surgery (i.e., TIMEPOINT_MONTHS[tp] < months_diff)
      - `after_surg`  : the first nominal timepoint whose scheduled scan date
                        falls AFTER surgery (i.e., TIMEPOINT_MONTHS[tp] >= months_diff)

    The original add_timepoint_columns.py used hardcoded intervals; here we
    compute directly from TIMEPOINT_MONTHS for generality.

    Subjects with months_diff <= 0 (surgery before or on M0 scan) or
    months_diff > 60 (surgery after all follow-up) are assigned None/None.

    Returns
    -------
    Same DataFrame with added `before_surg` and `after_surg` columns.
    """
    print('\n--- STEP 5: Assigning nominal before/after surgery timepoints ---')

    def _assign(months_diff):
        if pd.isna(months_diff) or months_diff <= 0 or months_diff > 60:
            return None, None
        # Last timepoint strictly before surgery
        before = None
        for tp in reversed(TIMEPOINTS):
            if TIMEPOINT_MONTHS[tp] < months_diff:
                before = tp
                break
        # First timepoint at or after surgery
        after = None
        for tp in TIMEPOINTS:
            if TIMEPOINT_MONTHS[tp] >= months_diff:
                after = tp
                break
        return before, after

    df = merged_df.copy()
    df[['before_surg', 'after_surg']] = df['months_diff_scan_to_surgery'].apply(
        lambda x: pd.Series(_assign(x))
    )

    assigned = df['before_surg'].notna()
    print(f'  Subjects with assignable timepoints: {assigned.sum()}')
    print('  before_surg distribution:')
    print(df.loc[assigned, 'before_surg'].value_counts().to_string())
    print('  after_surg distribution:')
    print(df.loc[assigned, 'after_surg'].value_counts().to_string())
    return df


# ============================================================================
# STEP 6 — Verify which timepoints have actual MRI data
# ============================================================================

def verify_data_existence(df, perslice_dir, structure='cord'):
    """
    For each subject, check which timepoints actually have rows in the
    per-slice MRI metric CSVs. Then determine:

      - `before_surg_found` : the most recent timepoint BEFORE surgery that
                              has actual MRI data for this subject
      - `after_surg_found`  : the earliest timepoint AFTER surgery that
                              has actual MRI data for this subject

    This step is critical because nominal timepoints (before_surg, after_surg)
    may not have MRI data — the subject may have missed the session or data may
    not have been processed. Using only confirmed data avoids computing deltas
    against missing measurements.

    Data-existence lookup:
        File: T2w_ax_{structure}_metrics_perslice_{tp}_data.csv
        A subject is present if their sub-XXX ID appears in the 'Filename' column.

    Returns
    -------
    Same DataFrame with added `before_surg_found` and `after_surg_found` columns.
    """
    print('\n--- STEP 6: Verifying actual MRI data per subject per timepoint ---')

    # Build a set of subjects with data at each timepoint
    subjects_with_data = {}
    for tp in TIMEPOINTS:
        fpath = Path(perslice_dir) / f'T2w_ax_{structure}_metrics_perslice_{tp}_data.csv'
        if not fpath.exists():
            print(f'  Warning: {fpath.name} not found, skipping {tp}')
            subjects_with_data[tp] = set()
            continue
        csv = pd.read_csv(fpath, usecols=['Filename'])
        subs = set(csv['Filename'].str.extract(r'(sub-\d+)')[0].dropna().unique())
        subjects_with_data[tp] = subs
        print(f'  {tp}: {len(subs)} subjects with data')

    def _find_before(subject_bids, months_diff):
        """Most recent timepoint before surgery with actual data."""
        if pd.isna(months_diff) or months_diff <= 0:
            return None
        tps_before = [tp for tp in TIMEPOINTS if TIMEPOINT_MONTHS[tp] < months_diff]
        for tp in reversed(tps_before):  # most recent first
            if subject_bids in subjects_with_data.get(tp, set()):
                return tp
        return None

    def _find_after(subject_bids, months_diff):
        """Earliest timepoint after surgery with actual data."""
        if pd.isna(months_diff) or months_diff <= 0:
            return None
        tps_after = [tp for tp in TIMEPOINTS if TIMEPOINT_MONTHS[tp] >= months_diff]
        for tp in tps_after:  # earliest first
            if subject_bids in subjects_with_data.get(tp, set()):
                return tp
        return None

    df = df.copy()
    df['before_surg_found'] = df.apply(
        lambda r: _find_before(r['subject_bids'], r['months_diff_scan_to_surgery']),
        axis=1
    )
    df['after_surg_found'] = df.apply(
        lambda r: _find_after(r['subject_bids'], r['months_diff_scan_to_surgery']),
        axis=1
    )

    both_found = df['before_surg_found'].notna() & df['after_surg_found'].notna()
    print(f'\n  Subjects with both before and after confirmed: {both_found.sum()}')
    print('  before_surg_found distribution:')
    print(df.loc[both_found, 'before_surg_found'].value_counts().to_string())
    print('  after_surg_found distribution:')
    print(df.loc[both_found, 'after_surg_found'].value_counts().to_string())
    return df


# ============================================================================
# STEP 7 — Finalise and save
# ============================================================================

def finalise_and_save(df, output_path):
    """
    Select and rename output columns, then save to CSV.

    The output follows the same schema as baseline_surgery_dates_merged_clean.csv
    so it can be used as a drop-in replacement in downstream scripts.
    """
    print('\n--- STEP 7: Finalising output ---')

    # Drop the numeric 'subject' column before renaming subject_bids to 'subject'
    out = df.drop(columns=['subject']).rename(columns={'subject_bids': 'subject'}).copy()

    # Format dates as strings (YYYY-MM-DD) to match the original file
    out['date_of_scan'] = pd.to_datetime(out['date_of_scan']).dt.strftime('%Y-%m-%d')
    out['surgery_date'] = pd.to_datetime(out['surgery_date']).dt.strftime('%Y-%m-%d')

    out = out[[
        'subject', 'date_of_scan', 'surgery_date',
        'months_diff_scan_to_surgery',
        'before_surg', 'after_surg',
        'before_surg_found', 'after_surg_found',
    ]].sort_values('subject').reset_index(drop=True)

    if os.path.exists(output_path):
        raise FileExistsError(
            f'Output file already exists: {output_path}\n'
            f'Choose a different --output path to avoid overwriting.'
        )

    out.to_csv(output_path, index=False)
    print(f'  Saved: {output_path}')
    print(f'  Total rows: {len(out)}')
    print(f'  Rows with surgery date: {out["surgery_date"].notna().sum()}')
    print(f'  Rows with both _found timepoints: '
          f'{(out["before_surg_found"].notna() & out["after_surg_found"].notna()).sum()}')
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    parser = get_parser()
    args = parser.parse_args()

    print('=' * 70)
    print('Build surgery scan mapping')
    print('=' * 70)
    print(f'Clinical scores : {args.clinical_scores}')
    print(f'Participants    : {args.participants}')
    print(f'Per-slice dir   : {args.perslice_dir}')
    print(f'Output          : {args.output}')

    # Step 1: Extract surgery dates
    surgery_df = extract_surgery_dates(args.clinical_scores)

    # Step 2: Load M0 scan dates
    baseline_df = load_baseline_scan_dates(args.participants)

    # Step 3: Merge
    merged = merge_scan_and_surgery(baseline_df, surgery_df)

    # Step 4: Compute months_diff (M0 scan → surgery)
    merged = compute_months_diff(merged)

    # Step 5: Assign nominal before/after timepoints from months_diff
    merged = assign_nominal_timepoints(merged)

    # Step 6: Verify actual data existence per timepoint
    merged = verify_data_existence(merged, args.perslice_dir, structure=args.structure)

    # Step 7: Save (refuses to overwrite)
    finalise_and_save(merged, args.output)

    print('\n' + '=' * 70)
    print(f'Done. Output saved to: {args.output}')
    print('=' * 70)


if __name__ == '__main__':
    main()
