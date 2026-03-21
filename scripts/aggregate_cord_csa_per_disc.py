#
# Aggregate spinal cord CSA per intervertebral disc for each subject.
#
# Discs are identified as the transition between two consecutive VertLevels.
# For each disc, the two slices flanking the transition are selected and the
# minimum CSA (MEAN(area)) is stored.
#
# Output structure:
#   rows    = subjects (Filename)
#   columns = discs C2/C3 ... C6/C7  (CSA value + two slice indices per disc)
#
# Usage example:
#   python aggregate_cord_csa_per_disc.py \
#       -i /path/to/T2w_ax_cord_metrics_perslice.csv \
#       -o /path/to/output_dir \
#       -exclude-file /path/to/exclude_dcm-zurich.yml
#
# Author: Jan Valosek

import argparse
import logging
import os
import re
import sys

import pandas as pd
import yaml

FNAME_LOG = 'log_aggregate_cord_csa_per_disc.txt'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

# Disc name → (VertLevel of the inferior vertebra, VertLevel of the superior vertebra)
# Slices run I→S so VertLevel decreases with increasing slice index.
# The disc sits between the last slice of the inferior level and the first slice
# of the superior level.
DISCS = {
    'C2/C3': (3, 2),
    'C3/C4': (4, 3),
    'C4/C5': (5, 4),
    'C5/C6': (6, 5),
    'C6/C7': (7, 6),
}

# Metrics to extract from the selected slice (the one with minimum CSA)
METRICS = [
    'MEAN(area)',
    'MEAN(diameter_AP)',
    'MEAN(diameter_RL)',
    'MEAN(eccentricity)',
    'MEAN(solidity)',
]

# Short column name suffix for each metric
METRIC_SUFFIX = {
    'MEAN(area)':         'area',
    'MEAN(diameter_AP)':  'diameter_AP',
    'MEAN(diameter_RL)':  'diameter_RL',
    'MEAN(eccentricity)': 'eccentricity',
    'MEAN(solidity)':     'solidity',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate spinal cord CSA at each intervertebral disc level."
    )
    parser.add_argument(
        '-i',
        required=True,
        metavar='<file_path>',
        help="Path to T2w_ax_cord_metrics_perslice.csv"
    )
    parser.add_argument(
        '-o',
        required=True,
        metavar='<dir_path>',
        help="Output directory"
    )
    parser.add_argument(
        '-exclude-file',
        required=False,
        type=str,
        default=os.path.expandvars('$HOME/code/dcm-metric-normalization/scripts/exclude_dcm-zurich.yml'),
        help="YAML file with subjects to exclude (uses the 't2_ax' key)"
    )
    return parser


def extract_sub_ses(filename):
    """Return (subject, session) extracted from a NIfTI file path.

    Examples
    --------
    .../sub-002/ses-M0/... → ('sub-002', 'ses-M0')
    .../sub-002/...        → ('sub-002', None)
    """
    sub_match = re.search(r'/(sub-\d+)/', filename)
    ses_match = re.search(r'/(ses-\w+)/', filename)
    subject = sub_match.group(1) if sub_match else None
    session = ses_match.group(1) if ses_match else None
    return subject, session


def load_exclude_list(exclude_file):
    """Return a set of (subject, session) tuples to exclude.

    Entries without a session (e.g. 'sub-156') set session to None and will
    match all sessions for that subject.
    """
    exclude_set = set()
    if not exclude_file or not os.path.isfile(exclude_file):
        logger.warning(f"Exclude file not found: {exclude_file}")
        return exclude_set

    with open(exclude_file, 'r') as f:
        data = yaml.safe_load(f)

    entries = data.get('t2_ax', [])
    for entry in entries:
        entry = str(entry).strip()
        # Remove inline comments (everything after '#')
        entry = entry.split('#')[0].strip()
        parts = entry.split('/')
        subject = parts[0].strip()
        session = parts[1].strip() if len(parts) > 1 else None
        exclude_set.add((subject, session))
        logger.info(f"  Excluding: subject={subject}, session={session}")

    return exclude_set


def should_exclude(subject, session, exclude_set):
    """Return True if the subject/session combination should be excluded."""
    # Exact match (subject + session)
    if (subject, session) in exclude_set:
        return True
    # Subject-only entry (session=None) → exclude all sessions
    if (subject, None) in exclude_set:
        return True
    return False


def find_disc_slices(df_sub):
    """For a single subject's DataFrame (sorted by slice I→S), find the two
    slices that bracket each target disc.

    Slice selection is based on minimum CSA (MEAN(area)). All other metrics
    are taken from that same selected slice.

    Returns a dict keyed by disc name with value
        {'slice_inf': int, 'slice_sup': int, 'slice_used': int,
         '<metric_suffix>': float, ...}
    or None when the disc transition is not found in the data.
    """
    df_sub = df_sub.sort_values('Slice (I->S)').reset_index(drop=True)
    results = {}

    for disc, (level_inf, level_sup) in DISCS.items():
        rows_inf = df_sub[df_sub['VertLevel'] == level_inf]
        rows_sup = df_sub[df_sub['VertLevel'] == level_sup]

        if rows_inf.empty or rows_sup.empty:
            results[disc] = None
            continue

        # Last slice of the inferior level (highest slice index within that level)
        row_last_inf = rows_inf.loc[rows_inf['Slice (I->S)'].idxmax()]
        # First slice of the superior level (lowest slice index within that level)
        row_first_sup = rows_sup.loc[rows_sup['Slice (I->S)'].idxmin()]

        slice_inf = int(row_last_inf['Slice (I->S)'])
        slice_sup = int(row_first_sup['Slice (I->S)'])

        # Select the slice with the smaller CSA
        csa_inf = row_last_inf['MEAN(area)']
        csa_sup = row_first_sup['MEAN(area)']
        row_selected = row_last_inf if csa_inf <= csa_sup else row_first_sup
        slice_used = int(row_selected['Slice (I->S)'])

        entry = {'slice_inf': slice_inf, 'slice_sup': slice_sup, 'slice_used': slice_used}

        # Extract all metrics from the selected slice
        for col in METRICS:
            suffix = METRIC_SUFFIX[col]
            entry[suffix] = row_selected[col]

        # Compute compression ratio from the selected slice
        ap = row_selected['MEAN(diameter_AP)']
        rl = row_selected['MEAN(diameter_RL)']
        entry['compression_ratio'] = ap / rl if rl != 0 else float('nan')

        results[disc] = entry

    return results


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)

    # Set up file logging
    log_path = os.path.join(args.o, FNAME_LOG)
    fh = logging.FileHandler(log_path)
    logging.root.addHandler(fh)

    logger.info(f"Reading: {args.i}")
    df = pd.read_csv(args.i)

    # Drop rows without a valid VertLevel
    df = df.dropna(subset=['VertLevel'])
    df['VertLevel'] = df['VertLevel'].astype(int)

    # Extract subject and session from Filename
    df[['subject', 'session']] = df['Filename'].apply(
        lambda fn: pd.Series(extract_sub_ses(fn))
    )

    logger.info(f"\nLoading exclude list from: {args.exclude_file}")
    exclude_set = load_exclude_list(args.exclude_file)

    # Flag rows to exclude
    df['_exclude'] = df.apply(
        lambda row: should_exclude(row['subject'], row['session'], exclude_set),
        axis=1
    )
    n_before = df['Filename'].nunique()
    excluded_subjects = df[df['_exclude']]['Filename'].unique()
    df = df[~df['_exclude']].drop(columns=['_exclude'])
    n_after = df['Filename'].nunique()
    logger.info(f"Subjects before exclusion: {n_before}, after: {n_after} "
                f"(excluded {n_before - n_after})")
    for fn in excluded_subjects:
        logger.info(f"  Excluded: {fn}")

    # Build output rows
    records = []
    for filename, df_sub in df.groupby('Filename'):
        subject, session = extract_sub_ses(filename)
        disc_data = find_disc_slices(df_sub)

        out_row = {'Filename': os.path.basename(filename), 'subject': subject, 'session': session}
        for disc in DISCS:
            info = disc_data.get(disc)
            if info is None:
                for suffix in list(METRIC_SUFFIX.values()) + ['compression_ratio']:
                    out_row[f'{disc}_{suffix}'] = float('nan')
                out_row[f'{disc}_slice_inf'] = float('nan')
                out_row[f'{disc}_slice_sup'] = float('nan')
                out_row[f'{disc}_slice_used'] = float('nan')
                logger.warning(f"  {subject}/{session}: disc {disc} not found")
            else:
                for suffix in METRIC_SUFFIX.values():
                    out_row[f'{disc}_{suffix}'] = info[suffix]
                out_row[f'{disc}_compression_ratio'] = info['compression_ratio']
                out_row[f'{disc}_slice_inf'] = info['slice_inf']
                out_row[f'{disc}_slice_sup'] = info['slice_sup']
                out_row[f'{disc}_slice_used'] = info['slice_used']

        records.append(out_row)

    out_df = pd.DataFrame(records)

    # Reorder columns: Filename, subject, session, then per disc: metrics, slices
    col_order = ['Filename', 'subject', 'session']
    metric_suffixes = list(METRIC_SUFFIX.values()) + ['compression_ratio']
    for disc in DISCS:
        for suffix in metric_suffixes:
            col_order.append(f'{disc}_{suffix}')
        col_order += [f'{disc}_slice_inf', f'{disc}_slice_sup', f'{disc}_slice_used']
    out_df = out_df[col_order]

    # Cast slice columns to nullable integer (preserves NaN without promoting to float)
    slice_cols = [c for c in out_df.columns if c.endswith(('_slice_inf', '_slice_sup', '_slice_used'))]
    out_df[slice_cols] = out_df[slice_cols].astype('Int64')

    out_path = os.path.join(args.o, 'T2w_ax_cord_CSA_per_disc.csv')
    out_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved {len(out_df)} subjects → {out_path}")

    # Print a quick summary (area only)
    logger.info("\nCSA summary (mean ± std across subjects):")
    for disc in DISCS:
        col = out_df[f'{disc}_area'].dropna()
        logger.info(f"  {disc}: {col.mean():.2f} ± {col.std():.2f}  (n={len(col)})")


if __name__ == '__main__':
    main()