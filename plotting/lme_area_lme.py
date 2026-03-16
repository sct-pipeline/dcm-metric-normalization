"""
Analyze longitudinal relationship between baseline spinal cord area and mJOA scores using Linear Mixed-Effects Models.

This script uses Linear Mixed-Effects Models (LME) to:
1. Analyze the relationship between baseline area ('MEAN(area)'), separately for C2 and C3
2. Model longitudinal repeated measures of mJOA scores (['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth'])
3. Use random intercepts and slopes for time
4. Adjust for age, sex, myelopathy, MCL and aSCOR as fixed effects
5. Include an interaction term (area × Time) to capture dynamic effects

The script applies the same subject filtering as generate_figure_PAM50_multiple_subjects.py:
- Excludes subjects with severe myelopathy (mJOA < 12)
- Excludes subjects based on exclude file (if provided)
- Excludes subjects based on C2/C3 level file (if provided)
- Drops subjects with highest stenosis at C2/C3 or C3/C4
- Drops subjects with 4 stenosis levels

Example usage:
python plotting/lme_area_lme.py \
    -clinical-file data/clinical_scores.xlsx \
    -morphometrics-file data/morphometrics.csv \
    -ascor-file data/T2w_ax_aSCOR_metrics_perslice_PAM50.csv \
    -exclude-file etc/exclude.yml \
    -c2c3-file data/c2c3_levels.txt \
    -structure cord \
    -o results/lme_mjoa_area
"""

import os
import argparse

import numpy as np
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

# Import utilities from existing scripts
from generate_figure_PAM50_multiple_subjects import (
    read_clinical_file,
    read_morphometrics_file,
    merge_morphometrics_and_clinical_data,
    read_exclude_file_and_exclude_subjects,
    read_c2c3_file_and_apply_exclusions,
    drop_highest_stenosis
)


def get_parser():
    """
    Get parser for command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Analyze longitudinal mJOA-area relationship using Linear Mixed-Effects Models',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-clinical-file',
        type=str,
        required=True,
        help='Path to clinical data file (Excel format)'
    )
    parser.add_argument(
        '-morphometrics-file',
        type=str,
        required=True,
        help='Path to morphometrics CSV file containing area measurements'
    )
    parser.add_argument(
        '-ascor-file',
        type=str,
        default=None,
        help='Path to aSCOR metrics CSV file (optional)'
    )
    parser.add_argument('-exclude-file', required=False, type=str,
                        default='$HOME/code/dcm-metric-normalization/scripts/exclude_dcm-zurich.yml',
                        help="YAML file with subjects to exclude")
    parser.add_argument('-c2c3-file', required=False, type=str,
                        default='$HOME/code/dcm-metric-normalization/scripts/dcm-zurich_T2w_ax_ses-M0_canal_analysis.txt',
                        help="File with list of subjects to use C2 or C3 vert level.")
    parser.add_argument(
        '-structure',
        type=str,
        default='cord',
        choices=['cord', 'canal'],
        help='Structure to analyze (cord or canal)'
    )
    parser.add_argument(
        '-o', '--outdir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    return parser


def analyze_longitudinal_mjoa_area(subjects_df, path_ascor_file, output_dir, structure):
    """
    Analyse the relationship between baseline area ('MEAN(area)'), separately for C2 and C3, and longitudinal
    repeated measures of mJOA scores (['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']) usings
    a linear mixed-effects model with random intercepts and slopes for time. Area is used as fixed effects,
    with adjustments for age, sex, myelopathy, MCL and aSCOR (i.e., T2w_ax_aSCOR_metrics_perslice_PAM50.csv is also read).
    An interaction term (area × Time) is used to capture these dynamic effects in the model.
    In addition, to account for the correlation between mJOA measurements taken at different time points,
    an autoregressive structure of order 1 is used.

    :param subjects_df: pandas.DataFrame containing morphometric and clinical data
    :param path_ascor_file: str, path to the aSCOR metrics CSV file
    :param output_dir: str, directory to save analysis results
    :param structure: str, structure to analyze ('cord' or 'canal')
    :return: dict, dictionary containing model results for C2 and C3 levels
    """

    print("\n" + "="*80)
    print("LONGITUDINAL mJOA-AREA ANALYSIS")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load aSCOR data if provided
    df_ascor = None
    if path_ascor_file and os.path.exists(path_ascor_file):
        try:
            df_ascor = read_morphometrics_file(os.path.abspath(path_ascor_file))
            print(f"Loaded aSCOR data from: {path_ascor_file}")
        except Exception as e:
            print(f"Warning: Could not load aSCOR data: {e}")

    # Filter for baseline area at C2 and C3 levels only
    baseline_area_df = subjects_df[(subjects_df['VertLevel'].isin([2, 3])) & (subjects_df['session_id'] == 'ses-M0')].copy()
    baseline_ascor_df = df_ascor[(df_ascor['VertLevel'].isin([2, 3])) & (df_ascor['session_id'] == 'ses-M0')].copy() if df_ascor is not None else pd.DataFrame()

    # Prepare longitudinal mJOA data
    # mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth', 'total_mjoa_12mth']
    mjoa_columns = ['total_mjoa_BL', 'total_mjoa_6mth']
    time_points = [0, 6]#, 12]  # months

    results = {}

    # Analyze separately for C2 and C3
    # for level in [2, 3]:
    for level in [3]:
        level_name = f"C{level}"
        print(f"\n--- Analysis for {level_name} ---")

        # Get baseline area for this level - aggregate per participant per level
        level_data_area = baseline_area_df[baseline_area_df['VertLevel'] == level].copy()
        level_data_acor = baseline_ascor_df[baseline_ascor_df['VertLevel'] == level].copy() if not baseline_ascor_df.empty else pd.DataFrame()

        # Compute per-participant per-level mean for the area (similar to violin plot approach)
        level_grouped_area = level_data_area[['participant_id', 'VertLevel', 'MEAN(area)']].dropna().groupby(['participant_id', 'VertLevel'], as_index=False).mean()
        level_grouped_ascor = level_data_acor[['participant_id', 'VertLevel', 'aSCOR']].dropna().groupby(['participant_id', 'VertLevel'], as_index=False).mean() if not level_data_acor.empty else pd.DataFrame()

        # Merge back with clinical data (get one row per participant with baseline clinical data)
        clinical_cols = ['participant_id', 'age', 'sex', 'Myelopathy', 'MCL', 'therapeutic_decision'] + mjoa_columns
        available_clinical_cols = [col for col in clinical_cols if col in level_data_area.columns]
        clinical_data = level_data_area[available_clinical_cols].drop_duplicates('participant_id')

        # Merge grouped area with clinical data
        level_merged = level_grouped_area.merge(clinical_data, on='participant_id', how='inner')
        # Merge aSCOR data if available
        if not level_grouped_ascor.empty:
            level_merged = level_merged.merge(level_grouped_ascor, on='participant_id', how='left')

        # Create longitudinal dataset
        long_data = []

        for _, row in level_merged.iterrows():
            participant_id = row['participant_id']
            baseline_area = row['MEAN(area)']

            # Extract covariates (use baseline values)
            age = row.get('age', np.nan)
            sex = row.get('sex', 'unknown')
            myelopathy = row.get('Myelopathy', 'unknown')
            mcl = row.get('MCL', 'unknown')
            therapeutic_decision = row.get('therapeutic_decision', 'unknown')
            ascor_value = row.get('aSCOR', np.nan)

            # Create rows for each time point
            for time_idx, (mjoa_col, time_months) in enumerate(zip(mjoa_columns, time_points)):
                mjoa_score = row.get(mjoa_col, np.nan)

                if not pd.isna(mjoa_score):
                    long_data.append({
                        'participant_id': participant_id,
                        'time': time_months,
                        'time_categorical': f"M{time_months}",
                        'mjoa_score': mjoa_score,
                        'baseline_area': baseline_area,
                        'age': age,
                        'sex': sex,
                        'Myelopathy': myelopathy,
                        'mcl': mcl,
                        'therapeutic_decision': therapeutic_decision,
                        'ascor': ascor_value,
                        'level': level_name
                    })

        if not long_data:
            print(f"No longitudinal mJOA data available for {level_name}")
            continue

        # Convert to DataFrame
        long_df = pd.DataFrame(long_data)

        # Remove participants with insufficient data (need at least 2 time points)
        print(f"Number of participants before filtering: {long_df['participant_id'].nunique()}")
        participant_counts = long_df['participant_id'].value_counts()
        valid_participants = participant_counts[participant_counts >= 2].index
        long_df = long_df[long_df['participant_id'].isin(valid_participants)]
        print(f"Removed participants with less than 2 time points. Remaining participants: {len(valid_participants)}")

        print(f"Analysis dataset for {level_name}:")
        print(f"  - Participants: {long_df['participant_id'].nunique()}")
        print(f"  - Total observations: {len(long_df)}")
        print(f"  - Time points per participant: {long_df['participant_id'].value_counts().describe()}")

        # Center continuous variables
        long_df['baseline_area_c'] = long_df['baseline_area'] - long_df['baseline_area'].mean()
        # long_df['time_c'] = long_df['time'] - long_df['time'].mean()

        # Handle age centering
        age_valid = pd.to_numeric(long_df['age'], errors='coerce').notna()
        if age_valid.any():
            long_df['age_c'] = pd.to_numeric(long_df['age'], errors='coerce')
            long_df['age_c'] = long_df['age_c'] - long_df['age_c'].mean()

        # Handle aSCOR centering
        ascor_valid = pd.to_numeric(long_df['ascor'], errors='coerce').notna()
        if ascor_valid.any():
            long_df['ascor_c'] = pd.to_numeric(long_df['ascor'], errors='coerce')
            long_df['ascor_c'] = long_df['ascor_c'] - long_df['ascor_c'].mean()

        # Build model formula
        fixed_effects = ['baseline_area_c', 'time', 'baseline_area_c:time']

        # FIXED EFFECTS - in the main formula
        # Add covariates if available and have sufficient variation
        if 'age_c' in long_df.columns and pd.to_numeric(long_df['age_c'], errors='coerce').notna().any():
            fixed_effects.append('age_c')

        if long_df['sex'].nunique() > 1 and 'unknown' not in long_df['sex'].values:
            fixed_effects.append('C(sex)')

        if long_df['Myelopathy'].nunique() > 1 and 'unknown' not in long_df['Myelopathy'].values:
            fixed_effects.append('C(Myelopathy)')

        if long_df['mcl'].nunique() > 1 and 'unknown' not in long_df['mcl'].values:
            fixed_effects.append('C(mcl)')

        if 'therapeutic_decision' in long_df.columns and long_df['therapeutic_decision'].nunique() > 1:
            fixed_effects.append('C(therapeutic_decision)')

        # if 'ascor_c' in long_df.columns and pd.to_numeric(long_df['ascor_c'], errors='coerce').notna().any():
        #     fixed_effects.append('ascor_c')

        formula = f"mjoa_score ~ {' + '.join(fixed_effects)}"

        try:
            # Fit mixed-effects model with random intercepts and slopes
            print(f"\nFitting model: {formula}")
            print(f"Random effects: Random intercepts and slopes for time by participant")

            # Remove rows with missing values for the model
            model_df = long_df.dropna(subset=['mjoa_score'] +
                                    [col for col in ['baseline_area_c', 'time', 'age_c', 'ascor_c']
                                     if col in long_df.columns])

            if len(model_df) < 10:
                print(f"Insufficient data for modeling {level_name} (n={len(model_df)})")
                continue

            # RANDOM EFFECTS - specified separately in re_formula
            # Fit the mixed-effects model
            # Random intercepts and slopes for time
            model = MixedLM.from_formula(
                formula,
                data=model_df,
                groups=model_df["participant_id"],
                re_formula="1 + time"  # Random intercept (1) and slope (time)
                # - By allowing this intercept to vary across patients, the model lets each person start at a different disability level.
                # - Without a random intercept, the model would assume everyone starts at the same mJOA value, which is unrealistic.
                # - Allowing this slope to vary means some patients worsen quickly, some slowly, and some may barely change.
                # - Without a random slope, the model would force all patients to worsen at the same rate.
            )

            try:
                result = model.fit(method='lbfgs', maxiter=1000)
            except:
                # Fallback to simpler fitting method
                result = model.fit()

            # Extract results
            model_summary = {
                'level': level_name,
                'n_participants': model_df['participant_id'].nunique(),
                'n_observations': len(model_df),
                'formula': formula,
                'converged': result.converged,
                'log_likelihood': result.llf,
                'aic': result.aic,
                'bic': result.bic
            }

            # Extract coefficient results
            coef_results = []
            for param in result.params.index:
                coef_results.append({
                    'parameter': param,
                    'coefficient': result.params[param],
                    'std_error': result.bse[param],
                    'z_value': result.tvalues[param],
                    'p_value': result.pvalues[param],
                    'ci_lower': result.conf_int().loc[param, 0],
                    'ci_upper': result.conf_int().loc[param, 1]
                })

            model_summary['coefficients'] = pd.DataFrame(coef_results)

            # Print results
            print(f"\nModel Results for {level_name}:")
            print(f"  Converged: {result.converged}")
            print(f"  Participants: {model_summary['n_participants']}")
            print(f"  Observations: {model_summary['n_observations']}")
            print(f"  AIC: {result.aic:.2f}")
            print(f"  BIC: {result.bic:.2f}")

            print(f"\nFixed Effects:")
            for _, row in model_summary['coefficients'].iterrows():
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                print(f"  {row['parameter']}: β = {row['coefficient']:.4f} ± {row['std_error']:.4f}, "
                      f"p = {row['p_value']:.4f}{significance} "
                      f"[95% CI: {row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")

            # baseline_area_c - "Does area predict mJOA on average?" -- effect of baseline CSA on mJOA
            # time - "Does mJOA improve over time?" -- effect of time on mJOA
            # baseline_area_c:time - does baseline area modify the rate of mJOA change over time -- - larger area may slow disability progression

            # Main effect (baseline_area_c): Tests if area predicts starting point (baseline mJOA)
            # Interaction (baseline__area_c:time): Tests if area predicts rate of change (slope over time)

            # Group Var - The between-participant variance in baseline mJOA scores (1.01), indicating substantial individual differences in neurological function
            # Group x time Cov - The covariance between individual baselines and slopes, showing how baseline mJOA relates to rate of change over time
            # time Var - The between-participant variance in rates of mJOA change over time (0.0035), indicating some individuals improve faster than others

            # Save detailed results
            results_file = os.path.join(output_dir, f"mixed_effects_mjoa_{structure}_area_{level_name}.txt")
            with open(results_file, 'w') as f:
                f.write(f"Mixed-Effects Model Results: mJOA vs Area ({level_name})\n")
                f.write("="*60 + "\n\n")
                f.write(f"Model: {formula}\n")
                f.write(f"Random Effects: Random intercepts and slopes for time by participant\n\n")
                f.write(f"Sample Size:\n")
                f.write(f"  Participants: {model_summary['n_participants']}\n")
                f.write(f"  Observations: {model_summary['n_observations']}\n\n")
                f.write(f"Model Fit:\n")
                f.write(f"  Converged: {result.converged}\n")
                f.write(f"  Log-Likelihood: {result.llf:.4f}\n")
                f.write(f"  AIC: {result.aic:.2f}\n")
                f.write(f"  BIC: {result.bic:.2f}\n\n")
                f.write(str(result.summary()))

            # Save coefficient table
            coef_file = os.path.join(output_dir, f"mixed_effects_coefficients_{level_name}.csv")
            model_summary['coefficients'].to_csv(coef_file, index=False)

            # Save model data for further analysis
            data_file = os.path.join(output_dir, f"mixed_effects_data_{level_name}.csv")
            model_df.to_csv(data_file, index=False)

            results[level_name] = model_summary

            print(f"Results saved:")
            print(f"  - Detailed summary: {results_file}")
            print(f"  - Coefficients: {coef_file}")
            print(f"  - Model data: {data_file}")

        except Exception as e:
            print(f"Error fitting model for {level_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary report
    if results:
        summary_file = os.path.join(output_dir, "mixed_effects_summary_report.txt")
        with open(summary_file, 'w') as f:
            f.write("LONGITUDINAL mJOA-AREA ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")

            for level_name, result in results.items():
                f.write(f"{level_name} Results:\n")
                f.write(f"  Sample: {result['n_participants']} participants, {result['n_observations']} observations\n")
                f.write(f"  Model fit: AIC={result['aic']:.2f}, BIC={result['bic']:.2f}\n")

                # Key findings
                coef_df = result['coefficients']

                # Baseline area effect
                baseline_effect = coef_df[coef_df['parameter'] == 'baseline_area_c']
                if not baseline_effect.empty:
                    row = baseline_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Baseline area effect: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                # Time effect
                time_effect = coef_df[coef_df['parameter'] == 'time']
                if not time_effect.empty:
                    row = time_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Time effect: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                # Interaction effect
                interaction_effect = coef_df[coef_df['parameter'] == 'baseline_area_c:time']
                if not interaction_effect.empty:
                    row = interaction_effect.iloc[0]
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
                    f.write(f"  Area × Time interaction: β={row['coefficient']:.4f}, p={row['p_value']:.4f}{sig}\n")

                f.write("\n")

        print(f"\nSummary report saved to: {summary_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)

    return results


def main():
    """Main execution function."""

    args = get_parser().parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Read data
    print("Reading clinical data...")
    df_clinical, clinical_columns = read_clinical_file(os.path.abspath(args.clinical_file))

    print("Reading morphometric data...")
    df_morphometrics = read_morphometrics_file(os.path.abspath(args.morphometrics_file))

    # Merge clinical and morphometric data
    print("Merging clinical and morphometric data...")
    subjects_df = merge_morphometrics_and_clinical_data(df_morphometrics, df_clinical, clinical_columns)

    # Get number of unique subjects
    n_subjects = len(subjects_df['participant_id'].unique())
    print(f"Number of unique subjects: {n_subjects}")

    # ----
    # Apply the same filtering as in generate_figure_PAM50_multiple_subjects.py
    # ----

    # Exclude subjects based on exclude file
    if args.exclude_file:
        exclude_file = os.path.expandvars(args.exclude_file)
        subjects_df = read_exclude_file_and_exclude_subjects(subjects_df, exclude_file)

    # Apply C2/C3 level exclusions
    if args.c2c3_file:
        c2c3_file = os.path.expandvars(args.c2c3_file)
        subjects_df = read_c2c3_file_and_apply_exclusions(subjects_df, c2c3_file)

    # Drop highest stenosis (C2/C3, C3/C4) and num_of_stenosis == 4
    subjects_df = drop_highest_stenosis(subjects_df)

    print(f'Total number of unique subjects after filtering: {len(subjects_df["participant_id"].unique())}')

    # Run longitudinal mJOA-area analysis
    results = analyze_longitudinal_mjoa_area(
        subjects_df=subjects_df,
        path_ascor_file=args.ascor_file,
        output_dir=args.outdir,
        structure=args.structure
    )

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {args.outdir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

