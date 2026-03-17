"""
Analyze and visualize clinical trajectories stratified by myelopathy status and therapeutic decision.

This script uses Linear Mixed-Effects Models (LME) to:
1. Compare clinical outcomes between groups defined by two variables:
   - Hyperintensity: T2w+ vs T2w-
   - Therapeutic decision: conservative vs operative
2. Account for baseline covariates: spinal cord area, age, sex, MCL level, single vs multi stenosis
3. Model both baseline and 6-month timepoints with random effects for participants

Example usage:
python plotting/lme_myelopathy_therapeutic_decision_two_variables.py \
    -clinical-file data/clinical_scores.xlsx \
    -morphometrics-file data/morphometrics.csv \
    -participants-to-use data/participants.txt \
    -o figures/myelopathy_therapeutic_lme
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Import utilities from existing scripts
from utils import read_clinical_file, read_morphometrics_file

# Plot fonts
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 16

# Clinical scores to analyze
CLINICAL_SCORES = {
    'mJOA': {
        'baseline_col': 'total_mjoa_BL',
        'followup_col': 'total_mjoa_6mth',
        'y_label': 'mJOA',
        'ylim': (11.5, 18.5)
    },
    'Motor_UE': {
        'baseline_col': 'motor_dysfunction_UE_bl_BL',
        'followup_col': 'motor_dysfunction_UE_6mth_6mth',
        'y_label': 'Motor Dysfunction UE',
        'ylim': None
    },
    'Motor_LE': {
        'baseline_col': 'motor_dysfunction_LE_bl_BL',
        'followup_col': 'motor_dysfunction_LE_6mth_6mth',
        'y_label': 'Motor Dysfunction LE',
        'ylim': None
    },
    'Sensory_UE': {
        'baseline_col': 'sensory_dysfunction_UE_bl_BL',
        'followup_col': 'sensory_dysfunction_UE_6mth_6mth',
        'y_label': 'Sensory Dysfunction UE',
        'ylim': None
    },
    'Sphincter': {
        'baseline_col': 'sphincter_dysfunction_bl_BL',
        'followup_col': 'sphincter_dysfunction_6mth_6mth',
        'y_label': 'Sphincter Dysfunction',
        'ylim': None
    },
}

# Group colors - T2w- = green, T2w+ = red
GROUP_COLORS = {
    'conservative_T2w-': '#2ca02c',    # green
    'conservative_T2w+': '#d62728',    # red
    'operative_T2w+': '#d62728',       # red
    'operative_T2w-': '#2ca02c',       # green
}

# Line styles - conservative = dashed, operative = solid
GROUP_LINE_STYLES = {
    'conservative_T2w-': '--',    # dashed
    'conservative_T2w+': '--',    # dashed
    'operative_T2w+': '-',        # solid
    'operative_T2w-': '-',        # solid
}

# Group order for plotting (matches dodge offset order)
GROUP_ORDER = ['conservative_T2w-', 'conservative_T2w+', 'operative_T2w-', 'operative_T2w+']

# Group labels for display
GROUP_LABELS = {
    'conservative_T2w-': 'Conservative T2w-',
    'conservative_T2w+': 'Conservative T2w+',
    'operative_T2w+': 'Operative T2w+',
    'operative_T2w-': 'Operative T2w-',
}


def get_parser():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze clinical trajectories with LME models stratified by myelopathy and therapeutic decision.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-clinical-file',
        required=True,
        type=str,
        help='Path to Excel file with clinical scores'
    )
    parser.add_argument(
        '-morphometrics-file',
        required=True,
        type=str,
        help='Path to CSV file with morphometric data (spinal cord area)'
    )
    parser.add_argument(
        '-participants-to-use',
        required=True,
        type=str,
        help='Path to text file with participant IDs to include (one ID per line)'
    )
    parser.add_argument(
        '-sessions', required=False, type=int,
        help='Number of sessions to include. '
             '2 sessions: baseline and 6 month follow up. '
             '3 sessions: baseline, 6 month follow up, and 12 month follow up.',
        choices=[2, 3], default=3)
    parser.add_argument(
        '-o', '--outdir',
        required=True,
        type=str,
        help='Output directory for figures and results'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[2, 3],
        help='Vertebral level to use for baseline spinal cord area (default: 3 for C3)'
    )
    return parser


def assign_group(row):
    """
    Assign participant to one of four plotting groups based on therapeutic decision and myelopathy status.

    :param row: DataFrame row with 'therapeutic_decision' and 'myelopathy' columns
    :return: group label or None if cannot be classified
    """
    therapeutic = row.get('therapeutic_decision', None)
    myelopathy = row.get('myelopathy', None)

    # Conservative groups
    if therapeutic == 'conservative':
        if myelopathy == 'yes':
            return 'conservative_T2w+'
        elif myelopathy == 'no':
            return 'conservative_T2w-'
    # Operative groups
    elif therapeutic == 'operative':
        if myelopathy == 'yes':
            return 'operative_T2w+'
        elif myelopathy == 'no':
            return 'operative_T2w-'

    return None


def log_print(message, log_file=None):
    """
    Print message to console and optionally to log file.

    :param message: Message to print
    :param log_file: Optional file handle to write to
    """
    print(message)
    if log_file is not None:
        log_file.write(message + '\n')
        log_file.flush()


def get_covariates_suffix(formula):
    """
    Extract covariates from formula and create a suffix string for filenames.

    :param formula: Formula string used in the LME model
    :return: String suffix with covariate names (e.g., "_cov-area_age_sex")
    """
    # Extract the right side of the formula (after ~)
    if '~' not in formula:
        return ""

    right_side = formula.split('~')[1].strip()

    # List of covariate terms we want to track (excluding group, time, and interactions)
    covariates = []

    if 'baseline_area_c' in right_side:
        covariates.append('area')
    if 'age_c' in right_side:
        covariates.append('age')
    if 'C(sex)' in right_side:
        covariates.append('sex')
    if 'C(maximum_stenosis)' in right_side:
        covariates.append('MCL')
    if 'C(stenosis)' in right_side:
        covariates.append('stenosis')

    if covariates:
        return f"_cov-{'_'.join(covariates)}"
    else:
        return "_cov-none"


def prepare_longitudinal_data(df_clinical, df_morphometrics, level=3, sessions=2):
    """
    Prepare longitudinal dataset for LME analysis.

    Reshapes data to long format with:
    - Each row = one observation (participant × timepoint)
    - Baseline covariates: spinal cord area at specified level, age, sex, myelopathy, MCL, single_vs_multi_stenosis
    - Outcome: clinical score at each timepoint

    :param df_clinical: Clinical data with baseline and 6-month scores
    :param df_morphometrics: Morphometric data with spinal cord area
    :param level: Vertebral level for baseline area (2 or 3)
    :param sessions: Number of sessions to include (2 or 3). If 3, will include 12-month follow-up if available.
    :return: Dictionary of DataFrames, one per clinical score
    """

    # Filter morphometrics for baseline and specified level
    df_morph_baseline = df_morphometrics[
        (df_morphometrics['session_id'] == 'ses-M0') &
        (df_morphometrics['VertLevel'] == level)
    ].copy()

    # Compute mean area per participant at this level
    df_area = df_morph_baseline.groupby('participant_id')['MEAN(area)'].mean().reset_index()
    df_area.columns = ['participant_id', f'baseline_area_C{level}']

    # Merge with clinical data
    df_merged = df_clinical.merge(df_area, on='participant_id', how='left')

    # Assign plotting groups
    df_merged['group'] = df_merged.apply(assign_group, axis=1)

    # Filter out participants without valid factor assignments
    df_merged = df_merged[
        df_merged['group'].notna() &
        df_merged['myelopathy'].notna() &
        df_merged['therapeutic_decision'].notna()
    ].copy()

    print(f"\nGroup distribution:")
    print(df_merged['group'].value_counts())

    # Prepare longitudinal data for each clinical score
    long_data_dict = {}

    for score_name, score_info in CLINICAL_SCORES.items():
        baseline_col = score_info['baseline_col']
        followup_col = score_info['followup_col']

        # Check if columns exist
        if baseline_col not in df_merged.columns or followup_col not in df_merged.columns:
            print(f"Warning: Skipping {score_name} - columns not found")
            continue

        # Create long format
        long_data = []

        for _, row in df_merged.iterrows():
            participant_id = row['participant_id']
            group = row['group']

            # Baseline covariates
            baseline_area = row.get(f'baseline_area_C{level}', np.nan)
            age = row.get('age', np.nan)
            sex = row.get('sex', 'unknown')
            myelopathy = row.get('myelopathy', 'unknown')
            therapeutic_decision = row.get('therapeutic_decision', 'unknown')
            maximum_stenosis = row.get('maximum_stenosis', 'unknown')
            stenosis = row.get('single_vs_multi_stenosis', 'unknown')

            # Baseline timepoint
            score_bl = row.get(baseline_col, np.nan)
            if not pd.isna(score_bl):
                long_data.append({
                    'participant_id': participant_id,
                    'group': group,
                    'myelopathy': myelopathy,
                    'therapeutic_decision': therapeutic_decision,
                    'time': 0,  # baseline
                    'time_label': 'Baseline',
                    'score': float(score_bl),
                    'baseline_area': baseline_area,
                    'age': age,
                    'sex': sex,
                    'maximum_stenosis': maximum_stenosis,
                    'stenosis': stenosis
                })

            # 6-month timepoint
            score_6m = row.get(followup_col, np.nan)
            if not pd.isna(score_6m):
                long_data.append({
                    'participant_id': participant_id,
                    'group': group,
                    'myelopathy': myelopathy,
                    'therapeutic_decision': therapeutic_decision,
                    'time': 1,  # 6 months
                    'time_label': '6-month',
                    'score': float(score_6m),
                    'baseline_area': baseline_area,
                    'age': age,
                    'sex': sex,
                    'maximum_stenosis': maximum_stenosis,
                    'stenosis': stenosis
                })

            # 12-month timepoint (if sessions=3)
            if sessions == 3:
                score_12m = row.get(followup_col.replace('6mth', '12mth'), np.nan)
                if not pd.isna(score_12m):
                    long_data.append({
                        'participant_id': participant_id,
                        'group': group,
                        'myelopathy': myelopathy,
                        'therapeutic_decision': therapeutic_decision,
                        'time': 2,  # 12 months
                        'time_label': '12-month',
                        'score': float(score_12m),
                        'baseline_area': baseline_area,
                        'age': age,
                        'sex': sex,
                        'maximum_stenosis': maximum_stenosis,
                        'stenosis': stenosis
                    })

        if not long_data:
            print(f"Warning: No data for {score_name}")
            continue

        df_long = pd.DataFrame(long_data)

        # Remove participants with only one timepoint
        participant_counts = df_long['participant_id'].value_counts()
        valid_participants = participant_counts[participant_counts >= 2].index
        df_long = df_long[df_long['participant_id'].isin(valid_participants)]

        print(f"\n{score_name}:")
        print(f"  Participants with both timepoints: {df_long['participant_id'].nunique()}")
        print(f"  Total observations: {len(df_long)}")

        long_data_dict[score_name] = df_long

    return long_data_dict


def fit_lme_model(df_long, score_name, log_file=None):
    """
    Fit Linear Mixed-Effects Model for a clinical score.

    Model specification:
    - Fixed effects: myelopathy, therapeutic decision, time, all interactions,
      baseline_area, age, sex, MCL, stenosis
    - Random effects: random intercept and slope for time by participant

    :param df_long: Long-format DataFrame
    :param score_name: Name of clinical score being analyzed
    :param log_file: Optional file handle to write log outputs
    :return: Fitted model result, cleaned data, and formula string
    """

    # Center continuous variables
    df_model = df_long.copy()

    # Set reference categories
    df_model['myelopathy'] = pd.Categorical(
        df_model['myelopathy'],
        categories=['no', 'yes'],
        ordered=False
    )
    df_model['therapeutic_decision'] = pd.Categorical(
        df_model['therapeutic_decision'],
        categories=['conservative', 'operative'],
        ordered=False
    )

    # Center baseline area
    if 'baseline_area' in df_model.columns:
        df_model['baseline_area_c'] = df_model['baseline_area'] - df_model['baseline_area'].mean()

    # Center age
    age_numeric = pd.to_numeric(df_model['age'], errors='coerce')
    if age_numeric.notna().any():
        df_model['age_c'] = age_numeric - age_numeric.mean()

    # Build formula with fixed effects
    # C(): categorical variable
    # _c: centered continuous variable
    fixed_terms = [
        'C(myelopathy)',
        'C(therapeutic_decision)',
        'time',
        # 'C(myelopathy):C(therapeutic_decision)',  # less important
        # 'C(myelopathy):time',     # less important
        'C(therapeutic_decision):time',
        'C(myelopathy):C(therapeutic_decision):time'
    ]

    # Add covariates if available
    if 'baseline_area_c' in df_model.columns and df_model['baseline_area_c'].notna().any():
        fixed_terms.append('baseline_area_c')

    if 'age_c' in df_model.columns and df_model['age_c'].notna().any():
        fixed_terms.append('age_c')

    if df_model['sex'].nunique() > 1 and 'unknown' not in df_model['sex'].values:
        fixed_terms.append('C(sex)')

    # MCL
    if df_model['maximum_stenosis'].nunique() > 1 and 'unknown' not in df_model['maximum_stenosis'].values:
        fixed_terms.append('C(maximum_stenosis)')

    # single vs multi stenosis
    if df_model['stenosis'].nunique() > 1 and 'unknown' not in df_model['stenosis'].values:
        fixed_terms.append('C(stenosis)')

    formula = f"score ~ {' + '.join(fixed_terms)}"

    # Remove rows with missing values
    required_cols = ['score', 'time', 'myelopathy', 'therapeutic_decision']
    optional_cols = ['baseline_area_c', 'age_c']

    drop_cols = required_cols + [col for col in optional_cols if col in df_model.columns]
    df_model = df_model.dropna(subset=drop_cols)

    if len(df_model) < 10:
        log_print(f"Insufficient data for {score_name} (n={len(df_model)})", log_file)
        return None, None, None

    log_print(f"\n{'='*80}", log_file)
    log_print(f"Fitting LME Model for {score_name}", log_file)
    log_print(f"{'='*80}", log_file)
    log_print(f"Formula: {formula}", log_file)
    log_print(f"Random effects: 1 + time | participant_id", log_file)
    log_print(f"Participants: {df_model['participant_id'].nunique()}", log_file)
    log_print(f"Observations: {len(df_model)}", log_file)

    # Fit the model
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)

            model = MixedLM.from_formula(
                formula,
                data=df_model,
                groups=df_model['participant_id'],
                re_formula='1 + time'  # Random intercept and slope
            )

            result = model.fit(method='lbfgs', maxiter=1000)

    except Exception as e:
        log_print(f"Error fitting model for {score_name}: {e}", log_file)
        return None, None, None

    # Print results
    log_print(f"\nModel converged: {result.converged}", log_file)
    log_print(f"AIC: {result.aic:.2f}", log_file)
    log_print(f"BIC: {result.bic:.2f}", log_file)
    log_print(f"Log-Likelihood: {result.llf:.2f}", log_file)

    log_print(f"\nFixed Effects:", log_file)
    log_print("-" * 80, log_file)
    for param in result.params.index:
        coef = result.params[param]
        se = result.bse[param]
        pval = result.pvalues[param]
        ci_lower, ci_upper = result.conf_int().loc[param]

        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

        log_print(f"{param:40s}: β = {coef:7.4f} ± {se:6.4f}, p = {pval:7.4f}{sig:3s} "
              f"[95% CI: {ci_lower:7.4f}, {ci_upper:7.4f}]", log_file)

    return result, df_model, formula


def create_trajectory_plot(df_long, lme_result, score_name, score_info, output_dir, covariates_suffix="", log_file=None):
    """
    Create trajectory plot with LME-fitted lines.

    Shows:
    - Individual participant trajectories (thin lines)
    - Group-averaged trajectories from LME model (thick lines)

    :param df_long: Long-format DataFrame
    :param lme_result: Fitted LME model result
    :param score_name: Name of clinical score
    :param score_info: Dictionary with score metadata
    :param output_dir: Output directory for figures
    :param covariates_suffix: Suffix indicating covariates used
    :param log_file: Optional file handle to write log outputs
    """

    log_print(f"\nCreating trajectory plot for {score_name}...", log_file)
    log_print(f"  Model converged: {lme_result.converged if lme_result else 'N/A'}", log_file)

    mpl.rcParams['font.family'] = 'Arial'

    sessions = sorted(df_long['time'].unique())
    width = max(5, int(2 * len(sessions)))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4))

    # Calculate dodge offsets for each combination
    treatment_strata = ['conservative', 'operative']
    myelopathy_strata = ['no', 'yes']

    n_combinations = len(treatment_strata) * len(myelopathy_strata)
    dodge_width = 0.04
    dodge_offsets = {}
    combo_idx = 0
    for treatment in treatment_strata:
        for myelopathy in myelopathy_strata:
            offset = (combo_idx - (n_combinations - 1) / 2) * (dodge_width / max(1, n_combinations - 1))
            dodge_offsets[(myelopathy, treatment)] = offset
            combo_idx += 1

    # Plot individual trajectories
    for group in GROUP_ORDER:
        group_data = df_long[df_long['group'] == group]

        if group_data.empty:
            continue

        if 'T2w-' in group:
            myelopathy = 'no'
        else:
            myelopathy = 'yes'

        if 'conservative' in group:
            treatment = 'conservative'
        else:
            treatment = 'operative'

        offset = dodge_offsets[(myelopathy, treatment)]

        for participant_id in group_data['participant_id'].unique():
            participant_data = group_data[group_data['participant_id'] == participant_id].sort_values('time')

            if len(participant_data) >= 2:
                ax.plot(
                    participant_data['time'] + 1 + offset,
                    participant_data['score'],
                    color=GROUP_COLORS[group],
                    linestyle=GROUP_LINE_STYLES[group],
                    alpha=0.3,
                    linewidth=0.3,
                    zorder=3
                )

    # Plot LME-fitted trajectories
    if lme_result is None:
        raise ValueError(f"Cannot plot {score_name}: LME model result is None")

    if not lme_result.converged:
        warnings.warn(f"LME model for {score_name} did not fully converge. "
                      f"Plotting trajectories anyway, but interpret results with caution.",
                      UserWarning)
        print(f"  ⚠ WARNING: Model did not fully converge for {score_name}")

    time_points = np.array([0, 1])
    plot_time_points = np.array([1, 2])

    params = lme_result.params

    log_print(f"\nAvailable model parameters for {score_name}:", log_file)
    for param_name in params.index:
        log_print(f"  {param_name}: {params[param_name]:.4f}", log_file)

    if 'Intercept' not in params.index:
        raise KeyError(f"Missing 'Intercept' parameter in LME model for {score_name}")
    if 'time' not in params.index:
        raise KeyError(f"Missing 'time' parameter in LME model for {score_name}")

    baseline = params['Intercept']
    time_effect = params['time']

    log_print(f"\nBaseline intercept: {baseline:.4f}", log_file)
    log_print(f"Time effect: {time_effect:.4f}", log_file)

    # Factor-specific coefficients
    myelopathy_baseline = params.get('C(myelopathy)[T.yes]', 0)
    treatment_baseline = params.get('C(therapeutic_decision)[T.operative]', 0)
    myelopathy_treatment_baseline = params.get(
        'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]', 0
    )
    myelopathy_time = params.get('C(myelopathy)[T.yes]:time', 0)
    treatment_time = params.get('C(therapeutic_decision)[T.operative]:time', 0)
    myelopathy_treatment_time = params.get(
        'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]:time', 0
    )

    log_print(f"\nFactor-specific coefficients:", log_file)
    log_print(f"  myelopathy baseline effect: {myelopathy_baseline:.4f}", log_file)
    log_print(f"  treatment baseline effect: {treatment_baseline:.4f}", log_file)
    log_print(f"  myelopathy:treatment baseline effect: {myelopathy_treatment_baseline:.4f}", log_file)
    log_print(f"  myelopathy:time effect: {myelopathy_time:.4f}", log_file)
    log_print(f"  treatment:time effect: {treatment_time:.4f}", log_file)
    log_print(f"  myelopathy:treatment:time effect: {myelopathy_treatment_time:.4f}", log_file)

    fitted_any_group = False
    for group in GROUP_ORDER:
        group_data = df_long[df_long['group'] == group]

        if group_data.empty:
            log_print(f"\nWarning: No data for group {group}, skipping", log_file)
            continue

        if 'T2w-' in group:
            myelopathy = 'no'
        else:
            myelopathy = 'yes'

        if 'conservative' in group:
            treatment = 'conservative'
        else:
            treatment = 'operative'

        baseline_adjustment = 0
        slope_adjustment = 0

        if myelopathy == 'yes':
            baseline_adjustment += myelopathy_baseline
            slope_adjustment += myelopathy_time

        if treatment == 'operative':
            baseline_adjustment += treatment_baseline
            slope_adjustment += treatment_time

        if myelopathy == 'yes' and treatment == 'operative':
            baseline_adjustment += myelopathy_treatment_baseline
            slope_adjustment += myelopathy_treatment_time

        predictions = np.zeros(len(time_points))
        for i, t in enumerate(time_points):
            predictions[i] = baseline + baseline_adjustment + (time_effect + slope_adjustment) * t

        offset = dodge_offsets[(myelopathy, treatment)]
        n_subjects = group_data[group_data['time'] == 1]['participant_id'].nunique()

        log_print(f"\nPlotting {group}:", log_file)
        log_print(f"  n={n_subjects}", log_file)
        log_print(f"  Baseline prediction: {predictions[0]:.4f}", log_file)
        log_print(f"  6-month prediction: {predictions[1]:.4f}", log_file)
        log_print(f"  Color: {GROUP_COLORS[group]}", log_file)
        log_print(f"  Line style: {GROUP_LINE_STYLES[group]}", log_file)
        log_print(f"  Dodge offset: {offset:.3f}", log_file)

        label = f"{GROUP_LABELS[group]} (n={n_subjects})"
        ax.plot(
            plot_time_points + offset,
            predictions,
            color=GROUP_COLORS[group],
            linestyle=GROUP_LINE_STYLES[group],
            linewidth=2,
            marker='o',
            markersize=3,
            label=label,
            zorder=6
        )

        fitted_any_group = True

    if not fitted_any_group:
        raise ValueError(f"Failed to plot any group trajectories for {score_name}")

    ax.set_ylabel(score_info['y_label'], fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', '6-month'])

    if score_info['ylim'] is not None:
        ax.set_ylim(score_info['ylim'])

    color_handles = [
        Line2D([0], [0], color='#2ca02c', lw=2, marker='o', markersize=3, label='T2w-'),
        Line2D([0], [0], color='#d62728', lw=2, marker='o', markersize=3, label='T2w+')
    ]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='conservative'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='operative')
    ]

    leg1 = ax.legend(handles=color_handles,
                     loc='center left',
                     bbox_to_anchor=(0.1, 0.1),
                     fontsize=TICK_FONT_SIZE,
                     frameon=True)
    leg1.get_frame().set_alpha(0.85)
    leg1.get_frame().set_facecolor('white')
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=style_handles,
                     loc='center right',
                     bbox_to_anchor=(0.9, 0.1),
                     fontsize=TICK_FONT_SIZE,
                     frameon=True)
    leg2.get_frame().set_alpha(0.85)
    leg2.get_frame().set_facecolor('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    fname = os.path.join(output_dir, f'trajectory_lme_{score_name}{covariates_suffix}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved trajectory plot: {fname}")
    plt.close()


def save_model_results(lme_results, output_dir, covariates_suffix=""):
    """
    Save LME model results to CSV files.

    :param lme_results: Dictionary of LME results (score_name -> result)
    :param output_dir: Output directory
    :param covariates_suffix: Suffix indicating covariates used
    """

    all_results = []

    for score_name, result in lme_results.items():
        if result is None:
            continue

        for param in result.params.index:
            all_results.append({
                'score': score_name,
                'parameter': param,
                'coefficient': result.params[param],
                'std_error': result.bse[param],
                'z_value': result.tvalues[param],
                'p_value': result.pvalues[param],
                'ci_lower': result.conf_int().loc[param, 0],
                'ci_upper': result.conf_int().loc[param, 1],
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.llf,
                'converged': result.converged
            })

    if all_results:
        df_results = pd.DataFrame(all_results)
        fname = os.path.join(output_dir, f'lme_results_all_scores{covariates_suffix}.csv')
        df_results.to_csv(fname, index=False)
        print(f"\nSaved model results to: {fname}")

        fname_summary = os.path.join(output_dir, f'lme_results_summary{covariates_suffix}.txt')
        with open(fname_summary, 'w') as f:
            for score_name, result in lme_results.items():
                if result is None:
                    continue

                f.write(f"\n{'='*80}\n")
                f.write(f"{score_name}\n")
                f.write(f"{'='*80}\n")
                f.write(str(result.summary()))
                f.write('\n\n')

        print(f"Saved detailed summary to: {fname_summary}")


def create_comparison_table(lme_results, output_dir, covariates_suffix=""):
    """
    Create a comparison table of predicted values at 6 months.

    :param lme_results: Dictionary of LME results
    :param output_dir: Output directory
    :param covariates_suffix: Suffix indicating covariates used
    """

    comparison_data = []

    for score_name, result in lme_results.items():
        if result is None:
            continue

        params = result.params

        baseline = params.get('Intercept', 0)
        time_effect = params.get('time', 0)
        myelopathy_baseline = params.get('C(myelopathy)[T.yes]', 0)
        treatment_baseline = params.get('C(therapeutic_decision)[T.operative]', 0)
        myelopathy_treatment_baseline = params.get(
            'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]', 0
        )
        myelopathy_time = params.get('C(myelopathy)[T.yes]:time', 0)
        treatment_time = params.get('C(therapeutic_decision)[T.operative]:time', 0)
        myelopathy_treatment_time = params.get(
            'C(myelopathy)[T.yes]:C(therapeutic_decision)[T.operative]:time', 0
        )

        conservative_t2w_minus_6m = baseline + time_effect
        conservative_t2w_plus_6m = baseline + myelopathy_baseline + time_effect + myelopathy_time
        operative_t2w_minus_6m = baseline + treatment_baseline + time_effect + treatment_time
        operative_t2w_plus_6m = (
            baseline + myelopathy_baseline + treatment_baseline +
            myelopathy_treatment_baseline + time_effect +
            myelopathy_time + treatment_time + myelopathy_treatment_time
        )

        comparison_data.append({
            'score': score_name,
            'conservative_T2w-_6m_predicted': conservative_t2w_minus_6m,
            'conservative_T2w+_6m_predicted': conservative_t2w_plus_6m,
            'operative_T2w+_6m_predicted': operative_t2w_plus_6m,
            'operative_T2w-_6m_predicted': operative_t2w_minus_6m,
            'diff_op_T2w+_vs_cons_T2w-': operative_t2w_plus_6m - conservative_t2w_minus_6m,
            'diff_op_T2w-_vs_cons_T2w-': operative_t2w_minus_6m - conservative_t2w_minus_6m,
            'diff_cons_T2w+_vs_cons_T2w-': conservative_t2w_plus_6m - conservative_t2w_minus_6m,
            'diff_op_T2w+_vs_op_T2w-': operative_t2w_plus_6m - operative_t2w_minus_6m,
            'diff_op_T2w+_vs_cons_T2w+': operative_t2w_plus_6m - conservative_t2w_plus_6m,
            'diff_op_T2w-_vs_cons_T2w+': operative_t2w_minus_6m - conservative_t2w_plus_6m,
        })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        fname = os.path.join(output_dir, f'group_comparison_6months{covariates_suffix}.csv')
        df_comparison.to_csv(fname, index=False, float_format='%.3f')
        print(f"\nSaved group comparison table to: {fname}")

        print(f"\n{'='*80}")
        print("Group Comparisons at 6 Months (LME-predicted values)")
        print(f"{'='*80}")
        print(df_comparison.to_string(index=False))


def main():
    """Main execution function."""

    args = get_parser().parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # 2 sessions: baseline and 6 month follow up
    # 3 sessions: baseline, 6 month and 12 month follow up
    num_of_sessions = args.sessions

    # Read data
    print("Reading clinical data...")
    df_clinical, _ = read_clinical_file(os.path.abspath(args.clinical_file), sessions=num_of_sessions)

    print("Reading morphometric data...")
    df_morphometrics = read_morphometrics_file(os.path.abspath(args.morphometrics_file))

    # Read participant IDs to include
    print("Reading participant list...")
    with open(args.participants_to_use, 'r') as f:
        participant_ids = [line.strip() for line in f if line.strip()]

    # Filter data
    df_clinical = df_clinical[df_clinical['participant_id'].isin(participant_ids)].copy()
    df_morphometrics = df_morphometrics[df_morphometrics['participant_id'].isin(participant_ids)].copy()

    # ----
    # Logging
    # ----
    print(f'Total number of unique subjects in clinical file for analysis: {len(df_clinical["participant_id"].unique())}')
    print(f'Mean age: {df_clinical["age"].mean():.2f} ± {df_clinical["age"].std():.2f}')
    print(f'Sex distribution: {df_clinical["sex"].value_counts().to_dict()}')
    print(f'Therapeutic decision distribution: {df_clinical["therapeutic_decision"].value_counts().to_dict()}')

    # Remap myelopathy values
    if 'myelopathy' not in df_clinical.columns and 'Myelopathy' in df_clinical.columns:
        df_clinical['myelopathy'] = df_clinical['Myelopathy']
    df_clinical['myelopathy'] = df_clinical['myelopathy'].map({0: 'no', 1: 'yes', '0': 'no', '1': 'yes', 'no': 'no', 'yes': 'yes'})

    print(f'Myelopathy distribution: {df_clinical["myelopathy"].value_counts().to_dict()}')
    print(f"Myelopathy distribution by therapeutic decision:")
    myelo_therapeutic_dist = df_clinical.groupby('therapeutic_decision')['myelopathy'].value_counts().unstack(fill_value=0)
    print(myelo_therapeutic_dist)
    no_vals = myelo_therapeutic_dist.get('no', pd.Series(dtype=int))
    yes_vals = myelo_therapeutic_dist.get('yes', pd.Series(dtype=int))

    groups = ['conservative', 'operative']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(groups, no_vals, label="T2w-", color="green")
    ax.bar(groups, yes_vals, bottom=no_vals, label="T2w+", color="red")
    ax.set_ylabel("Number of subjects", fontsize=LABEL_FONT_SIZE+3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=LABEL_FONT_SIZE+3)
    ax.legend(title="T2w hyperintensity", fontsize=LABEL_FONT_SIZE+3, title_fontsize=LABEL_FONT_SIZE+3)
    for i, (n_no, n_yes) in enumerate(zip(no_vals, yes_vals)):
        ax.text(i, n_no / 2, str(n_no), ha="center", va="center", fontsize=16)
        ax.text(i, n_no + n_yes / 2, str(n_yes), ha="center", va="center", fontsize=16)
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(os.path.join(args.outdir, 'myelopathy_distribution_by_therapeutic_decision.png'), dpi=300)
    plt.close()

    print(f"\nIncluded {len(participant_ids)} participants")
    print(f"Clinical data: {len(df_clinical)} rows")
    print(f"Morphometric data: {len(df_morphometrics)} rows")

    # Prepare longitudinal data
    print(f"\nPreparing longitudinal data (using C{args.level} baseline area)...")
    long_data_dict = prepare_longitudinal_data(df_clinical, df_morphometrics, level=args.level, sessions=num_of_sessions)

    if not long_data_dict:
        print("Error: No data available for analysis")
        return

    # Fit LME models and create plots
    lme_results = {}
    covariates_suffix = ""

    for score_name, df_long in long_data_dict.items():
        if score_name == 'mJOA':
            result, df_model, formula = fit_lme_model(df_long, score_name, log_file=None)
            if result is not None:
                covariates_suffix = get_covariates_suffix(formula)
                lme_results[score_name] = result

                log_file_path = os.path.join(args.outdir, f'lme_analysis_log{covariates_suffix}.txt')

                with open(log_file_path, 'w') as log_file:
                    log_print(f"LME Analysis Log", log_file)
                    log_print(f"{'='*80}", log_file)
                    log_print(f"Analysis started at C{args.level} vertebral level", log_file)
                    log_print(f"Output directory: {args.outdir}", log_file)
                    log_print(f"Covariates: {covariates_suffix}", log_file)
                    log_print(f"{'='*80}\n", log_file)

                    result, df_model, formula = fit_lme_model(df_long, score_name, log_file)

                    create_trajectory_plot(
                        df_model,
                        result,
                        score_name,
                        CLINICAL_SCORES[score_name],
                        args.outdir,
                        covariates_suffix,
                        log_file
                    )

                print(f"\nLog file saved to: {log_file_path}")
            else:
                print(f"\nWarning: Model fitting failed for {score_name}")

    if lme_results:
        save_model_results(lme_results, args.outdir, covariates_suffix)
        create_comparison_table(lme_results, args.outdir, covariates_suffix)
    else:
        print("\nWarning: No models were successfully fitted")
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {args.outdir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
