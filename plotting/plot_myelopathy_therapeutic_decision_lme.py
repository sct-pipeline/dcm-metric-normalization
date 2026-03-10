"""
Analyze and visualize clinical trajectories stratified by myelopathy status and therapeutic decision.

This script uses Linear Mixed-Effects Models (LME) to:
1. Compare clinical outcomes between four groups:
   - Conservative T2w+ (conservative treatment with myelopathy)
   - Conservative T2w- (conservative treatment without myelopathy)
   - Operative T2w+ (operative with myelopathy)
   - Operative T2w- (operative without myelopathy)
2. Account for baseline covariates: spinal cord area, age, sex, myelopathy, MCL level, single vs multi stenosis
3. Model both baseline and 6-month timepoints with random effects for participants

Example usage:
python plotting/plot_myelopathy_therapeutic_decision_lme.py \
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
from generate_figure_PAM50_multiple_subjects import (
    read_clinical_file,
    read_morphometrics_file
)

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
    Assign participant to one of four groups based on therapeutic decision and myelopathy status.

    Groups:
    1. conservative_T2w-: Conservative treatment without myelopathy
    2. conservative_T2w+: Conservative treatment with myelopathy (T2w hyperintensity)
    3. operative_T2w+: Operative treatment with myelopathy (T2w hyperintensity)
    4. operative_T2w-: Operative treatment without myelopathy

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


def prepare_longitudinal_data(df_clinical, df_morphometrics, level=3):
    """
    Prepare longitudinal dataset for LME analysis.

    Reshapes data to long format with:
    - Each row = one observation (participant × timepoint)
    - Baseline covariates: spinal cord area at specified level, age, sex, myelopathy, MCL, single_vs_multi_stenosis
    - Outcome: clinical score at each timepoint

    :param df_clinical: Clinical data with baseline and 6-month scores
    :param df_morphometrics: Morphometric data with spinal cord area
    :param level: Vertebral level for baseline area (2 or 3)
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

    # Assign groups
    df_merged['group'] = df_merged.apply(assign_group, axis=1)

    # Filter out participants without group assignment
    df_merged = df_merged[df_merged['group'].notna()].copy()

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
            maximum_stenosis = row.get('maximum_stenosis', 'unknown')
            stenosis = row.get('single_vs_multi_stenosis', 'unknown')

            # Baseline timepoint
            score_bl = row.get(baseline_col, np.nan)
            if not pd.isna(score_bl):
                long_data.append({
                    'participant_id': participant_id,
                    'group': group,
                    'time': 1,  # baseline
                    'time_label': 'Baseline',
                    'score': float(score_bl),
                    'baseline_area': baseline_area,
                    'age': age,
                    'sex': sex,
                    'myelopathy': myelopathy,
                    'maximum_stenosis': maximum_stenosis,
                    'stenosis': stenosis
                })

            # 6-month timepoint
            score_6m = row.get(followup_col, np.nan)
            if not pd.isna(score_6m):
                long_data.append({
                    'participant_id': participant_id,
                    'group': group,
                    'time': 2,  # 6 months
                    'time_label': '6-month',
                    'score': float(score_6m),
                    'baseline_area': baseline_area,
                    'age': age,
                    'sex': sex,
                    'myelopathy': myelopathy,
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
    - Fixed effects: group, time, group × time interaction, baseline_area, age, sex, myelopathy, MCL, stenosis
    - Random effects: random intercept and slope for time by participant

    :param df_long: Long-format DataFrame
    :param score_name: Name of clinical score being analyzed
    :param log_file: Optional file handle to write log outputs
    :return: Fitted model result and cleaned data
    """

    # Center continuous variables
    df_model = df_long.copy()

    # Set reference category for group (ensure conservative_T2w- is the reference)
    df_model['group'] = pd.Categorical(
        df_model['group'],
        categories=['conservative_T2w-', 'conservative_T2w+', 'operative_T2w+', 'operative_T2w-'],
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
    fixed_terms = ['C(group)', 'time', 'C(group):time']  # Main effects and interaction

    # Add covariates if available
    if 'baseline_area_c' in df_model.columns and df_model['baseline_area_c'].notna().any():
        fixed_terms.append('baseline_area_c')

    if 'age_c' in df_model.columns and df_model['age_c'].notna().any():
        fixed_terms.append('age_c')

    if df_model['sex'].nunique() > 1 and 'unknown' not in df_model['sex'].values:
        fixed_terms.append('C(sex)')

    # if df_model['myelopathy'].nunique() > 1 and 'unknown' not in df_model['myelopathy'].values:
    #     fixed_terms.append('C(myelopathy)')
    #
    # MCL
    if df_model['maximum_stenosis'].nunique() > 1 and 'unknown' not in df_model['maximum_stenosis'].values:
        fixed_terms.append('C(maximum_stenosis)')

    # single vs multi stenosis
    if df_model['stenosis'].nunique() > 1 and 'unknown' not in df_model['stenosis'].values:
        fixed_terms.append('C(stenosis)')

    formula = f"score ~ {' + '.join(fixed_terms)}"

    # Remove rows with missing values
    required_cols = ['score', 'time', 'group']
    optional_cols = ['baseline_area_c', 'age_c']

    drop_cols = required_cols + [col for col in optional_cols if col in df_model.columns]
    df_model = df_model.dropna(subset=drop_cols)

    if len(df_model) < 10:
        log_print(f"Insufficient data for {score_name} (n={len(df_model)})", log_file)
        return None, None

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
        return None, None

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

    return result, df_model


def create_trajectory_plot(df_long, lme_result, score_name, score_info, output_dir, log_file=None):
    """
    Create trajectory plot with LME-fitted lines.

    Shows:
    - Individual participant trajectories (thin lines)
    - Group-averaged trajectories from LME model (thick lines)
    - 95% confidence intervals (shaded)

    :param df_long: Long-format DataFrame
    :param lme_result: Fitted LME model result
    :param score_name: Name of clinical score
    :param score_info: Dictionary with score metadata
    :param output_dir: Output directory for figures
    :param log_file: Optional file handle to write log outputs
    """

    log_print(f"\nCreating trajectory plot for {score_name}...", log_file)
    log_print(f"  Model converged: {lme_result.converged if lme_result else 'N/A'}", log_file)

    mpl.rcParams['font.family'] = 'Arial'

    sessions = sorted(df_long['time'].unique())
    width = max(5, int(2 * len(sessions)))
    fig, ax = plt.subplots(1, 1, figsize=(width, 4))

    # Calculate dodge offsets for each group to prevent overlap
    # Define stratification variables
    # Loop order: treatment (conservative, operative), then myelopathy (no, yes)
    # This creates the visual grouping: cons T2w-, cons T2w+, op T2w+, op T2w-
    treatment_strata = ['conservative', 'operative']
    myelopathy_strata = ['no', 'yes']  # T2w- (no), T2w+ (yes)

    # Calculate dodge offsets for each combination
    n_combinations = len(treatment_strata) * len(myelopathy_strata)
    dodge_width = 0.04  # Total width for dodging
    dodge_offsets = {}
    combo_idx = 0
    for treatment in treatment_strata:
        for myelopathy in myelopathy_strata:
            # Center the offsets around 0
            offset = (combo_idx - (n_combinations - 1) / 2) * (dodge_width / max(1, n_combinations - 1))
            dodge_offsets[(myelopathy, treatment)] = offset
            combo_idx += 1

    # Plot individual trajectories
    for group in GROUP_ORDER:
        group_data = df_long[df_long['group'] == group]

        if group_data.empty:
            continue

        # Map group to myelopathy and treatment combination
        if 'T2w-' in group:
            myelopathy = 'no'
        else:
            myelopathy = 'yes'

        if 'conservative' in group:
            treatment = 'conservative'
        else:
            treatment = 'operative'

        # Get dodge offset for this combination
        offset = dodge_offsets[(myelopathy, treatment)]

        # Individual trajectories (thin, semi-transparent)
        for participant_id in group_data['participant_id'].unique():
            participant_data = group_data[group_data['participant_id'] == participant_id].sort_values('time')

            if len(participant_data) >= 2:
                ax.plot(
                    participant_data['time'] + offset,
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

    time_points = np.array([1, 2])

    # Extract coefficients from the model
    params = lme_result.params

    # Debug: print all available parameters
    log_print(f"\nAvailable model parameters for {score_name}:", log_file)
    for param_name in params.index:
        log_print(f"  {param_name}: {params[param_name]:.4f}", log_file)

    # Check for required parameters
    if 'Intercept' not in params.index:
        raise KeyError(f"Missing 'Intercept' parameter in LME model for {score_name}")
    if 'time' not in params.index:
        raise KeyError(f"Missing 'time' parameter in LME model for {score_name}")

    baseline = params['Intercept']
    time_effect = params['time']

    log_print(f"\nBaseline intercept: {baseline:.4f}", log_file)
    log_print(f"Time effect: {time_effect:.4f}", log_file)

    # Group-specific coefficients
    group_coeffs = {
        'conservative_T2w-': {'baseline': 0, 'time': 0},  # Reference group
        'conservative_T2w+': {
            'baseline': params.get('C(group)[T.conservative_T2w+]', 0),
            'time': params.get('C(group)[T.conservative_T2w+]:time', 0)
        },
        'operative_T2w+': {
            'baseline': params.get('C(group)[T.operative_T2w+]', 0),
            'time': params.get('C(group)[T.operative_T2w+]:time', 0)
        },
        'operative_T2w-': {
            'baseline': params.get('C(group)[T.operative_T2w-]', 0),
            'time': params.get('C(group)[T.operative_T2w-]:time', 0)
        }
    }

    log_print(f"\nGroup-specific coefficients:", log_file)
    for group, coeffs in group_coeffs.items():
        log_print(f"  {group}: baseline_effect={coeffs['baseline']:.4f}, time_effect={coeffs['time']:.4f}", log_file)

    fitted_any_group = False
    for group in GROUP_ORDER:
        group_data = df_long[df_long['group'] == group]

        if group_data.empty:
            log_print(f"\nWarning: No data for group {group}, skipping", log_file)
            continue

        # Calculate predictions manually from coefficients
        # Prediction at mean covariates (all centered covariates = 0)
        group_baseline_effect = group_coeffs[group]['baseline']
        group_time_effect = group_coeffs[group]['time']

        predictions = np.zeros(len(time_points))
        for i, t in enumerate(time_points):
            predictions[i] = baseline + group_baseline_effect + (time_effect + group_time_effect) * t

        # Map group to myelopathy and treatment combination
        if 'T2w-' in group:
            myelopathy = 'no'
        else:
            myelopathy = 'yes'

        if 'conservative' in group:
            treatment = 'conservative'
        else:
            treatment = 'operative'

        # Get dodge offset for this combination
        offset = dodge_offsets[(myelopathy, treatment)]

        # Count unique participants at 6-month
        n_subjects = group_data[group_data['time'] == 6]['participant_id'].nunique()

        log_print(f"\nPlotting {group}:", log_file)
        log_print(f"  n={n_subjects}", log_file)
        log_print(f"  Baseline prediction: {predictions[0]:.4f}", log_file)
        log_print(f"  6-month prediction: {predictions[1]:.4f}", log_file)
        log_print(f"  Color: {GROUP_COLORS[group]}", log_file)
        log_print(f"  Line style: {GROUP_LINE_STYLES[group]}", log_file)
        log_print(f"  Dodge offset: {offset:.3f}", log_file)

        # Plot fitted line with dodge offset applied to x-coordinates
        label = f"{GROUP_LABELS[group]} (n={n_subjects})"
        ax.plot(
            time_points + offset,
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

    # Customize plot
    # ax.set_xlabel('Time (months)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(score_info['y_label'], fontsize=LABEL_FONT_SIZE)
    # ax.set_title(f'{score_info["y_label"]} Trajectories by Group', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)

    # Set x-axis
    # ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', '6-month'])

    # Set y-axis limits if specified
    if score_info['ylim'] is not None:
        ax.set_ylim(score_info['ylim'])

    # Build separate legends: one for colors (T2w status) and one for line styles (treatment type)
    color_handles = [
        Line2D([0], [0], color='#2ca02c', lw=2, marker='o', markersize=3, label='T2w-'),
        Line2D([0], [0], color='#d62728', lw=2, marker='o', markersize=3, label='T2w+')
    ]
    style_handles = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='conservative'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='operative')
    ]

    # Place legends inside the axes to avoid cropping
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

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    fname = os.path.join(output_dir, f'trajectory_lme_{score_name}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"\nSaved trajectory plot: {fname}")
    plt.close()


def save_model_results(lme_results, output_dir):
    """
    Save LME model results to CSV files.

    :param lme_results: Dictionary of LME results (score_name -> result)
    :param output_dir: Output directory
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
        fname = os.path.join(output_dir, 'lme_results_all_scores.csv')
        df_results.to_csv(fname, index=False)
        print(f"\nSaved model results to: {fname}")

        # Also save summary statistics
        fname_summary = os.path.join(output_dir, 'lme_results_summary.txt')
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


def create_comparison_table(lme_results, output_dir):
    """
    Create a comparison table of group differences at 6 months.

    Extracts and compares the main group effects and time × group interactions.

    :param lme_results: Dictionary of LME results
    :param output_dir: Output directory
    """

    comparison_data = []

    for score_name, result in lme_results.items():
        if result is None:
            continue

        # Extract group effects at 6 months
        # This involves combining intercept + time + group + group:time terms

        params = result.params

        # Get baseline intercept (conservative_T2w- group at time 0, the reference)
        baseline = params.get('Intercept', 0)

        # Time effect (change from baseline to 6m for conservative_T2w- group)
        time_effect = params.get('time', 0)

        # Group effects (difference from conservative_T2w- at baseline)
        group_cons_t2w_plus = params.get('C(group)[T.conservative_T2w+]', 0)
        group_op_t2w_plus = params.get('C(group)[T.operative_T2w+]', 0)
        group_op_t2w_minus = params.get('C(group)[T.operative_T2w-]', 0)

        # Interaction effects (additional change over time for each group)
        interact_cons_t2w_plus = params.get('C(group)[T.conservative_T2w+]:time', 0)
        interact_op_t2w_plus = params.get('C(group)[T.operative_T2w+]:time', 0)
        interact_op_t2w_minus = params.get('C(group)[T.operative_T2w-]:time', 0)

        # Calculate predicted values at 6 months for each group
        conservative_t2w_minus_6m = baseline + time_effect * 6
        conservative_t2w_plus_6m = baseline + group_cons_t2w_plus + (time_effect + interact_cons_t2w_plus) * 6
        operative_t2w_plus_6m = baseline + group_op_t2w_plus + (time_effect + interact_op_t2w_plus) * 6
        operative_t2w_minus_6m = baseline + group_op_t2w_minus + (time_effect + interact_op_t2w_minus) * 6

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
        fname = os.path.join(output_dir, 'group_comparison_6months.csv')
        df_comparison.to_csv(fname, index=False, float_format='%.3f')
        print(f"\nSaved group comparison table to: {fname}")

        # Print to console
        print(f"\n{'='*80}")
        print("Group Comparisons at 6 Months (LME-predicted values)")
        print(f"{'='*80}")
        print(df_comparison.to_string(index=False))


def main():
    """Main execution function."""

    args = get_parser().parse_args()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Read data
    print("Reading clinical data...")
    df_clinical, _ = read_clinical_file(os.path.abspath(args.clinical_file))

    print("Reading morphometric data...")
    df_morphometrics = read_morphometrics_file(os.path.abspath(args.morphometrics_file))

    # Read participant IDs to include
    print("Reading participant list...")
    with open(args.participants_to_use, 'r') as f:
        participant_ids = [line.strip() for line in f if line.strip()]

    # Filter data
    df_clinical = df_clinical[df_clinical['participant_id'].isin(participant_ids)].copy()
    df_morphometrics = df_morphometrics[df_morphometrics['participant_id'].isin(participant_ids)].copy()

    print(f"\nIncluded {len(participant_ids)} participants")
    print(f"Clinical data: {len(df_clinical)} rows")
    print(f"Morphometric data: {len(df_morphometrics)} rows")

    # Remap myelopathy values
    df_clinical['myelopathy'] = df_clinical['myelopathy'].map({0: 'no', 1: 'yes', '0': 'no', '1': 'yes'})

    # Prepare longitudinal data
    print(f"\nPreparing longitudinal data (using C{args.level} baseline area)...")
    long_data_dict = prepare_longitudinal_data(df_clinical, df_morphometrics, level=args.level)

    if not long_data_dict:
        print("Error: No data available for analysis")
        return

    # Create log file
    log_file_path = os.path.join(args.outdir, 'lme_analysis_log.txt')
    with open(log_file_path, 'w') as log_file:
        log_print(f"LME Analysis Log", log_file)
        log_print(f"{'='*80}", log_file)
        log_print(f"Analysis started at C{args.level} vertebral level", log_file)
        log_print(f"Output directory: {args.outdir}", log_file)
        log_print(f"{'='*80}\n", log_file)

        # Fit LME models and create plots
        lme_results = {}

        for score_name, df_long in long_data_dict.items():
            if score_name == 'mJOA':
                # Fit model
                result, df_model = fit_lme_model(df_long, score_name, log_file)
                if result is not None:
                    lme_results[score_name] = result

                    # Create trajectory plot
                    create_trajectory_plot(
                        df_model,
                        result,
                        score_name,
                        CLINICAL_SCORES[score_name],
                        args.outdir,
                        log_file
                    )

        # Save results
        if lme_results:
            save_model_results(lme_results, args.outdir)
            create_comparison_table(lme_results, args.outdir)
        else:
            print("\nWarning: No models were successfully fitted")

    print(f"\nLog file saved to: {log_file_path}")
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {args.outdir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

