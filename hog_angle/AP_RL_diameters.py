#
# Script for comparing AP and RL diameters measured using different methods (skimage.regionprops vs. HOG).
# Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4958/
# This script generates scatter plots with regression lines to visualize the correlation
# between different measurement techniques.
#
# You can use SCT's conda environment to run this script:
#       # Go to the SCT directory
#       cd $SCT_DIR
#       # Activate SCT conda environment
#       source ./python/etc/profile.d/conda.sh
#       conda activate venv_sct
#
# Example usage on a single subject:
#       python AP_RL_diameters.py -i /path/to/morphometrics_plotting.csv -o /path/to/output/morphometrics.png
#
# Authors: Jan Valosek
#

import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress


def get_parser():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create scatter plots with regression lines for AP and RL diameters measured using different methods.")
    parser.add_argument('-i', required=True, type=str,
                        help="Path to the CSV file with diameter measurements.")
    parser.add_argument('-o', required=True, type=str,
                        help="Path to save the output figure.")
    return parser


def plot_with_regression(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, ax: plt.Axes) -> None:
    """Plot scatter with regression line"""
    # Drop NaNs
    df = pd.DataFrame({xlabel: x, ylabel: y}).dropna()
    x_clean, y_clean = df[xlabel], df[ylabel]

    # Regression:
    #   slope -- Slope of the regression line
    #   intercept  -- Intercept of the regression line
    #   r -- The Pearson correlation coefficient
    slope, intercept, r, _, _ = linregress(x_clean, y_clean)
    reg_line = slope * x_clean + intercept

    mpl.rcParams['font.family'] = 'Arial'

    # Scatter
    ax.scatter(x_clean, y_clean, s=20, edgecolor='k', facecolor='gray', alpha=0.7)
    ax.plot(x_clean, reg_line, 'r-', linewidth=2)

    # Identity line
    min_val, max_val = min(x_clean.min(), y_clean.min()), max(x_clean.max(), y_clean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

    # Labels and text
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val*0.95, max_val*1.05)
    ax.set_ylim(min_val*0.95, max_val*1.05)
    ax.text(0.05, 0.95,
            f'$r$ = {r:.2f}\n$y$ = {slope:.2f}$x$ + {intercept:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9)

    # Style
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def main() -> None:
    """Read CSV and generate comparison plots for AP and RL diameters."""
    parser = get_parser()
    args = parser.parse_args()

    # Use command line arguments
    csv_path = args.i
    out_path = args.o

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    plot_with_regression(
        df['MEAN(diameter_AP)'],
        df['MEAN(diameter_AP_hog)'],
        'AP diameter (skimage.regionprops)',
        'AP diameter (HOG)',
        axes[0]
    )
    plot_with_regression(
        df['MEAN(diameter_RL)'],
        df['MEAN(diameter_RL_hog)'],
        'RL diameter (skimage.regionprops)',
        'RL diameter (HOG)',
        axes[1]
    )
    plt.tight_layout()

    # Save figure
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Figure saved: {out_path}")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
