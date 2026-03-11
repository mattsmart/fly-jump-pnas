"""
Analyze within-fly posterior correlations between alpha and beta.

This script calculates the correlation between alpha and beta in the posterior
for each individual fly, then visualizes the distribution across genotypes and days.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import sys

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


# Import settings
from settings import DIR_FITS, DIR_OUTPUT

##########################
# Configuration
##########################
SAVE_FIGURES = True

# Load the fitted data to get fly information
df_file = DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv'
df = pd.read_csv(df_file)

# Filter out excluded flies (same as in other scripts)
excluded_flies = [31, 121, 80, 76]
df = df[~((df['genotype'] == 'KK') & (df['fly_id'].isin(excluded_flies)))]

print(f"Loaded {len(df)} observations from {df['genotype'].nunique()} genotypes")
print(f"Genotypes: {df['genotype'].unique()}")
print(f"Days: {sorted(df['day'].unique())}")

##########################
# Calculate within-fly correlations
##########################

def calculate_within_fly_correlation(genotype, day, fly_id):
    """
    Calculate correlation between alpha and beta in the posterior for a single fly.

    Parameters
    ----------
    genotype : str
        Genotype (KK or GD)
    day : int
        Day (7, 14, or 21)
    fly_id : int
        Fly ID

    Returns
    -------
    float
        Pearson correlation coefficient between alpha and beta posterior samples
    """
    # Load posterior draws
    draws_file = DIR_FITS + os.sep + f'{genotype}_day{day}_3d_draws.npz'

    if not os.path.exists(draws_file):
        return np.nan

    draws_data = np.load(draws_file, allow_pickle=True)

    # Find the fly in the draws
    # The draws are stored with fly_id_X as key
    fly_key = f'fly_id_{fly_id}'

    if fly_key not in draws_data:
        return np.nan

    # Extract posterior samples for this fly
    # Shape should be (n_draws, 3) where columns are [alpha, beta, p0]
    posterior_samples = draws_data[fly_key]

    if len(posterior_samples) < 2:
        return np.nan

    alpha_samples = posterior_samples[:, 0]
    beta_samples = posterior_samples[:, 1]

    # Calculate correlation
    corr, _ = pearsonr(alpha_samples, beta_samples)

    return corr


# Calculate correlations for all flies
print("\nCalculating within-fly posterior correlations...")

correlations = []

for idx, row in df.iterrows():
    genotype = row['genotype']
    day = row['day']
    fly_id = row['fly_id']

    corr = calculate_within_fly_correlation(genotype, day, fly_id)

    correlations.append({
        'genotype': genotype,
        'day': day,
        'fly_id': fly_id,
        'correlation': corr
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(df)} flies...")

df_corr = pd.DataFrame(correlations)

# Remove any NaN correlations
df_corr = df_corr.dropna(subset=['correlation'])

print(f"\nCalculated correlations for {len(df_corr)} flies")

##########################
# Summary statistics
##########################

print("\n" + "="*70)
print("WITHIN-FLY POSTERIOR CORRELATION SUMMARY")
print("="*70)

for genotype in ['GD', 'KK']:
    print(f"\n{genotype}:")
    for day in [7, 14, 21]:
        subset = df_corr[(df_corr['genotype'] == genotype) & (df_corr['day'] == day)]
        if len(subset) > 0:
            print(f"  Day {day}:")
            print(f"    Mean:   {subset['correlation'].mean():7.3f}")
            print(f"    Median: {subset['correlation'].median():7.3f}")
            print(f"    Std:    {subset['correlation'].std():7.3f}")
            print(f"    Range:  [{subset['correlation'].min():6.3f}, {subset['correlation'].max():6.3f}]")
            print(f"    N:      {len(subset)}")

##########################
# Visualization
##########################

print("\nCreating visualizations...")

# Color scheme
geno_colors = {
    'GD': '#66C2A5',  # Teal
    'KK': '#8DA0CB'   # Purple
}

# Figure: 2 rows (genotypes) × 3 columns (days)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
days = [7, 14, 21]
genotypes = ['KK', 'GD']

for row_idx, genotype in enumerate(genotypes):
    for col_idx, day in enumerate(days):
        ax = axes[row_idx, col_idx]

        # Get data for this genotype-day combination
        subset = df_corr[(df_corr['genotype'] == genotype) & (df_corr['day'] == day)]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            continue

        # Histogram
        ax.hist(subset['correlation'], bins=20, color=geno_colors[genotype],
               alpha=0.7, edgecolor='black', linewidth=0.5)

        # Add vertical lines for mean and median
        mean_val = subset['correlation'].mean()
        median_val = subset['correlation'].median()

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean={mean_val:.2f}', zorder=10)
        ax.axvline(median_val, color='black', linestyle='-', linewidth=2,
                  label=f'Median={median_val:.2f}', zorder=10)

        # Labels and title
        ax.set_xlabel(r'$\rho(\alpha, \beta)$ (within-fly)', fontsize=10)
        ax.set_ylabel('Number of flies', fontsize=10)

        # Title for top row only
        if row_idx == 0:
            ax.set_title(f'Day {day}', fontsize=12, fontweight='bold')

        # Genotype label on left
        if col_idx == 0:
            ax.text(-0.3, 0.5, genotype, transform=ax.transAxes,
                   fontsize=13, fontweight='bold', va='center', ha='center',
                   rotation=90)

        # Legend
        ax.legend(fontsize=8, loc='upper left')

        # Grid
        ax.grid(alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Add sample size annotation
        ax.text(0.98, 0.98, f'n={len(subset)}',
               transform=ax.transAxes, fontsize=9,
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.8, edgecolor='gray'))

plt.tight_layout()

if SAVE_FIGURES:
    output_path = DIR_OUTPUT + os.sep + 'posterior_correlation_alpha_beta_histograms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_svg = DIR_OUTPUT + os.sep + 'posterior_correlation_alpha_beta_histograms.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f"Saved: {output_path_svg}")

plt.show()

##########################
# Overall distribution (combined)
##########################

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

# Histogram for each genotype
for genotype in ['GD', 'KK']:
    subset = df_corr[df_corr['genotype'] == genotype]
    ax.hist(subset['correlation'], bins=30, color=geno_colors[genotype],
           alpha=0.5, edgecolor='black', linewidth=0.5, label=genotype)

# Overall mean and median
mean_val = df_corr['correlation'].mean()
median_val = df_corr['correlation'].median()

ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
          label=f'Overall Mean={mean_val:.2f}', zorder=10)
ax.axvline(median_val, color='black', linestyle='-', linewidth=2.5,
          label=f'Overall Median={median_val:.2f}', zorder=10)

ax.set_xlabel(r'$\rho(\alpha, \beta)$ (within-fly posterior correlation)', fontsize=12)
ax.set_ylabel('Number of flies', fontsize=12)
ax.set_title('Distribution of Within-Fly Posterior Correlations\nAcross All Flies',
            fontsize=13, fontweight='bold')

ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')
ax.set_axisbelow(True)

# Add sample size
ax.text(0.02, 0.98, f'Total n={len(df_corr)}',
       transform=ax.transAxes, fontsize=10,
       ha='left', va='top',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                alpha=0.9, edgecolor='gray'))

plt.tight_layout()

if SAVE_FIGURES:
    output_path = DIR_OUTPUT + os.sep + 'posterior_correlation_alpha_beta_overall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

plt.show()

print("\n[SUCCESS] Analysis complete!")
