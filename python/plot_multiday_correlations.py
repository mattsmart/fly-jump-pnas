import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import DIR_FITS, DIR_OUTPUT, OMIT_FLY_IDS

# Import necessary utility functions
from plot_multiday_utils import (
    process_phenotype_data,
    plot_parameter_consistency,
    examine_null_correlation_distributions
)

# --- User‐configurable ---
genotype = 'KK'

ages = [7, 14, 21]
age_pairs = [(7, 14),
             (14, 21),
             (7, 21)]

if genotype == 'KK':
    HIGHLIGHT_FLY_IDS = [44, 105]
else:
    HIGHLIGHT_FLY_IDS = []

# Parameters to analyze for stability
PARAMETERS_TO_ANALYZE = ['p0', 'p_ss', 'sra_mean', 'mag_abs', 'mag_rel', 'k_star']  # note: sra_mean is 'p_reactivity'


# --- Data Loading and Preprocessing ---
# Create output directory if it doesn't exist
output_dir = os.path.join(DIR_OUTPUT, 'multiday_correlations')
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess summary data
df_summary = pd.read_csv(os.path.join(
    DIR_FITS, "fly-stability-days-detailed-3d-habscores.csv"))

# Load posterior draws (3D model)
draws_over_age = {
    g: {d: np.load(os.path.join(DIR_FITS, f"{g}_day{d}_3d_draws.npz"), allow_pickle=True)
        for d in ages}
    for g in ['KK', 'GD']
}

# --- Analysis Loop ---
for param in PARAMETERS_TO_ANALYZE:
    print(f"--- Analyzing parameter: {param} ---")

    # Process phenotype data. We need to re-run this for each parameter if it's used as X_VAR.
    # A dummy Y_VAR is provided as the function expects it.
    df_pheno, _, _ = process_phenotype_data(
        df_summary, genotype, ages, draws_over_age,
        X_VAR=param, Y_VAR='p_ss', OMIT_FLY_IDS=OMIT_FLY_IDS
    )

    # 1) Per-day-pair correlation plots
    print("Generating parameter consistency plots...")
    fig_consistency = plot_parameter_consistency(
        df_pheno,
        param_name=param,
        age_pairs=age_pairs,
        highlight_fly_ids=HIGHLIGHT_FLY_IDS,
        quantile_bands=False,
        figsize=(15, 5)
    )

    # new suptitle line here
    fig_consistency.suptitle(f'Parameter Consistency for {param} ({genotype})', fontsize=16, y=1.02)

    plt.savefig(os.path.join(output_dir, f'parameter_consistency_{param}.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'parameter_consistency_{param}.svg'),
                bbox_inches='tight')
    plt.close(fig_consistency)

    # 2) Null checks and p-value plots
    print("Analyzing null correlation distributions...")
    results_df, _, null_diag_fig = examine_null_correlation_distributions(
        df_pheno,
        [param],  # The function can take a list of parameters
        ages=ages,
        n_permutations=1000
    )

    # Save the diagnostic figure
    plt.figure(null_diag_fig.number)
    plt.suptitle(f'Null Correlation Diagnostics for {param} ({genotype})', fontsize=16, y=1.02)
    plt.savefig(os.path.join(output_dir, f'null_diagnostics_{param}.png'),
               dpi=300, bbox_inches='tight')
    plt.close(null_diag_fig)

    # Print and save summary table
    print(f"\nNull Correlation Summary for {param}:")
    print(results_df)
    results_df.to_csv(os.path.join(output_dir, f'null_summary_{param}.csv'), index=False)

print("\nCorrelation analysis complete.")
print(f"Output saved to: {output_dir}")
