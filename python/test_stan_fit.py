"""
Test script to verify Stan fits work on new machine and produce similar results.

This runs a single fit (KK day 7) and compares with pre-computed results.
"""
import cmdstanpy as csp
import numpy as np
import os
import pandas as pd
import warnings

from fit_experimental_data import no_pooled, fit, to_file, parse_fly_data
from settings import DIR_STAN, DIR_FITS

# Reduce warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
csp.utils.get_logger().setLevel('ERROR')


def test_single_fit():
    """Run a single fit and compare with pre-computed results."""

    print("=" * 70)
    print("TESTING STAN FIT REPRODUCIBILITY")
    print("=" * 70)

    # Configuration
    genotype = 'KK'
    day = 7

    print(f"\nFitting model for {genotype} day {day}...")
    print("-" * 70)

    # Load the Stan model
    model_dir = DIR_STAN + os.sep + 'dynamics' + os.sep
    model, init_fun = no_pooled(model_dir)

    # Run the fit
    print(f"Running Stan sampling (2 chains, 400 warmup + 400 sampling)...")
    sample = fit(model, init_fun, genotype, day, show_progress=True,
                 filtered=True, detailed_format=True)

    # Extract posterior means
    alpha_draws = sample.stan_variable('alpha')
    beta_draws = sample.stan_variable('beta')
    p0_draws = sample.stan_variable('p0')

    alpha_hat_new = alpha_draws.mean(axis=0)
    beta_hat_new = beta_draws.mean(axis=0)
    p0_hat_new = p0_draws.mean(axis=0)

    print(f"\n✓ Fit completed successfully")
    print(f"  Number of flies: {len(alpha_hat_new)}")
    print(f"  Number of posterior samples per fly: {alpha_draws.shape[0]}")

    # Load pre-computed results
    print(f"\nLoading pre-computed results...")
    precomputed_csv = DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv'
    df_old = pd.read_csv(precomputed_csv)
    df_old_filtered = df_old[(df_old['genotype'] == genotype) & (df_old['day'] == day)]

    alpha_hat_old = df_old_filtered['alpha'].values
    beta_hat_old = df_old_filtered['beta'].values
    p0_hat_old = df_old_filtered['p0'].values

    print(f"  Pre-computed results for {len(alpha_hat_old)} flies loaded")

    # Compare results
    print(f"\n" + "=" * 70)
    print("COMPARISON: New fit vs Pre-computed")
    print("=" * 70)

    # Compute correlations
    corr_alpha = np.corrcoef(alpha_hat_new, alpha_hat_old)[0, 1]
    corr_beta = np.corrcoef(beta_hat_new, beta_hat_old)[0, 1]
    corr_p0 = np.corrcoef(p0_hat_new, p0_hat_old)[0, 1]

    print(f"\nCorrelations between new and pre-computed parameter estimates:")
    print(f"  alpha: {corr_alpha:.4f}")
    print(f"  beta:  {corr_beta:.4f}")
    print(f"  p0:    {corr_p0:.4f}")

    # Compute mean absolute differences
    mad_alpha = np.mean(np.abs(alpha_hat_new - alpha_hat_old))
    mad_beta = np.mean(np.abs(beta_hat_new - beta_hat_old))
    mad_p0 = np.mean(np.abs(p0_hat_new - p0_hat_old))

    print(f"\nMean absolute differences:")
    print(f"  alpha: {mad_alpha:.6f} (relative: {mad_alpha/np.mean(alpha_hat_old):.2%})")
    print(f"  beta:  {mad_beta:.6f} (relative: {mad_beta/np.mean(beta_hat_old):.2%})")
    print(f"  p0:    {mad_p0:.6f} (relative: {mad_p0/np.mean(p0_hat_old):.2%})")

    # Summary statistics comparison
    print(f"\nSummary statistics comparison:")
    print(f"  Parameter | New mean ± std | Old mean ± std")
    print(f"  --------- | -------------- | --------------")
    print(f"  alpha     | {np.mean(alpha_hat_new):.4f} ± {np.std(alpha_hat_new):.4f} | {np.mean(alpha_hat_old):.4f} ± {np.std(alpha_hat_old):.4f}")
    print(f"  beta      | {np.mean(beta_hat_new):.4f} ± {np.std(beta_hat_new):.4f} | {np.mean(beta_hat_old):.4f} ± {np.std(beta_hat_old):.4f}")
    print(f"  p0        | {np.mean(p0_hat_new):.4f} ± {np.std(p0_hat_new):.4f} | {np.mean(p0_hat_old):.4f} ± {np.std(p0_hat_old):.4f}")

    # Interpretation
    print(f"\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if all([corr_alpha > 0.95, corr_beta > 0.95, corr_p0 > 0.95]):
        print("\n✓ EXCELLENT: Correlations > 0.95 for all parameters")
        print("  The new fit is highly consistent with pre-computed results.")
    elif all([corr_alpha > 0.90, corr_beta > 0.90, corr_p0 > 0.90]):
        print("\n✓ GOOD: Correlations > 0.90 for all parameters")
        print("  The new fit is consistent with pre-computed results.")
    elif all([corr_alpha > 0.80, corr_beta > 0.80, corr_p0 > 0.80]):
        print("\n~ ACCEPTABLE: Correlations > 0.80 for all parameters")
        print("  Some variation expected due to MCMC sampling, but results are similar.")
    else:
        print("\n⚠ WARNING: Some correlations < 0.80")
        print("  Results may differ from pre-computed fits. Consider:")
        print("  - Different random seed")
        print("  - Different Stan/compiler version")
        print("  - Need for more sampling iterations")

    print("\nNote: Exact reproduction is not expected due to random MCMC sampling.")
    print("High correlations (>0.90) indicate the Stan setup is working correctly.")

    return sample, (alpha_hat_new, beta_hat_new, p0_hat_new), (alpha_hat_old, beta_hat_old, p0_hat_old)


if __name__ == '__main__':
    sample, new_estimates, old_estimates = test_single_fit()

    print(f"\n" + "=" * 70)
    print("Stan diagnostics:")
    print("=" * 70)
    print(sample.diagnose())
