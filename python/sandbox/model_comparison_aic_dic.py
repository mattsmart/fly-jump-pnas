"""
Model comparison using AIC and DIC for 1D, 2D, 3D habituation models.

Computes information criteria to evaluate whether model complexity is justified.
Properly accounts for posterior uncertainty by computing expected log-likelihood.

Reports:
    - Summary table comparing models
    - Histograms of per-fly log-likelihoods
    - Bar plots comparing total AIC/DIC

Addresses Reviewer 3 Major Comment #4: Justify model complexity via formal
model comparison criteria (AIC/DIC are in-sample approximations of cross-validation).
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


from functions_common import likelihood_func_vec
from settings import DIR_FITS, DIR_OUTPUT, OMIT_FLY_IDS

##########################
# Configuration
##########################
# Use thinned posterior? Set to None for all draws, or number for thinning
N_DRAWS_TO_USE = 800  # Use 800 for speed, or None for all (~8000)


def compute_log_likelihood_single_fly(jumpdata_str, alpha, beta, p0, is_1d_model=False):
    """
    Compute log-likelihood for a single fly's data given parameters.

    Handles both scalar and array inputs (for computing across posterior draws).

    Args:
        jumpdata_str: String of 0s and 1s (1050 trials)
        alpha, beta, p0: Model parameters (scalars or arrays for multiple draws)
        is_1d_model: If True, use constant p0 (no habituation). If False, use habituation model.

    Returns:
        log_lik: Log-likelihood (scalar if params are scalars, array if arrays)
    """
    jumpdata = np.array([int(c) for c in jumpdata_str])

    # Ensure parameters are arrays
    if np.isscalar(alpha):
        alpha = np.array([alpha])
        beta = np.array([beta])
        p0 = np.array([p0])
        squeeze_output = True
    else:
        squeeze_output = False

    n_draws = len(p0)
    log_lik_total = np.zeros(n_draws)

    # For 1D model: constant probability across all trials
    if is_1d_model:
        # Clip p0 to avoid log(0)
        p0_clipped = np.clip(p0, 1e-10, 1 - 1e-10)

        # All 1050 trials have same probability p0
        for trial_idx in range(1050):
            y = jumpdata[trial_idx]
            log_lik_total += y * np.log(p0_clipped) + (1 - y) * np.log(1 - p0_clipped)

    else:
        # For 2D/3D models: use habituation dynamics

        # 5 habituation blocks (200 trials each, period=1)
        for block_idx in range(5):
            start_idx = block_idx * 200
            end_idx = start_idx + 200
            block_data = jumpdata[start_idx:end_idx]

            # Compute probabilities for all trials in this block
            p_block = likelihood_func_vec(np.arange(200), alpha, beta, p0, pulse_period=1)

            # Clip to avoid log(0)
            p_block = np.clip(p_block, 1e-10, 1 - 1e-10)

            # Log-likelihood: sum over trials
            for trial_idx in range(200):
                y = block_data[trial_idx]
                p = p_block[:, trial_idx]
                log_lik_total += y * np.log(p) + (1 - y) * np.log(1 - p)

        # SRA block (50 trials, period=5)
        sra_data = jumpdata[1000:1050]
        p_sra = likelihood_func_vec(np.arange(50), alpha, beta, p0, pulse_period=5)
        p_sra = np.clip(p_sra, 1e-10, 1 - 1e-10)

        for trial_idx in range(50):
            y = sra_data[trial_idx]
            p = p_sra[:, trial_idx]
            log_lik_total += y * np.log(p) + (1 - y) * np.log(1 - p)

    if squeeze_output:
        return log_lik_total[0]
    return log_lik_total


def compute_aic_dic_for_model(model_type):
    """
    Compute AIC and DIC for a given model.

    Uses proper Bayesian approach: averages log-likelihood across posterior draws
    to avoid Jensen's inequality bias.

    Returns:
        DataFrame with per-fly results: fly_id, genotype, day, log_lik_mean,
                                        log_lik_at_mean, pD, AIC, DIC
    """
    print(f"\n{'='*70}")
    print(f"Processing {model_type.upper()} model")
    print('='*70)

    # Load fitted parameters (summary statistics)
    csv_path = DIR_FITS + os.sep + f'fly-stability-days-detailed-{model_type}.csv'
    df = pd.read_csv(csv_path)

    # Load posterior draws for all genotype-day combinations
    draws_dict = {}
    for genotype in ['KK', 'GD']:
        for day in [7, 14, 21]:
            draws_file = DIR_FITS + os.sep + f'{genotype}_day{day}_{model_type}_draws.npz'
            if os.path.exists(draws_file):
                draws_dict[f'{genotype}_{day}'] = np.load(draws_file, allow_pickle=True)
            else:
                print(f"  Warning: {draws_file} not found")

    # Exclude flies with technical issues
    for genotype, fly_ids in OMIT_FLY_IDS.items():
        if fly_ids:
            before = len(df)
            df = df[~((df['genotype'] == genotype) & (df['fly_id'].isin(fly_ids)))]
            after = len(df)
            if before > after:
                print(f"  Excluded {before - after} {genotype} flies: {fly_ids}")

    print(f"  Total flies: {len(df)}")

    # Number of parameters per fly
    if model_type == '1d':
        k = 1  # p0 only
    elif model_type == '2d':
        k = 2  # p0, beta (alpha fixed)
    else:  # 3d
        k = 3  # p0, alpha, beta

    results = []

    for idx, row in df.iterrows():
        fly_id = row['fly_id']
        genotype = row['genotype']
        day = row['day']
        jumpdata = row['jumpdata']

        # Get posterior draws for this fly
        draws_key = f'{genotype}_{day}'
        if draws_key not in draws_dict:
            print(f"  Warning: No draws for {genotype} day {day}")
            continue

        draws = draws_dict[draws_key]
        fly_key = f'fly_id_{fly_id}'

        if fly_key not in draws.keys():
            print(f"  Warning: {fly_key} not found in draws for {genotype} day {day}")
            continue

        posterior_samples = draws[fly_key]  # Shape: (n_draws, n_params)

        # Apply thinning if requested
        if N_DRAWS_TO_USE is not None and len(posterior_samples) > N_DRAWS_TO_USE:
            thin_factor = len(posterior_samples) // N_DRAWS_TO_USE
            indices = np.arange(0, len(posterior_samples), thin_factor)[:N_DRAWS_TO_USE]
            posterior_samples = posterior_samples[indices]

        # Extract parameters based on model structure
        # 1D model: stores [p0] (shape: n_draws × 1)
        # 2D model: stores [β, p0] (shape: n_draws × 2), alpha from CSV
        # 3D model: stores [α, β, p0] (shape: n_draws × 3)
        if posterior_samples.shape[1] == 1:
            # 1D model: only p0
            p0_samples = posterior_samples[:, 0]
            alpha_samples = np.zeros(len(p0_samples))  # Not used for 1D
            beta_samples = np.zeros(len(p0_samples))   # Not used for 1D
            is_1d = True
        elif posterior_samples.shape[1] == 2:
            # 2D model: beta and p0 from draws, alpha from CSV (fixed)
            beta_samples = posterior_samples[:, 0]
            p0_samples = posterior_samples[:, 1]
            alpha_fixed = row['alpha']  # Get fixed alpha from CSV
            alpha_samples = np.full(len(p0_samples), alpha_fixed)
            is_1d = False
        elif posterior_samples.shape[1] == 3:
            # 3D model: alpha, beta, p0
            alpha_samples = posterior_samples[:, 0]
            beta_samples = posterior_samples[:, 1]
            p0_samples = posterior_samples[:, 2]
            is_1d = False
        else:
            print(f"  Warning: Unexpected parameter shape for {fly_key}: {posterior_samples.shape}")
            continue

        # Compute log-likelihood for each posterior draw (proper Bayesian approach)
        log_liks = compute_log_likelihood_single_fly(jumpdata, alpha_samples,
                                                     beta_samples, p0_samples,
                                                     is_1d_model=is_1d)

        # Mean log-likelihood across posterior (for DIC, avoids Jensen's inequality)
        log_lik_mean = np.mean(log_liks)

        # Log-likelihood at posterior mean (for pD calculation)
        alpha_mean = np.mean(alpha_samples)
        beta_mean = np.mean(beta_samples)
        p0_mean = np.mean(p0_samples)
        log_lik_at_mean = compute_log_likelihood_single_fly(jumpdata, alpha_mean,
                                                            beta_mean, p0_mean,
                                                            is_1d_model=is_1d)

        # Effective number of parameters (pD)
        # pD = 2 * (log_lik_mean - log_lik_at_mean)
        # This measures how much the posterior mean differs from the full posterior
        pD = 2 * (log_lik_mean - log_lik_at_mean)

        # AIC: 2k - 2*log_lik
        # Using Bayesian posterior mean of log-likelihood
        AIC = 2 * k - 2 * log_lik_mean

        # DIC: -2*log_lik_mean + 2*pD
        DIC = -2 * log_lik_mean + 2 * pD

        results.append({
            'fly_id': fly_id,
            'genotype': genotype,
            'day': day,
            'log_lik_mean': log_lik_mean,
            'log_lik_at_mean': log_lik_at_mean,
            'pD': pD,
            'k': k,
            'AIC': AIC,
            'DIC': DIC
        })

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(df)} flies...")

    results_df = pd.DataFrame(results)

    print(f"\n  Computed AIC/DIC for {len(results_df)} flies")
    print(f"  Mean log-likelihood: {results_df['log_lik_mean'].mean():.2f}")
    print(f"  Mean AIC: {results_df['AIC'].mean():.2f}")
    print(f"  Mean DIC: {results_df['DIC'].mean():.2f}")
    print(f"  Mean pD: {results_df['pD'].mean():.2f} (nominal k={k})")

    return results_df


def create_summary_table(results_dict):
    """Create summary table comparing models."""

    summary_rows = []

    for model_type in ['1d', '2d', '3d']:
        df = results_dict[model_type]

        # Overall statistics
        row = {
            'Model': model_type.upper(),
            'k': df['k'].iloc[0],
            'n_flies': len(df),
            'Mean_LogLik': df['log_lik_mean'].mean(),
            'Total_AIC': df['AIC'].sum(),
            'Total_DIC': df['DIC'].sum(),
            'Mean_pD': df['pD'].mean()
        }
        summary_rows.append(row)

        # By genotype
        for genotype in ['KK', 'GD']:
            subset = df[df['genotype'] == genotype]
            if len(subset) > 0:
                row = {
                    'Model': f"{model_type.upper()}_{genotype}",
                    'k': subset['k'].iloc[0],
                    'n_flies': len(subset),
                    'Mean_LogLik': subset['log_lik_mean'].mean(),
                    'Total_AIC': subset['AIC'].sum(),
                    'Total_DIC': subset['DIC'].sum(),
                    'Mean_pD': subset['pD'].mean()
                }
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Add delta AIC and DIC (relative to best model)
    # Filter to overall models only (not genotype-specific)
    overall_mask = summary_df['Model'].str.match(r'^\d[dD]$')
    best_aic = summary_df[overall_mask]['Total_AIC'].min()
    best_dic = summary_df[overall_mask]['Total_DIC'].min()

    summary_df['Delta_AIC'] = summary_df['Total_AIC'] - best_aic
    summary_df['Delta_DIC'] = summary_df['Total_DIC'] - best_dic

    return summary_df


def plot_log_lik_distributions(results_dict, save=True):
    """Plot histograms of per-fly log-likelihoods."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {'1d': '#E74C3C', '2d': '#3498DB', '3d': '#2ECC71'}

    for idx, model_type in enumerate(['1d', '2d', '3d']):
        ax = axes[idx]
        df = results_dict[model_type]

        # Plot histogram
        ax.hist(df['log_lik_mean'], bins=30, alpha=0.7, color=colors[model_type],
                edgecolor='black', linewidth=0.5)

        # Add mean line
        mean_val = df['log_lik_mean'].mean()
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.1f}')

        ax.set_xlabel('Log-Likelihood', fontsize=12)
        ax.set_ylabel('Number of flies', fontsize=12)
        ax.set_title(f'{model_type.upper()} Model (k={df["k"].iloc[0]})',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle('Per-Fly Log-Likelihood Distributions', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison_aic_dic'
        os.makedirs(output_dir, exist_ok=True)
        out_path = output_dir + os.sep + 'log_likelihood_distributions.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {out_path}")

    plt.close()


def plot_aic_dic_comparison(results_dict, save=True):
    """Plot AIC and DIC comparisons across models."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'1d': '#E74C3C', '2d': '#3498DB', '3d': '#2ECC71'}
    models = ['1d', '2d', '3d']
    model_labels = ['1D\n(k=1)', '2D\n(k=2)', '3D\n(k=3)']

    # AIC comparison
    ax = axes[0]
    total_aics = [results_dict[m]['AIC'].sum() for m in models]
    bars = ax.bar(model_labels, total_aics, color=[colors[m] for m in models],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total AIC', fontsize=14, fontweight='bold')
    ax.set_title('AIC Comparison\n(lower is better)', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, total_aics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # DIC comparison
    ax = axes[1]
    total_dics = [results_dict[m]['DIC'].sum() for m in models]
    bars = ax.bar(model_labels, total_dics, color=[colors[m] for m in models],
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total DIC', fontsize=14, fontweight='bold')
    ax.set_title('DIC Comparison\n(lower is better)', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, total_dics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison_aic_dic'
        os.makedirs(output_dir, exist_ok=True)
        out_path = output_dir + os.sep + 'aic_dic_comparison.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")

    plt.close()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MODEL COMPARISON: AIC AND DIC")
    print("="*70)
    print("\nComputing information criteria using proper Bayesian approach:")
    print("  - Averages log-likelihood across posterior draws (avoids Jensen's bias)")
    print("  - AIC and DIC approximate leave-one-out cross-validation")
    print("  - Lower values indicate better models")

    # Compute AIC/DIC for each model
    results_dict = {}
    for model_type in ['1d', '2d', '3d']:
        results_dict[model_type] = compute_aic_dic_for_model(model_type)

    # Create summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    summary_df = create_summary_table(results_dict)

    # Format for nice display
    print("\n" + summary_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # Save summary table
    output_dir = DIR_OUTPUT + os.sep + 'model_comparison_aic_dic'
    os.makedirs(output_dir, exist_ok=True)
    summary_path = output_dir + os.sep + 'aic_dic_summary.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.2f')
    print(f"\nSaved summary table: {summary_path}")

    # Save detailed per-fly results
    for model_type, df in results_dict.items():
        detail_path = output_dir + os.sep + f'aic_dic_detailed_{model_type}.csv'
        df.to_csv(detail_path, index=False, float_format='%.4f')
        print(f"Saved detailed results: {detail_path}")

    # Create plots
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    plot_log_lik_distributions(results_dict, save=True)
    plot_aic_dic_comparison(results_dict, save=True)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nInterpretation:")
    print("  - Lower AIC/DIC indicates better model (balances fit vs. complexity)")
    print("  - Delta > 10 indicates strong evidence for model preference")
    print("  - pD estimates effective number of parameters:")
    print("    - pD ~= k suggests parameters are well-identified")
    print("    - pD < k suggests some parameters are constrained by data")
    print("    - pD > k suggests overfitting or parameter redundancy")
    print("\nGenerated outputs:")
    print("  - aic_dic_summary.csv: Overall comparison table")
    print("  - aic_dic_detailed_*.csv: Per-fly results for each model")
    print("  - log_likelihood_distributions.png: Histograms of fit quality")
    print("  - aic_dic_comparison.png: Bar plots comparing models")
