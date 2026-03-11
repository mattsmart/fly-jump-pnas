"""
Model comparison validation: Compare 1D, 2D, 3D model prediction quality.

Creates systematic scatter plots comparing empirical vs model-predicted metrics
for 1D (p0 only), 2D (p0, beta), and 3D (p0, alpha, beta) models.

Outputs:
    - Full figures: 6 figs (2 genotypes × 3 models), each 4 rows × N cols
      Rows: all days, day 7, day 14, day 21
      Cols: metrics in chronological order (p_transient, p_ss, p_reactivity)

    - Compact figures: 2 figs (2 genotypes), rotated layout
      Layout: N rows (metrics) × 3 cols (1D, 2D, 3D models)
      Metrics shown in chronological order: p_transient → p_ss → p_reactivity

    - Summary table: CSV with R², MAE, RMSE for all models/metrics/conditions
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats

# Import metric computation functions from existing script
from plot_fig3_model_validation_scatters import (
    compute_empirical_metrics
)
from data_tools import get_TTC_canonical
from functions_common import likelihood_func_vec
from settings import DIR_FITS, DIR_OUTPUT, OMIT_FLY_IDS

##########################
# Configuration
##########################
SHOW_BEST_FIT_LINE = False
INCLUDE_TOTAL_JUMPS = False  # Set to False to exclude total_jumps (uninformative metric)
P_TRANSIENT_CUTOFF_INDEX = 10  # Number of trials to use for transient phase (early dynamics)
USE_FULL_POSTERIOR = True  # If True, propagate full posterior (compute metrics per draw, then average)
N_DRAWS_TO_USE = 800  # Number of posterior draws to use (for speed). Set to None for all (~8000)

# Model colors (hex codes)
COLOR_1D = '#E74C3C'  # Red for 1D model (p₀ only)
COLOR_2D = '#3498DB'  # Blue for 2D model (p₀, β)
COLOR_3D = '#2ECC71'  # Green for 3D model (p₀, α, β)

# Compact figure layout options
COMPACT_FIGSIZE = (10, 7.5)  # Figure size for compact plots
COMPACT_MARKERSIZE = 25  # Scatter plot marker size
COMPACT_FONTSIZE_LABEL = 11  # Axis label font size
COMPACT_FONTSIZE_TITLE = 12  # Subplot title font size
COMPACT_FONTSIZE_SUPTITLE = 14  # Main title font size
COMPACT_FONTSIZE_STATS = 9  # Statistics text box font size
COMPACT_FONTSIZE_TICK = 9  # Tick label font size
HIDE_INTERIOR_TICKLABELS = True  # If True, remove tick labels from interior plots

# Build metrics list based on flag (in chronological order)
if INCLUDE_TOTAL_JUMPS:
    METRICS_SUBSET = ['total_jumps', 'p_transient', 'p_ss', 'p_reactivity']
else:
    METRICS_SUBSET = ['p_transient', 'p_ss', 'p_reactivity']  # Chronological order: transient → steady-state → reactivity


def compute_model_metrics_generic(df, model_type='3d'):
    """
    Compute model-predicted metrics for any model type (1D, 2D, or 3D).

    If USE_FULL_POSTERIOR=True:
        Properly propagates posterior uncertainty by computing metrics for each
        posterior draw, then averaging (avoids Jensen's inequality bias).
    If USE_FULL_POSTERIOR=False:
        Uses posterior mean parameters (faster but biased).

    Handles differences:
    - 1D: p0 only (constant jump probability, no habituation)
    - 2D: p0, beta fitted; alpha is fixed (alpha_global)
    - 3D: p0, alpha, beta all fitted per fly

    Args:
        df: DataFrame with fitted parameters
        model_type: '1d', '2d', or '3d'
    """

    if USE_FULL_POSTERIOR:
        # Load posterior draws for all genotype-day combinations
        print(f"  Loading posterior draws for {model_type.upper()} model...")
        draws_dict = {}
        for genotype in ['KK', 'GD']:
            for day in [7, 14, 21]:
                draws_file = DIR_FITS + os.sep + f'{genotype}_day{day}_{model_type}_draws.npz'
                if os.path.exists(draws_file):
                    draws_dict[f'{genotype}_{day}'] = np.load(draws_file, allow_pickle=True)
                else:
                    print(f"  Warning: {draws_file} not found")

    p_reactivity_model_list = []
    p_ss_model_list = []
    p_transient_model_list = []
    total_jumps_model_list = []
    m_abs_model_list = []
    m_rel_model_list = []
    ttc_model_list = []

    for idx, row in df.iterrows():
        if USE_FULL_POSTERIOR:
            # Extract posterior draws for this fly
            fly_id = row['fly_id']
            genotype = row['genotype']
            day = row['day']
            draws_key = f'{genotype}_{day}'
            fly_key = f'fly_id_{fly_id}'

            if draws_key not in draws_dict:
                print(f"  Warning: No draws for {genotype} day {day}, using posterior mean")
                USE_MEAN_FALLBACK = True
            elif fly_key not in draws_dict[draws_key].keys():
                print(f"  Warning: {fly_key} not found, using posterior mean")
                USE_MEAN_FALLBACK = True
            else:
                USE_MEAN_FALLBACK = False
                posterior_samples = draws_dict[draws_key][fly_key]

                # Apply thinning if requested
                if N_DRAWS_TO_USE is not None and len(posterior_samples) > N_DRAWS_TO_USE:
                    thin_factor = len(posterior_samples) // N_DRAWS_TO_USE
                    indices = np.arange(0, len(posterior_samples), thin_factor)[:N_DRAWS_TO_USE]
                    posterior_samples = posterior_samples[indices]

                # Extract parameters based on model structure
                if posterior_samples.shape[1] == 1:
                    # 1D model: only p0
                    p0_samples = posterior_samples[:, 0]
                    alpha_samples = np.zeros(len(p0_samples))
                    beta_samples = np.zeros(len(p0_samples))
                elif posterior_samples.shape[1] == 2:
                    # 2D model: beta and p0 from draws, alpha from CSV (fixed)
                    beta_samples = posterior_samples[:, 0]
                    p0_samples = posterior_samples[:, 1]
                    alpha_fixed = row['alpha']
                    alpha_samples = np.full(len(p0_samples), alpha_fixed)
                elif posterior_samples.shape[1] == 3:
                    # 3D model: alpha, beta, p0
                    alpha_samples = posterior_samples[:, 0]
                    beta_samples = posterior_samples[:, 1]
                    p0_samples = posterior_samples[:, 2]
                else:
                    print(f"  Warning: Unexpected shape {posterior_samples.shape}, using posterior mean")
                    USE_MEAN_FALLBACK = True

            if USE_MEAN_FALLBACK:
                # Fall back to posterior mean approach
                alpha = row['alpha']
                beta = row['beta'] if model_type in ['2d', '3d'] else 0.0
                p0 = row['p0']
                alpha_samples = np.array([alpha])
                beta_samples = np.array([beta])
                p0_samples = np.array([p0])
        else:
            # Use posterior mean approach (original behavior)
            alpha = row['alpha']
            beta = row['beta'] if model_type in ['2d', '3d'] else 0.0
            p0 = row['p0']
            alpha_samples = np.array([alpha])
            beta_samples = np.array([beta])
            p0_samples = np.array([p0])

        # Compute metrics for each posterior draw
        n_draws = len(p0_samples)

        if model_type == '1d':
            # 1D model: constant p0, no habituation
            p_reactivity_per_draw = p0_samples
            p_ss_per_draw = p0_samples
            p_transient_per_draw = p0_samples
            total_jumps_per_draw = 1050 * p0_samples

        else:
            # 2D and 3D: full habituation dynamics
            p_reactivity_per_draw = np.zeros(n_draws)
            p_ss_per_draw = np.zeros(n_draws)
            p_transient_per_draw = np.zeros(n_draws)
            total_jumps_per_draw = np.zeros(n_draws)

            # Compute for all draws at once (vectorized)
            # 1. p_reactivity: mean of SRA phase (period=5, trials 0-49)
            sra_curve = likelihood_func_vec(np.arange(50), alpha_samples, beta_samples,
                                           p0_samples, pulse_period=5)
            p_reactivity_per_draw = sra_curve.mean(axis=1)  # Mean over trials for each draw

            # 2. p_ss: steady-state probability (last 50 trials of hab block)
            hab_curve = likelihood_func_vec(np.arange(200), alpha_samples, beta_samples,
                                           p0_samples, pulse_period=1)
            p_ss_per_draw = hab_curve[:, 150:200].mean(axis=1)  # Last 50 trials

            # 3. p_transient: transient phase (first P_TRANSIENT_CUTOFF_INDEX trials)
            p_transient_per_draw = hab_curve[:, 0:P_TRANSIENT_CUTOFF_INDEX].mean(axis=1)

            # 4. total_jumps: expected total across all 1050 trials
            total_jumps_per_draw = 5 * hab_curve.sum(axis=1) + sra_curve.sum(axis=1)

        # Average metrics across posterior draws
        p_reactivity_model = p_reactivity_per_draw.mean()
        p_ss_model = p_ss_per_draw.mean()
        p_transient_model = p_transient_per_draw.mean()
        total_jumps_model = total_jumps_per_draw.mean()

        # Additional metrics: m_abs, m_rel (computed from averaged metrics)
        m_abs_model = p_reactivity_model - p_ss_model
        m_rel_model = m_abs_model / p_reactivity_model if p_reactivity_model > 0 else 0

        # TTC: For 1D model (no habituation), set to NaN
        if model_type == '1d':
            ttc_model = np.nan
        else:
            # For TTC, use posterior mean parameters (stochastic metric, expensive to average)
            alpha_mean = alpha_samples.mean()
            beta_mean = beta_samples.mean()
            p0_mean = p0_samples.mean()
            hab_curve_full = likelihood_func_vec(np.arange(200), np.array([alpha_mean]),
                                                 np.array([beta_mean]), np.array([p0_mean]),
                                                 pulse_period=1)
            simulated_data = np.random.binomial(1, hab_curve_full)
            ttc_model = get_TTC_canonical(''.join(str(x) for x in simulated_data[0, :]), 5)

        p_reactivity_model_list.append(p_reactivity_model)
        p_ss_model_list.append(p_ss_model)
        p_transient_model_list.append(p_transient_model)
        total_jumps_model_list.append(total_jumps_model)
        m_abs_model_list.append(m_abs_model)
        m_rel_model_list.append(m_rel_model)
        ttc_model_list.append(ttc_model)

        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(df)} flies...")

    df[f'p_reactivity_model'] = p_reactivity_model_list
    df[f'p_ss_model'] = p_ss_model_list
    df[f'p_transient_model'] = p_transient_model_list
    df[f'total_jumps_model'] = total_jumps_model_list
    df[f'm_abs_model'] = m_abs_model_list
    df[f'm_rel_model'] = m_rel_model_list
    df[f'ttc_model'] = ttc_model_list

    return df


def load_and_process_all_models():
    """
    Load all three model fits and compute empirical + model metrics.

    Returns:
        dict: {'1d': df_1d, '2d': df_2d, '3d': df_3d}
    """
    print("\n" + "="*70)
    print("LOADING AND PROCESSING ALL MODEL FITS")
    print("="*70)

    dfs = {}

    for model_type in ['1d', '2d', '3d']:
        print(f"\n{model_type.upper()} Model:")
        print("-" * 40)

        # Load fitted data
        csv_path = DIR_FITS + os.sep + f'fly-stability-days-detailed-{model_type}.csv'
        df = pd.read_csv(csv_path)

        # Exclude flies with technical issues
        for genotype, fly_ids in OMIT_FLY_IDS.items():
            if fly_ids:
                before = len(df[df['genotype'] == genotype])
                df = df[~((df['genotype'] == genotype) & (df['fly_id'].isin(fly_ids)))]
                after = len(df[df['genotype'] == genotype])
                if before > after:
                    print(f"  Excluded {before - after} {genotype} flies: {fly_ids}")

        print(f"  Final: {df[df['genotype']=='GD']['fly_id'].nunique()} GD flies, "
              f"{df[df['genotype']=='KK']['fly_id'].nunique()} KK flies")

        # Compute empirical metrics (same for all models)
        # Pass the cutoff parameter to ensure consistency with model metrics
        df = compute_empirical_metrics(df, p_transient_cutoff=P_TRANSIENT_CUTOFF_INDEX)

        # Compute model-predicted metrics (model-specific)
        df = compute_model_metrics_generic(df, model_type=model_type)

        dfs[model_type] = df

    return dfs


def plot_full_comparison_figure(dfs, genotype, model_type, save=True):
    """
    Create full 4-row × N-col scatter plot figure for one genotype and one model.

    Rows: all days aggregated, day 7, day 14, day 21
    Cols: metrics in chronological order (p_transient, p_ss, p_reactivity)
          Plus total_jumps if INCLUDE_TOTAL_JUMPS is True

    Args:
        dfs: dict with '1d', '2d', '3d' DataFrames
        genotype: 'KK' or 'GD'
        model_type: '1d', '2d', or '3d'
    """
    df = dfs[model_type]
    df_geno = df[df['genotype'] == genotype]

    days_list = [None, 7, 14, 21]  # None = all days aggregated
    row_labels = ['All days', 'Day 7', 'Day 14', 'Day 21']

    metrics = METRICS_SUBSET
    metric_labels = {
        'total_jumps': 'Total jumps',
        'p_reactivity': r'$p_{reactivity}$',
        'p_ss': r'$p_{ss}$',
        'p_transient': r'$p_{transient}$'
    }

    colors = {'1d': COLOR_1D, '2d': COLOR_2D, '3d': COLOR_3D}
    model_names = {'1d': '1D Model (p₀ only)', '2d': '2D Model (p₀, β)', '3d': '3D Model (p₀, α, β)'}

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'{model_names[model_type]} - {genotype}', fontsize=16, fontweight='bold', y=0.995)

    for row_idx, (day, row_label) in enumerate(zip(days_list, row_labels)):
        # Filter by day (or use all if day is None)
        if day is None:
            subset = df_geno
        else:
            subset = df_geno[df_geno['day'] == day]

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Get empirical and model values
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values

            # Remove NaN/inf
            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
            emp_vals = emp_vals[valid_mask]
            model_vals = model_vals[valid_mask]

            if len(emp_vals) < 2:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(emp_vals)})',
                       ha='center', va='center', transform=ax.transAxes, fontsize=9)
                continue

            # Scatter plot
            ax.scatter(emp_vals, model_vals, alpha=0.6, color=colors[model_type],
                      s=40, edgecolor='black', linewidth=0.5)

            # Calculate R²
            r_squared = stats.pearsonr(emp_vals, model_vals)[0]**2
            r_corr = stats.pearsonr(emp_vals, model_vals)[0]

            # Identity line
            min_val = min(emp_vals.min(), model_vals.min())
            max_val = max(emp_vals.max(), model_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5)

            # Labels
            if row_idx == 3:  # Bottom row
                ax.set_xlabel(f'Empirical {metric_labels[metric]}', fontsize=10)
            if col_idx == 0:  # Left column
                ax.set_ylabel(f'Model {metric_labels[metric]}', fontsize=10)

            # Title (top row only)
            if row_idx == 0:
                ax.set_title(metric_labels[metric], fontsize=11, fontweight='bold')

            # Row label (left column only)
            if col_idx == 0:
                ax.text(-0.45, 0.5, row_label, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', va='center', rotation=90)

            # R² text
            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nr = {r_corr:.3f}\nn = {len(emp_vals)}',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison_validation_fig3'
        os.makedirs(output_dir, exist_ok=True)
        out_base = f'{output_dir}{os.sep}full_{model_type}_{genotype}'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"  Saved: {out_base}.*")

    plt.close()
    return fig


def plot_compact_comparison_figure(dfs, genotype, save=True):
    """
    Create compact scatter plot figure for one genotype.

    Layout: 3 cols (models: 1D, 2D, 3D) × N rows (metrics)

    Args:
        dfs: dict with '1d', '2d', '3d' DataFrames
        genotype: 'KK' or 'GD'
    """
    metrics = METRICS_SUBSET
    metric_labels = {
        'total_jumps': 'Total jumps',
        'p_reactivity': r'$p_{reactivity}$',
        'p_ss': r'$p_{ss}$',
        'p_transient': r'$p_{transient}$'
    }

    colors = {'1d': COLOR_1D, '2d': COLOR_2D, '3d': COLOR_3D}
    model_names = {'1d': '1D', '2d': '2D', '3d': '3D'}
    model_types = ['1d', '2d', '3d']

    # Figure layout: models as columns, metrics as rows
    n_models = len(model_types)
    n_metrics = len(metrics)
    nrows, ncols = n_metrics, n_models
    fig, axes = plt.subplots(nrows, ncols, figsize=COMPACT_FIGSIZE)

    # Ensure axes is 2D array even if only 1 row or col
    if nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(nrows, ncols)

    # Count total samples for title
    total_n = len(dfs['3d'][dfs['3d']['genotype'] == genotype])

    fig.suptitle(f'Model Comparison - {genotype} (all days, n={total_n})',
                 fontsize=COMPACT_FONTSIZE_SUPTITLE, fontweight='bold', y=0.995)

    # Iterate through all subplots (row=metric, col=model)
    for i in range(nrows):
        for j in range(ncols):
            metric_idx = i
            model_idx = j
            metric = metrics[metric_idx]
            model_type = model_types[model_idx]
            ax = axes[i, j]

            # Get data for this model and genotype
            df = dfs[model_type]
            subset = df[df['genotype'] == genotype]

            # Get empirical and model values
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values

            # Remove NaN/inf
            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
            emp_vals = emp_vals[valid_mask]
            model_vals = model_vals[valid_mask]

            if len(emp_vals) < 2:
                ax.text(0.5, 0.5, f'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=COMPACT_FONTSIZE_STATS)
                ax.set_aspect('equal', adjustable='box')
                continue

            # Scatter plot
            ax.scatter(emp_vals, model_vals, alpha=0.6, color=colors[model_type],
                      s=COMPACT_MARKERSIZE, edgecolor='black', linewidth=0.5)

            # Calculate statistics
            r_corr = stats.pearsonr(emp_vals, model_vals)[0]
            r_squared = r_corr**2
            mae = np.mean(np.abs(emp_vals - model_vals))
            rmse = np.sqrt(np.mean((emp_vals - model_vals)**2))

            # Identity line
            min_val = min(emp_vals.min(), model_vals.min())
            max_val = max(emp_vals.max(), model_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5)

            # Axis labels (ALL rows get x-labels, each row is a different metric)
            ax.set_xlabel(f'Empirical {metric_labels[metric]}',
                         fontsize=COMPACT_FONTSIZE_LABEL)

            # Left column gets y-labels
            if j == 0:
                ax.set_ylabel(f'Model {metric_labels[metric]}',
                             fontsize=COMPACT_FONTSIZE_LABEL)

            # Column titles (model names) on top row
            if i == 0:
                ax.set_title(model_names[model_type], fontsize=COMPACT_FONTSIZE_TITLE,
                           fontweight='bold')

            # Row labels (metric names) on left
            if j == 0:
                ax.text(-0.45, 0.5, metric_labels[metric], transform=ax.transAxes,
                       fontsize=COMPACT_FONTSIZE_TITLE, fontweight='bold',
                       va='center', rotation=90)

            # Statistics text box (no n value here, it's in title)
            stats_text = f'$R^2$ = {r_squared:.3f}\nMAE = {mae:.3f}'
            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes, fontsize=COMPACT_FONTSIZE_STATS,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=COMPACT_FONTSIZE_TICK)

            # Hide y-tick labels on non-left columns (show x-tick labels on ALL rows)
            if HIDE_INTERIOR_TICKLABELS and j > 0:
                ax.set_yticklabels([])

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison_validation_fig3'
        os.makedirs(output_dir, exist_ok=True)
        out_base = f'{output_dir}{os.sep}compact_{genotype}_rotated'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"  Saved: {out_base}.*")

    plt.close()
    return fig


def compute_r2_summary_table(dfs):
    """
    Compute R², MAE, RMSE for all models, metrics, genotypes, and days.

    Returns:
        DataFrame with columns: model, genotype, day, metric, r_squared, r, mae, rmse, n
    """
    results = []

    for model_type in ['1d', '2d', '3d']:
        df = dfs[model_type]

        for genotype in ['GD', 'KK']:
            for day in [None, 7, 14, 21]:  # None = all days
                # Filter data
                subset = df[df['genotype'] == genotype]
                if day is not None:
                    subset = subset[subset['day'] == day]

                day_label = 'all' if day is None else str(day)

                for metric in METRICS_SUBSET:
                    emp_vals = subset[f'{metric}_emp'].values
                    model_vals = subset[f'{metric}_model'].values

                    # Remove NaN/inf
                    valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
                    emp_vals = emp_vals[valid_mask]
                    model_vals = model_vals[valid_mask]

                    if len(emp_vals) > 1:
                        r = stats.pearsonr(emp_vals, model_vals)[0]
                        r_squared = r**2
                        mae = np.mean(np.abs(emp_vals - model_vals))
                        rmse = np.sqrt(np.mean((emp_vals - model_vals)**2))
                        n = len(emp_vals)
                    else:
                        r = np.nan
                        r_squared = np.nan
                        mae = np.nan
                        rmse = np.nan
                        n = 0

                    results.append({
                        'model': model_type,
                        'genotype': genotype,
                        'day': day_label,
                        'metric': metric,
                        'r_squared': r_squared,
                        'r': r,
                        'mae': mae,
                        'rmse': rmse,
                        'n': n
                    })

    return pd.DataFrame(results)


def print_r2_summary(r2_df):
    """Print formatted summary of R², MAE, and RMSE."""

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY: R², MAE, RMSE")
    print("="*80)

    for metric in METRICS_SUBSET:
        print(f"\n{metric.upper()}:")
        print("-" * 80)

        subset = r2_df[(r2_df['metric'] == metric) & (r2_df['day'] == 'all')]

        # R² pivot table
        print("\nR²:")
        pivot_r2 = subset.pivot_table(values='r_squared', index='genotype', columns='model')
        print(pivot_r2.to_string())

        # MAE pivot table
        print("\nMAE:")
        pivot_mae = subset.pivot_table(values='mae', index='genotype', columns='model')
        print(pivot_mae.to_string())

        # RMSE pivot table
        print("\nRMSE:")
        pivot_rmse = subset.pivot_table(values='rmse', index='genotype', columns='model')
        print(pivot_rmse.to_string())
        print()

    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    print("\n" + "="*70)
    print("MODEL COMPARISON VALIDATION: 1D vs 2D vs 3D")
    print("="*70)
    print(f"Using first {P_TRANSIENT_CUTOFF_INDEX} trials for p_transient (early dynamics)\n")

    # Load and process all models
    dfs = load_and_process_all_models()

    # ========================================
    # Full figures (4 rows × 4 cols each)
    # ========================================
    print("\n" + "="*70)
    print("CREATING FULL FIGURES (4 rows × 4 cols)")
    print("="*70)

    for genotype in ['GD', 'KK']:
        for model_type in ['1d', '2d', '3d']:
            print(f"\n{model_type.upper()} - {genotype}:")
            plot_full_comparison_figure(dfs, genotype, model_type, save=True)

    # ========================================
    # Compact figures (rotated layout)
    # ========================================
    print("\n" + "="*70)
    print("CREATING COMPACT FIGURES (rotated layout)")
    print("="*70)

    for genotype in ['GD', 'KK']:
        print(f"\n{genotype}:")
        plot_compact_comparison_figure(dfs, genotype, save=True)

    # ========================================
    # R² summary table
    # ========================================
    print("\n" + "="*70)
    print("COMPUTING R² SUMMARY TABLE")
    print("="*70)

    r2_df = compute_r2_summary_table(dfs)

    # Save to CSV
    output_dir = DIR_OUTPUT + os.sep + 'model_comparison_validation_fig3'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir + os.sep + 'r2_summary.csv'
    r2_df.to_csv(csv_path, index=False)
    print(f"\nSaved R² summary to: {csv_path}")

    # Print formatted summary
    print_r2_summary(r2_df)

    print("\n" + "="*80)
    print("MODEL COMPARISON VALIDATION COMPLETE")
    print("="*80)
    print("\nGenerated outputs:")
    print("  - Full figures: 6 files (2 genotypes × 3 models)")
    n_metrics_compact = len(METRICS_SUBSET)
    n_models_compact = 3
    print(f"  - Compact figures: 2 files (2 genotypes, rotated layout: {n_metrics_compact} rows × {n_models_compact} cols)")
    print("  - Summary table: r2_summary.csv (includes R², MAE, RMSE)")
