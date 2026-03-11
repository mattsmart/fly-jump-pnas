"""
Model validation: scatter plots comparing model predictions vs empirical data.

Creates scatter plots with R² scores for 7 key metrics:
- p_reactivity: Jump probability during SRA phase
- p_ss: Steady-state jump probability (habituation tail)
- p_transient: Transient phase jump probability (first 50 trials per hab block)
- total_jumps: Total number of jumps across all 1050 trials
- m_rel: Relative habituation magnitude
- m_abs: Absolute habituation magnitude
- TTC: Time to criterion (habituation speed)

Addresses Reviewer 2 Major #1: Demonstrate model quality via observed vs predicted comparisons.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
from matplotlib import colors, cm

from data_tools import get_TTC_canonical
from data_format_add_score_columns import (compute_p_ss, compute_hab_magnitude_abs, compute_hab_magnitude_rel)
from functions_common import likelihood_func_vec
from settings import DIR_FITS, DIR_OUTPUT, OMIT_FLY_IDS

##########################
# Plotting options
##########################
SHOW_BEST_FIT_LINE = False  # If True, show red best-fit line in addition to identity line
INCLUDE_TOTAL_JUMPS = False  # Set to False to exclude total_jumps (uninformative metric)
P_TRANSIENT_CUTOFF_INDEX = 10  # Number of trials to use for transient phase (early dynamics)
USE_FULL_POSTERIOR = True  # If True, propagate full posterior (compute metrics per draw, then average)
N_DRAWS_TO_USE = 800  # Number of posterior draws to use (for speed). Set to None for all (~8000)

# By-age plot options (3×3 grid: metrics × ages for single genotype)
MAKE_BY_AGE_PLOTS = True  # If True, create validation plots across ages
USE_P0_COLORMAP = True  # If True, color by p0. If False, use solid color
SOLID_COLOR = 'grey'  # Color to use when USE_P0_COLORMAP = False
COLORMAP_NAME = 'Spectral'  # Colormap for p0 coloring
YLABEL_LEFT_ONLY = False  # If True, only show y-labels on left column
SHOW_RMSE = False  # If True, show RMSE in stats box
# By-age plot styling
FIGSIZE_BY_AGE = (7.5, 7.5)  # ~30% smaller than original (10, 10)
MARKERSIZE_BY_AGE = 28  # Scaled proportionally with figure size
FONTSIZE_LABEL_BY_AGE = 11  # Keep same for legibility
FONTSIZE_TITLE_BY_AGE = 12  # Keep same for legibility
FONTSIZE_SUPTITLE_BY_AGE = 14  # Keep same for legibility
FONTSIZE_STATS_BY_AGE = 9  # Keep same for legibility
FONTSIZE_TICK_BY_AGE = 9  # Keep same for legibility


def compute_empirical_metrics(df, p_transient_cutoff=None):
    """Compute all empirical metrics from jumpdata.

    Args:
        df: DataFrame with jumpdata column
        p_transient_cutoff: Number of trials for transient phase. If None, uses P_TRANSIENT_CUTOFF_INDEX
    """
    # Use global constant if not specified
    if p_transient_cutoff is None:
        p_transient_cutoff = P_TRANSIENT_CUTOFF_INDEX

    p_reactivity_emp_list = []
    p_ss_emp_list = []
    p_transient_emp_list = []
    m_rel_emp_list = []
    m_abs_emp_list = []
    ttc_emp_list = []
    total_jumps_emp_list = []

    for idx, row in df.iterrows():
        jumpdata_str = row['jumpdata']
        jumpdata = np.array([int(c) for c in jumpdata_str])

        # 1. p_reactivity: SRA mean (trials 1000-1049)
        p_reactivity_emp = jumpdata[1000:1050].mean()
        p_reactivity_emp_list.append(p_reactivity_emp)

        # 2. p_ss: steady-state, mean of last 50 trials of each hab block
        hab_tail_1 = jumpdata[ 50:200].mean()
        hab_tail_2 = jumpdata[250:400].mean()
        hab_tail_3 = jumpdata[450:600].mean()
        hab_tail_4 = jumpdata[650:800].mean()
        hab_tail_5 = jumpdata[850:1000].mean()
        hab_tail_per_trial = [hab_tail_1, hab_tail_2, hab_tail_3, hab_tail_4, hab_tail_5]
        p_ss_emp = np.mean(hab_tail_per_trial)
        p_ss_emp_list.append(p_ss_emp)

        # 3. p_transient: transient phase, mean of first p_transient_cutoff trials of each hab block
        hab_transient_1 = jumpdata[  0:  0+p_transient_cutoff].mean()
        hab_transient_2 = jumpdata[200:200+p_transient_cutoff].mean()
        hab_transient_3 = jumpdata[400:400+p_transient_cutoff].mean()
        hab_transient_4 = jumpdata[600:600+p_transient_cutoff].mean()
        hab_transient_5 = jumpdata[800:800+p_transient_cutoff].mean()
        p_transient_emp = np.mean([hab_transient_1, hab_transient_2, hab_transient_3,
                                    hab_transient_4, hab_transient_5])
        p_transient_emp_list.append(p_transient_emp)

        # 4. m_rel: relative magnitude (sra_mean - hab_tail) / sra_mean, averaged across blocks
        m_rel_hab1 = (p_reactivity_emp - hab_tail_1) / p_reactivity_emp if p_reactivity_emp > 0 else 0
        m_rel_hab2 = (p_reactivity_emp - hab_tail_2) / p_reactivity_emp if p_reactivity_emp > 0 else 0
        m_rel_hab3 = (p_reactivity_emp - hab_tail_3) / p_reactivity_emp if p_reactivity_emp > 0 else 0
        m_rel_hab4 = (p_reactivity_emp - hab_tail_4) / p_reactivity_emp if p_reactivity_emp > 0 else 0
        m_rel_hab5 = (p_reactivity_emp - hab_tail_5) / p_reactivity_emp if p_reactivity_emp > 0 else 0
        m_rel_emp = np.mean([m_rel_hab1, m_rel_hab2, m_rel_hab3, m_rel_hab4, m_rel_hab5])
        m_rel_emp_list.append(m_rel_emp)

        # 5. m_abs: absolute magnitude sra_mean - hab_tail, averaged across blocks
        m_abs_emp = p_reactivity_emp - p_ss_emp
        m_abs_emp_list.append(m_abs_emp)

        # 6. TTC: time to criterion for each block, averaged
        ttc_hab1 = get_TTC_canonical(''.join(str(int(x)) for x in jumpdata[  0:200]), 5)
        ttc_hab2 = get_TTC_canonical(''.join(str(int(x)) for x in jumpdata[200:400]), 5)
        ttc_hab3 = get_TTC_canonical(''.join(str(int(x)) for x in jumpdata[400:600]), 5)
        ttc_hab4 = get_TTC_canonical(''.join(str(int(x)) for x in jumpdata[600:800]), 5)
        ttc_hab5 = get_TTC_canonical(''.join(str(int(x)) for x in jumpdata[800:1000]), 5)
        ttc_emp = np.mean([ttc_hab1, ttc_hab2, ttc_hab3, ttc_hab4, ttc_hab5])
        ttc_emp_list.append(ttc_emp)

        # 7. total_jumps: sum of all 1050 trials
        total_jumps_emp = jumpdata.sum()
        total_jumps_emp_list.append(total_jumps_emp)

    df['p_reactivity_emp'] = p_reactivity_emp_list
    df['p_ss_emp'] = p_ss_emp_list
    df['p_transient_emp'] = p_transient_emp_list
    df['m_rel_emp'] = m_rel_emp_list
    df['m_abs_emp'] = m_abs_emp_list
    df['ttc_emp'] = ttc_emp_list
    df['total_jumps_emp'] = total_jumps_emp_list

    return df


def compute_model_metrics(df):
    """
    Compute model-predicted metrics using fitted parameters.

    If USE_FULL_POSTERIOR=True:
        Properly propagates posterior uncertainty by computing metrics for each
        posterior draw, then averaging (avoids Jensen's inequality bias).
    If USE_FULL_POSTERIOR=False:
        Uses posterior mean parameters (faster but biased).
    """

    if USE_FULL_POSTERIOR:
        # Load posterior draws for all genotype-day combinations
        print("  Loading posterior draws for 3D model...")
        draws_dict = {}
        for genotype in ['KK', 'GD']:
            for day in [7, 14, 21]:
                draws_file = DIR_FITS + os.sep + f'{genotype}_day{day}_3d_draws.npz'
                if os.path.exists(draws_file):
                    draws_dict[f'{genotype}_{day}'] = np.load(draws_file, allow_pickle=True)
                else:
                    print(f"  Warning: {draws_file} not found")

    p_reactivity_model_list = []
    p_ss_model_list = []
    p_transient_model_list = []
    m_rel_model_list = []
    m_abs_model_list = []
    ttc_model_list = []
    total_jumps_model_list = []

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

                # Extract parameters: 3D model has [α, β, p0]
                if posterior_samples.shape[1] == 3:
                    alpha_samples = posterior_samples[:, 0]
                    beta_samples = posterior_samples[:, 1]
                    p0_samples = posterior_samples[:, 2]
                else:
                    print(f"  Warning: Unexpected shape {posterior_samples.shape}, using posterior mean")
                    USE_MEAN_FALLBACK = True

            if USE_MEAN_FALLBACK:
                # Fall back to posterior mean approach
                alpha_samples = np.array([row['alpha']])
                beta_samples = np.array([row['beta']])
                p0_samples = np.array([row['p0']])
        else:
            # Use posterior mean approach (original behavior)
            alpha_samples = np.array([row['alpha']])
            beta_samples = np.array([row['beta']])
            p0_samples = np.array([row['p0']])

        # Compute metrics for each posterior draw (vectorized)
        n_draws = len(p0_samples)

        # 1. p_reactivity: mean of SRA phase (period=5, trials 0-49)
        sra_curve = likelihood_func_vec(np.arange(50), alpha_samples, beta_samples,
                                       p0_samples, pulse_period=5)
        p_reactivity_per_draw = sra_curve.mean(axis=1)  # Mean over trials for each draw
        p_reactivity_model = p_reactivity_per_draw.mean()  # Average across draws
        p_reactivity_model_list.append(p_reactivity_model)

        # 2. p_ss: steady-state probability (these functions handle arrays)
        p_ss_per_draw = compute_p_ss(alpha_samples, beta_samples, p0_samples, T=1)
        p_ss_model = p_ss_per_draw.mean()  # Average across draws
        p_ss_model_list.append(p_ss_model)

        # 3. p_transient: transient phase, mean of first P_TRANSIENT_CUTOFF_INDEX trials
        hab_transient_curve = likelihood_func_vec(np.arange(P_TRANSIENT_CUTOFF_INDEX),
                                                   alpha_samples, beta_samples, p0_samples,
                                                   pulse_period=1)
        p_transient_per_draw = hab_transient_curve.mean(axis=1)  # Mean over trials for each draw
        p_transient_model = p_transient_per_draw.mean()  # Average across draws
        p_transient_model_list.append(p_transient_model)

        # 4. m_rel: relative magnitude (function handles arrays)
        m_rel_per_draw = compute_hab_magnitude_rel(alpha_samples, beta_samples, T=1)
        m_rel_model = m_rel_per_draw.mean()  # Average across draws
        m_rel_model_list.append(m_rel_model)

        # 5. m_abs: absolute magnitude (function handles arrays)
        m_abs_per_draw = compute_hab_magnitude_abs(alpha_samples, beta_samples, p0_samples, T=1)
        m_abs_model = m_abs_per_draw.mean()  # Average across draws
        m_abs_model_list.append(m_abs_model)

        # 6. TTC: simulate data and compute time to criterion
        # For TTC, use posterior mean (stochastic metric, expensive to average)
        alpha_mean = alpha_samples.mean()
        beta_mean = beta_samples.mean()
        p0_mean = p0_samples.mean()
        likelihood = likelihood_func_vec(np.arange(200), np.array([alpha_mean]),
                                        np.array([beta_mean]), np.array([p0_mean]),
                                        pulse_period=1)
        simulated_data = np.random.binomial(1, likelihood[0, :])
        ttc_model = get_TTC_canonical(''.join(str(int(x)) for x in simulated_data), 5)
        ttc_model_list.append(ttc_model)

        # 7. total_jumps: expected total across all 1050 trials
        # 5 hab blocks (200 trials each) + 1 SRA block (50 trials)
        hab_curve = likelihood_func_vec(np.arange(200), alpha_samples, beta_samples,
                                       p0_samples, pulse_period=1)
        total_jumps_per_draw = 5 * hab_curve.sum(axis=1) + sra_curve.sum(axis=1)
        total_jumps_model = total_jumps_per_draw.mean()  # Average across draws
        total_jumps_model_list.append(total_jumps_model)

        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(df)} flies...")

    df['p_reactivity_model'] = p_reactivity_model_list
    df['p_ss_model'] = p_ss_model_list
    df['p_transient_model'] = p_transient_model_list
    df['m_rel_model'] = m_rel_model_list
    df['m_abs_model'] = m_abs_model_list
    df['ttc_model'] = ttc_model_list
    df['total_jumps_model'] = total_jumps_model_list

    return df


def load_and_process_data():
    """Load fitted data, exclude problematic flies, compute all metrics."""
    csv_path = DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv'
    df = pd.read_csv(csv_path)

    # Exclude flies with technical issues (imported from settings)
    for genotype, fly_ids in OMIT_FLY_IDS.items():
        if fly_ids:
            before = len(df[df['genotype'] == genotype])
            df = df[~((df['genotype'] == genotype) & (df['fly_id'].isin(fly_ids)))]
            after = len(df[df['genotype'] == genotype])
            print(f"Excluded {before - after} rows for {genotype} flies {fly_ids}")

    print(f"Final dataset: {df[df['genotype']=='GD']['fly_id'].nunique()} GD flies, "
          f"{df[df['genotype']=='KK']['fly_id'].nunique()} KK flies")

    # Compute empirical metrics
    print("Computing empirical metrics...")
    df = compute_empirical_metrics(df)

    # Compute model-predicted metrics
    print("Computing model-predicted metrics...")
    df = compute_model_metrics(df)

    return df


def plot_validation_scatterplots(df, save=True):
    """Create scatter plots comparing model vs empirical for all metrics."""

    if INCLUDE_TOTAL_JUMPS:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'total_jumps', 'm_rel', 'm_abs', 'ttc']
    else:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'm_rel', 'm_abs', 'ttc']
    metric_labels = {
        'p_reactivity': r'$p_{reactivity}$',
        'p_ss': r'$p_{ss}$',
        'p_transient': r'$p_{transient}$',
        'total_jumps': 'Total jumps',
        'm_rel': r'$m_{rel}$',
        'm_abs': r'$m_{abs}$',
        'ttc': 'TTC'
    }

    fig, axes = plt.subplots(2, 7, figsize=(30, 8))
    genotypes = ['GD', 'KK']
    geno_colors = {'GD': '#1B9E77', 'KK': '#7570B3'}

    for row_idx, genotype in enumerate(genotypes):
        subset = df[df['genotype'] == genotype]

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Get empirical and model values
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values

            # Remove any NaN or infinite values
            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
            emp_vals = emp_vals[valid_mask]
            model_vals = model_vals[valid_mask]

            # Scatter plot
            ax.scatter(emp_vals, model_vals, alpha=0.6, color=geno_colors[genotype],
                      s=40, edgecolor='black', linewidth=0.5)

            # Calculate R² and correlation
            r_squared = stats.pearsonr(emp_vals, model_vals)[0]**2
            r_corr = stats.pearsonr(emp_vals, model_vals)[0]

            # Identity line (perfect prediction)
            min_val = min(emp_vals.min(), model_vals.min())
            max_val = max(emp_vals.max(), model_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5,
                   label='Identity')

            # Best fit line (optional)
            if SHOW_BEST_FIT_LINE:
                slope, intercept = np.polyfit(emp_vals, model_vals, 1)
                fit_line = slope * emp_vals + intercept
                ax.plot(emp_vals, fit_line, 'r-', alpha=0.7, linewidth=1.5, label='Best fit')

            # Labels and title
            ax.set_xlabel(f'Empirical {metric_labels[metric]}', fontsize=10)
            ax.set_ylabel(f'Model {metric_labels[metric]}', fontsize=10)

            if row_idx == 0:
                ax.set_title(metric_labels[metric], fontsize=12, fontweight='bold')

            # Add R² text
            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nr = {r_corr:.3f}\nn = {len(emp_vals)}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add genotype label on left
            if col_idx == 0:
                ax.text(-0.35, 0.5, genotype, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='center', rotation=90)

            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Only show legend on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    if save:
        output_path = DIR_OUTPUT + os.sep + 'model_validation_scatterplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def print_validation_statistics(df):
    """Print R² and correlation statistics for model validation."""

    print("\n" + "="*70)
    print("MODEL VALIDATION STATISTICS")
    print("="*70)

    if INCLUDE_TOTAL_JUMPS:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'm_rel', 'm_abs', 'ttc', 'total_jumps']
    else:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'm_rel', 'm_abs', 'ttc']
    metric_names = {
        'p_reactivity': 'p_reactivity (SRA jump probability)',
        'p_ss': 'p_ss (steady-state jump probability)',
        'p_transient': 'p_transient (transient phase jump probability)',
        'm_rel': 'm_rel (relative habituation magnitude)',
        'm_abs': 'm_abs (absolute habituation magnitude)',
        'ttc': 'TTC (time to criterion)',
        'total_jumps': 'total_jumps (total number of jumps)'
    }

    for metric in metrics:
        print(f"\n{metric_names[metric].upper()}:")

        # Overall
        emp_all = df[f'{metric}_emp'].values
        model_all = df[f'{metric}_model'].values
        valid_mask = np.isfinite(emp_all) & np.isfinite(model_all)
        emp_all = emp_all[valid_mask]
        model_all = model_all[valid_mask]

        r_all = stats.pearsonr(emp_all, model_all)[0]
        r2_all = r_all**2
        rmse_all = np.sqrt(np.mean((emp_all - model_all)**2))

        print(f"  Overall: R² = {r2_all:.4f}, r = {r_all:.4f}, RMSE = {rmse_all:.4f}, n = {len(emp_all)}")

        # By genotype
        for genotype in ['GD', 'KK']:
            subset = df[df['genotype'] == genotype]
            emp = subset[f'{metric}_emp'].values
            model = subset[f'{metric}_model'].values
            valid_mask = np.isfinite(emp) & np.isfinite(model)
            emp = emp[valid_mask]
            model = model[valid_mask]

            r = stats.pearsonr(emp, model)[0]
            r2 = r**2
            rmse = np.sqrt(np.mean((emp - model)**2))

            print(f"  {genotype}: R² = {r2:.4f}, r = {r:.4f}, RMSE = {rmse:.4f}, n = {len(emp)}")

    print("\n" + "="*70)


def plot_validation_scatterplots_by_day(df, day, save=True):
    """Create scatter plots for a specific day."""

    df_day = df[df['day'] == day]

    if INCLUDE_TOTAL_JUMPS:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'total_jumps', 'm_rel', 'm_abs', 'ttc']
    else:
        metrics = ['p_reactivity', 'p_ss', 'p_transient', 'm_rel', 'm_abs', 'ttc']
    metric_labels = {
        'p_reactivity': r'$p_{reactivity}$',
        'p_ss': r'$p_{ss}$',
        'p_transient': r'$p_{transient}$',
        'total_jumps': 'Total jumps',
        'm_rel': r'$m_{rel}$',
        'm_abs': r'$m_{abs}$',
        'ttc': 'TTC'
    }

    fig, axes = plt.subplots(2, 7, figsize=(30, 8))
    genotypes = ['GD', 'KK']
    geno_colors = {'GD': '#1B9E77', 'KK': '#7570B3'}

    for row_idx, genotype in enumerate(genotypes):
        subset = df_day[df_day['genotype'] == genotype]

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Get empirical and model values
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values

            # Remove any NaN or infinite values
            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
            emp_vals = emp_vals[valid_mask]
            model_vals = model_vals[valid_mask]

            # Scatter plot
            ax.scatter(emp_vals, model_vals, alpha=0.6, color=geno_colors[genotype],
                      s=40, edgecolor='black', linewidth=0.5)

            # Calculate R² and correlation
            r_squared = stats.pearsonr(emp_vals, model_vals)[0]**2
            r_corr = stats.pearsonr(emp_vals, model_vals)[0]

            # Identity line (perfect prediction)
            min_val = min(emp_vals.min(), model_vals.min())
            max_val = max(emp_vals.max(), model_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5,
                   label='Identity')

            # Best fit line (optional)
            if SHOW_BEST_FIT_LINE:
                slope, intercept = np.polyfit(emp_vals, model_vals, 1)
                fit_line = slope * emp_vals + intercept
                ax.plot(emp_vals, fit_line, 'r-', alpha=0.7, linewidth=1.5, label='Best fit')

            # Labels and title
            ax.set_xlabel(f'Empirical {metric_labels[metric]}', fontsize=10)
            ax.set_ylabel(f'Model {metric_labels[metric]}', fontsize=10)

            if row_idx == 0:
                ax.set_title(f'{metric_labels[metric]}\nDay {day}', fontsize=12, fontweight='bold')

            # Add R² text
            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}\nr = {r_corr:.3f}\nn = {len(emp_vals)}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Add genotype label on left
            if col_idx == 0:
                ax.text(-0.35, 0.5, genotype, transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='center', rotation=90)

            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Only show legend on first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    if save:
        output_path = DIR_OUTPUT + os.sep + f'model_validation_scatterplots_day{day}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_validation_by_age(df, genotype, save=True):
    """
    Create 3×3 grid: metrics (rows) × ages (cols) for a single genotype.
    Points colored by p0 (if USE_P0_COLORMAP = True).

    Shows validation across ages for three key metrics: p_transient, p_ss, p_reactivity.
    """

    print(f"\n{'='*70}")
    print(f"CREATING VALIDATION BY AGE FIGURE FOR {genotype}")
    print('='*70)

    # Filter for this genotype
    df_geno = df[df['genotype'] == genotype]

    # Metrics to show (chronological order)
    metrics = ['p_transient', 'p_ss', 'p_reactivity']
    metric_labels = {
        'p_transient': r'$p_{transient}$',
        'p_ss': r'$p_{ss}$',
        'p_reactivity': r'$p_{reactivity}$'
    }
    ages = [7, 14, 21]
    age_labels = {7: 'Day 7', 14: 'Day 14', 21: 'Day 21'}

    nrows = len(metrics)
    ncols = len(ages)

    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_BY_AGE)

    # Track n per age (for adding to column titles)
    n_per_age = {}

    # Setup colormap if using p0 coloring
    if USE_P0_COLORMAP:
        all_p0 = df_geno['p0'].values
        vmin = np.nanmin(all_p0)
        vmax = np.nanmax(all_p0)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(COLORMAP_NAME)
        print(f"  p0 range for colormap: {vmin:.3f} - {vmax:.3f}")

    # Pre-compute axis limits per metric for row consistency
    axis_limits = {}
    for metric in metrics:
        all_emp = []
        all_model = []
        for age in ages:
            subset = df_geno[df_geno['day'] == age]
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values

            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals)
            all_emp.extend(emp_vals[valid_mask])
            all_model.extend(model_vals[valid_mask])

        if len(all_emp) > 0:
            global_min = min(min(all_emp), min(all_model))
            global_max = max(max(all_emp), max(all_model))
            axis_limits[metric] = (global_min, global_max)
            print(f"  {metric}: axis range [{global_min:.3f}, {global_max:.3f}]")

    # Create subplots
    for row_idx, metric in enumerate(metrics):
        for col_idx, age in enumerate(ages):
            ax = axes[row_idx, col_idx]

            # Get data for this age
            subset = df_geno[df_geno['day'] == age]

            # Get empirical and model values
            emp_vals = subset[f'{metric}_emp'].values
            model_vals = subset[f'{metric}_model'].values
            p0_vals = subset['p0'].values

            # Remove NaN/inf
            valid_mask = np.isfinite(emp_vals) & np.isfinite(model_vals) & np.isfinite(p0_vals)
            emp_vals = emp_vals[valid_mask]
            model_vals = model_vals[valid_mask]
            p0_vals = p0_vals[valid_mask]

            # Store n for this age (for column titles)
            if age not in n_per_age:
                n_per_age[age] = len(emp_vals)

            if len(emp_vals) < 2:
                ax.text(0.5, 0.5, f'Insufficient data\n(n={len(emp_vals)})',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=FONTSIZE_STATS_BY_AGE)
                ax.set_aspect('equal', adjustable='box')
                continue

            # Determine colors
            if USE_P0_COLORMAP:
                point_colors = cmap(norm(p0_vals))
            else:
                point_colors = SOLID_COLOR

            # Scatter plot
            ax.scatter(emp_vals, model_vals, alpha=0.6, c=point_colors,
                      s=MARKERSIZE_BY_AGE, edgecolor='black', linewidth=0.5)

            # Calculate statistics
            r_corr = stats.pearsonr(emp_vals, model_vals)[0]
            r_squared = r_corr**2
            mae = np.mean(np.abs(emp_vals - model_vals))
            rmse = np.sqrt(np.mean((emp_vals - model_vals)**2))

            # Identity line using row-consistent limits
            if metric in axis_limits:
                min_val, max_val = axis_limits[metric]
            else:
                min_val = min(emp_vals.min(), model_vals.min())
                max_val = max(emp_vals.max(), model_vals.max())

            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            # Labels
            # X-labels on all rows (each row is different metric)
            ax.set_xlabel(f'Empirical {metric_labels[metric]}', fontsize=FONTSIZE_LABEL_BY_AGE)

            # Y-labels: controlled by YLABEL_LEFT_ONLY flag
            if YLABEL_LEFT_ONLY:
                # Only show ylabel on left column
                if col_idx == 0:
                    ax.set_ylabel(f'Model {metric_labels[metric]}', fontsize=FONTSIZE_LABEL_BY_AGE)
            else:
                # Show ylabel on all panels
                ax.set_ylabel(f'Model {metric_labels[metric]}', fontsize=FONTSIZE_LABEL_BY_AGE)

            # Column titles (ages) on top row, with n values
            if row_idx == 0:
                title_text = f"{age_labels[age]} (n={n_per_age.get(age, '?')})"
                ax.set_title(title_text, fontsize=FONTSIZE_TITLE_BY_AGE, fontweight='bold')

            # Row labels (metrics) on left
            if col_idx == 0:
                ax.text(-0.45, 0.5, metric_labels[metric], transform=ax.transAxes,
                       fontsize=FONTSIZE_TITLE_BY_AGE, fontweight='bold',
                       va='center', rotation=90)

            # Statistics text box
            if SHOW_RMSE:
                stats_text = f'$R^2$ = {r_squared:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
            else:
                stats_text = f'$R^2$ = {r_squared:.3f}\nMAE = {mae:.3f}'

            ax.text(0.95, 0.05, stats_text,
                   transform=ax.transAxes, fontsize=FONTSIZE_STATS_BY_AGE,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax.grid(alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=FONTSIZE_TICK_BY_AGE)

            # Hide y-tick labels on non-left columns (only if YLABEL_LEFT_ONLY is True)
            if YLABEL_LEFT_ONLY and col_idx > 0:
                ax.set_yticklabels([])

    # Main title
    fig.suptitle(f'3D Model Validation Across Ages - {genotype}',
                 fontsize=FONTSIZE_SUPTITLE_BY_AGE, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_validation_by_age'
        os.makedirs(output_dir, exist_ok=True)

        color_suffix = 'p0color' if USE_P0_COLORMAP else 'solid'
        out_base = f'{output_dir}{os.sep}validation_3d_{genotype}_{color_suffix}'

        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"  Saved: {out_base}.*")

    # Create colorbar if using p0 colormap
    if USE_P0_COLORMAP and save:
        fig_cbar, ax_cbar = plt.subplots(figsize=(1, 6))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='vertical')
        cbar.set_label(r'Inferred $p_0$', fontsize=12)

        cbar_path = f'{output_dir}{os.sep}colorbar_p0_{genotype}.svg'
        plt.savefig(cbar_path, dpi=300, bbox_inches='tight')
        print(f"  Saved colorbar: {cbar_path}")
        plt.close(fig_cbar)

    plt.close(fig)


if __name__ == '__main__':
    print("Loading and processing data...")
    df = load_and_process_data()

    print("\nCreating validation scatter plots (all days combined)...")
    plot_validation_scatterplots(df)

    print("\nCreating validation scatter plots by day...")
    for day in [7, 14, 21]:
        print(f"  Day {day}...")
        plot_validation_scatterplots_by_day(df, day)

    if MAKE_BY_AGE_PLOTS:
        print("\nCreating validation by age plots (3×3 grid: metrics × ages)...")
        for genotype in ['KK', 'GD']:
            plot_validation_by_age(df, genotype, save=True)

    print_validation_statistics(df)

    print("\n[SUCCESS] Model validation complete!")
