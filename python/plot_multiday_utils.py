import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_format_add_score_columns import compute_p_ss, compute_hab_magnitude_rel, compute_hab_time_half_rel
from functions_common import likelihood_func_vec


# ==== MAIN DATA PROCESSING FUNCTIONS ====

def process_phenotype_data(df_summary, genotype, ages, draws_over_age,
                           X_VAR='p0', Y_VAR='p_ss', OMIT_FLY_IDS=None):
    """
    Process the phenotype data and calculate paths.

    Returns:
    - df_pheno: DataFrame with phenotype data
    - df_paths: DataFrame with path classifications
    - all_draws: Dictionary with posterior draws
    """
    # Filter by genotype and age
    ndf = df_summary[(df_summary.genotype == genotype) & (df_summary.day.isin(ages))].copy()

    # Remove omitted fly IDs
    if OMIT_FLY_IDS and genotype in OMIT_FLY_IDS:
        ndf = ndf[~ndf.fly_id.isin(OMIT_FLY_IDS[genotype])].reset_index(drop=True)

    # Build phenotype DataFrame
    records = []
    all_draws = {day: {'x': [], 'y': []} for day in ages}

    for day in ages:
        draws = draws_over_age[genotype][day]
        sub = ndf[ndf.day == day]

        for _, row in sub.iterrows():
            key = f"fly_id_{int(row.fly_id)}"
            if key not in draws:
                continue

            # Compute means and confidence intervals
            x_mu, y_mu, x_lo, x_hi, y_lo, y_hi, x_draws, y_draws = summarize_fly(row, draws, X_VAR, Y_VAR)

            # Store record
            records.append({
                'fly_id': int(row.fly_id),
                'day': day,
                X_VAR: x_mu,
                Y_VAR: y_mu,
                f"{X_VAR}_ci_low": x_lo,
                f"{X_VAR}_ci_high": x_hi,
                f"{Y_VAR}_ci_low": y_lo,
                f"{Y_VAR}_ci_high": y_hi
            })

            # Store draws
            all_draws[day]['x'].append(x_draws)
            all_draws[day]['y'].append(y_draws)

    # Flatten draw lists into arrays
    for day in ages:
        if all_draws[day]['x']:
            all_draws[day]['x'] = np.concatenate(all_draws[day]['x'])
            all_draws[day]['y'] = np.concatenate(all_draws[day]['y'])

    df_pheno = pd.DataFrame(records)

    # Calculate paths
    paths = []
    for fid, grp in df_pheno.groupby('fly_id'):
        grp = grp.sort_values('day')
        if len(grp) < 3:
            continue

        x1 = classify_step(grp[X_VAR].iloc[1], grp[f"{X_VAR}_ci_low"].iloc[0], grp[f"{X_VAR}_ci_high"].iloc[0])
        x2 = classify_step(grp[X_VAR].iloc[2], grp[f"{X_VAR}_ci_low"].iloc[1], grp[f"{X_VAR}_ci_high"].iloc[1])
        y1 = classify_step(grp[Y_VAR].iloc[1], grp[f"{Y_VAR}_ci_low"].iloc[0], grp[f"{Y_VAR}_ci_high"].iloc[0])
        y2 = classify_step(grp[Y_VAR].iloc[2], grp[f"{Y_VAR}_ci_low"].iloc[1], grp[f"{Y_VAR}_ci_high"].iloc[1])

        paths.append({
            'fly_id': fid,
            'path_code': x1 + x2 + y1 + y2,
            f"{X_VAR}_step1": x1,
            f"{X_VAR}_step2": x2,
            f"{Y_VAR}_step1": y1,
            f"{Y_VAR}_step2": y2,
        })

    df_paths = pd.DataFrame(paths)

    return df_pheno, df_paths, all_draws


def summarize_fly(day_row, posterior_draws, X_VAR, Y_VAR):
    """
    Calculate summary statistics for a fly based on posterior draws.
    """
    if isinstance(day_row, pd.Series):
        day_row = day_row.to_frame().T

    fid = int(day_row.fly_id.iloc[0])
    samples = posterior_draws[f"fly_id_{fid}"]  # (ndraws,3)
    alpha, beta, p0_draws = samples.T

    # Compute X draws
    if X_VAR == 'p0':
        x_draws = p0_draws
    elif X_VAR == 'p_ss':
        x_draws = compute_p_ss(alpha, beta, p0_draws, T=1)
    elif X_VAR == 'sra_mean':
        x_draws = np.mean(likelihood_func_vec(np.arange(50), alpha, beta, p0_draws, pulse_period=5), axis=1)
    elif X_VAR == 'mag_abs':
        p_ss_draws = compute_p_ss(alpha, beta, p0_draws, T=1)
        mag_abs_draws = p0_draws - p_ss_draws
        x_draws = mag_abs_draws
    elif X_VAR == 'mag_rel':
        mag_rel_draws = compute_hab_magnitude_rel(alpha, beta, T=1)
        x_draws = mag_rel_draws
    elif X_VAR == 'k_star':
        # compute normalized (relative) halftime
        hab_halftime_rel_draws = compute_hab_time_half_rel(alpha, beta, T=1)
        x_draws = hab_halftime_rel_draws
    else:
        raise ValueError(f"Unknown X_VAR: {X_VAR}")

    # Compute Y draws
    if Y_VAR == 'p_ss':
        y_draws = compute_p_ss(alpha, beta, p0_draws, T=1)
    elif Y_VAR == 'sra_mean':
        y_draws = np.mean(likelihood_func_vec(np.arange(50), alpha, beta, p0_draws, pulse_period=5), axis=1)
    elif Y_VAR == 'p_ss_normed':
        p_ss_draws = compute_p_ss(alpha, beta, p0_draws, T=1)
        y_draws = p_ss_draws / p0_draws
    else:
        raise ValueError(f"Unknown Y_VAR: {Y_VAR}")

    # Calculate means and confidence intervals
    x_mu, y_mu = x_draws.mean(), y_draws.mean()
    x_lo, x_hi = np.percentile(x_draws, [2.5, 97.5])
    y_lo, y_hi = np.percentile(y_draws, [2.5, 97.5])

    return x_mu, y_mu, x_lo, x_hi, y_lo, y_hi, x_draws, y_draws


def classify_step(mu, lo, hi):
    """Classify a step as +, -, or 0 based on confidence intervals."""
    return '+' if mu > hi else ('-' if mu < lo else '0')


# ==== PLOTTING ====

def plot_parameter_consistency(df_pheno, param_name, age_pairs=None, highlight_fly_ids=None,
                               quantile_bands=True, band_quantiles=[0.25, 0.5, 0.75],
                               figsize=(10, 10)):
    """
    Create a plot showing parameter consistency across ages with quantile bands.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Default age pairs if not provided
    if age_pairs is None:
        ages = sorted(df_pheno['day'].unique())
        age_pairs = [(ages[i], ages[i + 1]) for i in range(len(ages) - 1)]
        age_pairs.append((ages[0], ages[-1]))  # Add first to last

    # Create figure
    fig, axes = plt.subplots(1, len(age_pairs), figsize=figsize, sharey=True)
    if len(age_pairs) == 1:
        axes = [axes]

    # Process each age pair
    for i, (age1, age2) in enumerate(age_pairs):
        ax = axes[i]

        # Get data for this age pair
        df_age1 = df_pheno[df_pheno['day'] == age1]
        df_age2 = df_pheno[df_pheno['day'] == age2]

        # Find common fly IDs
        common_fly_ids = set(df_age1['fly_id']).intersection(set(df_age2['fly_id']))

        # Extract paired values
        x_values = []
        y_values = []
        ids = []

        for fly_id in common_fly_ids:
            val1 = df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
            val2 = df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
            x_values.append(val1)
            y_values.append(val2)
            ids.append(fly_id)

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Calculate correlation
        corr = np.corrcoef(x_values, y_values)[0, 1]

        # Create a background heatmap showing quantile bands if requested
        if quantile_bands:
            # Calculate quantiles for x and y
            x_quantiles = [np.quantile(x_values, q) for q in band_quantiles]
            y_quantiles = [np.quantile(y_values, q) for q in band_quantiles]

            # Get min/max
            x_min, x_max = np.min(x_values), np.max(x_values)
            y_min, y_max = np.min(y_values), np.max(y_values)

            # Create background quantile grid
            x_edges = np.array([x_min] + x_quantiles + [x_max])
            y_edges = np.array([y_min] + y_quantiles + [y_max])

            # Draw grid lines
            for edge in x_edges:
                ax.axvline(edge, color='gray', linestyle='--', alpha=0.3)
            for edge in y_edges:
                ax.axhline(edge, color='gray', linestyle='--', alpha=0.3)

            # Shade background based on quantile correspondence
            n_bands = len(band_quantiles) + 1
            cmap = plt.cm.Greens

            # Shade diagonal regions (perfect correspondence)
            for i in range(n_bands):
                x_start = x_edges[i]
                x_end = x_edges[i + 1]
                y_start = y_edges[i]
                y_end = y_edges[i + 1]

                # Create a polygon for the quantile box
                ax.fill_between(
                    [x_start, x_end],
                    [y_start, y_start],
                    [y_end, y_end],
                    color=cmap(0.2 + 0.6 * i / n_bands),
                    alpha=0.2
                )

            # Add text labels for quantiles
            for i in range(n_bands):
                center_x = (x_edges[i] + x_edges[i + 1]) / 2
                center_y = (y_edges[i] + y_edges[i + 1]) / 2

                if i == 0:
                    label = f"0-{band_quantiles[0] * 100:.0f}%"
                elif i == n_bands - 1:
                    label = f"{band_quantiles[-1] * 100:.0f}-100%"
                else:
                    label = f"{band_quantiles[i - 1] * 100:.0f}-{band_quantiles[i] * 100:.0f}%"

                ax.text(center_x, center_y, label, ha='center', va='center',
                        fontsize=8, color='darkgreen', alpha=0.7)

        # Plot identity line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)

        # Plot data points
        ax.scatter(x_values, y_values, alpha=0.6, s=30, color='C0')

        # Highlight specific flies if requested
        if highlight_fly_ids:
            for fly_id in highlight_fly_ids:
                if fly_id in common_fly_ids:
                    idx = list(common_fly_ids).index(fly_id)
                    ax.scatter(x_values[idx], y_values[idx], s=100,
                               color=f'C{fly_id % 10}', edgecolor='black',
                               linewidth=1.5, zorder=10)

                    # Fix: Use annotate() instead of text() with xytext
                    ax.annotate(f"#{fly_id}",
                                xy=(x_values[idx], y_values[idx]),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=10, fontweight='bold')

        # Add correlation statistic
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add labels
        ax.set_xlabel(f"Day {age1} {param_name}")
        ax.set_ylabel(f"Day {age2} {param_name}")
        ax.set_title(f"Day {age1} vs {age2}")

        # Add grid
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Parameter Consistency: {param_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def examine_null_correlation_distributions(df_pheno, param_names,
                                           ages=[7, 14, 21], n_permutations=1000):
    """
    Analyze the null correlation distributions to check if they're reasonable.

    Returns a summary table and diagnostic plots.
    """
    # Calculate age pairs
    age_pairs = [(ages[i], ages[i + 1]) for i in range(len(ages) - 1)]
    age_pairs.append((ages[0], ages[-1]))  # Add first to last

    # Storage for results
    results = []
    null_distributions = {}

    # Process each parameter
    for param_name in param_names:
        # Process each age pair
        for age1, age2 in age_pairs:
            # Get data for these ages
            df_age1 = df_pheno[df_pheno['day'] == age1]
            df_age2 = df_pheno[df_pheno['day'] == age2]

            # Find common fly IDs
            common_fly_ids = list(set(df_age1['fly_id']).intersection(set(df_age2['fly_id'])))

            # Extract values
            x_values = np.array([df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
                                 for fly_id in common_fly_ids])
            y_values = np.array([df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
                                 for fly_id in common_fly_ids])

            # Calculate actual correlation
            actual_r = np.corrcoef(x_values, y_values)[0, 1]

            # Generate null distribution with detailed statistics
            null_r_values = []
            seeds = []  # For debugging

            for p in range(n_permutations):
                # Use different seed for each permutation
                seed = 10000 + p  # Unique seed
                seeds.append(seed)
                np.random.seed(seed)

                # Create permuted version
                perm_indices = np.random.permutation(len(y_values))
                perm_y = y_values[perm_indices]

                # Calculate correlation
                null_r = np.corrcoef(x_values, perm_y)[0, 1]
                null_r_values.append(null_r)

            # Calculate statistics
            null_mean = np.mean(null_r_values)
            null_std = np.std(null_r_values)
            null_min = np.min(null_r_values)
            null_max = np.max(null_r_values)

            # Calculate p-value
            p_value = np.mean(np.array(null_r_values) >= actual_r)

            # Store results
            key = f"{param_name}_{age1}_{age2}"
            results.append({
                'Parameter': param_name,
                'Age1': age1,
                'Age2': age2,
                'Actual_r': actual_r,
                'Null_Mean': null_mean,
                'Null_Std': null_std,
                'Null_Min': null_min,
                'Null_Max': null_max,
                'p_value': p_value,
                'N_Flies': len(common_fly_ids),
                'Value_Range': np.max(x_values) - np.min(x_values)
            })

            # Store distribution
            null_distributions[key] = {
                'values': null_r_values,
                'x_values': x_values,
                'y_values': y_values,
                'actual_r': actual_r,
                'seeds': seeds
            }

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create diagnostic plots
    fig, axes = plt.subplots(len(param_names), len(age_pairs), figsize=(15, 10))

    for i, param_name in enumerate(param_names):
        for j, (age1, age2) in enumerate(age_pairs):
            ax = axes[i, j] if len(param_names) > 1 else axes[j]

            key = f"{param_name}_{age1}_{age2}"
            null_r_values = null_distributions[key]['values']
            actual_r = null_distributions[key]['actual_r']

            # Plot null distribution
            sns.histplot(null_r_values, kde=True, ax=ax)

            # Add vertical line for actual r
            ax.axvline(actual_r, color='red', linestyle='--')

            # Add text with statistics
            ax.text(0.05, 0.95,
                    f"Actual r: {actual_r:.2f}\n"
                    f"Null: {np.mean(null_r_values):.2f}±{np.std(null_r_values):.2f}\n"
                    f"Range: [{np.min(null_r_values):.2f}, {np.max(null_r_values):.2f}]\n"
                    f"N flies: {results_df[(results_df.Parameter == param_name) & (results_df.Age1 == age1)].N_Flies.values[0]}",
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Set title and labels
            ax.set_title(f"{param_name}: Day {age1} → {age2}")
            ax.set_xlabel("Correlation (r)")
            ax.set_ylabel("Frequency")

    plt.tight_layout()

    return results_df, null_distributions, fig
