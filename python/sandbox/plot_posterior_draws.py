import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


from data_format_add_score_columns import (
    compute_hab_magnitude_rel, compute_hab_time_half_rel, helper_summarize_univariate_samples)
from functions_common import likelihood_func
from plot_common import plot_posterior_likelihood_summary_over_days
from settings import day_palette

# Override DIR_FITS for sandbox location (need to go up one extra level)
DIR_FITS   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fits"))
DIR_OUTPUT_SANDBOX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))

##########################
# PARAMETERS
##########################
N_DRAWS_TO_USE = 800  # Number of posterior draws to use (for speed, set to None for all ~8000 draws)


def compute_global_param_limits_from_draws(draws_over_age):

    # Helper function to compute parameter limits
    def compute_limits(alpha, beta, p0):
        return {
            'alpha': (alpha.min() - 0.1 * alpha.std(), alpha.max() + 0.1 * alpha.std()),
            'beta': (beta.min() - 0.1 * beta.std(), beta.max() + 0.1 * beta.std()),
            'p0': (p0.min() - 0.1 * p0.std(), p0.max() + 0.1 * p0.std())
        }

    # Initialize storage for all conditions
    collected_data = defaultdict(lambda: {'alpha': [], 'beta': [], 'p0': []})

    # Iterate over genotypes and days
    for genotype in draws_over_age:
        for day in draws_over_age[genotype]:
            day_data = draws_over_age[genotype][day]

            # Iterate over each fly within the day
            for fly_id, samples in day_data.items():

                # Extract data if stored in .npz format
                if isinstance(samples, np.lib.npyio.NpzFile):
                    samples = samples[list(samples.files)[0]]  # Extract first array

                # Check if samples are valid
                if isinstance(samples, np.ndarray) and samples.shape[1] == 3:
                    alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

                    # Collect data for each level
                    collected_data['all']['alpha'].extend(alpha)
                    collected_data['all']['beta'].extend(beta)
                    collected_data['all']['p0'].extend(p0)

                    collected_data[genotype]['alpha'].extend(alpha)
                    collected_data[genotype]['beta'].extend(beta)
                    collected_data[genotype]['p0'].extend(p0)

                    key_geno_day = f"{genotype}_d{day}"
                    collected_data[key_geno_day]['alpha'].extend(alpha)
                    collected_data[key_geno_day]['beta'].extend(beta)
                    collected_data[key_geno_day]['p0'].extend(p0)
                else:
                    print(f"Warning: Unexpected format for {genotype} Day {day} Fly {fly_id}")

    # Compute limits for each condition
    param_limits = {}
    for condition, data in collected_data.items():
        alpha_arr = np.array(data['alpha'])
        beta_arr = np.array(data['beta'])
        p0_arr = np.array(data['p0'])
        param_limits[condition] = compute_limits(alpha_arr, beta_arr, p0_arr)

    return param_limits


def plot_posterior_3d_scatter(alpha_draws, beta_draws, p0_draws):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(alpha_draws, beta_draws, p0_draws, c='r', marker='o')

    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_zlabel('p0')

    plt.show()


def plot_pairwise_scatters_by_day(draws_over_age, genotype, fly_id, days_to_show=[7, 14, 21],
                                   param_limits=None, conv_to_scores=False, kde_overlay='contour',
                                   save_path=None):
    """
    Create 3x3 grid of pairwise scatter plots: rows=parameter pairs, cols=days.

    Args:
        draws_over_age: Dictionary of draws {genotype: {day: {fly_id: samples}}}
        genotype: 'KK' or 'GD'
        fly_id: Fly ID number
        days_to_show: List of days to plot (one per column)
        param_limits: Optional dict for axis limits {param: (min, max)}
        conv_to_scores: If True, convert alpha, beta to magnitude, halftime
        kde_overlay: 'contour', 'fill', or None
        save_path: If provided, save figure (without extension)
    """
    # Define color mapping for each day
    colors = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}

    # Define labels
    if conv_to_scores:
        labels = ['magnitude', 'halftime', 'p0']
        labels_latex = ['$M$', '$T_{1/2}$', r'$p_0$']
    else:
        labels = ['alpha', 'beta', 'p0']
        labels_latex = [r'$\alpha$', r'$\beta$', r'$p_0$']

    if param_limits is None:
        param_limits = {k: None for k in labels}

    # Define pairwise combinations (column order)
    pairs = [
        (0, 1),  # (param1, param2): e.g., (alpha, beta)
        (0, 2),  # (param1, param3): e.g., (alpha, p0)
        (1, 2)   # (param2, param3): e.g., (beta, p0)
    ]

    # Create figure: rows=pairs, cols=days
    n_days = len(days_to_show)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, n_days, figsize=(2.5 * n_days, 7))

    # Ensure axes is 2D
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    if n_days == 1:
        axes = axes.reshape(-1, 1)

    # Font sizes (increased for better readability)
    FONTSIZE_LABEL = 13
    FONTSIZE_TITLE = 14
    FONTSIZE_SUPTITLE = 16
    FONTSIZE_ROW_LABEL = 13
    FONTSIZE_TICK = 10

    # Process each pair (row) and day (column)
    for pair_idx, (idx_x, idx_y) in enumerate(pairs):
        param_x = labels[idx_x]
        param_y = labels[idx_y]

        for day_idx, day in enumerate(days_to_show):
            try:
                samples = draws_over_age[genotype][day]['fly_id_' + str(fly_id)]

                # Extract samples from .npz if needed
                if isinstance(samples, np.lib.npyio.NpzFile):
                    samples = samples[list(samples.files)[0]]

                # Apply thinning
                if N_DRAWS_TO_USE is not None and len(samples) > N_DRAWS_TO_USE:
                    thin_factor = len(samples) // N_DRAWS_TO_USE
                    indices = np.arange(0, len(samples), thin_factor)[:N_DRAWS_TO_USE]
                    samples = samples[indices]

                # Ensure expected format
                if not (isinstance(samples, np.ndarray) and samples.shape[1] == 3):
                    print(f"Warning: Unexpected format for {genotype} Day {day} Fly {fly_id}")
                    continue

                alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

                # Convert to scores if requested
                if conv_to_scores:
                    mag_rel = compute_hab_magnitude_rel(alpha, beta, T=1)
                    halftime_rel = compute_hab_time_half_rel(alpha, beta, T=1)
                    data = [mag_rel, halftime_rel, p0]
                else:
                    data = [alpha, beta, p0]

                # Clip data to valid ranges
                for i, label in enumerate(labels):
                    if label == 'p0':
                        data[i] = np.clip(data[i], 0, 1)
                    else:
                        data[i] = np.clip(data[i], 0, None)

                # Compute means
                means = [np.mean(d) for d in data]

                # Get axis for this pair/day combination
                ax = axes[pair_idx, day_idx]

                x_data = data[idx_x]
                y_data = data[idx_y]

                # Scatter plot
                ax.scatter(x_data, y_data, alpha=0.3, s=10, color=colors[day],
                          edgecolor='none')

                # KDE overlay
                if kde_overlay in ['contour', 'fill']:
                    try:
                        # Create DataFrame for seaborn
                        df_kde = pd.DataFrame({param_x: x_data, param_y: y_data})

                        if kde_overlay == 'contour':
                            sns.kdeplot(data=df_kde, x=param_x, y=param_y, ax=ax,
                                       color=colors[day], levels=5, linewidths=1.5, alpha=0.8)
                        else:  # fill
                            sns.kdeplot(data=df_kde, x=param_x, y=param_y, ax=ax,
                                       color=colors[day], fill=True, alpha=0.3, levels=5)
                    except Exception as e:
                        print(f"Warning: KDE overlay failed for day {day}, pair {pair_idx}: {e}")

                # Plot mean
                ax.scatter([means[idx_x]], [means[idx_y]], color=colors[day],
                          edgecolor='black', s=120, marker='o', zorder=10, linewidth=2,
                          label=f'Day {day} mean')

                # Set axis limits
                # Default: start at 0, constrain p0 to [0, 1]
                if param_x == 'p0':
                    ax.set_xlim(0, 1)
                else:
                    xlim_current = ax.get_xlim()
                    ax.set_xlim(0, xlim_current[1])

                if param_y == 'p0':
                    ax.set_ylim(0, 1)
                else:
                    ylim_current = ax.get_ylim()
                    ax.set_ylim(0, ylim_current[1])

                # Apply user-specified limits if provided
                if param_limits[param_x] is not None:
                    xlim_user = param_limits[param_x]
                    if param_x == 'p0':
                        ax.set_xlim(max(0, xlim_user[0]), min(1, xlim_user[1]))
                    else:
                        ax.set_xlim(max(0, xlim_user[0]), xlim_user[1])

                if param_limits[param_y] is not None:
                    ylim_user = param_limits[param_y]
                    if param_y == 'p0':
                        ax.set_ylim(max(0, ylim_user[0]), min(1, ylim_user[1]))
                    else:
                        ax.set_ylim(max(0, ylim_user[0]), ylim_user[1])

                # Labels and titles
                # X-labels: bottom row only
                if pair_idx == n_pairs - 1:
                    ax.set_xlabel(labels_latex[idx_x], fontsize=FONTSIZE_LABEL)
                else:
                    ax.set_xlabel('')

                # Y-labels: left column only (same for all panels in a row)
                if day_idx == 0:
                    ax.set_ylabel(labels_latex[idx_y], fontsize=FONTSIZE_LABEL)
                else:
                    ax.set_ylabel('')

                # Column titles (day labels) - top row only
                if pair_idx == 0:
                    ax.set_title(f'Day {day}', fontsize=FONTSIZE_TITLE, fontweight='bold')

                # Row labels (parameter pairs) - left column only
                if day_idx == 0:
                    ax.text(-0.35, 0.5, f'{labels_latex[idx_x]} vs {labels_latex[idx_y]}',
                           transform=ax.transAxes, fontsize=FONTSIZE_ROW_LABEL, fontweight='bold',
                           va='center', rotation=90)

                # Style tick labels
                ax.tick_params(labelsize=FONTSIZE_TICK)

                # Remove right and top spines for cleaner look
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.grid(alpha=0.3, linewidth=0.5)
                # Don't force equal aspect - parameters can have very different scales
                # especially for scores (magnitude vs halftime can differ by orders of magnitude)

            except KeyError:
                print(f"Warning: No data for {genotype} Day {day} Fly {fly_id}")
                continue

    # Overall title
    score_type = 'Scores' if conv_to_scores else 'Parameters'
    overlay_str = f' ({kde_overlay} KDE)' if kde_overlay else ''
    fig.suptitle(f'Pairwise Posterior {score_type} - {genotype} Fly #{fly_id}{overlay_str}',
                fontsize=FONTSIZE_SUPTITLE, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        print(f"Saved: {save_path}.png and .svg")

    plt.show()


def plot_pairwise_scatters_multifly(draws_over_age, genotype, fly_ids, days_to_show=[7, 14, 21],
                                     param_limits=None, conv_to_scores=False, kde_overlay='contour',
                                     hide_interior_ticklabels=True, save_path=None):
    """
    Create grid of pairwise scatter plots: rows=flies, cols=parameter pairs.
    Each panel combines all days (with different colors).

    Args:
        draws_over_age: Dictionary of draws {genotype: {day: {fly_id: samples}}}
        genotype: 'KK' or 'GD'
        fly_ids: List of fly ID numbers (one per row)
        days_to_show: List of days to overlay in each panel
        param_limits: Optional dict for axis limits {param: (min, max)}
        conv_to_scores: If True, convert alpha, beta to magnitude, halftime
        kde_overlay: 'contour', 'fill', or None
        hide_interior_ticklabels: If True, hide x-tick labels except bottom row
        save_path: If provided, save figure (without extension)
    """
    # Define color mapping for each day
    colors = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}

    # Define labels
    if conv_to_scores:
        labels = ['magnitude', 'halftime', 'p0']
        labels_latex = ['$M$', '$T_{1/2}$', r'$p_0$']
    else:
        labels = ['alpha', 'beta', 'p0']
        labels_latex = [r'$\alpha$', r'$\beta$', r'$p_0$']

    if param_limits is None:
        param_limits = {k: None for k in labels}

    # Define pairwise combinations (column order)
    pairs = [
        (0, 1),  # (param1, param2): e.g., (alpha, beta)
        (0, 2),  # (param1, param3): e.g., (alpha, p0)
        (1, 2)   # (param2, param3): e.g., (beta, p0)
    ]

    # Create figure: rows=flies, cols=pairs
    n_flies = len(fly_ids)
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_flies, n_pairs, figsize=(7.5, 2.5 * n_flies))

    # Ensure axes is 2D
    if n_flies == 1:
        axes = axes.reshape(1, -1)
    if n_pairs == 1:
        axes = axes.reshape(-1, 1)

    # Font sizes
    FONTSIZE_TITLE = 14
    FONTSIZE_SUPTITLE = 16
    FONTSIZE_ROW_LABEL = 13
    FONTSIZE_TICK = 10
    FONTSIZE_LEGEND = 9

    # Process each fly (row) and pair (column)
    for fly_idx, fly_id in enumerate(fly_ids):
        for pair_idx, (idx_x, idx_y) in enumerate(pairs):
            param_x = labels[idx_x]
            param_y = labels[idx_y]

            ax = axes[fly_idx, pair_idx]

            # Loop through days and overlay them in this panel
            for day in days_to_show:
                try:
                    samples = draws_over_age[genotype][day]['fly_id_' + str(fly_id)]

                    # Extract samples from .npz if needed
                    if isinstance(samples, np.lib.npyio.NpzFile):
                        samples = samples[list(samples.files)[0]]

                    # Apply thinning
                    if N_DRAWS_TO_USE is not None and len(samples) > N_DRAWS_TO_USE:
                        thin_factor = len(samples) // N_DRAWS_TO_USE
                        indices = np.arange(0, len(samples), thin_factor)[:N_DRAWS_TO_USE]
                        samples = samples[indices]

                    # Ensure expected format
                    if not (isinstance(samples, np.ndarray) and samples.shape[1] == 3):
                        print(f"Warning: Unexpected format for {genotype} Day {day} Fly {fly_id}")
                        continue

                    alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

                    # Convert to scores if requested
                    if conv_to_scores:
                        mag_rel = compute_hab_magnitude_rel(alpha, beta, T=1)
                        halftime_rel = compute_hab_time_half_rel(alpha, beta, T=1)
                        data = [mag_rel, halftime_rel, p0]
                    else:
                        data = [alpha, beta, p0]

                    # Clip data to valid ranges
                    for i, label in enumerate(labels):
                        if label == 'p0':
                            data[i] = np.clip(data[i], 0, 1)
                        else:
                            data[i] = np.clip(data[i], 0, None)

                    # Compute means
                    means = [np.mean(d) for d in data]

                    x_data = data[idx_x]
                    y_data = data[idx_y]

                    # Scatter plot (smaller alpha since overlaying multiple days)
                    ax.scatter(x_data, y_data, alpha=0.2, s=8, color=colors[day],
                              edgecolor='none', label=f'Day {day}')

                    # KDE overlay
                    if kde_overlay in ['contour', 'fill']:
                        try:
                            df_kde = pd.DataFrame({param_x: x_data, param_y: y_data})

                            if kde_overlay == 'contour':
                                sns.kdeplot(data=df_kde, x=param_x, y=param_y, ax=ax,
                                           color=colors[day], levels=3, linewidths=1.0, alpha=0.6)
                            else:  # fill
                                sns.kdeplot(data=df_kde, x=param_x, y=param_y, ax=ax,
                                           color=colors[day], fill=True, alpha=0.2, levels=3)
                        except Exception as e:
                            print(f"Warning: KDE overlay failed for day {day}, fly {fly_id}: {e}")

                    # Plot mean
                    ax.scatter([means[idx_x]], [means[idx_y]], color=colors[day],
                              edgecolor='black', s=100, marker='o', zorder=10, linewidth=1.5)

                except KeyError:
                    print(f"Warning: No data for {genotype} Day {day} Fly {fly_id}")
                    continue

            # Set axis limits
            if param_x == 'p0':
                ax.set_xlim(0, 1)
            else:
                xlim_current = ax.get_xlim()
                ax.set_xlim(0, xlim_current[1])

            if param_y == 'p0':
                ax.set_ylim(0, 1)
            else:
                ylim_current = ax.get_ylim()
                ax.set_ylim(0, ylim_current[1])

            # Apply user-specified limits if provided
            if param_limits[param_x] is not None:
                xlim_user = param_limits[param_x]
                if param_x == 'p0':
                    ax.set_xlim(max(0, xlim_user[0]), min(1, xlim_user[1]))
                else:
                    ax.set_xlim(max(0, xlim_user[0]), xlim_user[1])

            if param_limits[param_y] is not None:
                ylim_user = param_limits[param_y]
                if param_y == 'p0':
                    ax.set_ylim(max(0, ylim_user[0]), min(1, ylim_user[1]))
                else:
                    ax.set_ylim(max(0, ylim_user[0]), ylim_user[1])

            # No x or y axis labels (titles show parameter names instead)
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Column titles (parameter pairs with x/y annotation) - top row only
            if fly_idx == 0:
                ax.set_title(f'x: {labels_latex[idx_x]}, y: {labels_latex[idx_y]}',
                           fontsize=FONTSIZE_TITLE, fontweight='bold')

            # Row labels (fly IDs) - left column only
            if pair_idx == 0:
                ax.text(-0.3, 0.5, f'Fly {fly_id}', transform=ax.transAxes,
                       fontsize=FONTSIZE_ROW_LABEL, fontweight='bold',
                       va='center', rotation=90)

            # Add legend to first panel only
            if fly_idx == 0 and pair_idx == 0:
                ax.legend(loc='upper right', fontsize=FONTSIZE_LEGEND, framealpha=0.8)

            # Hide interior tick labels if requested
            if hide_interior_ticklabels:
                # Hide x-tick labels except bottom row
                if fly_idx < n_flies - 1:
                    ax.set_xticklabels([])
                # Hide y-tick labels only on right column (since each column has different y-axis)
                # Left and middle columns keep y-tick labels since they plot different parameters
                if pair_idx == 2:  # Right column only
                    ax.set_yticklabels([])

            # Style
            ax.tick_params(labelsize=FONTSIZE_TICK)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.grid(alpha=0.3, linewidth=0.5)

    # Overall title
    score_type = 'Scores' if conv_to_scores else 'Parameters'
    overlay_str = f' ({kde_overlay} KDE)' if kde_overlay else ''
    days_str = ', '.join([str(d) for d in days_to_show])
    fig.suptitle(f'Pairwise Posterior {score_type} - {genotype} (Days {days_str}){overlay_str}',
                fontsize=FONTSIZE_SUPTITLE, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        print(f"Saved: {save_path}.png and .svg")

    plt.show()


def plot_overlayed_kde_multiple_days(draws_over_age, genotype, fly_id, days_to_show=[7, 14, 21], param_limits=None,
                                     fill_kde=True, conv_to_scores=False, save_path=None):
    """
    Overlays KDEs for the same fly across multiple days (7, 14, 21) with distinct colors.

    Args:
        draws_over_age: Dictionary of draws in the format {genotype: {day: {fly_id: samples}}}
        genotype: Genotype to plot ('KK' or 'GD')
        fly_id: Fly ID in the format 'fly_id_X'
        param_limits: Optional dict for axis limits
        fill_kde: Whether to fill the KDE plots
        conv_to_scores: If True, convert alpha, beta to ---> hab_mag_rel, hab_time_half_rel
        save_path: If provided, save figure as PNG and SVG (without extension)
    """
    # Define color mapping for each day
    colors = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}

    # Extract and organize data for the fly across all days
    dfs = []
    mean_values = {}

    if conv_to_scores:
        labels = ['magnitude', 'halftime', 'p0']
        labels_latex = ['$M$', '$T_{1/2}$', r'$p_0$']
    else:
        labels = ['alpha', 'beta', 'p0']
        labels_latex = [r'$\alpha$', r'$\beta$', r'$p_0$']
    if param_limits is None:
        param_limits = {k: None for k in labels}

    for day in days_to_show:
        try:
            samples = draws_over_age[genotype][day]['fly_id_' + str(fly_id)]

            # if samples are stored in .npz format, extract the array
            if isinstance(samples, np.lib.npyio.NpzFile):
                samples = samples[list(samples.files)[0]]

            # Apply thinning to speed up computation
            if N_DRAWS_TO_USE is not None and len(samples) > N_DRAWS_TO_USE:
                thin_factor = len(samples) // N_DRAWS_TO_USE
                indices = np.arange(0, len(samples), thin_factor)[:N_DRAWS_TO_USE]
                samples = samples[indices]

            # ensure the samples are in the expected format
            if isinstance(samples, np.ndarray) and samples.shape[1] == 3:
                alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]
                if conv_to_scores:
                    # magnitude (abs): "p0 - p_inf" hab magnitude (T=1s)
                    mag_rel = compute_hab_magnitude_rel(alpha, beta, T=1)
                    mag_abs = p0 * mag_rel
                    mag_choice = mag_rel  # choose which magnitude (abs <-> rel) to plot
                    # speed: relative halftime, i.e. 0.5 = R[k^*] = (p[k^*] - p_inf) / (p0 - p_inf)
                    halftime_rel = compute_hab_time_half_rel(alpha, beta, T=1)
                    # Store means for annotation
                    mean_values[day] = (np.mean(mag_choice), np.mean(halftime_rel), np.mean(p0))
                    # Create a DataFrame for Seaborn plotting
                    dict_for_df = {labels[0]: mag_choice, labels[1]: halftime_rel, labels[2]: p0, 'day': day}

                else:
                    # Store means for annotation
                    mean_values[day] = (np.mean(alpha), np.mean(beta), np.mean(p0))
                    # Create a DataFrame for Seaborn plotting
                    #df = pd.DataFrame({labels[0]: alpha, labels[1]: beta, labels[2]: p0, 'day': day})
                    dict_for_df = {labels[0]: alpha, labels[1]: beta, labels[2]: p0, 'day': day}

                df = pd.DataFrame(dict_for_df)
                dfs.append(df)
            else:
                print(f"Warning: Unexpected format for {genotype} Day {day} Fly {fly_id}")

        except KeyError:
            print(f"Warning: No data for {genotype} Day {day} Fly {fly_id}")
            continue

    # Combine DataFrames across all days
    combined_df = pd.concat(dfs, ignore_index=True)

    # Clip all data to valid ranges to ensure KDEs don't extend into invalid regions
    # This also ensures consistent "walls" at boundaries for all days
    for label in labels:
        if label == 'p0':
            combined_df[label] = combined_df[label].clip(lower=0, upper=1)
        else:
            # All other parameters/scores are non-negative
            combined_df[label] = combined_df[label].clip(lower=0)

    # Define clip ranges for KDE computation (non-negative support)
    clip_global = (0, None)

    # Generate Pairwise KDE Plot
    # clip=(0, None): constrains KDE evaluation domain to [0, ∞) - no negative support
    # We use cut=0 to prevent extending beyond data, but will manually set axis limits below
    g = sns.pairplot(
        combined_df,
        kind='kde',
        diag_kind='kde',
        corner=True,
        hue='day',
        palette=colors,
        plot_kws={'alpha': 0.4, 'fill': fill_kde, 'cut': 0, 'clip': clip_global},
        diag_kws={'cut': 0, 'clip': clip_global}
    )
    g.fig.subplots_adjust(wspace=0.3, hspace=0.2)  # Adjust horizontal/vertical spacing (def: 0.2)

    # Set consistent axis limits across all panels to avoid jarring visual inconsistencies
    # - All parameters start at 0 (non-negative support)
    # - p0 is constrained to [0, 1] (probability)
    for i, param_y in enumerate(labels):
        for j, param_x in enumerate(labels):
            if j <= i:  # Only lower triangle + diagonal
                ax = g.axes[i, j]

                # Set x-axis limits
                if param_x == 'p0':
                    # p0 is a probability: [0, 1]
                    current_xlim = ax.get_xlim()
                    ax.set_xlim(0, 1)
                else:
                    # All other params: start at 0, keep upper limit from data
                    current_xlim = ax.get_xlim()
                    ax.set_xlim(0, current_xlim[1])

                # Set y-axis limits for off-diagonal plots
                if i != j:
                    if param_y == 'p0':
                        # p0 is a probability: [0, 1]
                        current_ylim = ax.get_ylim()
                        ax.set_ylim(0, 1)
                    else:
                        # All other params: start at 0, keep upper limit from data
                        current_ylim = ax.get_ylim()
                        ax.set_ylim(0, current_ylim[1])

    # Plot 95% CI bounds and median on diagonal panels
    # Note: These are computed from the raw data, while KDE is smoothed
    for i, param in enumerate(labels):
        ax = g.axes[i, i]
        for day in days_to_show:

            # Access the data array for this day and parameter
            x = combined_df[combined_df['day'] == day][param].values
            pdf_summary = helper_summarize_univariate_samples(x)

            # Plot CI bounds (2.5th and 97.5th percentiles)
            ax.axvline(pdf_summary['CI_lower'], color=colors[day], linestyle='--',
                      linewidth=1, alpha=0.6, label=f'Day {day} 95% CI')
            ax.axvline(pdf_summary['CI_upper'], color=colors[day], linestyle='--',
                      linewidth=1, alpha=0.6)

            # Plot median for reference
            ax.axvline(pdf_summary['median'], color=colors[day], linestyle='-',
                      linewidth=1.5, alpha=0.8)

            # Summary stats computed by helper function:
            # - mean, median, std
            # - CI_lower: 2.5th percentile
            # - CI_upper: 97.5th percentile

    # Annotate Posterior Means for Each Day
    for day, (alpha_mean, beta_mean, p0_mean) in mean_values.items():
        for i, param_y in enumerate(labels):
            for j, param_x in enumerate(labels):
                if j < i:
                    mean_x = alpha_mean if param_x == labels[0] else beta_mean if param_x == labels[1] else p0_mean
                    mean_y = alpha_mean if param_y == labels[0] else beta_mean if param_y == labels[1] else p0_mean

                    ax = g.axes[i, j]
                    ax.scatter(
                        [mean_x], [mean_y],
                        color=colors[day],
                        edgecolor='black',
                        s=80,
                        label=f'Day {day} Mean',
                        zorder=20
                    )

                    # Apply parameter-specific axis limits if provided
                    # Ensure they respect the minimum bounds (0 for all, max 1 for p0)
                    if param_limits[param_x] is not None:
                        xlim_user = param_limits[param_x]
                        if param_x == 'p0':
                            # p0 must be in [0, 1]
                            ax.set_xlim(max(0, xlim_user[0]), min(1, xlim_user[1]))
                        else:
                            # Other params must be non-negative
                            ax.set_xlim(max(0, xlim_user[0]), xlim_user[1])

                    if param_limits[param_y] is not None:
                        ylim_user = param_limits[param_y]
                        if param_y == 'p0':
                            # p0 must be in [0, 1]
                            ax.set_ylim(max(0, ylim_user[0]), min(1, ylim_user[1]))
                        else:
                            # Other params must be non-negative
                            ax.set_ylim(max(0, ylim_user[0]), ylim_user[1])

    g._legend.set_bbox_to_anchor((0.85, 0.55))  # (x, y) position
    g._legend.set_title("Day")  # Optional: Set the title of the legend

    # Create annotation text for mean values
    annotation_text = "\n".join([
        f"Day {day}: {labels_latex[0]}={mean_values[day][0]:.3f}, {labels_latex[1]}={mean_values[day][1]:.3f}, {labels_latex[2]}={mean_values[day][2]:.3f}"
        for day in mean_values
    ])

    # Add annotation text to the top-right corner
    plt.gcf().text(0.55, 0.75, f"Fly ID: {fly_id}\nGenotype: {genotype}\n\n{annotation_text}",
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Final touches
    plt.suptitle(f"Posterior KDE for {genotype} fly #{fly_id}", y=0.99)

    # Add legend
    #handles, labels = ax.get_legend_handles_labels()
    #if handles:
    #    plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # Save figure if save_path provided
    if save_path is not None:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        print(f"Saved: {save_path}.png and .svg")

    plt.show()


if __name__ == '__main__':

    # Load posterior draws for multiple days/genotypes
    draws_GD_day7 = np.load(DIR_FITS + os.sep + "GD_day7_3d_draws.npz", allow_pickle=True)
    draws_GD_day14 = np.load(DIR_FITS + os.sep + "GD_day14_3d_draws.npz", allow_pickle=True)
    draws_GD_day21 = np.load(DIR_FITS + os.sep + "GD_day21_3d_draws.npz", allow_pickle=True)

    draws_KK_day7 = np.load(DIR_FITS + os.sep + "KK_day7_3d_draws.npz", allow_pickle=True)
    draws_KK_day14 = np.load(DIR_FITS + os.sep + "KK_day14_3d_draws.npz", allow_pickle=True)
    draws_KK_day21 = np.load(DIR_FITS + os.sep + "KK_day21_3d_draws.npz", allow_pickle=True)

    draws_over_age = {
        'KK': {7: draws_KK_day7, 14: draws_KK_day14, 21: draws_KK_day21},
        'GD': {7: draws_GD_day7, 14: draws_GD_day14, 21: draws_GD_day21}
    }

    # Compute global axis limits across all datasets
    param_limits_dict = compute_global_param_limits_from_draws(draws_over_age)
    print('param_limits_dict.keys():', param_limits_dict.keys())

    # now get draws for specific fly
    genotype = 'KK'
    fly_id = 74 # 40, 44, 74, 105
    day = 7

    # Axis x/y limits for subpanels
    # - note: param_limits is a dict with structure {alpha: (min, max), ...}
    #param_limits_manual = dict(alpha=None, beta=None, p0=[0.0, 1.0])
    param_limits_manual = dict(alpha=[0.0, 0.65], beta=[0.0, 1.0], p0=[0.0, 1.0])
    param_limits_manual_scores = dict(magnitude=None, halftime=[0.0, 30], p0=[0.0, 1.0])

    draws = draws_over_age[genotype][day]

    # Access posterior triplets for a specific fly (e.g., fly 3)
    fly_triplets = draws[f"fly_id_{fly_id}"]  # Shape: (n_draws, 3)

    # Apply thinning to speed up computation
    if N_DRAWS_TO_USE is not None and len(fly_triplets) > N_DRAWS_TO_USE:
        thin_factor = len(fly_triplets) // N_DRAWS_TO_USE
        indices = np.arange(0, len(fly_triplets), thin_factor)[:N_DRAWS_TO_USE]
        fly_triplets = fly_triplets[indices]

    alpha_draws = fly_triplets[:, 0]
    beta_draws  = fly_triplets[:, 1]
    p0_draws    = fly_triplets[:, 2]

    print('posterior draws shape:', fly_triplets.shape)
    ndraws, ndim = fly_triplets.shape
    assert ndim == 3

    # [unused] make a 3d scatter plot
    #plot_posterior_3d_scatter(alpha_draws, beta_draws, p0_draws)

    plot_overlayed_kde_multiple_days(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        #param_limits=param_limits_dict['all'],
        #param_limits=param_limits_dict[f'{genotype}'],
        param_limits=param_limits_manual,
        days_to_show=[14],
        fill_kde=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_kde_{genotype}_fly{fly_id}_d14_filled_params')
    )
    plot_overlayed_kde_multiple_days(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        # param_limits=param_limits_dict['all'],
        # param_limits=param_limits_dict[f'{genotype}'],
        param_limits=param_limits_manual_scores,
        days_to_show=[14],
        fill_kde=True,
        conv_to_scores=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_kde_{genotype}_fly{fly_id}_d14_filled_scores')
    )

    # Example input dictionaries with posterior draws for days 7, 14, 21
    plot_overlayed_kde_multiple_days(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        #param_limits=param_limits_dict['all'],
        #param_limits=param_limits_dict[f'{genotype}'],
        param_limits=param_limits_manual,
        days_to_show=[7, 14, 21],
        fill_kde=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_kde_{genotype}_fly{fly_id}_d7_14_21_filled_params')
    )

    plot_overlayed_kde_multiple_days(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        param_limits=param_limits_manual,
        days_to_show=[7, 14, 21],
        fill_kde=False,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_kde_{genotype}_fly{fly_id}_d7_14_21_outline_params')
    )

    plot_overlayed_kde_multiple_days(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        param_limits=param_limits_manual_scores,
        #param_limits=None,
        days_to_show=[7, 14, 21],
        fill_kde=False,
        conv_to_scores=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_kde_{genotype}_fly{fly_id}_d7_14_21_outline_scores')
    )

    # Assuming `draws` is the posterior sample array for a fly and `foo` computes p(t)
    # Call the updated function with the draws_over_age object
    plot_posterior_likelihood_summary_over_days(
        draws_over_age=draws_over_age,
        genotype='KK',
        fly_id=fly_id,
        #days_to_show=[7, 14, 21],
        days_to_show=[14],
        spaghetti=False,   # show all curves with spaghetti_alpha transparency
        ci_level=0.95,
        n_samples=N_DRAWS_TO_USE,
        show=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_ppc_{genotype}_fly{fly_id}_d14')
    )

    # now do same plot for this fly but over all three ages
    plot_posterior_likelihood_summary_over_days(
        draws_over_age=draws_over_age,
        genotype='KK',
        fly_id=fly_id,
        days_to_show=[7, 14, 21],
        spaghetti=False,  # show all curves with spaghetti_alpha transparency
        ci_level=0.95,
        n_samples=N_DRAWS_TO_USE,
        show=True,
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'posterior_ppc_{genotype}_fly{fly_id}_d7_14_21')
    )

    # ========================================
    # NEW: Pairwise scatter plots (3x3 grid)
    # ========================================
    print("\n" + "="*70)
    print("CREATING PAIRWISE SCATTER PLOTS (3 days × 3 pairs)")
    print("="*70)

    # Parameters with contour KDE
    plot_pairwise_scatters_by_day(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        days_to_show=[7, 14, 21],
        param_limits=param_limits_manual,
        conv_to_scores=False,
        kde_overlay='contour',
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_scatter_{genotype}_fly{fly_id}_params_contour')
    )

    # Parameters with filled KDE
    plot_pairwise_scatters_by_day(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        days_to_show=[7, 14, 21],
        param_limits=param_limits_manual,
        conv_to_scores=False,
        kde_overlay='fill',
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_scatter_{genotype}_fly{fly_id}_params_fill')
    )

    # Just scatters, no KDE overlay
    plot_pairwise_scatters_by_day(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        days_to_show=[7, 14, 21],
        param_limits=param_limits_manual,
        conv_to_scores=False,
        kde_overlay=None,  # 'contour' / 'fill' / None
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_scatter_{genotype}_fly{fly_id}_params_noKDE')
    )

    # Scores with contour KDE
    plot_pairwise_scatters_by_day(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_id=fly_id,
        days_to_show=[7, 14, 21],
        #param_limits=param_limits_manual_scores,
        conv_to_scores=True,
        kde_overlay='contour',
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_scatter_{genotype}_fly{fly_id}_scores_contour')
    )

    # ========================================
    # NEW: Multi-fly comparison (3 rows × N flies)
    # ========================================
    print("\n" + "="*70)
    print("CREATING MULTI-FLY COMPARISON PLOTS")
    print("="*70)

    # Define list of flies to compare
    #fly_ids_to_compare = [40, 44, 74, 105]  # Example flies
    fly_ids_to_compare = [20, 21, 22, 23]  # Example flies

    # Parameters with contour KDE
    plot_pairwise_scatters_multifly(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_ids=fly_ids_to_compare,
        days_to_show=[7, 14, 21],
        param_limits=param_limits_manual,
        conv_to_scores=False,
        kde_overlay=None,  # fill, contour, or None
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_multifly_{genotype}_params_contour')
    )

    # Scores with filled KDE
    plot_pairwise_scatters_multifly(
        draws_over_age=draws_over_age,
        genotype=genotype,
        fly_ids=fly_ids_to_compare,
        days_to_show=[7, 14, 21],
        #param_limits=param_limits_manual_scores,
        conv_to_scores=True,
        kde_overlay='fill',
        save_path=os.path.join(DIR_OUTPUT_SANDBOX, f'pairwise_multifly_{genotype}_scores_fill')
    )
