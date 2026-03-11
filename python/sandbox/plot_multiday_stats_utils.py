import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import random
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist, euclidean, squareform, pdist
from scipy.interpolate import interp1d
from scipy import stats
from seaborn import light_palette, rugplot

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

from data_tools import df_to_arr_jumps
from data_format_add_score_columns import compute_p_ss, compute_hab_magnitude_rel, compute_hab_time_half_rel
from functions_common import likelihood_func_vec
from settings import DIR_OUTPUT, DIR_FITS, days_palettes

# ==== CORE UTILITY FUNCTIONS ====

def filter_complete_flies(df_pheno, required_ages=[7, 14, 21]):
    """Get only flies that have data for all required ages."""
    grp = df_pheno.groupby('fly_id')['day'].unique()
    complete_ids = [fid for fid, days in grp.items()
                    if set(required_ages).issubset(set(days))]
    return complete_ids


def extract_parameters_by_age(df_pheno, fly_ids, param_names, ages=[7, 14, 21]):
    """Extract parameter values by age for specified flies."""
    params_by_age = {age: {} for age in ages}

    for age in ages:
        data_age = df_pheno[df_pheno['day'] == age].set_index('fly_id')
        for param in param_names:
            params_by_age[age][param] = np.array([data_age.loc[fid, param]
                                                  if fid in data_age.index else np.nan
                                                  for fid in fly_ids])

    return params_by_age


def calculate_fly_distances(df_pheno, param_names, ages=[7, 14, 21]):
    """Calculate within-fly and between-fly distances."""
    complete_ids = filter_complete_flies(df_pheno, ages)
    distances = {}

    for fly_id in complete_ids:
        coords = []
        for age in ages:
            data = df_pheno[(df_pheno['fly_id'] == fly_id) & (df_pheno['day'] == age)]
            if len(data) == 0:
                continue
            coords.append(np.array([data[param].values[0] for param in param_names]))

        if len(coords) != len(ages):
            continue

        # Calculate pairwise distances
        distances[fly_id] = {
            f'd{ages[0]}_{ages[1]}': euclidean(coords[0], coords[1]),
            f'd{ages[1]}_{ages[2]}': euclidean(coords[1], coords[2]),
            f'd{ages[0]}_{ages[2]}': euclidean(coords[0], coords[2]),
            'path_length': euclidean(coords[0], coords[1]) + euclidean(coords[1], coords[2])
        }

    return distances


# ==== CORE PLOTTING FUNCTIONS ====

def setup_scatter_figure(nrows, ncols, figsize=None, with_marginals=False):
    """Create a figure with optional marginal plots."""
    if with_marginals:
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        fig = plt.figure(figsize=figsize)
        outer_grid = gridspec.GridSpec(nrows, ncols, figure=fig,
                                       height_ratios=[1] * nrows,
                                       hspace=0.3, wspace=0.3)

        # Create arrays for axes
        main_axes = np.empty((nrows, ncols), dtype=object)
        top_margins = np.empty((nrows, ncols), dtype=object)
        right_margins = np.empty((nrows, ncols), dtype=object)

        # Set up each subplot with margins
        for row in range(nrows):
            for col in range(ncols):
                inner_grid = gridspec.GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=outer_grid[row, col],
                    height_ratios=[0.2, 0.8], width_ratios=[0.8, 0.2],
                    hspace=0.05, wspace=0.05
                )

                main_axes[row, col] = fig.add_subplot(inner_grid[1, 0])
                top_margins[row, col] = fig.add_subplot(inner_grid[0, 0], sharex=main_axes[row, col])
                top_margins[row, col].tick_params(axis='x', labelbottom=False)
                right_margins[row, col] = fig.add_subplot(inner_grid[1, 1], sharey=main_axes[row, col])
                right_margins[row, col].tick_params(axis='y', labelleft=False)

        return fig, main_axes, top_margins, right_margins

    else:
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(max(nrows, ncols), 1) if nrows > 1 else axes.reshape(1, ncols)

        return fig, axes, None, None


def plot_scatter_with_highlights(ax, x, y, highlight_indices=None, highlight_colors=None,
                                 regular_color='C0', show_error_bars=False,
                                 x_err=None, y_err=None, add_equality_line=True,
                                 point_labels=None):
    """Plot scatter with highlighted points and optional error bars."""
    # Determine regular vs highlighted indices
    if highlight_indices is None:
        highlight_indices = []

    all_indices = np.arange(len(x))
    regular_indices = np.setdiff1d(all_indices, highlight_indices)

    # Plot regular points
    if len(regular_indices) > 0:
        ax.scatter(x[regular_indices], y[regular_indices],
                   s=20, color=regular_color, alpha=0.7)

    # Plot highlighted points
    for i, idx in enumerate(highlight_indices):
        if idx >= len(x):
            continue

        color = highlight_colors[i] if highlight_colors is not None else f"C{(idx % 10) + 1}"

        # Plot point
        ax.scatter(x[idx], y[idx], s=80, color=color, alpha=0.8,
                   marker='o', edgecolor='black', linewidth=1.5)

        # Add error bars if requested
        if show_error_bars and x_err is not None and y_err is not None:
            ax.errorbar(x[idx], y[idx],
                        xerr=[[x_err[0][idx]], [x_err[1][idx]]] if isinstance(x_err, tuple) else [[x_err[idx]],
                                                                                                  [x_err[idx]]],
                        yerr=[[y_err[0][idx]], [y_err[1][idx]]] if isinstance(y_err, tuple) else [[y_err[idx]],
                                                                                                  [y_err[idx]]],
                        fmt='none', ecolor=color, elinewidth=1.5, capsize=4, alpha=0.8)

        # Add label if provided
        if point_labels is not None and idx < len(point_labels):
            ax.text(x[idx], y[idx] * 1.05, point_labels[idx],
                    ha='center', va='bottom', color=color,
                    fontweight='bold', fontsize=10)

    # Add equality line if requested
    if add_equality_line:
        xlim = ax.get_xlim() if ax.get_xlim() != (0, 1) else (min(x) * 0.9, max(x) * 1.1)
        ylim = ax.get_ylim() if ax.get_ylim() != (0, 1) else (min(y) * 0.9, max(y) * 1.1)
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)


def add_stats_textbox(ax, x, y, position=(0.05, 0.95), additional_stats=None):
    """Add correlation and other statistics in a text box."""
    # Calculate correlation
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 1:
        corr, p_value = stats.pearsonr(x[mask], y[mask])
        mae = np.mean(np.abs(y[mask] - x[mask]))

        stats_text = f'r = {corr:.2f}\np = {p_value:.3f}\nMAE = {mae:.3f}'

        if additional_stats:
            stats_text += '\n' + '\n'.join(additional_stats)

        ax.text(position[0], position[1], stats_text,
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def add_marginal_histograms(x, y, top_ax, right_ax, highlight_indices=None,
                            highlight_colors=None, bins=15, add_kde=False):
    """Add marginal histograms and optional KDE to the top and right axes."""
    if top_ax is not None and right_ax is not None:
        # Top margin (x axis)
        top_ax.hist(x, bins=bins, alpha=0.6, color='C0')

        # Right margin (y axis)
        right_ax.hist(y, bins=bins, alpha=0.6, color='C0', orientation='horizontal')

        # Add KDE if requested
        if add_kde and len(x) > 3:
            try:
                x_grid = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
                kde_x = stats.gaussian_kde(x)
                top_ax.plot(x_grid, kde_x(x_grid), color='black', alpha=0.7)

                y_grid = np.linspace(min(y) * 0.9, max(y) * 1.1, 100)
                kde_y = stats.gaussian_kde(y)
                right_ax.plot(kde_y(y_grid), y_grid, color='black', alpha=0.7)
            except:
                pass  # In case KDE fails

        # Add highlight markers if provided
        if highlight_indices is not None:
            for i, idx in enumerate(highlight_indices):
                if idx >= len(x):
                    continue

                color = highlight_colors[i] if highlight_colors is not None else f"C{(idx % 10) + 1}"

                # Add lines to marginals
                top_ax.axvline(x[idx], color=color, linestyle='-', linewidth=2, alpha=0.8)
                right_ax.axhline(y[idx], color=color, linestyle='-', linewidth=2, alpha=0.8)


def create_highlight_legend(ax, highlight_ids, label_prefix="Fly #"):
    """Create legend with highlighted elements."""
    handles = []
    labels = []

    for fly_id in highlight_ids:
        fly_color = f"C{fly_id % 10}"
        handle = Line2D([0], [0], marker='o', color=fly_color,
                        markeredgecolor='black', markerfacecolor=fly_color,
                        markersize=8, linestyle='None')
        handles.append(handle)
        labels.append(f'{label_prefix}{fly_id}')

    # Add legend if we have entries
    if handles:
        ax.legend(handles, labels, loc='lower right', frameon=True,
                  framealpha=0.8, edgecolor='black')

    return handles, labels


# ==== HIGHER-LEVEL PLOTTING FUNCTIONS ====

def plot_parameter_comparisons_across_ages(df_pheno, param_names, highlight_fly_ids=None,
                                           show_marginals=False, title=None):
    """
    Create scatter plots comparing parameters across ages.
    """
    ages = [7, 14, 21]
    age_pairs = [(7, 14), (14, 21), (7, 21)]

    # Setup figure with or without marginals
    fig, axes, top_margins, right_margins = setup_scatter_figure(
        len(param_names), len(age_pairs),
        figsize=(15, 5 * len(param_names)), with_marginals=show_marginals
    )

    # Process each parameter
    for param_idx, param_name in enumerate(param_names):
        param_axes = axes[param_idx]

        # Process each age pair
        for col_idx, (age1, age2) in enumerate(age_pairs):
            ax = param_axes[col_idx]

            # Get data for flies present at both ages
            df_age1 = df_pheno[df_pheno['day'] == age1]
            df_age2 = df_pheno[df_pheno['day'] == age2]

            # Find common fly IDs
            common_fly_ids = list(set(df_age1['fly_id']).intersection(set(df_age2['fly_id'])))

            # Extract data arrays
            x_values = []
            y_values = []
            x_err_low = []
            x_err_high = []
            y_err_low = []
            y_err_high = []

            for fly_id in common_fly_ids:
                data1 = df_age1[df_age1['fly_id'] == fly_id]
                data2 = df_age2[df_age2['fly_id'] == fly_id]

                if len(data1) == 0 or len(data2) == 0:
                    continue

                x_values.append(data1[param_name].values[0])
                y_values.append(data2[param_name].values[0])

                # Get error bars if available
                if f"{param_name}_ci_low" in data1.columns:
                    x_err_low.append(data1[param_name].values[0] - data1[f"{param_name}_ci_low"].values[0])
                    x_err_high.append(data1[f"{param_name}_ci_high"].values[0] - data1[param_name].values[0])
                    y_err_low.append(data2[param_name].values[0] - data2[f"{param_name}_ci_low"].values[0])
                    y_err_high.append(data2[f"{param_name}_ci_high"].values[0] - data2[param_name].values[0])

            # Convert to numpy arrays
            x_values = np.array(x_values)
            y_values = np.array(y_values)

            # Find highlighted indices
            highlight_indices = [i for i, fid in enumerate(common_fly_ids)
                                 if highlight_fly_ids is not None and fid in highlight_fly_ids]

            # Set consistent plot limits
            all_values = np.concatenate([df_pheno[param_name].values,
                                         x_values, y_values])
            min_val = np.nanmin(all_values) - np.nanstd(all_values) * 0.1
            max_val = np.nanmax(all_values) + np.nanstd(all_values) * 0.1

            # Plot the data
            plot_scatter_with_highlights(
                ax, x_values, y_values,
                highlight_indices=highlight_indices,
                show_error_bars=len(x_err_low) > 0,
                x_err=(np.array(x_err_low), np.array(x_err_high)) if len(x_err_low) > 0 else None,
                y_err=(np.array(y_err_low), np.array(y_err_high)) if len(y_err_low) > 0 else None
            )

            # Add statistics
            add_stats_textbox(ax, x_values, y_values)

            # Add labels and title
            ax.set_xlabel(f'Day {age1} {param_name}')
            ax.set_ylabel(f'Day {age2} {param_name}')
            ax.set_title(f'Day {age1} vs Day {age2}: {param_name}')

            # Set consistent limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            # Add marginal plots if requested
            if show_marginals:
                add_marginal_histograms(
                    x_values, y_values,
                    top_margins[param_idx, col_idx],
                    right_margins[param_idx, col_idx],
                    highlight_indices=highlight_indices
                )

    # Add legend to first subplot
    if highlight_fly_ids:
        create_highlight_legend(axes[0, 0], highlight_fly_ids)

    # Add overall title
    plt.suptitle(title or 'Parameter Stability Across Ages', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the title

    return fig


def plot_parameter_comparisons_with_null(df_pheno, param_names, highlight_fly_ids=None,
                                         n_permutations=1000, random_seed=42,
                                         create_null_example_fig=True):
    """
    Create scatter plots comparing parameters across ages with null model overlay.
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    ages = [7, 14, 21]
    age_pairs = [(7, 14), (14, 21), (7, 21)]

    # Setup main figure
    fig, axes, _, _ = setup_scatter_figure(len(param_names), len(age_pairs),
                                           figsize=(15, 5 * len(param_names)))

    # Storage for results and examples
    null_results = {}
    null_examples = {}

    # Process each parameter
    for param_idx, param_name in enumerate(param_names):
        param_axes = axes[param_idx]

        # Process each age pair
        for col_idx, (age1, age2) in enumerate(age_pairs):
            ax = param_axes[col_idx]

            # Get data for flies present at both ages
            df_age1 = df_pheno[df_pheno['day'] == age1]
            df_age2 = df_pheno[df_pheno['day'] == age2]

            # Find common fly IDs
            common_fly_ids = list(set(df_age1['fly_id']).intersection(set(df_age2['fly_id'])))

            # Extract data arrays
            x_values = np.array([df_age1[df_age1['fly_id'] == fid][param_name].values[0]
                                 for fid in common_fly_ids])
            y_values = np.array([df_age2[df_age2['fly_id'] == fid][param_name].values[0]
                                 for fid in common_fly_ids])

            # Handle error bars if available
            have_ci = f"{param_name}_ci_low" in df_age1.columns
            if have_ci:
                x_err_low = []
                x_err_high = []
                y_err_low = []
                y_err_high = []

                for fly_id in common_fly_ids:
                    x_val = df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
                    x_low = df_age1[df_age1['fly_id'] == fly_id][f"{param_name}_ci_low"].values[0]
                    x_high = df_age1[df_age1['fly_id'] == fly_id][f"{param_name}_ci_high"].values[0]

                    y_val = df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
                    y_low = df_age2[df_age2['fly_id'] == fly_id][f"{param_name}_ci_low"].values[0]
                    y_high = df_age2[df_age2['fly_id'] == fly_id][f"{param_name}_ci_high"].values[0]

                    x_err_low.append(x_val - x_low)
                    x_err_high.append(x_high - x_val)
                    y_err_low.append(y_val - y_low)
                    y_err_high.append(y_high - y_val)

                x_err_low = np.array(x_err_low)
                x_err_high = np.array(x_err_high)
                y_err_low = np.array(y_err_low)
                y_err_high = np.array(y_err_high)

            # Calculate actual correlation and MAE
            actual_r = np.corrcoef(x_values, y_values)[0, 1]
            actual_mae = np.mean(np.abs(y_values - x_values))

            # Store a single example scrambling for visualization
            example_null_y = np.random.permutation(y_values)
            null_examples[f"{param_name}_{age1}_{age2}"] = {
                'x_values': x_values,
                'y_values': y_values,
                'example_null_y': example_null_y,
                'common_fly_ids': common_fly_ids
            }

            # Run permutation test
            null_r_values = []
            null_mae_values = []

            for _ in range(n_permutations):
                scrambled_y = np.random.permutation(y_values)
                null_r = np.corrcoef(x_values, scrambled_y)[0, 1]
                null_mae = np.mean(np.abs(scrambled_y - x_values))

                null_r_values.append(null_r)
                null_mae_values.append(null_mae)

            # Calculate p-values
            p_value_r = np.mean(np.array(null_r_values) >= actual_r)
            p_value_mae = np.mean(np.array(null_mae_values) <= actual_mae)

            # Store results
            key = f"{param_name}_{age1}_{age2}"
            null_results[key] = {
                'actual_r': actual_r,
                'null_r_mean': np.mean(null_r_values),
                'null_r_std': np.std(null_r_values),
                'p_value_r': p_value_r,
                'actual_mae': actual_mae,
                'null_mae_mean': np.mean(null_mae_values),
                'null_mae_std': np.std(null_mae_values),
                'p_value_mae': p_value_mae
            }

            # Find highlighted indices
            highlight_indices = [i for i, fid in enumerate(common_fly_ids)
                                 if highlight_fly_ids is not None and fid in highlight_fly_ids]

            # Set consistent plot limits
            all_values = df_pheno[param_name].values
            min_val, max_val = np.min(all_values), np.max(all_values)
            range_val = max_val - min_val
            min_val -= range_val * 0.05
            max_val += range_val * 0.05

            # Plot the data
            plot_scatter_with_highlights(
                ax, x_values, y_values,
                highlight_indices=highlight_indices,
                show_error_bars=have_ci,
                x_err=(x_err_low, x_err_high) if have_ci else None,
                y_err=(y_err_low, y_err_high) if have_ci else None
            )

            # Add a few example permuted datasets (lightly colored)
            for i in range(3):
                scrambled_y = np.random.permutation(y_values)
                ax.plot(x_values, scrambled_y, 'o', alpha=0.1, color='gray', markersize=3)

            # Add stats with null model results
            result = null_results[key]
            ax.text(
                0.05, 0.95,
                f'r = {result["actual_r"]:.2f} vs null: {result["null_r_mean"]:.2f}±{result["null_r_std"]:.2f}\n'
                f'p = {result["p_value_r"]:.3f}\n'
                f'MAE = {result["actual_mae"]:.3f} vs null: {result["null_mae_mean"]:.3f}±{result["null_mae_std"]:.3f}\n'
                f'p = {result["p_value_mae"]:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )

            # Add labels and title
            ax.set_xlabel(f'Day {age1} {param_name}', fontsize=12)
            ax.set_ylabel(f'Day {age2} {param_name}', fontsize=12)
            ax.set_title(f'Day {age1} vs Day {age2}: {param_name}', fontsize=12)

            # Set consistent limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

    # Add legend
    handles, labels = create_highlight_legend(axes[0, 0],
                                              highlight_fly_ids or [])

    # Add null model to legend
    handles.append(Line2D([0], [0], marker='o', color='gray', alpha=0.3, markersize=6))
    labels.append('Null model')
    axes[0, 0].legend(handles, labels, loc='lower right', frameon=True,
                      framealpha=0.8, edgecolor='black')

    # Add overall title
    plt.suptitle('Parameter Stability vs Null Model (ID-Scrambled)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the title

    # Create null example figure if requested
    if create_null_example_fig:
        # Create a new figure with a side-by-side comparison layout
        null_fig, null_axes = plt.subplots(len(param_names), 6, figsize=(20, 4 * len(param_names)))
        if len(param_names) == 1:
            null_axes = null_axes.reshape(1, -1)

        # For each parameter and age pair
        for param_idx, param_name in enumerate(param_names):
            for col_idx, (age1, age2) in enumerate(age_pairs):
                # Each age pair gets two columns: real data and null model
                real_ax = null_axes[param_idx, col_idx * 2]
                null_ax = null_axes[param_idx, col_idx * 2 + 1]

                # Get data
                key = f"{param_name}_{age1}_{age2}"
                x_values = null_examples[key]['x_values']
                y_values = null_examples[key]['y_values']
                example_null_y = null_examples[key]['example_null_y']

                # Calculate stats
                real_r = np.corrcoef(x_values, y_values)[0, 1]
                real_mae = np.mean(np.abs(y_values - x_values))
                null_r = np.corrcoef(x_values, example_null_y)[0, 1]
                null_mae = np.mean(np.abs(example_null_y - x_values))

                # Set common attributes for both plots
                for ax in [real_ax, null_ax]:
                    # Plot equality line
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
                    ax.set_xlim(min_val, max_val)
                    ax.set_ylim(min_val, max_val)
                    ax.set_xlabel(f'Day {age1} {param_name}', fontsize=11)

                # Plot real data
                real_ax.scatter(x_values, y_values, s=20, color='C0', alpha=0.7)
                real_ax.set_ylabel(f'Day {age2} {param_name}', fontsize=11)
                real_ax.set_title(f'Real Data: Day {age1} vs {age2}', fontsize=11)

                # Add real data stats
                real_ax.text(
                    0.05, 0.95,
                    f'r = {real_r:.2f}\nMAE = {real_mae:.3f}',
                    transform=real_ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9
                )

                # Plot null example
                null_ax.scatter(x_values, example_null_y, s=20, color='gray', alpha=0.7)
                null_ax.set_ylabel(f'Day {age2} {param_name} (Scrambled)', fontsize=11)
                null_ax.set_title(f'Null Example: Day {age1} vs {age2}', fontsize=11)

                # Add null stats
                null_ax.text(
                    0.05, 0.95,
                    f'r = {null_r:.2f}\nMAE = {null_mae:.3f}',
                    transform=null_ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9
                )

        # Add overall title
        null_fig.suptitle('Real Data vs Example Null Model Comparison', fontsize=16)
        null_fig.tight_layout(rect=[0, 0, 1, 0.95])

        return fig, null_fig, null_results

    return fig, null_results


def plot_distance_comparison_scatters(df_pheno, param_names, distances=None, highlight_fly_ids=None):
    """
    Create scatter plots comparing distances between different age transitions.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame containing parameter values for different flies and ages
    param_names : list
        Names of the parameter columns in df_pheno
    distances : dict, optional
        Pre-calculated distances dictionary. If None, will calculate
    highlight_fly_ids : list, optional
        IDs of flies to highlight in the plots
    """

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # If distances not provided, calculate them
    if distances is None:
        distances = {}
        for fly_id in df_pheno['fly_id'].unique():
            # Get data for this fly at each age
            data_7 = df_pheno[(df_pheno['fly_id'] == fly_id) & (df_pheno['day'] == 7)]
            data_14 = df_pheno[(df_pheno['fly_id'] == fly_id) & (df_pheno['day'] == 14)]
            data_21 = df_pheno[(df_pheno['fly_id'] == fly_id) & (df_pheno['day'] == 21)]

            # Skip flies that don't have data for all ages
            if len(data_7) == 0 or len(data_14) == 0 or len(data_21) == 0:
                continue

            # Extract parameter values
            p_7 = np.array([data_7[param].values[0] for param in param_names])
            p_14 = np.array([data_14[param].values[0] for param in param_names])
            p_21 = np.array([data_21[param].values[0] for param in param_names])

            # Calculate distances
            d7_14 = euclidean(p_7, p_14)
            d14_21 = euclidean(p_14, p_21)
            d7_21 = euclidean(p_7, p_21)

            distances[fly_id] = {
                'd7_14': d7_14,
                'd14_21': d14_21,
                'd7_21': d7_21
            }

    # Convert to arrays for plotting
    fly_ids_with_data = list(distances.keys())
    d7_14_values = [distances[fid]['d7_14'] for fid in fly_ids_with_data]
    d14_21_values = [distances[fid]['d14_21'] for fid in fly_ids_with_data]
    d7_21_values = [distances[fid]['d7_21'] for fid in fly_ids_with_data]

    # Plot 1: d7->14 vs d14->21
    for i, fly_id in enumerate(fly_ids_with_data):
        if highlight_fly_ids is not None and fly_id in highlight_fly_ids:
            # Highlighted flies
            fly_color = f"C{fly_id % 10}"
            ax1.scatter(d7_14_values[i], d14_21_values[i],
                        color=fly_color, s=100, alpha=0.8, marker='o',
                        edgecolor='black', linewidth=1.5,
                        label=f'Fly #{fly_id}')
        else:
            # Regular flies
            ax1.scatter(d7_14_values[i], d14_21_values[i], color='C0', s=30, alpha=0.6)

    # Calculate and display correlation
    corr1 = np.corrcoef(d7_14_values, d14_21_values)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr1:.2f}', transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax1.set_xlabel('Distance Day 7→14', fontsize=12)
    ax1.set_ylabel('Distance Day 14→21', fontsize=12)
    ax1.set_title('Distance Comparison: 7→14 vs 14→21', fontsize=14)

    # Set equal aspect ratio and add diagonal line
    max_val1 = max(max(d7_14_values), max(d14_21_values)) * 1.1
    ax1.set_xlim(0, max_val1)
    ax1.set_ylim(0, max_val1)
    ax1.plot([0, max_val1], [0, max_val1], 'k--', alpha=0.5)

    # Plot 2: d7->14 vs d7->21
    for i, fly_id in enumerate(fly_ids_with_data):
        if highlight_fly_ids is not None and fly_id in highlight_fly_ids:
            # Highlighted flies
            fly_color = f"C{fly_id % 10}"
            ax2.scatter(d7_14_values[i], d7_21_values[i],
                        color=fly_color, s=100, alpha=0.8, marker='o',
                        edgecolor='black', linewidth=1.5,
                        label=f'Fly #{fly_id}')
        else:
            # Regular flies
            ax2.scatter(d7_14_values[i], d7_21_values[i], color='C0', s=30, alpha=0.6)

    # Calculate and display correlation
    corr2 = np.corrcoef(d7_14_values, d7_21_values)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr2:.2f}', transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add labels and title
    ax2.set_xlabel('Distance Day 7→14', fontsize=12)
    ax2.set_ylabel('Distance Day 7→21', fontsize=12)
    ax2.set_title('Distance Comparison: 7→14 vs 7→21', fontsize=14)

    # Set equal aspect ratio and add diagonal line
    max_val2 = max(max(d7_14_values), max(d7_21_values)) * 1.1
    ax2.set_xlim(0, max_val2)
    ax2.set_ylim(0, max_val2)
    ax2.plot([0, max_val2], [0, max_val2], 'k--', alpha=0.5)

    # Add legend to first subplot
    handles = []
    labels = []
    if highlight_fly_ids is not None:
        for fly_id in highlight_fly_ids:
            if fly_id in fly_ids_with_data:
                fly_color = f"C{fly_id % 10}"
                handle = Line2D([0], [0], marker='o', color=fly_color,
                                markeredgecolor='black', markerfacecolor=fly_color,
                                markersize=8, linestyle='None')
                handles.append(handle)
                labels.append(f'Fly #{fly_id}')

    # Add legend to first subplot
    if handles:
        ax1.legend(handles, labels, loc='upper right', frameon=True,
                   framealpha=0.8, edgecolor='black')

    # Add overall title
    param_names_str = "-".join(param_names)
    fig.suptitle(f'Parameter Distance Comparisons in {param_names_str} Space', fontsize=16, y=1.05)

    plt.tight_layout()
    return fig, distances


def plot_path_efficiency(df_pheno, param_names, highlight_fly_ids=None, figsize=(10, 8)):
    """Create path efficiency plot showing direct vs total distances."""
    # Create figure with gridspec for KDEs
    fig = plt.figure(figsize=figsize)

    # Create gridspec with space for KDEs
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[4, 1],
                           height_ratios=[1, 4],
                           wspace=0.05, hspace=0.05)

    # Create main plot areas
    ax_main = fig.add_subplot(gs[1, 0])  # Main scatter plot
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)  # Top KDE for x-axis
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)  # Right KDE for y-axis

    # Hide tick labels for KDE axes
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_right.tick_params(axis='y', labelleft=False)

    # Calculate path data for all flies
    path_data = {}
    fly_ids = []

    for fly_id in df_pheno['fly_id'].unique():
        # Check if fly has data for all ages
        ages = [7, 14, 21]
        fly_data = {age: df_pheno[(df_pheno['fly_id'] == fly_id) & (df_pheno['day'] == age)]
                    for age in ages}

        if all(len(fly_data[age]) > 0 for age in ages):
            fly_ids.append(fly_id)

            # Extract coordinates
            coords = {age: np.array([fly_data[age][param].values[0] for param in param_names])
                      for age in ages}

            # Calculate distances
            d7_14 = euclidean(coords[7], coords[14])
            d14_21 = euclidean(coords[14], coords[21])
            d7_21 = euclidean(coords[7], coords[21])

            path_data[fly_id] = {
                'coords': coords,
                'd7_14': d7_14,
                'd14_21': d14_21,
                'd7_21': d7_21,
                'path_length': d7_14 + d14_21,
                'efficiency': d7_21 / (d7_14 + d14_21) if (d7_14 + d14_21) > 0 else 1.0
            }

    # Extract data for plotting
    x_values = np.array([path_data[fid]['path_length'] for fid in fly_ids])
    y_values = np.array([path_data[fid]['d7_21'] for fid in fly_ids])

    # Find the maximum value for setting limits
    max_val = max(max(x_values), max(y_values)) * 1.1

    # Plot diagonal line (perfect efficiency)
    ax_main.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)

    # Find highlighted indices
    highlight_indices = [i for i, fid in enumerate(fly_ids)
                         if highlight_fly_ids is not None and fid in highlight_fly_ids]
    regular_indices = [i for i, fid in enumerate(fly_ids)
                       if highlight_fly_ids is None or fid not in highlight_fly_ids]

    # Plot regular points
    ax_main.scatter(x_values[regular_indices], y_values[regular_indices],
                    color='C0', s=40, alpha=0.6)

    # Plot highlighted points
    for i, idx in enumerate(highlight_indices):
        fly_id = fly_ids[idx]
        fly_color = f"C{fly_id % 10}"

        # Add highlighted point
        ax_main.scatter(x_values[idx], y_values[idx],
                        color=fly_color, s=120, alpha=0.8, marker='o',
                        edgecolor='black', linewidth=1.5,
                        label=f'Fly #{fly_id}')

        # Add fly ID label
        ax_main.text(x_values[idx], y_values[idx] * 1.05,
                     f"#{fly_id}", ha='center', va='bottom',
                     color=fly_color, fontweight='bold', fontsize=11)

    # Calculate statistics
    corr, p_value = stats.pearsonr(x_values, y_values)
    efficiency_values = [path_data[fid]['efficiency'] for fid in fly_ids]
    median_efficiency = np.median(efficiency_values)

    # Add stats text
    ax_main.text(0.05, 0.95,
                 f'r = {corr:.2f}\n'
                 f'p = {p_value:.3f}\n'
                 f'Median Efficiency = {median_efficiency:.2f}',
                 transform=ax_main.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Set axis limits
    ax_main.set_xlim(0, max_val)
    ax_main.set_ylim(0, max_val)

    # Add KDE plots
    if len(x_values) > 2:
        # Top KDE (x-axis)
        kde_x = stats.gaussian_kde(x_values)
        x_grid = np.linspace(0, max_val, 100)
        ax_top.plot(x_grid, kde_x(x_grid), color='black')
        ax_top.fill_between(x_grid, kde_x(x_grid), alpha=0.3, color='C0')

        # Add median line
        median_x = np.median(x_values)
        ax_top.axvline(median_x, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

        # Add highlighted points to KDE
        for i, idx in enumerate(highlight_indices):
            fly_id = fly_ids[idx]
            fly_color = f"C{fly_id % 10}"
            ax_top.axvline(x_values[idx], color=fly_color, linestyle='-', linewidth=2, alpha=0.8)

        # Right KDE (y-axis)
        kde_y = stats.gaussian_kde(y_values)
        y_grid = np.linspace(0, max_val, 100)
        ax_right.plot(kde_y(y_grid), y_grid, color='black')
        ax_right.fill_betweenx(y_grid, kde_y(y_grid), alpha=0.3, color='C0')

        # Add median line
        median_y = np.median(y_values)
        ax_right.axhline(median_y, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

        # Add highlighted points to KDE
        for i, idx in enumerate(highlight_indices):
            fly_id = fly_ids[idx]
            fly_color = f"C{fly_id % 10}"
            ax_right.axhline(y_values[idx], color=fly_color, linestyle='-', linewidth=2, alpha=0.8)

    # Add labels
    ax_main.set_xlabel('Total Path Length (7→14→21)', fontsize=12)
    ax_main.set_ylabel('Direct Distance (7→21)', fontsize=12)
    ax_top.set_ylabel('Density', fontsize=10)
    ax_right.set_xlabel('Density', fontsize=10)

    # Add title
    ax_main.set_title(f'Path Efficiency: {" vs ".join(param_names)}', fontsize=14)

    # Add explanatory annotations
    ax_main.text(max_val * 0.2, max_val * 0.2,
                 "Direct Path (Efficient)",
                 ha='left', va='bottom', fontsize=10, rotation=45,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.3, edgecolor='none'))

    ax_main.text(max_val * 0.75, max_val * 0.25,
                 "Indirect Path (Inefficient)",
                 ha='center', va='center', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round', pad=0.3, edgecolor='none'))

    # Add legend
    handles = []
    labels = []

    # Add highlighted flies
    if highlight_fly_ids:
        for fly_id in highlight_fly_ids:
            if fly_id in fly_ids:
                fly_color = f"C{fly_id % 10}"
                handles.append(Line2D([0], [0], marker='o', color=fly_color,
                                      markeredgecolor='black', markerfacecolor=fly_color,
                                      markersize=8, linestyle='None'))
                labels.append(f'Fly #{fly_id}')

    # Add diagonal line and median
    handles.append(Line2D([0], [0], linestyle='--', color='k', alpha=0.7))
    labels.append('Direct Path')
    handles.append(Line2D([0], [0], linestyle='-', color='black', alpha=0.8))
    labels.append('Population Median')

    # Add legend
    ax_main.legend(handles, labels, loc='upper left', fontsize=10)

    # Add grid
    ax_main.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    return fig, path_data


def plot_transition_heatmaps(df_paths, param_names):
    """Plot heatmaps of transitions between age groups."""
    order = ['-', '0', '+']

    # Create figure
    fig, axes = plt.subplots(1, len(param_names), figsize=(6 * len(param_names), 5))
    if len(param_names) == 1:
        axes = [axes]

    # For each parameter
    for ax, param_name in zip(axes, param_names):
        step1, step2 = f"{param_name}_step1", f"{param_name}_step2"

        # Create crosstab
        pivot = pd.crosstab(df_paths[step1], df_paths[step2])
        pivot = pivot.reindex(index=order, columns=order, fill_value=0)

        # Calculate percentages
        pct = pivot / pivot.values.sum() * 100
        annot = pct.round(1).astype(str) + '%'

        # Create heatmap
        sns.heatmap(pct, annot=annot, fmt='', cmap='Blues',
                    ax=ax, cbar=False, linewidths=0.5)

        # Adjust appearance
        ax.invert_yaxis()
        ax.set_xlabel('14→21')
        ax.set_ylabel('7→14')
        ax.set_title(f"{param_name} transitions")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig


# ==== MAIN ANALYSIS FUNCTION ====

def analyze_fly_stability(df_summary, genotype, ages=[7, 14, 21],
                          X_VAR='p0', Y_VAR='p_ss',
                          HIGHLIGHT_FLY_IDS=None, OMIT_FLY_IDS=None,
                          DIR_OUTPUT='./output'):
    """
    Main function to run all analyses for fly stability.

    Parameters:
    -----------
    df_summary : DataFrame
        Summary data with fly information
    genotype : str
        Genotype to analyze
    ages : list
        Ages to include in analysis
    X_VAR, Y_VAR : str
        Variables to analyze
    HIGHLIGHT_FLY_IDS : list
        IDs of flies to highlight
    OMIT_FLY_IDS : dict
        Dictionary of fly IDs to omit by genotype
    DIR_OUTPUT : str
        Directory for output files
    """
    # Load posterior draws (3D model)
    draws_over_age = {
        g: {d: np.load(os.path.join(DIR_FITS, f"{g}_day{d}_3d_draws.npz"), allow_pickle=True)
            for d in ages}
        for g in ['KK', 'GD']
    }

    # Process phenotype data
    df_pheno, df_paths, all_draws = process_phenotype_data(
        df_summary, genotype, ages, draws_over_age,
        X_VAR, Y_VAR, OMIT_FLY_IDS
    )

    # Define target paths if needed
    fly_paths = dict(zip(df_paths['fly_id'], df_paths['path_code']))
    to_highlight = set()  # This would normally be populated by TARGET_PATHS

    # Create output directory if it doesn't exist
    os.makedirs(DIR_OUTPUT, exist_ok=True)

    # 1) Parameter comparisons across ages
    fig = plot_parameter_comparisons_across_ages(
        df_pheno, [X_VAR, Y_VAR],
        highlight_fly_ids=HIGHLIGHT_FLY_IDS,
        show_marginals=True
    )
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY_across_ages.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY_across_ages.svg'),
                bbox_inches='tight')

    # 2) Parameter comparisons with null model
    fig_null, null_fig, null_results = plot_parameter_comparisons_with_null(
        df_pheno, [X_VAR, Y_VAR],
        highlight_fly_ids=HIGHLIGHT_FLY_IDS,
        n_permutations=1000,
        random_seed=42
    )
    plt.figure(fig_null.number)
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY.svg'),
                bbox_inches='tight')

    plt.figure(null_fig.number)
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY_sepnull.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_stability_XY_sepnull.svg'),
                bbox_inches='tight')

    # 3) Path efficiency
    fig_eff, path_data = plot_path_efficiency(
        df_pheno, [X_VAR, Y_VAR],
        highlight_fly_ids=HIGHLIGHT_FLY_IDS
    )
    plt.savefig(os.path.join(DIR_OUTPUT, f'path_efficiency_scatter.png'),
                dpi=300, bbox_inches='tight')

    # 4) Transition heatmaps
    fig_heat = plot_transition_heatmaps(df_paths, [X_VAR, Y_VAR])
    plt.savefig(os.path.join(DIR_OUTPUT, f'trajectory_heatmaps.png'),
                dpi=300, bbox_inches='tight')

    # 5) Original stability plot with lines and CI (if still needed)
    fig_stab = plot_stability_trajectories(
        df_pheno, X_VAR, Y_VAR,
        highlight_fly_ids=HIGHLIGHT_FLY_IDS,
        to_highlight=to_highlight,
        highlight_only=True,
        show_ci=True
    )
    plt.savefig(os.path.join(DIR_OUTPUT, f'stability_{genotype}_{X_VAR}_{Y_VAR}.png'),
                dpi=300, bbox_inches='tight')

    return {
        'df_pheno': df_pheno,
        'df_paths': df_paths,
        'all_draws': all_draws,
        'path_data': path_data,
        'null_results': null_results
    }


def plot_stability_trajectories(df_pheno, X_VAR, Y_VAR, highlight_fly_ids=None,
                                to_highlight=None, highlight_only=True, show_ci=True,
                                age_styles=None):
    """Plot stability trajectories for each fly."""
    if age_styles is None:
        age_styles = {7: {'marker': 'o'},
                      14: {'marker': 's'},
                      21: {'marker': '^'}}

    fig, ax = plt.subplots(figsize=(4, 4))

    for fid, grp in df_pheno.groupby('fly_id'):
        grp = grp.sort_values('day')
        if len(grp) < 3:
            continue

        # Determine if this fly should be highlighted
        if to_highlight is None:
            to_highlight = set()

        is_target = (highlight_fly_ids is not None and fid in highlight_fly_ids) or (fid in to_highlight)

        if highlight_only and not is_target:
            col, alpha, lw, ms, s, zz = 'lightgrey', .4, .7, 3, 25, 7
        else:
            col, alpha, lw, ms, s, zz = f"C{fid % 10}", 1, 2, 6, 50, 10

        # CI bars
        if is_target and show_ci:
            for _, r in grp.iterrows():
                ax.errorbar(
                    r[X_VAR], r[Y_VAR],
                    xerr=[[r[X_VAR] - r[f"{X_VAR}_ci_low"]], [r[f"{X_VAR}_ci_high"] - r[X_VAR]]],
                    yerr=[[r[Y_VAR] - r[f"{Y_VAR}_ci_low"]], [r[f"{Y_VAR}_ci_high"] - r[Y_VAR]]],
                    fmt='none', ecolor=col, alpha=alpha, capsize=3, zorder=zz
                )

        # Points & lines
        for _, r in grp.iterrows():
            ax.scatter(r[X_VAR], r[Y_VAR], color=col,
                       marker=age_styles[r.day]['marker'], s=s, lw=lw, alpha=alpha, zorder=zz)

        ax.plot(grp[X_VAR], grp[Y_VAR], color=col, lw=lw, alpha=alpha, zorder=zz,
                label=f"Fly {fid}" if highlight_fly_ids is not None and fid in highlight_fly_ids else None)

    # Add labels and title
    ax.set_xlabel(X_VAR)
    ax.set_ylabel(Y_VAR)
    ax.set_title(f"Stability: {X_VAR} vs {Y_VAR}")

    # Add constraint line if applicable
    if X_VAR == 'p0' and Y_VAR == 'p_ss':
        ax.plot([0, 1], [0, 1], '--', color='grey', alpha=.5)

    ax.grid(alpha=0.2)

    # Legend
    age_handles = [mlines.Line2D([], [], color='grey', marker=age_styles[d]['marker'],
                                 linestyle='None', markersize=8, label=str(d))
                   for d in age_styles.keys()]

    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h + age_handles, l + [h.get_label() for h in age_handles])

    plt.tight_layout()
    return fig


def plot_parameter_age_distributions(df_pheno, param_names,
                                     ages=[7, 14, 21],
                                     highlight_fly_ids=None,
                                     track_flies='highlighted',
                                     # Options: 'none', 'highlighted', 'all', 'random_subset'
                                     random_subset_size=10,
                                     plot_type='dotplot',  # Options: 'dotplot', 'histogram', 'violin', 'kde'
                                     figsize=(12, 8)):
    """
    Create longitudinal distribution plots showing parameter distributions across ages.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame containing parameter values for flies at different ages
    param_names : list
        Names of the parameter columns to plot
    ages : list
        List of ages to plot
    highlight_fly_ids : list, optional
        IDs of flies to highlight
    track_flies : str
        Which flies to track across ages with lines:
        - 'none': Don't show tracking lines
        - 'highlighted': Track only highlighted flies
        - 'all': Track all flies
        - 'random_subset': Track a random subset of flies
    random_subset_size : int
        Number of flies to track if track_flies='random_subset'
    plot_type : str
        Type of plot to use:
        - 'dotplot': Show individual points
        - 'histogram': Show histogram
        - 'violin': Show violin plot
        - 'kde': Show kernel density estimate
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import random

    # Input validation
    if plot_type not in ['dotplot', 'histogram', 'violin', 'kde']:
        raise ValueError(f"Invalid plot_type: {plot_type}. Must be one of: 'dotplot', 'histogram', 'violin', 'kde'")

    if track_flies not in ['none', 'highlighted', 'all', 'random_subset']:
        raise ValueError(
            f"Invalid track_flies: {track_flies}. Must be one of: 'none', 'highlighted', 'all', 'random_subset'")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize, sharey=True)
    if len(param_names) == 1:
        axes = [axes]

    # Get flies that have data for all ages
    complete_flies = filter_complete_flies(df_pheno, ages)

    # Determine which flies to track
    flies_to_track = []
    if track_flies == 'highlighted' and highlight_fly_ids is not None:
        flies_to_track = [f for f in highlight_fly_ids if f in complete_flies]
    elif track_flies == 'all':
        flies_to_track = complete_flies
    elif track_flies == 'random_subset':
        if len(complete_flies) <= random_subset_size:
            flies_to_track = complete_flies
        else:
            random.seed(42)  # For reproducibility
            flies_to_track = random.sample(complete_flies, random_subset_size)

    # Process each parameter
    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # Set up y-axis as categorical (ages)
        age_positions = range(len(ages))

        # Extract parameter values for each age
        all_values = []
        values_by_age = {}
        fly_positions = {}

        for age_idx, age in enumerate(ages):
            df_age = df_pheno[df_pheno['day'] == age]
            values = df_age[param_name].values
            fly_ids = df_age['fly_id'].values

            # Store values and positions
            all_values.extend(values)
            values_by_age[age] = values
            fly_positions[age] = {fly_id: (val, age_idx) for fly_id, val in zip(fly_ids, values)}

        # Set x-axis limits with padding
        all_values = np.array(all_values)
        xmin, xmax = np.nanmin(all_values), np.nanmax(all_values)
        xrange = xmax - xmin
        xmin -= xrange * 0.05
        xmax += xrange * 0.05

        # Plot distributions based on selected plot type
        for age_idx, age in enumerate(ages):
            values = values_by_age[age]

            if plot_type == 'dotplot':
                # Scatter plot (dotplot)
                ax.scatter(values, [age_idx] * len(values),
                           alpha=0.6, s=30, color=f'C{age_idx}',
                           edgecolor='none', label=f'Age {age}')

                # Highlight specific flies if requested
                if highlight_fly_ids is not None:
                    for fly_id in highlight_fly_ids:
                        if fly_id in fly_positions[age]:
                            val, pos = fly_positions[age][fly_id]
                            ax.scatter(val, pos, s=100, color=f'C{fly_id % 10}',
                                       edgecolor='black', linewidth=1.5, zorder=10)

            elif plot_type == 'histogram':
                # Horizontal histogram
                hist_heights = 0.6  # Height of each histogram
                bins = min(30, max(10, int(len(values) / 5)))  # Adaptive bin size

                # Calculate histogram
                counts, bin_edges = np.histogram(values, bins=bins, range=(xmin, xmax))
                max_count = counts.max()

                # Normalize and plot
                if max_count > 0:
                    norm_counts = counts / max_count * hist_heights
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    for j, (x, h) in enumerate(zip(bin_centers, norm_counts)):
                        ax.fill_between(
                            [bin_edges[j], bin_edges[j + 1]],
                            [age_idx - h / 2, age_idx - h / 2],
                            [age_idx + h / 2, age_idx + h / 2],
                            color=f'C{age_idx}', alpha=0.7
                        )

            elif plot_type == 'violin':
                # Violin plot (use seaborn)
                if len(values) >= 3:  # Need at least 3 points for KDE
                    try:
                        parts = ax.violinplot(
                            values, positions=[age_idx], vert=False,
                            showmeans=False, showextrema=False, showmedians=False
                        )
                        for pc in parts['bodies']:
                            pc.set_facecolor(f'C{age_idx}')
                            pc.set_alpha(0.7)
                    except:
                        # Fallback if violin plot fails
                        ax.scatter(values, [age_idx] * len(values),
                                   alpha=0.6, s=20, color=f'C{age_idx}')

            elif plot_type == 'kde':
                # KDE plot
                if len(values) >= 3:  # Need at least 3 points for KDE
                    try:
                        x_grid = np.linspace(xmin, xmax, 100)
                        kde = sns.kdeplot(
                            values, ax=ax, color=f'C{age_idx}',
                            label=f'Age {age}', alpha=0.7,
                        )
                    except:
                        # Fallback if KDE fails
                        ax.scatter(values, [age_idx] * len(values),
                                   alpha=0.6, s=20, color=f'C{age_idx}')

        # Draw tracking lines for selected flies
        if track_flies != 'none' and flies_to_track:
            for fly_id in flies_to_track:
                fly_points = []
                for age in ages:
                    if fly_id in fly_positions[age]:
                        fly_points.append(fly_positions[age][fly_id])

                if len(fly_points) > 1:
                    # Draw lines connecting the points
                    xs = [p[0] for p in fly_points]
                    ys = [p[1] for p in fly_points]

                    line_color = f'C{fly_id % 10}' if highlight_fly_ids is not None and fly_id in highlight_fly_ids else 'gray'
                    line_alpha = 0.8 if highlight_fly_ids is not None and fly_id in highlight_fly_ids else 0.3
                    line_width = 2 if highlight_fly_ids is not None and fly_id in highlight_fly_ids else 1
                    line_style = '-' if highlight_fly_ids is not None and fly_id in highlight_fly_ids else '--'

                    ax.plot(xs, ys, color=line_color, alpha=line_alpha,
                            linewidth=line_width, linestyle=line_style, zorder=5)

        # Set axis labels and limits
        ax.set_xlim(xmin, xmax)
        ax.set_title(param_name, fontsize=14)
        ax.set_xlabel(f'{param_name} Value', fontsize=12)

        # Only add y-ticks and label to the first subplot
        if i == 0:
            ax.set_ylabel('Age', fontsize=12)
            ax.set_yticks(age_positions)
            ax.set_yticklabels([f'Age {age}' for age in ages])

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Add overall title
    plt.suptitle('Parameter Distributions Across Ages', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_parameter_normalized_trajectories(df_pheno, param_names,
                                           ages=[7, 14, 21],
                                           highlight_fly_ids=None,
                                           track_flies='all',
                                           random_subset_size=15,
                                           normalize_method='z_score',  # 'z_score', 'percentile', 'relative_to_mean'
                                           figsize=(12, 8)):
    """
    Create plots showing normalized parameter trajectories across ages.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame containing parameter values for flies at different ages
    param_names : list
        Names of the parameter columns to plot
    ages : list
        List of ages to plot
    highlight_fly_ids : list, optional
        IDs of flies to highlight
    track_flies : str
        Which flies to track across ages with lines:
        - 'highlighted': Track only highlighted flies
        - 'all': Track all flies
        - 'random_subset': Track a random subset of flies
    random_subset_size : int
        Number of flies to track if track_flies='random_subset'
    normalize_method : str
        Method to normalize values:
        - 'z_score': Standard score (mean=0, std=1)
        - 'percentile': Percentile rank (0-100)
        - 'relative_to_mean': Value relative to age mean
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from scipy import stats

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize, sharey=True)
    if len(param_names) == 1:
        axes = [axes]

    # Get flies that have data for all ages
    complete_flies = filter_complete_flies(df_pheno, ages)

    # Determine which flies to track
    flies_to_track = []
    if track_flies == 'highlighted' and highlight_fly_ids is not None:
        flies_to_track = [f for f in highlight_fly_ids if f in complete_flies]
    elif track_flies == 'all':
        flies_to_track = complete_flies
    elif track_flies == 'random_subset':
        if len(complete_flies) <= random_subset_size:
            flies_to_track = complete_flies
        else:
            random.seed(42)  # For reproducibility
            flies_to_track = random.sample(complete_flies, random_subset_size)

    # Process each parameter
    for i, param_name in enumerate(param_names):
        ax = axes[i]

        # Extract and normalize parameter values by age
        normalized_values = {}
        original_values = {}

        for age in ages:
            df_age = df_pheno[df_pheno['day'] == age]
            values = df_age[param_name].values
            fly_ids = df_age['fly_id'].values

            # Store original values
            original_values[age] = {fly_id: val for fly_id, val in zip(fly_ids, values)}

            # Normalize based on selected method
            if normalize_method == 'z_score':
                # Z-score normalization
                mean, std = np.mean(values), np.std(values)
                if std > 0:
                    norm_vals = (values - mean) / std
                else:
                    norm_vals = values - mean  # If std=0, just center

            elif normalize_method == 'percentile':
                # Percentile ranking (0-100)
                norm_vals = np.array([stats.percentileofscore(values, v) for v in values])

            elif normalize_method == 'relative_to_mean':
                # Relative to age mean (mean=1)
                mean = np.mean(values)
                if mean > 0:
                    norm_vals = values / mean
                else:
                    norm_vals = values

            # Store normalized values
            normalized_values[age] = {fly_id: val for fly_id, val in zip(fly_ids, norm_vals)}

        # Plot tracks for flies
        for fly_id in flies_to_track:
            # Extract data points for this fly
            xs = ages
            ys = [normalized_values[age].get(fly_id, np.nan) for age in ages]
            original_ys = [original_values[age].get(fly_id, np.nan) for age in ages]

            # Skip if we don't have enough data
            if sum(~np.isnan(ys)) < 2:
                continue

            # Set line properties
            is_highlighted = highlight_fly_ids is not None and fly_id in highlight_fly_ids
            line_color = f'C{fly_id % 10}' if is_highlighted else 'gray'
            line_alpha = 0.9 if is_highlighted else 0.3
            line_width = 2 if is_highlighted else 1
            line_style = '-' if is_highlighted else '--'
            marker_size = 100 if is_highlighted else 30
            marker_edge = 1.5 if is_highlighted else 0

            # Plot the trajectory
            ax.plot(xs, ys, color=line_color, alpha=line_alpha,
                    linewidth=line_width, linestyle=line_style, zorder=5)

            # Add markers at each point
            for j, (x, y, orig_y) in enumerate(zip(xs, ys, original_ys)):
                if np.isnan(y):
                    continue

                ax.scatter(x, y, s=marker_size, color=line_color,
                           alpha=line_alpha, zorder=10,
                           edgecolor='black' if is_highlighted else None,
                           linewidth=marker_edge)

                # Add value annotation for highlighted flies
                if is_highlighted:
                    ax.annotate(f"{orig_y:.2f}",
                                (x, y),
                                xytext=(5, 5),
                                textcoords='offset points',
                                fontsize=8)

        # Add horizontal line at y=0 (mean) for reference
        if normalize_method in ['z_score', 'relative_to_mean']:
            reference_value = 0 if normalize_method == 'z_score' else 1
            ax.axhline(reference_value, color='black', linestyle='-', alpha=0.3, zorder=1)

        # Label the plot
        ylabel = {
            'z_score': 'Z-Score (σ)',
            'percentile': 'Percentile Rank',
            'relative_to_mean': 'Relative to Mean'
        }[normalize_method]

        ax.set_title(param_name, fontsize=14)
        ax.set_xlabel('Age (days)', fontsize=12)
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=12)

        # Set x-ticks to be the ages
        ax.set_xticks(ages)
        ax.set_xticklabels(ages)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')

    # Add title with normalization method
    title_suffix = {
        'z_score': '(Z-Score Normalized)',
        'percentile': '(Percentile Rank)',
        'relative_to_mean': '(Relative to Age Mean)'
    }[normalize_method]

    plt.suptitle(f'Parameter Trajectories Across Ages {title_suffix}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig



def plot_parameter_rank_stability(df_pheno, param_name, ages=None, highlight_fly_ids=None,
                                  percentiles=False, figsize=(12, 8)):
    """
    Create a plot showing parameter rank stability across ages.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Default ages if not provided
    if ages is None:
        ages = sorted(df_pheno['day'].unique())

    # Get flies that have data for all ages
    complete_flies = filter_complete_flies(df_pheno, ages)

    # Create rank DataFrame
    rank_data = []

    for age in ages:
        df_age = df_pheno[df_pheno['day'] == age]

        # Get values for flies with complete data
        values = []
        for fly_id in complete_flies:
            fly_data = df_age[df_age['fly_id'] == fly_id]
            if len(fly_data) > 0:
                values.append((fly_id, fly_data[param_name].values[0]))

        # Convert to numpy arrays
        fly_ids, param_values = zip(*values)
        param_values = np.array(param_values)

        # Calculate ranks or percentiles
        if percentiles:
            # Calculate percentile rank (0-100)
            ranks = np.array([stats.percentileofscore(param_values, v) for v in param_values])
        else:
            # Calculate integer ranks (1 to n)
            ranks = stats.rankdata(param_values, method='average')

        # Add to dataset
        for fly_id, rank in zip(fly_ids, ranks):
            rank_data.append({
                'fly_id': fly_id,
                'age': age,
                'rank': rank
            })

    rank_df = pd.DataFrame(rank_data)

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the rank lines for each fly
    for fly_id in complete_flies:
        fly_data = rank_df[rank_df['fly_id'] == fly_id]

        # Skip if missing data
        if len(fly_data) < len(ages):
            continue

        # Sort by age
        fly_data = fly_data.sort_values('age')

        # Set line properties
        is_highlighted = highlight_fly_ids is not None and fly_id in highlight_fly_ids
        line_color = f'C{fly_id % 10}' if is_highlighted else 'gray'
        line_alpha = 0.9 if is_highlighted else 0.3
        line_width = 2 if is_highlighted else 1
        line_style = '-' if is_highlighted else '--'

        # Plot rank line
        ax.plot(fly_data['age'], fly_data['rank'],
                color=line_color, alpha=line_alpha,
                linewidth=line_width, linestyle=line_style,
                marker='o' if is_highlighted else None,
                markersize=8 if is_highlighted else 4)

        # Add labels for highlighted flies
        if is_highlighted:
            for _, row in fly_data.iterrows():
                '''
                ax.text(row['age'], row['rank'], f"#{fly_id}",
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=line_color)'''
                # Fix: Use annotate() instead of text() with xytext
                ax.annotate(f"#{fly_id}",
                            xy=(row['age'], row['rank']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold', color=line_color)

    # Customize y-axis based on rank type
    if percentiles:
        ax.set_ylabel('Percentile Rank', fontsize=12)
        # Invert y-axis so higher percentiles are at the top
        ax.invert_yaxis()
    else:
        ax.set_ylabel('Rank', fontsize=12)
        # Calculate max rank for setting limits
        max_rank = rank_df['rank'].max()
        ax.set_ylim(max_rank + 1, 0.5)  # Invert axis so rank 1 is at the top

    # Set x-ticks to be the ages
    ax.set_xticks(ages)
    ax.set_xticklabels(ages)
    ax.set_xlabel('Age (days)', fontsize=12)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add title
    title_suffix = "(Percentile Rank)" if percentiles else "(Integer Rank)"
    plt.title(f"Parameter Rank Stability: {param_name} {title_suffix}", fontsize=16)

    # Add legend for highlighted flies if any
    if highlight_fly_ids:
        handles = []
        labels = []

        for fly_id in highlight_fly_ids:
            if fly_id in complete_flies:
                fly_color = f'C{fly_id % 10}'
                from matplotlib.lines import Line2D
                handle = Line2D([0], [0], marker='o', color=fly_color,
                                markeredgecolor='black', markerfacecolor=fly_color,
                                markersize=8, linestyle='-')
                handles.append(handle)
                labels.append(f'Fly #{fly_id}')

        if handles:
            ax.legend(handles, labels, loc='best', frameon=True,
                      framealpha=0.8, edgecolor='black')

    plt.tight_layout()
    return fig


def plot_parameter_transition_matrix(df_pheno, param_name, ages=None, n_quantiles=4,
                                     figsize=(16, 5), n_permutations=100):
    """
    Create a transition probability matrix showing how flies move between
    parameter quantiles across ages, with significance testing against null model.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame containing parameter values for flies at different ages
    param_name : str
        Name of the parameter column to analyze
    ages : list, optional
        List of ages to include
    n_quantiles : int
        Number of quantiles to divide the data into
    figsize : tuple
        Figure size (width, height)
    n_permutations : int
        Number of permutations for significance testing
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import random

    # Default ages if not provided
    if ages is None:
        ages = sorted(df_pheno['day'].unique())

    # We need at least 2 ages for transitions
    if len(ages) < 2:
        raise ValueError("Need at least 2 ages to compute transitions")

    # Compute all pairs of ages
    age_pairs = []
    for i in range(len(ages)):
        for j in range(i + 1, len(ages)):
            age_pairs.append((ages[i], ages[j]))

    # Compute quantiles for each age
    quantile_labels = []
    for i in range(n_quantiles):
        q_low = i / n_quantiles * 100
        q_high = (i + 1) / n_quantiles * 100
        quantile_labels.append(f"Q{i + 1}\n({q_low:.0f}-{q_high:.0f}%)")

    # Initialize transition matrices
    transition_matrices = []

    # Process each pair of ages
    for age1, age2 in age_pairs:
        # Get data for these ages
        df_age1 = df_pheno[df_pheno['day'] == age1]
        df_age2 = df_pheno[df_pheno['day'] == age2]

        # Find common fly IDs
        common_fly_ids = set(df_age1['fly_id']).intersection(set(df_age2['fly_id']))

        # Skip if not enough flies
        if len(common_fly_ids) < n_quantiles:
            continue

        # Extract values
        values1 = {}
        values2 = {}

        for fly_id in common_fly_ids:
            val1 = df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
            val2 = df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
            values1[fly_id] = val1
            values2[fly_id] = val2

        # Calculate quantile boundaries for age1
        all_values1 = np.array(list(values1.values()))
        quantile_bounds1 = np.quantile(all_values1, np.linspace(0, 1, n_quantiles + 1))

        # Calculate quantile boundaries for age2
        all_values2 = np.array(list(values2.values()))
        quantile_bounds2 = np.quantile(all_values2, np.linspace(0, 1, n_quantiles + 1))

        # Assign flies to quantiles
        quantile_indices1 = {}
        quantile_indices2 = {}

        for fly_id in common_fly_ids:
            val1 = values1[fly_id]
            val2 = values2[fly_id]

            # Find quantile for age1
            q1 = np.searchsorted(quantile_bounds1, val1, side='right') - 1
            if q1 >= n_quantiles:  # Handle edge case
                q1 = n_quantiles - 1

            # Find quantile for age2
            q2 = np.searchsorted(quantile_bounds2, val2, side='right') - 1
            if q2 >= n_quantiles:  # Handle edge case
                q2 = n_quantiles - 1

            quantile_indices1[fly_id] = q1
            quantile_indices2[fly_id] = q2

        # Build transition count matrix
        transition_matrix = np.zeros((n_quantiles, n_quantiles))

        for fly_id in common_fly_ids:
            q1 = quantile_indices1[fly_id]
            q2 = quantile_indices2[fly_id]
            transition_matrix[q1, q2] += 1

        # Convert to probability
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_prob = np.zeros_like(transition_matrix)

        # Avoid division by zero
        for i in range(n_quantiles):
            if row_sums[i] > 0:
                transition_prob[i] = transition_matrix[i] / row_sums[i]

        transition_matrices.append({
            'age1': age1,
            'age2': age2,
            'matrix': transition_prob,
            'raw_counts': transition_matrix
        })

    # Permutation testing for transition matrices
    null_transitions = []

    for trans in transition_matrices:
        age1, age2 = trans['age1'], trans['age2']

        # Get data for these ages
        df_age1 = df_pheno[df_pheno['day'] == age1]
        df_age2 = df_pheno[df_pheno['day'] == age2]

        # Find common fly IDs
        common_fly_ids = list(set(df_age1['fly_id']).intersection(set(df_age2['fly_id'])))

        # Extract values as arrays for faster permutation
        values1 = np.array([df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
                            for fly_id in common_fly_ids])
        values2 = np.array([df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
                            for fly_id in common_fly_ids])

        # Calculate original quantile assignments
        q1 = np.zeros(len(common_fly_ids), dtype=int)
        q2 = np.zeros(len(common_fly_ids), dtype=int)

        for i, val in enumerate(values1):
            idx = np.searchsorted(quantile_bounds1, val, side='right') - 1
            q1[i] = min(idx, n_quantiles - 1)

        for i, val in enumerate(values2):
            idx = np.searchsorted(quantile_bounds2, val, side='right') - 1
            q2[i] = min(idx, n_quantiles - 1)

        # Run permutations
        null_matrices = np.zeros((n_permutations, n_quantiles, n_quantiles))

        for p in range(n_permutations):
            # Shuffle the relationship between q1 and q2
            q2_perm = np.random.permutation(q2)

            # Build transition matrix
            matrix = np.zeros((n_quantiles, n_quantiles))
            for i in range(len(common_fly_ids)):
                matrix[q1[i], q2_perm[i]] += 1

            # Convert to probability
            for i in range(n_quantiles):
                row_sum = np.sum(matrix[i])
                if row_sum > 0:
                    matrix[i] /= row_sum

            null_matrices[p] = matrix

        # Calculate mean and standard deviation
        null_mean = np.mean(null_matrices, axis=0)
        null_std = np.std(null_matrices, axis=0)

        # Calculate p-values (two-tailed)
        p_values = np.zeros((n_quantiles, n_quantiles))
        for i in range(n_quantiles):
            for j in range(n_quantiles):
                # Count permutation values as or more extreme than observed
                count = np.sum(np.abs(null_matrices[:, i, j] - null_mean[i, j]) >=
                               np.abs(trans['matrix'][i, j] - null_mean[i, j]))
                p_values[i, j] = count / n_permutations

        null_transitions.append({
            'null_mean': null_mean,
            'null_std': null_std,
            'p_values': p_values
        })

    # Create the figure
    n_matrices = len(transition_matrices)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize, squeeze=False,
                             gridspec_kw={'wspace': 0.2})  # Add space between plots

    # Plot each transition matrix
    for i, (ax, trans, null) in enumerate(zip(axes[0], transition_matrices, null_transitions)):
        age1, age2 = trans['age1'], trans['age2']

        # Create heatmap
        sns.heatmap(
            trans['matrix'],
            annot=False,  # We'll add custom annotations
            cmap='Blues',
            vmin=0,
            vmax=1,
            cbar=(i == n_matrices - 1),  # Only last plot gets colorbar
            cbar_kws={'label': 'Transition Probability'} if i == n_matrices - 1 else {},
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor='white'
        )

        # Add custom annotations with significance markers
        for r in range(n_quantiles):
            for c in range(n_quantiles):
                count = trans['raw_counts'][r, c]
                prob = trans['matrix'][r, c]
                p_val = null['p_values'][r, c]

                if count > 0:
                    # Determine significance markers
                    sig_marker = ''
                    if p_val < 0.05:
                        sig_marker = '*'
                        if p_val < 0.01:
                            sig_marker = '**'
                            if p_val < 0.001:
                                sig_marker = '***'

                    text_color = 'white' if prob > 0.5 else 'black'

                    # Show probability with significance marker
                    ax.text(
                        c + 0.5, r + 0.5 - 0.15,
                        f"{prob:.2f}{sig_marker}",
                        ha='center', va='center',
                        color=text_color,
                        fontsize=9,
                        fontweight='bold'
                    )

                    # Show count in parentheses
                    ax.text(
                        c + 0.5, r + 0.5 + 0.15,
                        f"({int(count)})",
                        ha='center', va='center',
                        color=text_color,
                        fontsize=7
                    )

        # Set labels
        ax.set_xlabel(f'Day {age2} Quantiles', fontsize=12)
        ax.set_ylabel(f'Day {age1} Quantiles', fontsize=12)
        ax.set_title(f'Day {age1} → {age2}', fontsize=12)

        # Set tick labels
        ax.set_xticks(np.arange(n_quantiles) + 0.5)
        ax.set_yticks(np.arange(n_quantiles) + 0.5)
        ax.set_xticklabels(quantile_labels)
        ax.set_yticklabels(quantile_labels)

        # Move y-labels to right for middle plots (cleaner look)
        if i > 0:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    # Add significance legend at bottom
    fig.text(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001 versus null model",
             ha='center', fontsize=9)

    plt.suptitle(f'Parameter Transition Probabilities: {param_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for significance legend

    return fig, transition_matrices, null_transitions


def plot_transition_with_null_comparison(df_pheno, param_name, ages=None, n_quantiles=4,
                                         figsize=(20, 15), n_permutations=100,
                                         custom_column_order=None):
    """
    Create transition matrices with null model comparisons.
    Shows actual transition matrix, an example null model, and the mean null model.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame containing parameter values for flies at different ages
    param_name : str
        Name of the parameter column to analyze
    ages : list, optional
        List of ages to include
    n_quantiles : int
        Number of quantiles to divide the data into
    figsize : tuple
        Figure size (width, height)
    n_permutations : int
        Number of permutations for significance testing
    custom_column_order : list or None
        Custom ordering of columns (e.g., [0,2,1,3] to swap columns 2 and 3)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from scipy import stats

    # Default ages if not provided
    if ages is None:
        ages = sorted(df_pheno['day'].unique())

    # We need at least 2 ages for transitions
    if len(ages) < 2:
        raise ValueError("Need at least 2 ages to compute transitions")

    # If no custom order specified, use default order
    if custom_column_order is None:
        custom_column_order = list(range(n_quantiles))

    # Check validity of custom_column_order
    if set(custom_column_order) != set(range(n_quantiles)):
        raise ValueError(f"custom_column_order must contain each integer from 0 to {n_quantiles - 1} exactly once")

    # Compute all pairs of ages
    age_pairs = []
    for i in range(len(ages)):
        for j in range(i + 1, len(ages)):
            age_pairs.append((ages[i], ages[j]))

    # Compute quantile labels
    quantile_labels = []
    for i in range(n_quantiles):
        q_low = i / n_quantiles * 100
        q_high = (i + 1) / n_quantiles * 100
        quantile_labels.append(f"Q{i + 1}\n({q_low:.0f}-{q_high:.0f}%)")

    # Create reordered tick labels
    custom_tick_labels = [quantile_labels[i] for i in custom_column_order]

    # Generate transition data for each age pair
    transition_data = []

    for age1, age2 in age_pairs:
        # Generate transition matrices and null models
        transition_data.append(
            generate_transition_matrices(df_pheno, param_name, age1, age2,
                                         n_quantiles, n_permutations)
        )

    # Create the figure with 3 rows (actual, example null, mean null)
    n_matrices = len(transition_data)
    fig = plt.figure(figsize=figsize)

    # Create a grid layout with space for colorbar
    gs = plt.GridSpec(3, n_matrices + 1,
                      width_ratios=[1] * n_matrices + [0.1],
                      height_ratios=[1, 1, 1],
                      wspace=0.2, hspace=0.3,
                      figure=fig)

    # Common colormap settings
    vmin, vmax = 0, 1

    # Plot matrices for each transition
    for col, (data, (age1, age2)) in enumerate(zip(transition_data, age_pairs)):
        # Extract data
        orig_prob = data['orig_prob']
        raw_counts = data['raw_counts']
        example_null = data['example_null']
        null_mean = data['null_mean']
        p_values = data['p_values']
        expected_value = data['expected_value']

        # Reorder columns based on custom_column_order
        orig_prob_reordered = orig_prob[:, custom_column_order]
        raw_counts_reordered = raw_counts[:, custom_column_order]
        example_null_reordered = example_null[:, custom_column_order]
        null_mean_reordered = null_mean[:, custom_column_order]
        p_values_reordered = p_values[:, custom_column_order]

        # Row 1: Actual transition matrix
        ax1 = fig.add_subplot(gs[0, col])

        # Create heatmap
        sns.heatmap(
            orig_prob_reordered,
            annot=False,
            cmap='Blues',
            vmin=vmin, vmax=vmax,
            cbar=(col == n_matrices - 1),
            cbar_ax=fig.add_subplot(gs[0, -1]) if col == n_matrices - 1 else None,
            ax=ax1, square=True,
            linewidths=0.5, linecolor='white'
        )

        # Add custom annotations
        for r in range(n_quantiles):
            for c in range(n_quantiles):
                # Get reordered column index
                c_orig = custom_column_order[c]

                count = raw_counts[r, c_orig]
                prob = orig_prob[r, c_orig]
                p_val = p_values[r, c_orig]

                # Determine significance markers
                sig_marker = ''
                if p_val < 0.05:
                    sig_marker = '*'
                    if p_val < 0.01:
                        sig_marker = '**'
                        if p_val < 0.001:
                            sig_marker = '***'

                text_color = 'white' if prob > 0.5 else 'black'

                # Show probability with significance marker
                ax1.text(
                    c + 0.5, r + 0.5 - 0.15,
                    f"{prob:.2f}{sig_marker}",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=9,
                    fontweight='bold'
                )

                # Show count in parentheses
                ax1.text(
                    c + 0.5, r + 0.5 + 0.15,
                    f"({int(count)})",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=7
                )

        # Set title and labels
        ax1.set_title(f'Actual: Day {age1} → {age2}', fontsize=12)
        ax1.set_xlabel(f'Day {age1} Quantiles', fontsize=11)
        ax1.set_ylabel(f'Day {age2} Quantiles', fontsize=11)

        # Set tick labels
        ax1.set_xticks(np.arange(n_quantiles) + 0.5)
        ax1.set_yticks(np.arange(n_quantiles) + 0.5)
        ax1.set_xticklabels(custom_tick_labels, fontsize=8)
        ax1.set_yticklabels(quantile_labels, fontsize=8)

        # Row 2: Example null model
        ax2 = fig.add_subplot(gs[1, col])

        sns.heatmap(
            example_null_reordered,
            annot=True, fmt='.2f',
            cmap='Greys',
            vmin=vmin, vmax=vmax,
            cbar=(col == n_matrices - 1),
            cbar_ax=fig.add_subplot(gs[1, -1]) if col == n_matrices - 1 else None,
            ax=ax2, square=True,
            linewidths=0.5, linecolor='white'
        )

        ax2.set_title(f'Example Null: Day {age1} → {age2}', fontsize=12)
        ax2.set_xlabel(f'Day {age1} Quantiles', fontsize=11)
        ax2.set_ylabel(f'Day {age2} Quantiles', fontsize=11)
        ax2.set_xticks(np.arange(n_quantiles) + 0.5)
        ax2.set_yticks(np.arange(n_quantiles) + 0.5)
        ax2.set_xticklabels(custom_tick_labels, fontsize=8)
        ax2.set_yticklabels(quantile_labels, fontsize=8)

        # Row 3: Mean null model
        ax3 = fig.add_subplot(gs[2, col])

        sns.heatmap(
            null_mean_reordered,
            annot=True, fmt='.2f',
            cmap='Greys',
            vmin=vmin, vmax=vmax,
            cbar=(col == n_matrices - 1),
            cbar_ax=fig.add_subplot(gs[2, -1]) if col == n_matrices - 1 else None,
            ax=ax3, square=True,
            linewidths=0.5, linecolor='white'
        )

        # Add a horizontal line to the title showing expected value
        ax3.set_title(f'Mean Null: Day {age1} → {age2}\nExpected: {expected_value:.2f}', fontsize=12)
        ax3.set_xlabel(f'Day {age1} Quantiles', fontsize=11)
        ax3.set_ylabel(f'Day {age2} Quantiles', fontsize=11)
        ax3.set_xticks(np.arange(n_quantiles) + 0.5)
        ax3.set_yticks(np.arange(n_quantiles) + 0.5)
        ax3.set_xticklabels(custom_tick_labels, fontsize=8)
        ax3.set_yticklabels(quantile_labels, fontsize=8)

    # Add significance legend
    fig.text(0.5, 0.01, "* p<0.05, ** p<0.01, *** p<0.001 versus uniform expectation",
             ha='center', fontsize=9)

    plt.suptitle(f'Parameter Transition Analysis: {param_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, transition_data


def generate_transition_matrices(df_pheno, param_name, age1, age2, n_quantiles=4, n_permutations=100):
    """
    Generate transition matrices and null models for parameter values between two ages.

    Parameters:
    -----------
    df_pheno : pandas.DataFrame
        DataFrame with fly parameter data
    param_name : str
        Name of parameter column to analyze
    age1, age2 : int
        Ages to compare (age1 < age2)
    n_quantiles : int
        Number of quantiles to divide data into
    n_permutations : int
        Number of permutations for null model

    Returns:
    --------
    dict containing transition matrices and null model statistics
    """
    import numpy as np
    from scipy import stats

    # Get data for these ages
    df_age1 = df_pheno[df_pheno['day'] == age1]
    df_age2 = df_pheno[df_pheno['day'] == age2]

    # Find common fly IDs
    common_fly_ids = list(set(df_age1['fly_id']).intersection(set(df_age2['fly_id'])))

    # Skip if not enough flies
    if len(common_fly_ids) < n_quantiles:
        raise ValueError(f"Not enough flies ({len(common_fly_ids)}) for {n_quantiles} quantiles")

    # Extract values
    values1 = np.array([df_age1[df_age1['fly_id'] == fly_id][param_name].values[0]
                        for fly_id in common_fly_ids])
    values2 = np.array([df_age2[df_age2['fly_id'] == fly_id][param_name].values[0]
                        for fly_id in common_fly_ids])

    # Assign flies to quantiles using rank-based method
    # This ensures balanced quantiles
    ranks1 = stats.rankdata(values1, method='average')
    ranks2 = stats.rankdata(values2, method='average')

    # Convert ranks to quantile indices (0 to n_quantiles-1)
    # This guarantees equal numbers in each quantile (or as close as possible with ties)
    q1 = np.floor((ranks1 - 1) / len(ranks1) * n_quantiles).astype(int)
    q2 = np.floor((ranks2 - 1) / len(ranks2) * n_quantiles).astype(int)

    # Cap at n_quantiles-1 (for any edge cases)
    q1 = np.minimum(q1, n_quantiles - 1)
    q2 = np.minimum(q2, n_quantiles - 1)

    # Count occurrences in each quantile
    q1_counts = np.bincount(q1, minlength=n_quantiles)
    q2_counts = np.bincount(q2, minlength=n_quantiles)

    # Build raw count matrix (transposed for earlier age on X-axis, later on Y-axis)
    # rows = quantiles of age2, columns = quantiles of age1
    raw_counts = np.zeros((n_quantiles, n_quantiles))
    for i in range(len(common_fly_ids)):
        raw_counts[q2[i], q1[i]] += 1

    # Convert to conditional probability (each column sums to 1)
    # This represents P(age2 quantile | age1 quantile)
    orig_prob = np.zeros_like(raw_counts, dtype=float)
    for j in range(n_quantiles):
        col_sum = raw_counts[:, j].sum()
        if col_sum > 0:
            orig_prob[:, j] = raw_counts[:, j] / col_sum

    # Calculate theoretical expectation for uniform distribution
    expected_value = 1.0 / n_quantiles

    # Generate null models by permutation
    null_matrices = np.zeros((n_permutations, n_quantiles, n_quantiles))

    # Generate one example null for visualization
    np.random.seed(42)  # Fixed seed for reproducibility
    q2_perm = np.random.permutation(q2)
    example_null = np.zeros((n_quantiles, n_quantiles))
    for i in range(len(common_fly_ids)):
        example_null[q2_perm[i], q1[i]] += 1

    # Convert to probability
    for j in range(n_quantiles):
        col_sum = example_null[:, j].sum()
        if col_sum > 0:
            example_null[:, j] = example_null[:, j] / col_sum

    # Run permutations for null distribution
    for p in range(n_permutations):
        # Randomly permute the quantile assignments for age2
        perm_q2 = np.random.permutation(q2)

        # Build permuted matrix
        perm_matrix = np.zeros((n_quantiles, n_quantiles))
        for i in range(len(common_fly_ids)):
            perm_matrix[perm_q2[i], q1[i]] += 1

        # Convert to probability (column-normalized)
        perm_prob = np.zeros_like(perm_matrix, dtype=float)
        for j in range(n_quantiles):
            col_sum = perm_matrix[:, j].sum()
            if col_sum > 0:
                perm_prob[:, j] = perm_matrix[:, j] / col_sum

        null_matrices[p] = perm_prob

    # Calculate mean and standard deviation of null models
    null_mean = np.mean(null_matrices, axis=0)
    null_std = np.std(null_matrices, axis=0)

    # Verify null model is uniform (each cell should be close to 1/n_quantiles)
    # If not, there's an issue with the permutation approach
    deviation = np.abs(null_mean - expected_value).max()
    is_uniform = deviation < 0.05  # Allow small deviation due to sample size

    # Calculate p-values against uniform expectation
    p_values = np.zeros((n_quantiles, n_quantiles))
    for i in range(n_quantiles):
        for j in range(n_quantiles):
            # Count how many permutation values are as or more extreme vs uniform
            diff_obs = abs(orig_prob[i, j] - expected_value)
            more_extreme = np.sum(abs(null_matrices[:, i, j] - expected_value) >= diff_obs)
            p_values[i, j] = more_extreme / n_permutations

    return {
        'orig_prob': orig_prob,
        'raw_counts': raw_counts,
        'example_null': example_null,
        'null_matrices': null_matrices,
        'null_mean': null_mean,
        'null_std': null_std,
        'p_values': p_values,
        'q1_counts': q1_counts,
        'q2_counts': q2_counts,
        'is_uniform': is_uniform,
        'expected_value': expected_value,
        'deviation': deviation
    }
