"""
Exploratory model comparison visualizations (not used in final manuscript).

Model comparison visualization: 1D vs 2D vs 3D models

Part A: Posterior Predictive Checks (PPC) - Figure S2 style
    - Compare observed vs simulated jump data for each model
    - Heatmap visualizations showing model fit quality

Part B: Individual fly trajectories - Figure 2C,D style
    - Compare parameter posteriors across models for specific flies
    - Show how 1D/2D/3D models differ in their predictions
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


from functions_common import jump_prob
from plot_jump_data_vs_fit import simulate_from_fit, plot_full_data
from plot_multiday_one_fly import plot_multiday_one_fly
from settings import DIR_FITS, DIR_OUTPUT, OMIT_FLY_IDS, heatmap_0, heatmap_1

# ============================================================================
# CONFIGURATION
# ============================================================================

GENOTYPE = 'KK'
DAYS = [7, 14, 21]
EXAMPLE_FLY_IDS = [44, 105]  # From manuscript Figure 2

# ============================================================================
# PART A: POSTERIOR PREDICTIVE CHECKS (PPC)
# ============================================================================

def simulate_from_fit_model(genotype, day, model_type='3d'):
    """
    Simulate jump data from fitted model (1D, 2D, or 3D).

    Args:
        genotype: 'KK' or 'GD'
        day: 7, 14, or 21
        model_type: '1d', '2d', or '3d'

    Returns:
        jumps_arr: np.array of simulated jumps (n_flies, n_trials)
    """
    # Load fitted parameters
    csv_path = DIR_FITS + os.sep + f'fly-stability-days-detailed-{model_type}.csv'
    df = pd.read_csv(csv_path)
    df = df[(df['genotype'] == genotype) & (df['day'] == day)]

    # Omit specified fly IDs
    omit_ids = OMIT_FLY_IDS.get(genotype, [])
    if omit_ids:
        df = df[~df['fly_id'].isin(omit_ids)]

    print(f"\n{model_type.upper()} model: {genotype} day {day}")
    print(f"  N flies after exclusions: {len(df)}")

    # Simulate jumps for each fly
    jumps_arr = []
    for i, row in df.iterrows():
        alpha = row['alpha']
        beta = row['beta']
        p0 = row['p0']

        # For 1D model, alpha and beta are NaN
        if model_type == '1d':
            # No habituation dynamics - constant p0
            p = np.full(1050, p0)
        else:
            # Full habituation dynamics
            p = jump_prob(alpha, beta, p0)

        jumps = np.random.binomial(1, p)
        jumps_arr.append(jumps)

    return np.array(jumps_arr)


def plot_ppc_comparison(genotype='KK', day=14, save=True):
    """
    Create PPC comparison heatmaps for 1D, 2D, 3D models.

    Shows observed data vs simulated data for each model type.
    Includes timeseries showing mean jump probability with gaps between blocks.
    """
    print(f"\n{'='*70}")
    print(f"PART A: Posterior Predictive Check - {genotype} day {day}")
    print(f"{'='*70}")

    # Load observed data
    data_jumps_arr, _ = plot_full_data(genotype=genotype, day=day,
                                       filtered=True, detailed_format=True,
                                       use_mpl=False)

    # Simulate from each model
    sim_1d = simulate_from_fit_model(genotype, day, '1d')
    sim_2d = simulate_from_fit_model(genotype, day, '2d')
    sim_3d = simulate_from_fit_model(genotype, day, '3d')

    # Create comparison figure with heatmaps on top, timeseries on bottom
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 2], hspace=0.3)

    # Top row: Heatmaps
    cmap = mpl.colors.ListedColormap([heatmap_0, heatmap_1])

    # Plot observed
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(data_jumps_arr, cmap=cmap, aspect='auto', interpolation='none')
    ax0.set_title(f'Observed\n{genotype} day {day}', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Fly ID')
    ax0.set_xlabel('Trial')
    ax0.invert_yaxis()

    # Plot simulated from each model
    axes_heat = [ax0]
    for idx, (sim, label) in enumerate([(sim_1d, '1D'), (sim_2d, '2D'), (sim_3d, '3D')]):
        ax = fig.add_subplot(gs[0, idx + 1])
        ax.imshow(sim, cmap=cmap, aspect='auto', interpolation='none')
        ax.set_title(f'{label} Model\n(simulated)', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel('Trial')
        ax.invert_yaxis()
        axes_heat.append(ax)

    # Add trial markers to heatmaps
    for ax in axes_heat:
        ax.axvline(x=200, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=400, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=600, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=800, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=1000, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    # Bottom row: Timeseries with gaps between blocks
    ax_ts = fig.add_subplot(gs[1, :])

    # Compute mean jump probability across flies for each trial
    mean_obs = data_jumps_arr.mean(axis=0)
    mean_1d = sim_1d.mean(axis=0)
    mean_2d = sim_2d.mean(axis=0)
    mean_3d = sim_3d.mean(axis=0)

    # Define block boundaries and create x-axis positions with gaps
    # Hab blocks: 0-200, 200-400, 400-600, 600-800, 800-1000
    # SRA: 1000-1050
    gap_size = 20  # Gap between blocks in x-axis units
    block_starts = [0, 200, 400, 600, 800]
    block_ends = [200, 400, 600, 800, 1000]
    sra_start, sra_end = 1000, 1050

    x_positions = []
    current_x = 0

    # Map original trial indices to new x-positions with gaps
    trial_to_x = {}

    # Hab blocks 1-5
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        block_trials = np.arange(start, end)
        block_x = np.arange(current_x, current_x + len(block_trials))
        for trial, x in zip(block_trials, block_x):
            trial_to_x[trial] = x
        current_x += len(block_trials) + gap_size

    # SRA block (with larger gap)
    sra_trials = np.arange(sra_start, sra_end)
    sra_x = np.arange(current_x, current_x + len(sra_trials))
    for trial, x in zip(sra_trials, sra_x):
        trial_to_x[trial] = x

    # Extract x-positions in order
    all_trials = list(range(1050))
    x_vals = [trial_to_x[t] for t in all_trials]

    # Plot timeseries
    ax_ts.plot(x_vals, mean_obs, 'k-', linewidth=2, label='Observed', alpha=0.8)
    ax_ts.plot(x_vals, mean_1d, '-', color='#E74C3C', linewidth=1.5, label='1D Model', alpha=0.7)
    ax_ts.plot(x_vals, mean_2d, '-', color='#3498DB', linewidth=1.5, label='2D Model', alpha=0.7)
    ax_ts.plot(x_vals, mean_3d, '-', color='#2ECC71', linewidth=1.5, label='3D Model', alpha=0.7)

    # Add vertical lines to show block boundaries
    block_x_boundaries = []
    for i in range(len(block_starts)):
        block_end_x = trial_to_x[block_ends[i] - 1]
        if i < len(block_starts) - 1:
            block_x_boundaries.append(block_end_x)

    for x_bound in block_x_boundaries:
        ax_ts.axvline(x=x_bound, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Mark SRA region
    sra_start_x = trial_to_x[sra_start]
    ax_ts.axvline(x=sra_start_x, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_ts.text(sra_start_x + 10, ax_ts.get_ylim()[1] * 0.95, 'SRA',
               fontsize=10, color='red', fontweight='bold')

    ax_ts.set_xlabel('Trial (with gaps between blocks)', fontsize=11)
    ax_ts.set_ylabel('Mean Jump Probability', fontsize=11)
    ax_ts.set_title('Population-averaged jump probability across trials', fontsize=12, fontweight='bold')
    ax_ts.legend(loc='upper right', framealpha=0.9)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.set_ylim(-0.05, 1.05)

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison'
        os.makedirs(output_dir, exist_ok=True)

        out_base = f'{output_dir}{os.sep}ppc_comparison_{genotype}_day{day}'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"\nSaved PPC comparison to: {out_base}.*")

    return fig


def plot_within_block_dynamics(genotype='KK', save=True, show_credible_interval=False):
    """
    Create Figure 1E-style plot showing within-block habituation dynamics.

    For each day (3 panels in a row):
    - Plot mean timeseries for first 100 trials of each hab block (5 curves)
    - Plot mean timeseries for SRA (first 50 trials)
    - Overlay 3 model predictions (posterior mean) as thick colored lines
    - Optionally show credible intervals from posterior draws

    Args:
        genotype: 'KK' or 'GD'
        save: Whether to save figures
        show_credible_interval: If True, show 95% credible interval bands
    """
    print(f"\n{'='*70}")
    print(f"WITHIN-BLOCK DYNAMICS - {genotype}")
    print(f"{'='*70}")

    days = [7, 14, 21]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Updated color palette (hab 1-5 + SRA)
    colors_blocks = ['#583e81', '#8088ba', '#3e99cb', '#93b1c2', '#cacaca']
    color_sra = '#333333'  # Dark grey
    colors_models = {'1d': '#E74C3C', '2d': '#3498DB', '3d': '#2ECC71'}

    for day_idx, day in enumerate(days):
        ax = axes[day_idx]

        # Load observed data
        data_jumps_arr, _ = plot_full_data(genotype=genotype, day=day,
                                           filtered=True, detailed_format=True,
                                           use_mpl=False)

        # Load CSV files to get posterior mean parameters
        csv_1d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-1d.csv')
        csv_2d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-2d.csv')
        csv_3d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv')

        # Filter for this genotype/day
        df_1d = csv_1d[(csv_1d['genotype'] == genotype) & (csv_1d['day'] == day)]
        df_2d = csv_2d[(csv_2d['genotype'] == genotype) & (csv_2d['day'] == day)]
        df_3d = csv_3d[(csv_3d['genotype'] == genotype) & (csv_3d['day'] == day)]

        # Omit excluded flies
        omit_ids = OMIT_FLY_IDS.get(genotype, [])
        if omit_ids:
            df_1d = df_1d[~df_1d['fly_id'].isin(omit_ids)]
            df_2d = df_2d[~df_2d['fly_id'].isin(omit_ids)]
            df_3d = df_3d[~df_3d['fly_id'].isin(omit_ids)]

        # Compute model predictions: mean of curves (NOT curve of means!)
        # For nonlinear functions, mean(f(x)) != f(mean(x))

        # 1D: constant p0 for each fly
        p_1d_curves = []
        for i, row in df_1d.iterrows():
            p_1d_curves.append(np.full(1050, row['p0']))
        p_1d = np.mean(p_1d_curves, axis=0)

        # 2D: compute curve for each fly, then average
        p_2d_curves = []
        for i, row in df_2d.iterrows():
            p_2d_curves.append(jump_prob(row['alpha'], row['beta'], row['p0']))
        p_2d = np.mean(p_2d_curves, axis=0)

        # 3D: compute curve for each fly, then average
        p_3d_curves = []
        for i, row in df_3d.iterrows():
            p_3d_curves.append(jump_prob(row['alpha'], row['beta'], row['p0']))
        p_3d = np.mean(p_3d_curves, axis=0)

        # Optionally compute credible intervals from posterior draws
        if show_credible_interval:
            # Load NPZ files
            npz_2d = np.load(DIR_FITS + os.sep + f"{genotype}_day{day}_2d_draws.npz")
            npz_3d = np.load(DIR_FITS + os.sep + f"{genotype}_day{day}_3d_draws.npz")

            # Sample trajectories from posterior (e.g., 100 random draws)
            n_samples = 100
            p_2d_samples = []
            p_3d_samples = []

            for fly_key in npz_2d.files:
                draws_2d = npz_2d[fly_key]  # Shape: (n_draws, 2) -> [β, p₀]
                draws_3d = npz_3d[fly_key]  # Shape: (n_draws, 3) -> [α, β, p₀]

                # Random sample indices
                sample_idx = np.random.choice(len(draws_2d), size=n_samples, replace=False)

                for idx in sample_idx:
                    # 2D model
                    beta_2d_sample = draws_2d[idx, 0]
                    p0_2d_sample = draws_2d[idx, 1]
                    alpha_2d_sample = df_2d[df_2d['fly_id'] == int(fly_key.split('_')[-1])]['alpha'].values[0]
                    p_2d_samples.append(jump_prob(alpha_2d_sample, beta_2d_sample, p0_2d_sample))

                    # 3D model
                    alpha_3d_sample = draws_3d[idx, 0]
                    beta_3d_sample = draws_3d[idx, 1]
                    p0_3d_sample = draws_3d[idx, 2]
                    p_3d_samples.append(jump_prob(alpha_3d_sample, beta_3d_sample, p0_3d_sample))

            p_2d_samples = np.array(p_2d_samples)
            p_3d_samples = np.array(p_3d_samples)

            # Compute 95% credible intervals
            p_2d_lower = np.percentile(p_2d_samples, 2.5, axis=0)
            p_2d_upper = np.percentile(p_2d_samples, 97.5, axis=0)
            p_3d_lower = np.percentile(p_3d_samples, 2.5, axis=0)
            p_3d_upper = np.percentile(p_3d_samples, 97.5, axis=0)

        # Plot habituation blocks (first 100 trials of each 200-trial block)
        for block_idx in range(5):
            start_trial = block_idx * 200
            end_trial = start_trial + 100  # Only first 100 trials

            # Observed mean across flies
            obs_mean = data_jumps_arr[:, start_trial:end_trial].mean(axis=0)

            # Model predictions (posterior mean)
            p_1d_block = p_1d[start_trial:end_trial]
            p_2d_block = p_2d[start_trial:end_trial]
            p_3d_block = p_3d[start_trial:end_trial]

            x = np.arange(100)

            # Plot observed data (thin lines)
            ax.plot(x, obs_mean, '-', color=colors_blocks[block_idx],
                   linewidth=1.5, alpha=0.8, label=f'Hab {block_idx+1} (obs)')

            # Plot model predictions (thick lines, only label first block)
            if block_idx == 0:
                ax.plot(x, p_1d_block, '-', color=colors_models['1d'],
                       linewidth=3, alpha=0.8, label='1D model')
                ax.plot(x, p_2d_block, '-', color=colors_models['2d'],
                       linewidth=3, alpha=0.8, label='2D model')
                ax.plot(x, p_3d_block, '-', color=colors_models['3d'],
                       linewidth=3, alpha=0.8, label='3D model')

                # Add credible intervals if requested (only for 2D/3D)
                if show_credible_interval:
                    ax.fill_between(x, p_2d_lower[start_trial:end_trial],
                                   p_2d_upper[start_trial:end_trial],
                                   color=colors_models['2d'], alpha=0.2)
                    ax.fill_between(x, p_3d_lower[start_trial:end_trial],
                                   p_3d_upper[start_trial:end_trial],
                                   color=colors_models['3d'], alpha=0.2)
            else:
                ax.plot(x, p_1d_block, '-', color=colors_models['1d'],
                       linewidth=3, alpha=0.6)
                ax.plot(x, p_2d_block, '-', color=colors_models['2d'],
                       linewidth=3, alpha=0.6)
                ax.plot(x, p_3d_block, '-', color=colors_models['3d'],
                       linewidth=3, alpha=0.6)

                # Add credible intervals if requested
                if show_credible_interval:
                    ax.fill_between(x, p_2d_lower[start_trial:end_trial],
                                   p_2d_upper[start_trial:end_trial],
                                   color=colors_models['2d'], alpha=0.15)
                    ax.fill_between(x, p_3d_lower[start_trial:end_trial],
                                   p_3d_upper[start_trial:end_trial],
                                   color=colors_models['3d'], alpha=0.15)

        # Plot SRA block (trials 1000-1050, show as 0-50)
        sra_obs_mean = data_jumps_arr[:, 1000:1050].mean(axis=0)
        p_1d_sra = p_1d[1000:1050]
        p_2d_sra = p_2d[1000:1050]
        p_3d_sra = p_3d[1000:1050]

        x_sra = np.arange(50)
        ax.plot(x_sra, sra_obs_mean, '-', color=color_sra,
               linewidth=1.5, alpha=0.8, label='SRA (obs)')
        ax.plot(x_sra, p_1d_sra, '-', color=colors_models['1d'],
               linewidth=3, alpha=0.6)
        ax.plot(x_sra, p_2d_sra, '-', color=colors_models['2d'],
               linewidth=3, alpha=0.6)
        ax.plot(x_sra, p_3d_sra, '-', color=colors_models['3d'],
               linewidth=3, alpha=0.6)

        # Add credible intervals for SRA if requested
        if show_credible_interval:
            ax.fill_between(x_sra, p_2d_lower[1000:1050], p_2d_upper[1000:1050],
                           color=colors_models['2d'], alpha=0.15)
            ax.fill_between(x_sra, p_3d_lower[1000:1050], p_3d_upper[1000:1050],
                           color=colors_models['3d'], alpha=0.15)

        # Formatting
        ax.set_xlabel('Trial within block', fontsize=11)
        ax.set_ylabel('Mean jump probability', fontsize=11)
        ax.set_title(f'{genotype} Day {day}', fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison'
        os.makedirs(output_dir, exist_ok=True)

        out_base = f'{output_dir}{os.sep}within_block_dynamics_{genotype}'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"\nSaved within-block dynamics to: {out_base}.*")

    return fig


# ============================================================================
# PART B: INDIVIDUAL FLY TRAJECTORIES (Figure 2C,D style)
# ============================================================================

def load_draws_all_models(genotype, day, fly_id):
    """
    Load posterior draws for a single fly across all three models.

    Returns:
        dict with keys '1d', '2d', '3d', each containing draws array
    """
    draws = {}

    for model_type in ['1d', '2d', '3d']:
        npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_{model_type}_draws.npz"
        npz_data = np.load(npz_file)
        fly_key = f"fly_id_{fly_id}"

        if fly_key in npz_data.files:
            draws[model_type] = npz_data[fly_key]
        else:
            print(f"Warning: {fly_key} not found in {model_type} model for day {day}")
            draws[model_type] = None

    return draws


def plot_fly_model_comparison(genotype, fly_id, days, save=True):
    """
    Compare 1D/2D/3D model fits for a single fly across days.

    Creates Figure 2C,D style plots showing:
    - Parameter posteriors for each model
    - Model predictions compared to data
    """
    print(f"\n{'='*70}")
    print(f"PART B: Individual Fly Comparison - {genotype} fly {fly_id}")
    print(f"{'='*70}")

    # Load CSV summaries for all models
    csv_1d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-1d.csv')
    csv_2d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-2d.csv')
    csv_3d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv')

    # Filter for this fly
    df_1d = csv_1d[(csv_1d['genotype'] == genotype) & (csv_1d['fly_id'] == fly_id)]
    df_2d = csv_2d[(csv_2d['genotype'] == genotype) & (csv_2d['fly_id'] == fly_id)]
    df_3d = csv_3d[(csv_3d['genotype'] == genotype) & (csv_3d['fly_id'] == fly_id)]

    # Load posterior draws for all models and days
    draws_all = {
        '1d': {day: load_draws_all_models(genotype, day, fly_id)['1d'] for day in days},
        '2d': {day: load_draws_all_models(genotype, day, fly_id)['2d'] for day in days},
        '3d': {day: load_draws_all_models(genotype, day, fly_id)['3d'] for day in days}
    }

    print(f"\nLoaded draws for fly {fly_id}:")
    for model in ['1d', '2d', '3d']:
        for day in days:
            if draws_all[model][day] is not None:
                print(f"  {model.upper()} day {day}: shape = {draws_all[model][day].shape}")

    # Create figure with 3 rows (one per day) × 3 cols (p0, beta, alpha)
    fig, axes = plt.subplots(len(days), 3, figsize=(15, 4 * len(days)))
    fig.suptitle(f'Model Comparison: {genotype} fly {fly_id}', fontsize=16, y=0.995)

    colors = {'1d': '#E74C3C', '2d': '#3498DB', '3d': '#2ECC71'}
    model_labels = {'1d': '1D (p₀ only)', '2d': '2D (p₀, β)', '3d': '3D (p₀, α, β)'}

    # Collect data ranges for consistent axis limits
    all_p0_vals = []
    all_beta_vals = []
    all_alpha_vals = []

    for day in days:
        for model in ['1d', '2d', '3d']:
            if draws_all[model][day] is not None:
                all_p0_vals.extend(draws_all[model][day][:, -1])
                if model in ['2d', '3d']:
                    beta_col = 0 if model == '2d' else 1
                    all_beta_vals.extend(draws_all[model][day][:, beta_col])
                if model == '3d':
                    all_alpha_vals.extend(draws_all[model][day][:, 0])

    # Also include alpha_global from 2D
    for day in days:
        alpha_global = df_2d[df_2d['day'] == day]['alpha'].values[0]
        all_alpha_vals.append(alpha_global)

    # Compute shared x-limits (with padding)
    p0_lim = [max(0, min(all_p0_vals) - 0.05), min(1, max(all_p0_vals) + 0.05)]
    beta_lim = [max(0, min(all_beta_vals) - 0.01), max(all_beta_vals) + 0.01]
    alpha_lim = [max(0, min(all_alpha_vals) - 0.02), max(all_alpha_vals) + 0.02]

    for day_idx, day in enumerate(days):
        # p0 posterior
        ax_p0 = axes[day_idx, 0]
        for model in ['1d', '2d', '3d']:
            if draws_all[model][day] is not None:
                # Extract p0 from draws (last column for all models)
                p0_draws = draws_all[model][day][:, -1]
                ax_p0.hist(p0_draws, bins=50, alpha=0.5, color=colors[model],
                          label=model_labels[model], density=True)
        ax_p0.set_xlabel('p₀ (basal jump prob.)', fontsize=10)
        ax_p0.set_ylabel('Density', fontsize=10)
        ax_p0.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_p0.set_xlim(p0_lim)
        ax_p0.legend(fontsize=8)
        ax_p0.grid(True, alpha=0.3)

        # beta posterior
        ax_beta = axes[day_idx, 1]
        for model in ['2d', '3d']:  # 1D doesn't have beta
            if draws_all[model][day] is not None:
                # Extract beta: column 0 for 2D [β, p₀], column 1 for 3D [α, β, p₀]
                beta_col = 0 if model == '2d' else 1
                beta_draws = draws_all[model][day][:, beta_col]
                ax_beta.hist(beta_draws, bins=50, alpha=0.5, color=colors[model],
                            label=model_labels[model], density=True)
        ax_beta.set_xlabel('β (accumulation rate)', fontsize=10)
        ax_beta.set_ylabel('Density', fontsize=10)
        ax_beta.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_beta.set_xlim(beta_lim)
        ax_beta.legend(fontsize=8)
        ax_beta.grid(True, alpha=0.3)
        ax_beta.text(0.5, 0.95, '(1D: fixed at 0)', transform=ax_beta.transAxes,
                    ha='center', va='top', fontsize=8, style='italic', color='gray')

        # alpha posterior
        ax_alpha = axes[day_idx, 2]
        # Only 3D has fly-specific alpha; 2D uses alpha_global
        if draws_all['3d'][day] is not None:
            alpha_draws_3d = draws_all['3d'][day][:, 0]  # column 0 for 3D (alpha is first)
            ax_alpha.hist(alpha_draws_3d, bins=50, alpha=0.5, color=colors['3d'],
                         label=model_labels['3d'], density=True)

        # For 2D, show alpha_global as vertical line
        alpha_global_2d = df_2d[df_2d['day'] == day]['alpha'].values[0]
        ax_alpha.axvline(alpha_global_2d, color=colors['2d'], linestyle='--',
                        linewidth=2, label=f'2D (α_global = {alpha_global_2d:.3f})')

        ax_alpha.set_xlabel('α (habituation rate)', fontsize=10)
        ax_alpha.set_ylabel('Density', fontsize=10)
        ax_alpha.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_alpha.set_xlim(alpha_lim)
        ax_alpha.legend(fontsize=8)
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.text(0.5, 0.95, '(1D: no habituation)', transform=ax_alpha.transAxes,
                     ha='center', va='top', fontsize=8, style='italic', color='gray')

    # Sync y-limits within each row
    for row_idx in range(len(days)):
        y_max_row = max([axes[row_idx, col].get_ylim()[1] for col in range(3)])
        for col in range(3):
            axes[row_idx, col].set_ylim([0, y_max_row])

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison'
        os.makedirs(output_dir, exist_ok=True)

        out_base = f'{output_dir}{os.sep}fly_comparison_{genotype}_fly{fly_id}'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"\nSaved fly comparison to: {out_base}.*")

    return fig


# ============================================================================
# PART C: POPULATION PARAMETER DISTRIBUTIONS
# ============================================================================

def plot_population_distributions(genotype='KK', days=[7, 14, 21], save=True):
    """
    Plot distribution of posterior means across the fly population.

    Similar to Part B layout (3 rows × 3 cols), but instead of showing
    posterior draws for a single fly, shows histogram of posterior means
    across all flies in the population.

    Shows between-fly variability in fitted parameters.

    Args:
        genotype: 'KK' or 'GD'
        days: List of days to plot
        save: Whether to save figure
    """
    print(f"\n{'='*70}")
    print(f"PART C: Population Parameter Distributions - {genotype}")
    print(f"{'='*70}")

    # Load CSV summaries for all models
    csv_1d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-1d.csv')
    csv_2d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-2d.csv')
    csv_3d = pd.read_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv')

    # Filter for this genotype
    csv_1d = csv_1d[csv_1d['genotype'] == genotype]
    csv_2d = csv_2d[csv_2d['genotype'] == genotype]
    csv_3d = csv_3d[csv_3d['genotype'] == genotype]

    # Collect data ranges for consistent axis limits across all days
    all_p0_vals = []
    all_beta_vals = []
    all_alpha_vals = []

    for day in days:
        # Collect p0 from all models
        all_p0_vals.extend(csv_1d[csv_1d['day'] == day]['p0'].values)
        all_p0_vals.extend(csv_2d[csv_2d['day'] == day]['p0'].values)
        all_p0_vals.extend(csv_3d[csv_3d['day'] == day]['p0'].values)

        # Collect beta from 2D and 3D
        all_beta_vals.extend(csv_2d[csv_2d['day'] == day]['beta'].values)
        all_beta_vals.extend(csv_3d[csv_3d['day'] == day]['beta'].values)

        # Collect alpha from 3D (plus alpha_global from 2D)
        all_alpha_vals.extend(csv_3d[csv_3d['day'] == day]['alpha'].values)
        alpha_global = csv_2d[csv_2d['day'] == day]['alpha'].values[0]
        all_alpha_vals.append(alpha_global)

    # Compute shared x-limits (with padding)
    p0_lim = [max(0, min(all_p0_vals) - 0.05), min(1, max(all_p0_vals) + 0.05)]
    beta_lim = [max(0, min(all_beta_vals) - 0.01), max(all_beta_vals) + 0.01]
    alpha_lim = [max(0, min(all_alpha_vals) - 0.02), max(all_alpha_vals) + 0.02]

    # Create figure with 3 rows (days) × 3 cols (p0, beta, alpha)
    fig, axes = plt.subplots(len(days), 3, figsize=(15, 4 * len(days)))
    fig.suptitle(f'Population Parameter Distributions: {genotype}', fontsize=16, y=0.995)

    colors = {'1d': '#E74C3C', '2d': '#3498DB', '3d': '#2ECC71'}
    model_labels = {'1d': '1D (p₀ only)', '2d': '2D (p₀, β)', '3d': '3D (p₀, α, β)'}

    for day_idx, day in enumerate(days):
        # Filter for this day
        df_1d_day = csv_1d[csv_1d['day'] == day]
        df_2d_day = csv_2d[csv_2d['day'] == day]
        df_3d_day = csv_3d[csv_3d['day'] == day]

        print(f"\n  Day {day}:")
        print(f"    1D: {len(df_1d_day)} flies")
        print(f"    2D: {len(df_2d_day)} flies")
        print(f"    3D: {len(df_3d_day)} flies")

        # ===== p0 distribution =====
        ax_p0 = axes[day_idx, 0]

        # Histogram of posterior means across flies
        for model, df, label in [('1d', df_1d_day, model_labels['1d']),
                                  ('2d', df_2d_day, model_labels['2d']),
                                  ('3d', df_3d_day, model_labels['3d'])]:
            p0_means = df['p0'].values
            ax_p0.hist(p0_means, bins=20, alpha=0.5, color=colors[model],
                      label=label, density=True)

        ax_p0.set_xlabel('p₀ (basal jump prob.)', fontsize=10)
        ax_p0.set_ylabel('Density', fontsize=10)
        ax_p0.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_p0.set_xlim(p0_lim)
        ax_p0.legend(fontsize=8)
        ax_p0.grid(True, alpha=0.3)

        # ===== beta distribution =====
        ax_beta = axes[day_idx, 1]

        # Only 2D and 3D have fitted beta
        for model, df, label in [('2d', df_2d_day, model_labels['2d']),
                                  ('3d', df_3d_day, model_labels['3d'])]:
            beta_means = df['beta'].values
            ax_beta.hist(beta_means, bins=20, alpha=0.5, color=colors[model],
                        label=label, density=True)

        ax_beta.set_xlabel('β (accumulation rate)', fontsize=10)
        ax_beta.set_ylabel('Density', fontsize=10)
        ax_beta.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_beta.set_xlim(beta_lim)
        ax_beta.legend(fontsize=8)
        ax_beta.grid(True, alpha=0.3)
        ax_beta.text(0.5, 0.95, '(1D: no β parameter)', transform=ax_beta.transAxes,
                    ha='center', va='top', fontsize=8, style='italic', color='gray')

        # ===== alpha distribution =====
        ax_alpha = axes[day_idx, 2]

        # Only 3D has fly-specific alpha
        alpha_means_3d = df_3d_day['alpha'].values
        ax_alpha.hist(alpha_means_3d, bins=20, alpha=0.5, color=colors['3d'],
                     label=model_labels['3d'], density=True)

        # 2D uses alpha_global (same for all flies in this condition)
        alpha_global_2d = df_2d_day['alpha'].values[0]  # Same for all flies
        ax_alpha.axvline(alpha_global_2d, color=colors['2d'], linestyle='--',
                        linewidth=2, label=f'2D (α_global = {alpha_global_2d:.3f})')

        ax_alpha.set_xlabel('α (habituation rate)', fontsize=10)
        ax_alpha.set_ylabel('Density', fontsize=10)
        ax_alpha.set_title(f'Day {day}', fontsize=11, fontweight='bold')
        ax_alpha.set_xlim(alpha_lim)
        ax_alpha.legend(fontsize=8)
        ax_alpha.grid(True, alpha=0.3)
        ax_alpha.text(0.5, 0.95, '(1D: no habituation)', transform=ax_alpha.transAxes,
                     ha='center', va='top', fontsize=8, style='italic', color='gray')

    # Sync y-limits within each row
    for row_idx in range(len(days)):
        y_max_row = max([axes[row_idx, col].get_ylim()[1] for col in range(3)])
        for col in range(3):
            axes[row_idx, col].set_ylim([0, y_max_row])

    plt.tight_layout()

    if save:
        output_dir = DIR_OUTPUT + os.sep + 'model_comparison'
        os.makedirs(output_dir, exist_ok=True)

        out_base = f'{output_dir}{os.sep}population_distributions_{genotype}'
        plt.savefig(f'{out_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_base}.svg', bbox_inches='tight')
        print(f"\nSaved population distributions to: {out_base}.*")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    print("\n" + "="*70)
    print("MODEL COMPARISON VISUALIZATION: 1D vs 2D vs 3D")
    print("="*70)

    # ========================================
    # PART A: Posterior Predictive Checks
    # ========================================

    for day in DAYS:
        fig_ppc = plot_ppc_comparison(genotype=GENOTYPE, day=day, save=True)
        plt.show()

    # ========================================
    # Within-Block Dynamics (Fig 1E style)
    # ========================================

    # Set to True to show 95% credible interval bands from posterior draws
    SHOW_CREDIBLE_INTERVALS = False

    fig_dynamics = plot_within_block_dynamics(genotype=GENOTYPE, save=True,
                                              show_credible_interval=SHOW_CREDIBLE_INTERVALS)
    plt.show()

    # ========================================
    # PART B: Individual Fly Comparisons
    # ========================================

    for fly_id in EXAMPLE_FLY_IDS:
        fig_fly = plot_fly_model_comparison(genotype=GENOTYPE, fly_id=fly_id,
                                            days=DAYS, save=True)
        plt.show()

    # ========================================
    # PART C: Population Distributions (histogram of posterior means)
    # ========================================

    fig_pop = plot_population_distributions(genotype=GENOTYPE, days=DAYS, save=True)
    plt.show()

    print("\n" + "="*70)
    print("MODEL COMPARISON COMPLETE!")
    print("="*70)
