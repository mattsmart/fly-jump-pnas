import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, cm

from data_tools import df_to_arr_jumps, get_TTC_canonical, habituation_time_fisherexact
from data_format_add_score_columns import (compute_p_ss, compute_hab_time_half_rel, compute_hab_time_95_rel,
                                           compute_hab_magnitude_abs, compute_hab_magnitude_rel)
from functions_common import likelihood_func_vec
from settings import DIR_OUTPUT, DIR_DATA_EXPT, DIR_FITS, OMIT_FLY_IDS

##############################
# Toggle flags:
SORT_FIGURES = True    # if True, each individual figure (Figures 1–7) is sorted independently by its metric.
TALL_MODE = True      # if True, swap the x and y axes (so fly IDs appear on the vertical axis).
MEDIAN_MODE = False    # if True, use median (instead of mean) for central value calculations and ranking.
RASTER_SCATTER = True  # if True, use raster scatter plots (much smaller SVG files).
COLOR_BY_P0 = True     # if True, color each fly's scatter points by the mean value of p0 using a spectral colormap.
remove_spines = True   # if True, remove top and right ax spines (borders)

SHOW_CHAMBER_BOX_ID = False  # if True, show box/chamber ID labels on a twin axis

# Performance: Thin posterior draws to speed up computation
# Set to None to use all draws (~8000), or specify number of draws to use (e.g., 800)
# If less than total draws, will subsample evenly across the full posterior
N_DRAWS_TO_USE = 800  # Recommend 800 for faster execution, or None for full posterior (8000 orig)

# p_transient metric configuration
P_TRANSIENT_CUTOFF_INDEX = 10  # Number of trials to use for transient phase
##############################

# --- New Options: Highlight Fly IDs ---
# These lists use the numeric fly ID (extracted from the key "fly_id_X").
HIGHLIGHT_FLY_IDS = {
    #'KK': [91, 44],       # For genotype 'KK', highlight these fly numbers.
    'KK': [],
    'GD': []
}

CHAMBER_IDX_TO_BOX = {i: ((i - 1) // 8) for i in range(1, 33)}

# --- Helper function: central_value ---
def central_value(x):
    return np.median(x) if MEDIAN_MODE else np.mean(x)

# --- Load the parameters inferred from the fit ---
df_summary_csv = pd.read_csv(DIR_FITS + os.sep + "fly-stability-days-detailed-3d-habscores.csv")
print(df_summary_csv.head())

# Load posterior draws for multiple days/genotypes (3D model)
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

# Apply draw thinning if requested (for performance)
if N_DRAWS_TO_USE is not None:
    print(f"\n>>> Thinning posterior draws to {N_DRAWS_TO_USE} samples for performance...")
    for genotype in draws_over_age.keys():
        for day in draws_over_age[genotype].keys():
            draws_dict = draws_over_age[genotype][day]
            # Get original number of draws from first fly
            first_key = [k for k in draws_dict.keys() if k.startswith('fly_id_')][0]
            n_draws_original = draws_dict[first_key].shape[0]

            if N_DRAWS_TO_USE < n_draws_original:
                # Calculate thinning factor (take every Nth draw to get approximately N_DRAWS_TO_USE samples)
                thin_factor = max(1, n_draws_original // N_DRAWS_TO_USE)
                indices = np.arange(0, n_draws_original, thin_factor)[:N_DRAWS_TO_USE]

                # Create a new dictionary with thinned arrays (NpzFile doesn't support item assignment)
                thinned_dict = {}
                for key in draws_dict.keys():
                    if isinstance(draws_dict[key], np.ndarray) and len(draws_dict[key].shape) > 0:
                        # Only thin if first dimension matches n_draws_original
                        if draws_dict[key].shape[0] == n_draws_original:
                            thinned_dict[key] = draws_dict[key][indices]
                        else:
                            thinned_dict[key] = draws_dict[key]
                    else:
                        thinned_dict[key] = draws_dict[key]

                # Replace the NpzFile with the thinned dictionary
                draws_over_age[genotype][day] = thinned_dict

                print(f"  {genotype} day {day}: {n_draws_original} -> {len(indices)} draws (thin factor = {thin_factor})")
    print(">>> Draw thinning complete.\n")

# specify genotype and day
genotype = 'KK'
age = 14

# filter the summary df for genotype and day
df_summary_csv_genotype = df_summary_csv[df_summary_csv['genotype'] == genotype]
df_summary_csv_genotype_day = df_summary_csv_genotype[df_summary_csv_genotype['day'] == age]

draws = draws_over_age[genotype][age]
num_draws = draws[list(draws.keys())[0]].shape[0]

# Determine the number of flies (up to 128)
num_flies = 128
available_fly_keys = [key for key in draws.keys() if key.startswith("fly_id_")]
# Omit flies as specified for this genotype.
omit_ids = OMIT_FLY_IDS.get(genotype, [])
available_fly_keys = [k for k in available_fly_keys if int(k.split('_')[-1]) not in omit_ids]
available_fly_keys.sort(key=lambda x: int(x.split('_')[-1]))
# get chamber id for each fly id in available_fly_keys
available_fly_ids = [int(k.split('_')[-1]) for k in available_fly_keys]
# get chamber id and box id for each fly id in available_fly_keys
chambers_for_avail_fly_ids = df_summary_csv_genotype_day[df_summary_csv_genotype_day['fly_id'].isin(available_fly_ids)]['chamber_id'].values
boxes_for_avail_fly_ids = [CHAMBER_IDX_TO_BOX[chamber_id] for chamber_id in chambers_for_avail_fly_ids]
# helper dicts for plotting
fly_to_box     = {fly_key: box_id for fly_key, box_id in zip(available_fly_keys, boxes_for_avail_fly_ids)}
fly_to_chamber = {fly_key: chamber_id for fly_key, chamber_id in zip(available_fly_keys, chambers_for_avail_fly_ids)}

num_flies = min(num_flies, len(available_fly_keys))
default_xticklabels = [k.split('_')[-1] for k in available_fly_keys]

# --- Set up the color map for scatter points ---
scatter_cmap = plt.get_cmap('Spectral')
# Loop over all available flies to gather all p0 values.
all_p0 = []
for key in available_fly_keys:
    samples = draws[key]
    all_p0.extend(samples[:, 2])
vmin = np.min(all_p0)
vmax = np.max(all_p0)
print('p0: vmin/vmax (all draws) - ', vmin, vmax)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

metric_names = ['p_transient', 'hab_tail', 'sra_mean', 'hab_halftime_rel', 'hab_saturation_time_rel',
                'hab_ttc', 'hab_fisherexact', 'hab_magnitude_rel', 'hab_magnitude_abs',
                'alpha', 'beta', 'p0']
metrics_to_skip = ['hab_fisherexact']  # list of metrics to skip (e.g. if slow to compute)

data_for_out_csv = {a: {'data': np.zeros((num_flies, num_draws)),
                        'fly_id': available_fly_keys}
                    for a in metric_names}

data_for_chambers = {a: {chamber_id: {'z_model_data_by_fly': [],  # will be list of len: associated_fly_ids
                                      'expt_data_per_trial': [],  # check below if len == 0 for plotting cases
                                      'expt_data_mean': [],       # check below if len == 0 for plotting cases
                                      'associated_fly_ids': [],
                                      'associated_colors': []}
                           for chamber_id in range(1, 33)}
                     for a in metric_names}
data_for_boxes = {a: {box_id: {'z_model_data_by_fly': [],  # will be list of len: associated_fly_ids
                               'expt_data_per_trial': [],
                               'expt_data_mean': [],
                               'associated_fly_ids': [],
                               'associated_colors': []}
                        for box_id in [0, 1, 2, 3]}
                     for a in metric_names}

##############################################################################
# Precompute independent sorted positions for each metric (for FIGURES 1–7)
# We build a dictionary: sorted_positions[metric_name] = { fly_id_str: rank, ... }
# Now using central_value (mean or median based on MEDIAN_MODE).
##############################################################################
sorted_positions = {}
if SORT_FIGURES:
    for m in metric_names:

        print('Working on sort:', m, '...')
        fly_stats = []  # list of (fly_id_str, summary_value)
        for fly_id_str in available_fly_keys:
            samples = draws[fly_id_str]
            alpha = samples[:, 0]
            beta  = samples[:, 1]
            p0    = samples[:, 2]

            likelihood = likelihood_func_vec(np.arange(200), alpha, beta, p0, pulse_period=1)
            simulated_data = np.random.binomial(1, likelihood)  # shape: ndraw x ntime=200

            # Compute the metric value for this fly
            if m in metrics_to_skip:
                metric_val = 0
            else:
                if m == 'p_transient':
                    transient_curves = likelihood_func_vec(np.arange(P_TRANSIENT_CUTOFF_INDEX),
                                                          alpha, beta, p0, pulse_period=1)
                    metric_val = central_value(np.mean(transient_curves, axis=1))
                elif m == 'hab_tail':
                    metric_val = central_value(compute_p_ss(alpha, beta, p0, T=1))
                elif m == 'sra_mean':
                    sra_curves = likelihood_func_vec(np.arange(50), alpha, beta, p0, pulse_period=5)
                    metric_val = central_value(np.mean(sra_curves, axis=1))
                elif m == 'hab_halftime_rel':
                    metric_val = central_value(compute_hab_time_half_rel(alpha, beta, T=1))
                elif m == 'hab_saturation_time_rel':
                    metric_val = central_value(compute_hab_time_95_rel(alpha, beta, T=1))
                elif m == 'hab_ttc':
                    ttc_vals = [get_TTC_canonical(''.join(str(x) for x in simulated_data[i, :]), 5)
                                for i in range(simulated_data.shape[0])]
                    metric_val = central_value(ttc_vals)
                elif m == 'hab_fisherexact':
                    #fisherexact_vals = [habituation_time_fisherexact(simulated_data[i, :])
                    #                    for i in range(simulated_data.shape[0])]
                    fisherexact_vals = [habituation_time_fisherexact(simulated_data[i, :])
                                        for i in range(10)]  # use fewer samples because quite slow
                    metric_val = central_value(fisherexact_vals)
                elif m == 'hab_magnitude_rel':
                    metric_val = central_value(compute_hab_magnitude_rel(alpha, beta, T=1))
                elif m == 'hab_magnitude_abs':
                    metric_val = central_value(compute_hab_magnitude_abs(alpha, beta, p0, T=1))
                elif m == 'alpha':
                    metric_val = central_value(alpha)
                elif m == 'beta':
                    metric_val = central_value(beta)
                elif m == 'p0':
                    metric_val = central_value(p0)
                fly_stats.append((fly_id_str, metric_val))
        # Sort (lowest first) and assign ranks 1 ... num_flies
        fly_stats.sort(key=lambda x: x[1])
        sorted_positions[m] = {fs[0]: rank+1 for rank, fs in enumerate(fly_stats)}
else:
    sorted_positions = None

##############################################################################
# FIGURES 1–7: Individual Dot Plots for each metric (model draws plus data overlay)
# In the inner loop below, for each metric we use the fly’s independent sorted rank
# (if SORT_FIGURES is True) instead of idx+1.
##############################################################################

# Original sizing
if TALL_MODE:
    #figsize = (6, 18)
    #figsize = (4.5, 18)  # thinner if doing 4x instead of 3x cols
    figsize = (3.5, 18)  # even thinner for 4x (double col) or 2x (single col)
else:
    #figsize = (14, 6)   # (14, 6) is OK for powerpoint, but not for sub-figure
    figsize = (18, 6)
# fontsize settings
fontsize_flylabel = 18
fontsize_zlabel = 12
fontsize_yticks = 9.5
fontsize_zval_ticks = 16
fontsize_title = 14
# plot linewidth & markersize settings (linewidths reduced 50% for Illustrator scaling)
z_CI_lw = 0.6  # Posterior CI line width (was 1.2) - REDUCED
z_center_ms = 5.5  # Center point marker size - UNCHANGED
z_scatter_sz = 30  # Posterior scatter size - UNCHANGED
data_mean_sz = 8  # Mean data marker size - UNCHANGED
data_trial_sz = 4.75  # Per-trial data marker size - UNCHANGED
data_mean_mew = 0.5  # Mean marker edge width (was 1) - REDUCED
data_trial_mew = 0.375  # Trial marker edge width (was 0.75) - REDUCED
grid_alpha = 0.5
grid_lw = 0.175  # Grid line width (was 0.35) - REDUCED
# Axis and spine settings (50% reduction)
axis_linewidth = 0.5  # Axis spine width - REDUCED
tick_width = 0.5  # Tick mark width - REDUCED
tick_length = 2.5  # Tick mark length (was ~5) - REDUCED
''' 
# Half-width panel sizing
if TALL_MODE:
    figsize = (3, 7)
else:
    figsize = (7, 3)   # (14, 6) is OK for powerpoint, but not for sub-figure
# fontsize settings
fontsize_flylabel = 6
fontsize_zlabel = 6
fontsize_zval_ticks = 12
fontsize_yticks = 4
fontsize_title = 7
# plot linewidth & markersize settings
z_CI_lw = 1
z_center_ms = 2.5
z_scatter_sz = 5
data_mean_sz = 3
data_trial_sz = 1.5
data_trial_mew = 0.5
data_mean_mew = 0.5
grid_alpha = 0.5
grid_lw = 0.25
'''

# plot-size-independent settings
data_mean_mrkr  = 'd'  # 'd'/'D' diamond; 's' square; 'o' circle, 'X' x,
data_mean_c     = 'aliceblue'   # orig: 'green'  (didn't like lightgreen)
data_trial_mrkr = 'x'  # 'x'
data_trial_c    = 'aliceblue'       # orig: 'orange'

kwargs_data_mean  = dict(marker=data_mean_mrkr,  color=data_mean_c,  markersize=data_mean_sz,  markeredgecolor='k', zorder=20, markeredgewidth=data_mean_mew)
kwargs_data_trial = dict(marker=data_trial_mrkr, color=data_trial_c, markersize=data_trial_sz, markeredgecolor='k', zorder=19,  markeredgewidth=data_trial_mew)

fig1, ax1 = plt.subplots(figsize=figsize)
fig2, ax2 = plt.subplots(figsize=figsize)
fig3, ax3 = plt.subplots(figsize=figsize)
fig4, ax4 = plt.subplots(figsize=figsize)
fig5, ax5 = plt.subplots(figsize=figsize)
fig6, ax6 = plt.subplots(figsize=figsize)
fig7, ax7 = plt.subplots(figsize=figsize)
fig8, ax8 = plt.subplots(figsize=figsize)
fig9, ax9 = plt.subplots(figsize=figsize)
fig10, ax10 = plt.subplots(figsize=figsize)
fig11, ax11 = plt.subplots(figsize=figsize)
fig12, ax12 = plt.subplots(figsize=figsize)

for idx, fly_id_str in enumerate(available_fly_keys):
    print('Working on fly:', fly_id_str, '...')
    fly_id_int = int(fly_id_str.split('_')[-1])
    df_for_fly = df_summary_csv_genotype_day[df_summary_csv_genotype_day['fly_id'] == fly_id_int]

    # get chamber id and box id for this fly (e.g. group of 8 chambers constitutes a box)
    chamber_id = int(df_for_fly['chamber_id'].values[0])
    box_id = CHAMBER_IDX_TO_BOX[chamber_id]
    print('fly_id_int:', fly_id_int, 'chamber_id:', chamber_id, 'box_id:', box_id)\

    # Extract posterior samples for this fly.
    samples = draws[fly_id_str]  # shape: ndraw x nparam
    alpha = samples[:, 0]
    beta  = samples[:, 1]
    p0    = samples[:, 2]

    if COLOR_BY_P0:
        fly_color = scatter_cmap(norm(np.mean(p0)))
        #fly_color = scatter_cmap(norm(p0))
        #print(fly_id_str, 'p0', np.mean(p0))
    else:
        fly_color = 'cornflowerblue'

    # Compute various metrics for this fly:
    jumps = df_to_arr_jumps(df_for_fly, jump_col='jumpdata')[0, :]

    # p_transient: mean of first P_TRANSIENT_CUTOFF_INDEX trials from each habituation block
    p_transient_1 = np.mean(jumps[  0:  0+P_TRANSIENT_CUTOFF_INDEX])
    p_transient_2 = np.mean(jumps[200:200+P_TRANSIENT_CUTOFF_INDEX])
    p_transient_3 = np.mean(jumps[400:400+P_TRANSIENT_CUTOFF_INDEX])
    p_transient_4 = np.mean(jumps[600:600+P_TRANSIENT_CUTOFF_INDEX])
    p_transient_5 = np.mean(jumps[800:800+P_TRANSIENT_CUTOFF_INDEX])
    p_transient_per_trial = [p_transient_1, p_transient_2, p_transient_3, p_transient_4, p_transient_5]
    p_transient_mean = np.mean(p_transient_per_trial)
    transient_curves = likelihood_func_vec(np.arange(P_TRANSIENT_CUTOFF_INDEX), alpha, beta, p0, pulse_period=1)
    z_model_p_transient = np.mean(transient_curves, axis=1)

    hab_tail_1 = np.mean(jumps[ 50:200])
    hab_tail_2 = np.mean(jumps[250:400])
    hab_tail_3 = np.mean(jumps[450:600])
    hab_tail_4 = np.mean(jumps[650:800])
    hab_tail_5 = np.mean(jumps[850:1000])
    hab_tail_per_trial = [hab_tail_1, hab_tail_2, hab_tail_3, hab_tail_4, hab_tail_5]
    hab_tail_mean = np.mean(hab_tail_per_trial)
    z_model_habtail = compute_p_ss(alpha, beta, p0, T=1)

    sra_mean = np.mean(jumps[1000:1050])
    sra_curves = likelihood_func_vec(np.arange(50), alpha, beta, p0, pulse_period=5)
    z_model_sramean = np.mean(sra_curves, axis=1)

    z_model_hab_halftime_rel = compute_hab_time_half_rel(alpha, beta, T=1)
    z_model_hab_time95_rel = compute_hab_time_95_rel(alpha, beta, T=1)

    # simulate some data for the model to calc TTC and fisher exact
    likelihood = likelihood_func_vec(np.arange(200), alpha, beta, p0, pulse_period=1)
    simulated_data = np.random.binomial(1, likelihood)

    z_model_ttc5 = [get_TTC_canonical(''.join(str(x) for x in simulated_data[i, :]), 5)
                    for i in range(simulated_data.shape[0])]
    ttc_hab1 = get_TTC_canonical(''.join(str(x) for x in jumps[0:200]), 5)
    ttc_hab2 = get_TTC_canonical(''.join(str(x) for x in jumps[200:400]), 5)
    ttc_hab3 = get_TTC_canonical(''.join(str(x) for x in jumps[400:600]), 5)
    ttc_hab4 = get_TTC_canonical(''.join(str(x) for x in jumps[600:800]), 5)  # Note: updated index if needed.
    ttc_hab5 = get_TTC_canonical(''.join(str(x) for x in jumps[800:1000]), 5)
    ttc_per_trial = [ttc_hab1, ttc_hab2, ttc_hab3, ttc_hab4, ttc_hab5]
    ttc_hab_mean = np.mean(ttc_per_trial)

    if 'hab_fisherexact' not in metrics_to_skip:
        #z_model_fisherexact = [habituation_time_fisherexact(simulated_data[i, :])
        #                       for i in range(simulated_data.shape[0])]
        z_model_fisherexact = [habituation_time_fisherexact(simulated_data[i, :])
                               for i in range(20)]  # we sub-sample because its slow
        fisherexact_hab1 = habituation_time_fisherexact(jumps[0:200])
        fisherexact_hab2 = habituation_time_fisherexact(jumps[200:400])
        fisherexact_hab3 = habituation_time_fisherexact(jumps[400:600])
        fisherexact_hab4 = habituation_time_fisherexact(jumps[600:800])
        fisherexact_hab5 = habituation_time_fisherexact(jumps[800:1000])
        fisherexact_per_trial = [fisherexact_hab1, fisherexact_hab2, fisherexact_hab3, fisherexact_hab4, fisherexact_hab5]
        fisherexact_hab_mean = np.mean(fisherexact_per_trial)

    else:
        z_model_fisherexact = None

    # absolute and relative magnitudes (model and empirical estimates from data)
    z_model_hab_magnitude_abs = compute_hab_magnitude_abs(alpha, beta, p0, T=1)
    z_model_hab_magnitude_rel = compute_hab_magnitude_rel(alpha, beta, T=1)
    # compute proxies for both quantities (which do not need to respect theoretical bounds of 0<x<1 etc)
    # - mag. abs: do p_0 - p_ss         ~ empirically, compute - p_<SRA> - p_ss_tail_approx
    # - mag. rel: do (p_0 - p_ss) / p_0 ~ empirically, compute - 1 - p_ss_tail_approx / p_<SRA>
    approx_mag_abs_hab1 = sra_mean - hab_tail_1
    approx_mag_abs_hab2 = sra_mean - hab_tail_2
    approx_mag_abs_hab3 = sra_mean - hab_tail_3
    approx_mag_abs_hab4 = sra_mean - hab_tail_4
    approx_mag_abs_hab5 = sra_mean - hab_tail_5
    approx_mag_abs_per_trial = [approx_mag_abs_hab1, approx_mag_abs_hab2, approx_mag_abs_hab3, approx_mag_abs_hab4, approx_mag_abs_hab5]
    approx_mag_abs_hab_mean = np.mean(approx_mag_abs_per_trial)

    approx_mag_rel_hab1 = (sra_mean - hab_tail_1) / sra_mean
    approx_mag_rel_hab2 = (sra_mean - hab_tail_2) / sra_mean
    approx_mag_rel_hab3 = (sra_mean - hab_tail_3) / sra_mean
    approx_mag_rel_hab4 = (sra_mean - hab_tail_4) / sra_mean
    approx_mag_rel_hab5 = (sra_mean - hab_tail_5) / sra_mean
    approx_mag_rel_per_trial = [approx_mag_rel_hab1, approx_mag_rel_hab2, approx_mag_rel_hab3, approx_mag_rel_hab4,
                                approx_mag_rel_hab5]
    approx_mag_rel_hab_mean = np.mean(approx_mag_rel_per_trial)

    z_model_alpha = alpha
    z_model_beta = beta
    z_model_p0 = p0

    # Fill in data for CSV output
    data_for_out_csv['p_transient']['data'][idx, :] = z_model_p_transient
    data_for_out_csv['hab_tail']['data'][idx, :] = z_model_habtail
    data_for_out_csv['sra_mean']['data'][idx, :] = z_model_sramean
    data_for_out_csv['hab_halftime_rel']['data'][idx, :] = z_model_hab_halftime_rel
    data_for_out_csv['hab_saturation_time_rel']['data'][idx, :] = z_model_hab_time95_rel
    data_for_out_csv['hab_ttc']['data'][idx, :] = z_model_ttc5
    if 'hab_fisherexact' not in metrics_to_skip:
        data_for_out_csv['hab_fisherexact']['data'][idx, :len(z_model_fisherexact)] = z_model_fisherexact
    data_for_out_csv['hab_magnitude_rel']['data'][idx, :] = z_model_hab_magnitude_rel
    data_for_out_csv['hab_magnitude_abs']['data'][idx, :] = z_model_hab_magnitude_abs
    data_for_out_csv['alpha']['data'][idx, :] = z_model_alpha
    data_for_out_csv['beta']['data'][idx, :] = z_model_beta
    data_for_out_csv['p0']['data'][idx, :] = z_model_p0

    # For each metric, determine the plotting properties.
    print('\nBlock: Overlay model and data scatter on each figure...')
    print('=' * 30)
    for ax, metric_name, model_data in [
            (ax12, 'p_transient', z_model_p_transient),
            (ax1, 'hab_tail', z_model_habtail),
            (ax2, 'sra_mean', z_model_sramean),
            (ax3, 'hab_halftime_rel', z_model_hab_halftime_rel),
            (ax4, 'hab_saturation_time_rel', z_model_hab_time95_rel),
            (ax5, 'hab_ttc', z_model_ttc5),
            (ax6, 'hab_fisherexact', z_model_fisherexact),
            (ax7, 'hab_magnitude_rel', z_model_hab_magnitude_rel),
            (ax8, 'hab_magnitude_abs', z_model_hab_magnitude_abs),
            (ax9, 'alpha', z_model_alpha),
            (ax10, 'beta', z_model_beta),
            (ax11, 'p0', z_model_p0)]:

        if metric_name in metrics_to_skip:
            continue

        if SORT_FIGURES:
            pos = sorted_positions[metric_name][fly_id_str]
        else:
            pos = idx + 1

        lower = np.percentile(model_data, 2.5)
        upper = np.percentile(model_data, 97.5)
        central_val = central_value(model_data)
        if TALL_MODE:
            # In tall mode, swap: x becomes model value, y becomes position.
            y_jitter = np.random.normal(loc=pos, scale=0.05, size=len(model_data))
            ax.scatter(model_data, y_jitter, alpha=0.3, color=fly_color, s=z_scatter_sz,
                       rasterized=RASTER_SCATTER, zorder=4)
            ax.plot([lower, upper], [pos, pos], color='k', lw=z_CI_lw, zorder=5)
            ax.plot(central_val, pos, marker='o', color='k', markersize=z_center_ms, zorder=5)
            if fly_id_int in HIGHLIGHT_FLY_IDS.get(genotype, []):
                ax.plot(central_val, pos, marker='D', color='gold', markersize=z_center_ms*1.2,
                        markeredgecolor='k', zorder=16)
        else:
            x_jitter = np.random.normal(loc=pos, scale=0.05, size=len(model_data))
            ax.scatter(x_jitter, model_data, alpha=0.3, color=fly_color, s=z_scatter_sz,
                       rasterized=RASTER_SCATTER, zorder=4)
            ax.plot([pos, pos], [lower, upper], color='k', lw=z_CI_lw, zorder=5)
            ax.plot(pos, central_val, marker='o', color='k', markersize=z_center_ms, zorder=5)
            if fly_id_int in HIGHLIGHT_FLY_IDS.get(genotype, []):
                ax.plot(pos, central_val, marker='D', color='gold', markersize=z_center_ms*1.2,
                        markeredgecolor='k', zorder=16)

        # Fill in data for chamber and box data dicts
        data_for_chambers[metric_name][chamber_id]['z_model_data_by_fly'].append(model_data)
        data_for_chambers[metric_name][chamber_id]['associated_fly_ids'].append(fly_id_int)
        data_for_boxes[metric_name][box_id]['z_model_data_by_fly'].append(model_data)
        data_for_boxes[metric_name][box_id]['associated_fly_ids'].append(fly_id_int)
        # Fill in fly color data
        data_for_chambers[metric_name][chamber_id]['associated_colors'].append(fly_color)
        data_for_boxes[metric_name][box_id]['associated_colors'].append(fly_color)

        # Data overlays for specific axes:
        # - only overlay experimental data points for subset of plots
        if ax == ax12:
            if TALL_MODE:
                ax.plot(p_transient_1, pos, **kwargs_data_trial)
                ax.plot(p_transient_2, pos, **kwargs_data_trial)
                ax.plot(p_transient_3, pos, **kwargs_data_trial)
                ax.plot(p_transient_4, pos, **kwargs_data_trial)
                ax.plot(p_transient_5, pos, **kwargs_data_trial)
                ax.plot(p_transient_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, p_transient_1, **kwargs_data_trial)
                ax.plot(pos, p_transient_2, **kwargs_data_trial)
                ax.plot(pos, p_transient_3, **kwargs_data_trial)
                ax.plot(pos, p_transient_4, **kwargs_data_trial)
                ax.plot(pos, p_transient_5, **kwargs_data_trial)
                ax.plot(pos, p_transient_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['p_transient'][chamber_id]['expt_data_per_trial'].append(p_transient_per_trial)
            data_for_chambers['p_transient'][chamber_id]['expt_data_mean'].append(p_transient_mean)
            data_for_boxes['p_transient'][box_id]['expt_data_per_trial'].append(p_transient_per_trial)
            data_for_boxes['p_transient'][box_id]['expt_data_mean'].append(p_transient_mean)
        elif ax == ax1:
            if TALL_MODE:
                ax.plot(hab_tail_1, pos, **kwargs_data_trial)
                ax.plot(hab_tail_2, pos, **kwargs_data_trial)
                ax.plot(hab_tail_3, pos, **kwargs_data_trial)
                ax.plot(hab_tail_4, pos, **kwargs_data_trial)
                ax.plot(hab_tail_5, pos, **kwargs_data_trial)
                ax.plot(hab_tail_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, hab_tail_1, **kwargs_data_trial)
                ax.plot(pos, hab_tail_2, **kwargs_data_trial)
                ax.plot(pos, hab_tail_3, **kwargs_data_trial)
                ax.plot(pos, hab_tail_4, **kwargs_data_trial)
                ax.plot(pos, hab_tail_5, **kwargs_data_trial)
                ax.plot(pos, hab_tail_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['hab_tail'][chamber_id]['expt_data_per_trial'].append(hab_tail_per_trial)
            data_for_chambers['hab_tail'][chamber_id]['expt_data_mean'].append(hab_tail_mean)
            data_for_boxes['hab_tail'][box_id]['expt_data_per_trial'].append(hab_tail_per_trial)
            data_for_boxes['hab_tail'][box_id]['expt_data_mean'].append(hab_tail_mean)
        elif ax == ax2:
            if TALL_MODE:
                ax.plot(sra_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, sra_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['sra_mean'][chamber_id]['expt_data_mean'].append(sra_mean)
            data_for_boxes['sra_mean'][box_id]['expt_data_mean'].append(sra_mean)
        elif ax == ax5:
            if TALL_MODE:
                ax.plot(ttc_hab1, pos, **kwargs_data_trial)
                ax.plot(ttc_hab2, pos, **kwargs_data_trial)
                ax.plot(ttc_hab3, pos, **kwargs_data_trial)
                ax.plot(ttc_hab4, pos, **kwargs_data_trial)
                ax.plot(ttc_hab5, pos, **kwargs_data_trial)
                ax.plot(ttc_hab_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, ttc_hab1, **kwargs_data_trial)
                ax.plot(pos, ttc_hab2, **kwargs_data_trial)
                ax.plot(pos, ttc_hab3, **kwargs_data_trial)
                ax.plot(pos, ttc_hab4, **kwargs_data_trial)
                ax.plot(pos, ttc_hab5, **kwargs_data_trial)
                ax.plot(pos, ttc_hab_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['hab_ttc'][chamber_id]['expt_data_per_trial'].append(ttc_per_trial)
            data_for_chambers['hab_ttc'][chamber_id]['expt_data_mean'].append(ttc_hab_mean)
            data_for_boxes['hab_ttc'][box_id]['expt_data_per_trial'].append(ttc_per_trial)
            data_for_boxes['hab_ttc'][box_id]['expt_data_mean'].append(ttc_hab_mean)
        elif ax == ax6:
            if TALL_MODE:
                ax.plot(fisherexact_hab1, pos, **kwargs_data_trial)
                ax.plot(fisherexact_hab2, pos, **kwargs_data_trial)
                ax.plot(fisherexact_hab3, pos, **kwargs_data_trial)
                ax.plot(fisherexact_hab4, pos, **kwargs_data_trial)
                ax.plot(fisherexact_hab5, pos, **kwargs_data_trial)
                ax.plot(fisherexact_hab_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, fisherexact_hab1, **kwargs_data_trial)
                ax.plot(pos, fisherexact_hab2, **kwargs_data_trial)
                ax.plot(pos, fisherexact_hab3, **kwargs_data_trial)
                ax.plot(pos, fisherexact_hab4, **kwargs_data_trial)
                ax.plot(pos, fisherexact_hab5, **kwargs_data_trial)
                ax.plot(pos, fisherexact_hab_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['hab_fisherexact'][chamber_id]['expt_data_per_trial'].append(fisherexact_per_trial)
            data_for_chambers['hab_fisherexact'][chamber_id]['expt_data_mean'].append(fisherexact_hab_mean)
            data_for_boxes['hab_fisherexact'][box_id]['expt_data_per_trial'].append(fisherexact_per_trial)
            data_for_boxes['hab_fisherexact'][box_id]['expt_data_mean'].append(fisherexact_hab_mean)
        elif ax == ax7:
            if TALL_MODE:
                ax.plot(approx_mag_rel_hab1, pos, **kwargs_data_trial)
                ax.plot(approx_mag_rel_hab2, pos, **kwargs_data_trial)
                ax.plot(approx_mag_rel_hab3, pos, **kwargs_data_trial)
                ax.plot(approx_mag_rel_hab4, pos, **kwargs_data_trial)
                ax.plot(approx_mag_rel_hab5, pos, **kwargs_data_trial)
                ax.plot(approx_mag_rel_hab_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, approx_mag_rel_hab1, **kwargs_data_trial)
                ax.plot(pos, approx_mag_rel_hab2, **kwargs_data_trial)
                ax.plot(pos, approx_mag_rel_hab3, **kwargs_data_trial)
                ax.plot(pos, approx_mag_rel_hab4, **kwargs_data_trial)
                ax.plot(pos, approx_mag_rel_hab5, **kwargs_data_trial)
                ax.plot(pos, approx_mag_rel_hab_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['hab_magnitude_rel'][chamber_id]['expt_data_per_trial'].append(approx_mag_rel_per_trial)
            data_for_chambers['hab_magnitude_rel'][chamber_id]['expt_data_mean'].append(approx_mag_rel_hab_mean)
            data_for_boxes['hab_magnitude_rel'][box_id]['expt_data_per_trial'].append(approx_mag_rel_per_trial)
            data_for_boxes['hab_magnitude_rel'][box_id]['expt_data_mean'].append(approx_mag_rel_hab_mean)
        elif ax == ax8:
            if TALL_MODE:
                ax.plot(approx_mag_abs_hab1, pos, **kwargs_data_trial)
                ax.plot(approx_mag_abs_hab2, pos, **kwargs_data_trial)
                ax.plot(approx_mag_abs_hab3, pos, **kwargs_data_trial)
                ax.plot(approx_mag_abs_hab4, pos, **kwargs_data_trial)
                ax.plot(approx_mag_abs_hab5, pos, **kwargs_data_trial)
                ax.plot(approx_mag_abs_hab_mean, pos, **kwargs_data_mean)
            else:
                ax.plot(pos, approx_mag_abs_hab1, **kwargs_data_trial)
                ax.plot(pos, approx_mag_abs_hab2, **kwargs_data_trial)
                ax.plot(pos, approx_mag_abs_hab3, **kwargs_data_trial)
                ax.plot(pos, approx_mag_abs_hab4, **kwargs_data_trial)
                ax.plot(pos, approx_mag_abs_hab5, **kwargs_data_trial)
                ax.plot(pos, approx_mag_abs_hab_mean, **kwargs_data_mean)
            # Fill in data for chamber and box data dicts
            data_for_chambers['hab_magnitude_abs'][chamber_id]['expt_data_per_trial'].append(approx_mag_abs_per_trial)
            data_for_chambers['hab_magnitude_abs'][chamber_id]['expt_data_mean'].append(approx_mag_abs_hab_mean)
            data_for_boxes['hab_magnitude_abs'][box_id]['expt_data_per_trial'].append(approx_mag_abs_per_trial)
            data_for_boxes['hab_magnitude_abs'][box_id]['expt_data_mean'].append(approx_mag_abs_hab_mean)
    if TALL_MODE:
        ax3.set_xlim(-2, 50)
        ax4.set_xlim(-4, 150)
    else:
        ax3.set_ylim(-2, 50)
        ax4.set_ylim(-4, 150)

##############################################################################
# Customize each plot (FIGURES 1–7): Set tick labels based on the independent ordering.
##############################################################################
print('\nBlock: Figure customization...')
print('='*30)
for ax, plot_var in [
        (ax12, 'p_transient'),
        (ax1, 'hab_tail'),
        (ax2, 'sra_mean'),
        (ax3, 'hab_halftime_rel'),
        (ax4, 'hab_saturation_time_rel'),
        (ax5, 'hab_ttc'),
        (ax6, 'hab_fisherexact'),
        (ax7, 'hab_magnitude_rel'),
        (ax8, 'hab_magnitude_abs'),
        (ax9, 'alpha'),
        (ax10, 'beta'),
        (ax11, 'p0')]:

    if plot_var in metrics_to_skip:
        continue

    if TALL_MODE:
        ax.set_ylabel("Fly ID", fontsize=fontsize_flylabel)
        ax.set_xlabel("Posterior samples for z=%s" % plot_var, fontsize=fontsize_zlabel)
        ax.set_yticks(np.arange(1, num_flies + 1))
        if SORT_FIGURES:
            sorted_keys_metric                = sorted(available_fly_keys, key=lambda fid: sorted_positions[plot_var][fid])
            sorted_boxes    = [fly_to_box[fid] for fid in sorted_keys_metric]
            sorted_chambers = [fly_to_chamber[fid] for fid in sorted_keys_metric]

            tick_labels = [fid.split('_')[-1] for fid in sorted_keys_metric]
            twintick_labels = ['%d: %d' % (sorted_boxes[idx], sorted_chambers[idx])
                               for idx in range(len(sorted_chambers))]
        else:
            tick_labels = default_xticklabels  # no sorting
            twintick_labels = ['%d: %d' % (boxes_for_avail_fly_ids[idx], chambers_for_avail_fly_ids[idx])
                               for idx in range(len(chambers_for_avail_fly_ids))]

        ax.set_yticklabels(tick_labels, fontsize=fontsize_yticks)
        # add twin axis
        if SHOW_CHAMBER_BOX_ID:
            ax2 = ax.twinx()
            ax2.set_yticks(np.arange(1, num_flies + 1))
            ax2.set_yticklabels(twintick_labels, fontsize=fontsize_yticks)
            #ax2.tick_params(axis='y', labelsize=fontsize_zval_ticks)

        ax.tick_params(axis='x', labelsize=fontsize_zval_ticks, width=tick_width, length=tick_length)
        ax.tick_params(axis='y', width=tick_width, length=tick_length)
        ax.set_ylim(0.5, num_flies + 0.5)
        ax.grid(alpha=grid_alpha, linewidth=grid_lw)
    else:
        ax.set_xlabel("Fly ID", fontsize=fontsize_flylabel)
        ax.set_ylabel("Posterior samples for z=%s" % plot_var, fontsize=fontsize_zlabel)
        ax.set_xticks(np.arange(1, num_flies + 1))
        if SORT_FIGURES:
            sorted_keys_metric = sorted(available_fly_keys, key=lambda fid: sorted_positions[plot_var][fid])
            tick_labels = [fid.split('_')[-1] for fid in sorted_keys_metric]
            twintick_labels = ['%d: %d' % (boxes_for_avail_fly_ids[fid], chambers_for_avail_fly_ids[fid])
                               for fid in sorted_keys_metric]
        else:
            tick_labels = default_xticklabels  # no sorting
            twintick_labels = ['%d: %d' % (boxes_for_avail_fly_ids[idx], chambers_for_avail_fly_ids[idx])
                               for idx in range(len(chambers_for_avail_fly_ids))]
        ax.set_xticklabels(tick_labels, rotation=70, fontsize=fontsize_yticks)
        # add twin axis
        if SHOW_CHAMBER_BOX_ID:
            ax2 = ax.twiny()
            ax2.set_xticks(np.arange(1, num_flies + 1))
            ax2.set_xticklabels(twintick_labels, rotation=70, fontsize=fontsize_yticks)
            # ax2.tick_params(axis='y', labelsize=fontsize_zval_ticks)

        ax.tick_params(axis='y', labelsize=fontsize_zval_ticks, width=tick_width, length=tick_length)
        ax.tick_params(axis='x', width=tick_width, length=tick_length)
        ax.set_xlim(0.5, num_flies + 0.5)
        ax.grid(alpha=grid_alpha, linewidth=grid_lw)
    if remove_spines:
        ax.spines[['right', 'top']].set_visible(False)
    # Set spine linewidths
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

#fig1.tight_layout()  # can look better, but leave off for now
#fig2.tight_layout()
print('\nBlock: Saving matplotlib figures and Data CSVs')
print('='*30)
for fig, metric in [
    (fig12, 'p_transient'),
    (fig1, 'hab_tail'),
    (fig2, 'sra_mean'),
    (fig3, 'hab_halftime_rel'),
    (fig4, 'hab_saturation_time_rel'),
    (fig5, 'hab_ttc'),
    (fig6, 'hab_fisherexact'),
    (fig7, 'hab_magnitude_rel'),
    (fig8, 'hab_magnitude_abs'),
    (fig9, 'alpha'),
    (fig10, 'beta'),
    (fig11, 'p0')]:

    # save data CSV
    if metric in metrics_to_skip:
        continue
    else:
        # 1) Save figure
        fig.savefig(DIR_OUTPUT + os.sep + "metric_by_fly_id_%s_%d_%s.png" % (genotype, age, metric))
        fig.savefig(DIR_OUTPUT + os.sep + "metric_by_fly_id_%s_%d_%s.svg" % (genotype, age, metric), dpi=600)
        fig.savefig(DIR_OUTPUT + os.sep + "metric_by_fly_id_%s_%d_%s.pdf" % (genotype, age, metric), dpi=600)

        # 2) Output data csv
        # Retrieve the stored data and fly IDs.
        data_array = data_for_out_csv[metric]['data']  # shape: (num_flies, num_draws)
        fly_ids = data_for_out_csv[metric]['fly_id']  # list of fly IDs (length num_flies)
        # Transpose the array so that rows correspond to draws and columns to flies.
        # Then create a DataFrame with fly_ids as column headers.
        df_metric = pd.DataFrame(data_array.T, columns=fly_ids)
        # Optionally add an index name (e.g. "draw")
        df_metric.index.name = 'draw'
        # Save the DataFrame to a CSV file.
        df_metric.to_csv(DIR_OUTPUT + os.sep + "metric_by_fly_id_%s_%d_%s.csv" % (genotype, age, metric))

plt.show()

##############################################################################
# Aggregation block: plot same data but now clumped by both chamber and box (separately)

# For each metric, we will create two additional dotplot figures
#  1) chamber id (1 to 32) -- up to  4 flies per chamber are used - overlay the mean for each fly
#  2) box id  (0, 1, 2, 3) -- up to 32 flies per box     are used - overlay the mean for each fly

# Notes:
# - here we ignore tall mode (all plots horizontal)
# - here we do not sort by ranks (since we expect no ordering)
##############################################################################
figsize_clump_to_chamber = (18, 6)
figsize_clump_to_box     = (18, 6)

print('\nBlock: Figures grouped by chamber ID and Box ID')
print('='*30)
for idx, (metric,) in enumerate([
    ('hab_tail',),
    ('sra_mean',),
    ('hab_halftime_rel',),
    ('hab_saturation_time_rel',),
    ('hab_ttc',),
    ('hab_fisherexact',),
    ('hab_magnitude_rel',),
    ('hab_magnitude_abs',),
    ('alpha',),
    ('beta',),
    ('p0',)]):

    if metric in metrics_to_skip:
        continue
    else:
        # Create new figures for chamber and box aggregations.
        fig_chamber, ax_chamber = plt.subplots(figsize=figsize_clump_to_chamber)
        fig_chamber_boxplot, ax_chamber_boxplot = plt.subplots(figsize=figsize_clump_to_chamber)
        fig_chamber_violin, ax_chamber_violin = plt.subplots(figsize=figsize_clump_to_chamber)
        fig_box, ax_box = plt.subplots(figsize=figsize_clump_to_box)

        # get unique chamber ids
        chamber_indices = list(range(1,33))
        for c_idx, chamber in enumerate(chamber_indices):
            # Get the list of model data arrays for flies in this chamber.
            chamber_z_model_data_by_fly = data_for_chambers[metric][chamber]['z_model_data_by_fly']
            associated_fly_ids = data_for_chambers[metric][chamber]['associated_fly_ids']
            nflies_in_group = len(associated_fly_ids)

            if nflies_in_group == 0:
                continue  # No flies in this chamber

            # Stack to form an array of shape (num_flies_in_group, num_draws)
            fly_stack = np.vstack(chamber_z_model_data_by_fly)
            ### In this object, just concatenate all the arrays into one
            ###fly_stack = np.concatenate(group_list, axis=0)

            # Aggregate across flies (here by mean)
            z_mean_per_fly_in_group = np.mean(fly_stack, axis=1)  # size: (num_flies_in_group,)

            '''
            ax.scatter(x_jitter, model_data, alpha=0.3, color=fly_color, s=z_scatter_sz,
                       rasterized=RASTER_SCATTER, zorder=4)'''
            print('Working on fly:', fly_id_str, '...')
            fly_id_int = int(fly_id_str.split('_')[-1])
            df_for_fly = df_summary_csv_genotype_day[df_summary_csv_genotype_day['fly_id'] == fly_id_int]

            # get chamber id and box id for this fly (e.g. group of 8 chambers constitutes a box)
            chamber_id = int(df_for_fly['chamber_id'].values[0])
            box_id = CHAMBER_IDX_TO_BOX[chamber_id]
            print('fly_id_int:', fly_id_int, 'chamber_id:', chamber_id, 'box_id:', box_id)

            # Determine the x position for this group
            x_pos = chamber

            # Plot each draw as a scatter point, adding some jitter
            draws_idx = np.arange(1, num_draws + 1)
            jitter = np.random.normal(loc=0, scale=0.15, size=draws_idx.shape)

            # for ax chamber simple, we will instead plot a boxplot with compression of fly stack
            # reshape the fly stack into a single dim array and do boxplot
            ax_chamber_boxplot.boxplot(
                fly_stack.flatten(), positions=[x_pos],
                widths=0.5,
                notch=False,
                patch_artist=True,
                showmeans=True,
                showfliers=False
            )
            # make violin plot
            vplots = ax_chamber_violin.violinplot(fly_stack.flatten(),
                                                  positions=[x_pos],
                                                  showmeans=False,
                                                  showmedians=True)
            # change colors in violin plot
            for pc in vplots['bodies']:
                pc.set_facecolor('tab:blue')
                pc.set_edgecolor('black')
            vplots['cmedians'].set_colors('black')

            # Plot: for each fly, plot the scatter points for each draw, colored by that flies mean val of p0
            for k, assoc_fly_id in enumerate(associated_fly_ids):
                assoc_color = data_for_chambers[metric][chamber]['associated_colors'][k]
                fly_specific_data = fly_stack[k, :]
                x_pos = chamber + (k - 1.5) * 0.1  # shift it a bit for each fly; assume 4 flies
                print('check chamber, x_pos, k, assoc_fly_id:', chamber, x_pos, k, assoc_fly_id)
                x_jitter = np.random.normal(loc=0, scale=0.05, size=len(fly_specific_data))
                x_pos_jittered = x_pos + x_jitter

                # Plot the individual draws for each fly in this group
                ax_chamber.scatter(
                    x_pos_jittered,
                    fly_specific_data,
                    alpha=0.3,
                    color=assoc_color,
                    rasterized=RASTER_SCATTER,
                    s=z_scatter_sz*0.8,
                    edgecolor='none')  # should be coloring the scatter by the per-fly mean

                # Per-fly in chamber: plot central value
                # Per-fly in chamber: plot CI
                central_val = central_value(fly_specific_data)
                lower = np.percentile(fly_specific_data, 2.5)
                upper = np.percentile(fly_specific_data, 97.5)
                ax_chamber.plot(x_pos, central_val, marker='o', color='k', markersize=z_center_ms, zorder=5)
                ax_chamber_boxplot.plot(x_pos, central_val, marker='o', color='k', markersize=z_center_ms, zorder=5)
                ax_chamber_violin.plot(x_pos, central_val, marker='o', color='k', markersize=z_center_ms, zorder=5)
                ax_chamber.plot([x_pos, x_pos], [lower, upper], color='k', lw=z_CI_lw, zorder=5)

                print(len(data_for_chambers[metric][chamber]['expt_data_per_trial']))

                # Data overlays for specific axes:
                # - only overlay experimental data points for subset of plots
                if metric == 'hab_tail':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][0], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][1], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][2], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][3], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][4], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)
                elif metric == 'sra_mean':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)
                elif metric == 'hab_ttc':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][0], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][1], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][2], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][3], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][4], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)
                elif metric == 'hab_fisherexact':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][0], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][1], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][2], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][3], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][4], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)
                elif metric == 'hab_magnitude_rel':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][0], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][1], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][2], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][3], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][4], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)
                elif metric == 'hab_magnitude_abs':
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][0], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][1], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][2], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][3], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_per_trial'][k][4], **kwargs_data_trial)
                    ax_chamber.plot(x_pos, data_for_chambers[metric][chamber]['expt_data_mean'][k], **kwargs_data_mean)

        if metric == 'hab_halftime_rel':
            ax_chamber.set_ylim(-2, 50)
            ax_chamber_boxplot.set_ylim(-2, 50)
            ax_chamber_violin.set_ylim(-2, 50)
        elif metric == 'hab_saturation_time_rel':
            ax_chamber.set_ylim(-4, 150)
            ax_chamber_boxplot.set_ylim(-4, 150)
            ax_chamber_violin.set_ylim(-4, 150)

        # Customize the chamber figure axes.
        for ax in [ax_chamber, ax_chamber_boxplot, ax_chamber_violin]:

            ax.set_ylabel("Draws (z = %s)" % metric, fontsize=12)
            ax.set_xlabel("Chamber ID", fontsize=12)
            ax.set_title(f"z={metric} Aggregated by Chamber for {genotype}-{age}", fontsize=14)
            ax.set_xticks(chamber_indices)
            ax.grid(alpha=grid_alpha, linewidth=grid_lw * 2)

            # add vlines for box boundaries
            for idx in range(0, 5):
                vv = 8*idx + 0.5
                ax.axvline(vv, color='gray', linestyle='--', lw=1.75)



        for ff, flabel in [(fig_chamber, ''),
                           (fig_chamber_boxplot, '_boxplot'),
                           (fig_chamber_violin, '_violin')]:
            ##ff.tight_layout()
            ff.savefig(DIR_OUTPUT + os.sep + f"aggregated_{metric}_by_chamber_{genotype}_{age}{flabel}.png", dpi=600)
            ff.savefig(DIR_OUTPUT + os.sep + f"aggregated_{metric}_by_chamber_{genotype}_{age}{flabel}.svg", dpi=600)
            ff.savefig(DIR_OUTPUT + os.sep + f"aggregated_{metric}_by_chamber_{genotype}_{age}{flabel}.pdf", dpi=600)

        plt.show()


if COLOR_BY_P0:
    # Create a new figure for the colorbar
    fig_colorbar = plt.figure(figsize=(1, 6))
    # Create an axes for the colorbar. The list defines [left, bottom, width, height] in figure fraction.
    ax_colorbar = fig_colorbar.add_axes([0.3, 0.05, 0.4, 0.9])
    # Create the ScalarMappable (sm) using your existing cmap and norm
    sm = cm.ScalarMappable(cmap=scatter_cmap, norm=norm)
    sm.set_array([])  # Only needed for the colorbar; no data required.
    cbar = fig_colorbar.colorbar(sm, cax=ax_colorbar, orientation='vertical')
    cbar.set_label(r"mean $p_0$", fontsize=12)
    plt.savefig(DIR_OUTPUT + os.sep + "colorbar_p0_%s_%s.svg" % (genotype, age), dpi=300)
    plt.show()

    # also make a horizontal version of cbar
    fig_colorbar_h = plt.figure(figsize=(6, 1))
    ax_colorbar_h = fig_colorbar_h.add_axes([0.05, 0.3, 0.9, 0.4])
    sm = cm.ScalarMappable(cmap=scatter_cmap, norm=norm)
    sm.set_array([])  # Only needed for the colorbar; no data required.
    cbar = fig_colorbar_h.colorbar(sm, cax=ax_colorbar_h, orientation='horizontal')
    cbar.set_label(r"mean $p_0$", fontsize=12)
    plt.savefig(DIR_OUTPUT + os.sep + "colorbar_p0_horizontal_%s_%s.svg" % (genotype, age), dpi=300)

# --- COMBINED FIGURE WITH TWIN AXES ---
# Use twinx() for normal mode and twiny() for tall mode so that the y-axis (fly ID) is shared.
fig_combined, ax1 = plt.subplots(figsize=(figsize))
if TALL_MODE:
    ax2 = ax1.twiny()  # share the y-axis
else:
    ax2 = ax1.twinx()

if SORT_FIGURES:
    fly_stats = []
    for fly_id_str in available_fly_keys[:num_flies]:
        samples = draws[fly_id_str]
        alpha = samples[:, 0]
        beta  = samples[:, 1]
        p0    = samples[:, 2]
        mag_abs = compute_hab_magnitude_abs(alpha, beta, p0, T=1)
        mean_or_median = central_value(mag_abs)
        fly_stats.append((fly_id_str, mean_or_median))
    fly_stats.sort(key=lambda x: x[1])
    sorted_fly_keys = [fs[0] for fs in fly_stats]
    xticklabels_comb = [fly_id_str.split('_')[-1] for fly_id_str in sorted_fly_keys]
else:
    sorted_fly_keys = available_fly_keys[:num_flies]
    xticklabels_comb = default_xticklabels

offset = 0.15

for idx, fly_id_str in enumerate(sorted_fly_keys):
    fly_num = int(fly_id_str.split('_')[-1])
    samples = draws[fly_id_str]
    alpha = samples[:, 0]
    beta = samples[:, 1]
    p0 = samples[:, 2]
    sat_time = compute_hab_time_95_rel(alpha, beta, T=1)
    mag_abs = compute_hab_magnitude_abs(alpha, beta, p0, T=1)
    x_left = idx + 1 - offset
    x_right = idx + 1 + offset
    if TALL_MODE:
        jitter_mag = np.random.normal(loc=x_left, scale=0.02, size=len(mag_abs))
        ax1.scatter(mag_abs, np.random.normal(loc=x_left, scale=0.02, size=len(mag_abs)),
                    color='cornflowerblue', alpha=0.3, s=z_scatter_sz,
                    rasterized=RASTER_SCATTER,
                    label='Magnitude Abs' if idx == 0 else "")
        mean_or_median_mag = central_value(mag_abs)
        lower_mag = np.percentile(mag_abs, 2.5)
        upper_mag = np.percentile(mag_abs, 97.5)
        ax1.plot([lower_mag, upper_mag], [x_left, x_left], color='black', lw=z_CI_lw, zorder=5)
        ax1.plot(mean_or_median_mag, x_left, marker='o', color='black', markersize=z_center_ms, zorder=6)
        if fly_num in HIGHLIGHT_FLY_IDS.get(genotype, []):
            ax1.plot(mean_or_median_mag, x_left, marker='D', color='gold', markersize=z_center_ms*1.2, zorder=16)

        jitter_sat = np.random.normal(loc=x_right, scale=0.02, size=len(sat_time))
        ax2.scatter(sat_time, np.random.normal(loc=x_right, scale=0.02, size=len(sat_time)),
                    color='salmon', alpha=0.3, s=z_scatter_sz,
                    rasterized=RASTER_SCATTER,
                    label='Saturation Time' if idx == 0 else "")
        mean_or_median_sat = central_value(sat_time)
        lower_sat = np.percentile(sat_time, 2.5)
        upper_sat = np.percentile(sat_time, 97.5)
        ax2.plot([lower_sat, upper_sat], [x_right, x_right], color='black', lw=z_CI_lw, zorder=5)
        ax2.plot(mean_or_median_sat, x_right, marker='o', color='black', markersize=z_center_ms, zorder=5)
        if fly_num in HIGHLIGHT_FLY_IDS.get(genotype, []):
            ax2.plot(mean_or_median_sat, x_right, marker='D', color='gold', markersize=z_center_ms*1.2, zorder=16)
    else:
        jitter_mag = np.random.normal(loc=x_left, scale=0.02, size=len(mag_abs))
        ax1.scatter(jitter_mag, mag_abs, color='cornflowerblue', alpha=0.3, s=z_scatter_sz,
                    rasterized=RASTER_SCATTER,
                    label='Magnitude Abs' if idx == 0 else "")
        mean_or_median_mag = central_value(mag_abs)
        lower_mag = np.percentile(mag_abs, 2.5)
        upper_mag = np.percentile(mag_abs, 97.5)
        ax1.plot([x_left, x_left], [lower_mag, upper_mag], color='black', lw=z_CI_lw, zorder=5)
        ax1.plot(x_left, mean_or_median_mag, marker='o', color='black', markersize=z_center_ms, zorder=5)
        if fly_num in HIGHLIGHT_FLY_IDS.get(genotype, []):
            ax1.plot(x_left, mean_or_median_mag, marker='D', color='gold', markersize=z_center_ms*1.2, zorder=16)

        jitter_sat = np.random.normal(loc=x_right, scale=0.02, size=len(sat_time))
        ax2.scatter(jitter_sat, sat_time, color='salmon', alpha=0.3, s=z_scatter_sz,
                    rasterized=RASTER_SCATTER,
                    label='Saturation Time' if idx == 0 else "")
        mean_or_median_sat = central_value(sat_time)
        lower_sat = np.percentile(sat_time, 2.5)
        upper_sat = np.percentile(sat_time, 97.5)
        ax2.plot([x_right, x_right], [lower_sat, upper_sat], color='black', lw=z_CI_lw, zorder=5)
        ax2.plot(x_right, mean_or_median_sat, marker='o', color='black', markersize=z_center_ms, zorder=5)
        if fly_num in HIGHLIGHT_FLY_IDS.get(genotype, []):
            ax2.plot(x_right, mean_or_median_sat, marker='D', color='gold', markersize=z_center_ms*1.2, zorder=16)

if TALL_MODE:
    ax1.set_ylabel("Fly ID", fontsize=fontsize_flylabel, color='blue')
    ax1.set_xlabel("Magnitude Abs", fontsize=fontsize_zlabel, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yticks(np.arange(1, num_flies + 1))
    ax1.set_yticklabels(xticklabels_comb, rotation=0, fontsize=fontsize_yticks)
    ax1.set_ylim(0.5, num_flies + 0.5)
    ax2.set_xlabel("Saturation Time", fontsize=fontsize_zlabel, color='red')
    ax2.tick_params(axis='x', labelcolor='red', width=tick_width, length=tick_length)
    ax2.set_xlim(0, 200)
else:
    ax1.set_xlabel("Fly ID", fontsize=fontsize_flylabel, color='blue')
    ax1.set_ylabel("Magnitude Abs", fontsize=fontsize_zlabel, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', width=tick_width, length=tick_length)
    ax1.set_xticks(np.arange(1, num_flies + 1))
    ax1.set_xticklabels(xticklabels_comb, rotation=70, fontsize=fontsize_yticks)
    ax1.set_xlim(0.5, num_flies + 0.5)
    ax2.set_ylabel("Saturation Time", fontsize=fontsize_zlabel, color='red')
    ax2.tick_params(axis='y', labelcolor='red', width=tick_width, length=tick_length)
    ax2.set_ylim(0, 200)

ax1.set_title("Combined Plot: Magnitude Abs (blue) & Saturation Time (red)", fontsize=fontsize_title)
# Set spine linewidths for combined plot
for spine in ax1.spines.values():
    spine.set_linewidth(axis_linewidth)
if not TALL_MODE:
    ax1.set_xticks(np.arange(1, num_flies + 1))
    ax1.set_xticklabels(xticklabels_comb, rotation=70, fontsize=fontsize_yticks)
    ax1.set_xlim(0.5, num_flies + 0.5)
else:
    ax1.set_xlim(0, 1)

ax1.grid(alpha=grid_alpha, linewidth=grid_lw)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

fig_combined.tight_layout()
plt.show()
