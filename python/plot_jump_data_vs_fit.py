import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn

from fit_experimental_data import to_file, parse_fly_data
from functions_common import jump_prob
from data_tools import filter_df_by_filterdict, build_dataframe_from_data, build_mergetrials_dataframe, df_to_arr_jumps
from settings import DIR_DATA_EXPT, DIR_FITS, DIR_OUTPUT, heatmap_0, heatmap_1, OMIT_FLY_IDS


def plot_full_data_all_days(genotype='KK', show=True, verbose=False):
    assert genotype in ['GD', 'KK']

    df_full = build_dataframe_from_data(data_version='v3')
    df_full_mergetrials = build_mergetrials_dataframe(df_full, data_version='v3')
    df_to_use = filter_df_by_filterdict(df_full_mergetrials, dict(gene_bgr=[genotype])).copy()

    # Omit specified fly IDs
    # Note: OMIT_FLY_IDS uses global fly_id (1-128), but df has idx_fly (fly_id_64, 1-64) and var
    # Conversion: fly_id = idx_fly (if var=1) or idx_fly + 64 (if var=2)
    df_to_use['global_fly_id'] = df_to_use['idx_fly'] + (df_to_use['var'] - 1) * 64

    omit_ids = OMIT_FLY_IDS.get(genotype, [])
    if omit_ids:
        df_to_use = df_to_use[~df_to_use['global_fly_id'].isin(omit_ids)]

    # Only include flies that completed all 3 ages (to match behavior of plot_full_data with filtered CSVs)
    fly_age_counts = df_to_use.groupby('global_fly_id')['age'].nunique()
    flies_all_3_ages = fly_age_counts[fly_age_counts == 3].index
    df_to_use = df_to_use[df_to_use['global_fly_id'].isin(flies_all_3_ages)]

    df_to_use = df_to_use.drop(columns=['global_fly_id'])

    jumps = df_to_arr_jumps(df_to_use)
    M, n = jumps.shape
    aspect = n / M * 0.6

    if verbose:
        print('Data shape')
        print('\tfull:', jumps.shape)

    plt.figure(figsize=(6, 12))
    cmap = mpl.colors.ListedColormap([heatmap_0, heatmap_1])  # no data (-1) | no jump (0) | jump (1)
    plt.imshow(jumps, aspect=aspect, interpolation='none', cmap=cmap)
    plt.gca().invert_yaxis()
    plt.savefig(DIR_OUTPUT + os.sep + '%s-alldays-jump-heatmap_mpl.svg' % (genotype))
    plt.savefig(DIR_OUTPUT + os.sep + '%s-alldays-jump-heatmap_mpl.pdf' % (genotype))

    if show:
        plt.show()

    return jumps


def plot_full_data(genotype='KK', day=14, filtered=True, detailed_format=True, use_mpl=True):
    assert genotype in ['GD', 'KK']
    assert day in [7, 14, 21]

    file_path = to_file(genotype, day, detailed_format=detailed_format, filtered=filtered)
    data_dict = parse_fly_data(file_path, detailed_format=detailed_format, omit_fly_ids=OMIT_FLY_IDS[genotype])

    num_experiments = data_dict['num_experiments']
    num_trials = data_dict['num_trials_per_experiment']
    jumps_arr = data_dict['jump']

    df = pd.DataFrame(jumps_arr, index=range(1, num_experiments+1), columns=range(num_trials))
    df = df.reset_index().melt(id_vars='index', var_name='trial', value_name='jump')
    df.rename(columns={'index': 'fly'}, inplace=True)

    if use_mpl:
        plot = plt.figure(figsize=(8,3))
        cmap = mpl.colors.ListedColormap([heatmap_0, heatmap_1])  # no data (-1) | no jump (0) | jump (1)
        plt.imshow(jumps_arr, cmap=cmap, aspect=5.99, interpolation='none')
        plt.gca().invert_yaxis()
        plt.savefig(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap_mpl.svg' % (genotype, day))
        plt.savefig(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap_mpl.pdf' % (genotype, day))
    else:
        plot = plot_jumps(df, f"Observed Fly Jumps ({genotype=} {day=})")
        plot.save(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap.svg' % (genotype, day), width=9, height=3)
        plot.save(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap.pdf' % (genotype, day), width=9, height=3)

    return jumps_arr, plot


def plot_jumps(df, title):
    df['jump'] = df['jump'].astype(str)
    df['fly'] = df['fly'].astype(int)
    df['trial'] = df['trial'].astype(float)
    plot = (
        pn.ggplot(df, pn.aes(x='trial', y='fly', fill='jump'))
        + pn.geom_tile()
        + pn.scale_fill_manual(values=[heatmap_0, heatmap_1])  # darkblue: 0 (no-jump), orange: 1 (jump)
        #+ pn.scale_fill_manual(values=["purple", "yellow"])  # purple: 0 (no-jump), yellow: 1 (jump)
        + pn.geom_vline(xintercept=[0, 200, 400, 600, 800, 1000],
                            color="black", size=0.5, linetype="solid")
        + pn.labs(title=title, x='Trial', y='Fly ID', fill='Jump')
        + pn.scale_x_continuous(breaks=[0, 200, 400, 600, 800, 1000],
            labels=[0, 200, 600, 400, 800, 1000])
        + pn.theme(
            panel_background=pn.element_rect(fill="white", color=None),
            panel_grid_major=pn.element_blank(),
            panel_grid_minor=pn.element_blank(),
            axis_ticks_major=pn.element_blank(),
            )
        )
    return plot


def simulate_from_fit(genotype='KK', day=14, fit_path=DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv',
                      use_mpl=True):
    """
    Simulate jumps from the inferred model for genotype KK on day 14.
    - Load the fly stability data for a genotype (e.g. KK) and age (e.g. day 14).
    - For each fly, calculate the jump probability for each trial.
    - Simulate a single trajectory (one jumps for each trial) using the calculated probabilities.
    """
    assert genotype in ['GD', 'KK']
    assert day in [7, 14, 21]

    # must create fly-stability-days.csv first
    df = pd.read_csv(fit_path)
    df = filter_df_by_filterdict(df, {'day': [day], 'genotype': [genotype]})

    # Omit specified fly IDs
    omit_ids = OMIT_FLY_IDS.get(genotype, [])
    if omit_ids:
        df = df[~df['fly_id'].isin(omit_ids)]

    flies = []
    jumps_arr = []
    for i, row in df.iterrows():
        alpha = row['alpha']
        beta = row['beta']
        p0 = row['p0']
        fly = row['fly']
        genotype = row['genotype']
        day = row['day']

        p = jump_prob(alpha, beta, p0)
        jumps = np.random.binomial(1, p)
        jumps_arr.append(jumps)

        for trial_idx, jump_val in enumerate(jumps):
            flies.append({
                'genotype': genotype,
                'day': day,
                'fly': fly,
                'trial': trial_idx,
                'jump': jump_val
            })
    jumps_arr = np.array(jumps_arr)
    sim_df = pd.DataFrame(flies)  # this df is in the 'melted' format expected by plot_jumps(...)

    print('Num. rows in df (expt filtered to genotype=%s, age=%d):' % (genotype, day), len(df))
    print('Num. rows in sim_df:', len(sim_df) // 1050)
    sim_df = sim_df.sort_values(by='fly')

    if use_mpl:
        plot = plt.figure(figsize=(8, 3))
        cmap = mpl.colors.ListedColormap([heatmap_0, heatmap_1])  # no jump (0) | jump (1)
        plt.imshow(jumps_arr, cmap=cmap, aspect=5.99, interpolation='none')
        plt.gca().invert_yaxis()
        plt.savefig(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap-sim_mpl.svg' % (genotype, day))
        plt.savefig(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap-sim_mpl.pdf' % (genotype, day))
    else:
        plot = plot_jumps(sim_df, f"Simulated Fly Jumps ({genotype=} {day=})")
        plot.save(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap-sim.svg' % (genotype, day), width=9, height=3)
        plot.save(DIR_OUTPUT + os.sep + '%s-%s-jump-heatmap-sim.pdf' % (genotype, day), width=9, height=3)
    return sim_df, jumps_arr, plot


if __name__ == '__main__':

    genotype = 'KK'

    # 1) plot full dataset across all days
    jumps = plot_full_data_all_days(genotype=genotype, show=True)
    print('full_data_all_days jumps shape:', jumps.shape)

    # 2) Posterior predictive check: experiment vs simulation on particular day
    for day in [7, 14, 21]:
        # - Plot the observed jumps (experimental data) for a given genotype and day
        data_jumps_arr, data_plot = plot_full_data(genotype=genotype, day=day, detailed_format=True)
        data_plot.show()

        # - Simulate jumps from the inferred model
        fit_path = DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv'
        sim_df, sim_jumps_arr, plot = simulate_from_fit(genotype=genotype, day=day, fit_path=fit_path)
        plot.show()
