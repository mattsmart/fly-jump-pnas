import matplotlib.pyplot as plt
import numpy as np

import os
import sys

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


from python.data_tools import (build_dataframe_from_data, build_mergetrials_dataframe,
                              df_to_arr_jumps, filter_df_by_filterdict)


def make_one_plot_per_expt_slice(df_full_mergetrials, convert_pm1=True):
    """
    plot where the x axis is time index from 0 to 200 and y axis is cumulative number of jumps
    - repeat for each fly, each experiment
    - each line is one fly in one experiment (e.g. GD age 14, Hab 1 of 5)

    convert_pm1 = True  # if True, convert to 0, 1 ---> -1, 1
    """
    df_merge_GD_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [7]})
    df_merge_GD_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [14]})
    df_merge_GD_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [21]})

    df_merge_KK_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [7]})
    df_merge_KK_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [14]})
    df_merge_KK_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [21]})

    for df_expt, label_expt in [
        (df_merge_GD_7, 'GD_a7'), (df_merge_GD_14, 'GD_a14'), (df_merge_GD_21, 'GD_a21'),
        (df_merge_KK_7, 'KK_a7'), (df_merge_KK_14, 'KK_a14'), (df_merge_KK_21, 'KK_a21')]:

        # get an array of chamber IDs from dataframe
        chamber_groups = df_expt['chamber_group'].values

        # cmap = plt.get_cmap('tab20')
        # colors_chamber = [cmap(i)[0:3] for i in chamber_groups]  # strip alpha
        cmap = ['#44AA99', '#AA4499', '#999933', '#FF3D00']  # color for chambers 0, 1, 2, 3
        colors_chamber = [cmap[i] for i in chamber_groups]  # strip alpha

        for slice, label_slice in [([0, 200], 'Hab 1'),
                                   ([200, 400], 'Hab 2'),
                                   ([400, 600], 'Hab 3'),
                                   ([600, 800], 'Hab 4'),
                                   ([800, 1000], 'Hab 5'),
                                   ([1000, 1050], 'Reactivity')]:

            npts = slice[1] - slice[0]
            assert npts in [200, 50]

            jump_data = df_to_arr_jumps(df_expt)[:, slice[0]:slice[1]]
            if convert_pm1:
                jump_data = jump_data * 2 - 1  # convert to -1, 1

            cumulative_jumps = jump_data.cumsum(axis=1)  # shape is (n_flies, npts)
            cumulative_jumps_mean = cumulative_jumps.mean(axis=0)

            # Create figure for 1 hab expt for 1 gene_bgr / day (e.g. GD, age 14)
            plt.figure(figsize=(6, 8))

            for k in range(cumulative_jumps.shape[0]):
                plt.plot(cumulative_jumps[k], alpha=0.2, c=colors_chamber[k])
            # plt.plot(cumulative_jumps.T, alpha=0.4, c=colors_chamber)

            # plot overall mean
            plt.plot(cumulative_jumps_mean, color='black', linewidth=4, label='mean (all)', zorder=10)
            # get indices for chamber group, take mean, plot group mean
            for cg in range(4):
                indices = np.where(chamber_groups == cg)[0]
                cumulative_jumps_cg = cumulative_jumps[indices, :]
                cumulative_jumps_cg_mean = cumulative_jumps_cg.mean(axis=0)
                plt.plot(cumulative_jumps_cg_mean, '-', color=cmap[cg], linewidth=4, label='chamber %d' % cg)

            # plot baselines
            plt.plot([0, npts - 1], [1, npts], '--', color='red', linewidth=2, zorder=20, label=r'100% jump')
            if convert_pm1:
                plt.axhline(0, linestyle='--', color='k', linewidth=2, zorder=20, label='fair coin')
                plt.plot([0, npts - 1], [-1, -npts], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')
            else:
                plt.plot([0, npts - 1], [0, npts / 2.0], linestyle='--', color='k', linewidth=2, zorder=20,
                         label='fair coin')
                plt.plot([0, npts - 1], [0, 0], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')

            plt.title('Cumulative jumps per fly: %s - %s' % (label_expt, label_slice))
            plt.xlabel('Stimulus index')
            plt.ylabel('Cumulative sum jumps%s' % (r' (converted to $\pm 1$)' if convert_pm1 else ''))
            plt.legend()

            # zoom via x, y limits
            zoom = npts + 0.5  # default: npts + 0.5 - no zoom
            plt.xlim(-1, zoom)
            if convert_pm1:
                plt.ylim(-zoom, zoom)
            else:
                plt.ylim(0, zoom)

            ax2 = plt.gca().twinx()
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Final jump fraction')

            plt.show()


def make_one_plot_per_genebgr(df_full_mergetrials, convert_pm1=True, show_legend=True):
    """
    convert_pm1 = True  # if True, convert to 0, 1 ---> -1, 1

    See make_one_plot_per_expt_slice
    """
    df_merge_GD_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [7]})
    df_merge_GD_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [14]})
    df_merge_GD_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [21]})

    df_merge_KK_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [7]})
    df_merge_KK_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [14]})
    df_merge_KK_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [21]})

    for plot_block, gene_bgr in [
        [((df_merge_GD_7, 'a7'), (df_merge_GD_14, 'a14'), (df_merge_GD_21, 'a21')), 'GD'],
        [((df_merge_KK_7, 'a7'), (df_merge_KK_14, 'a14'), (df_merge_KK_21, 'a21')), 'KK']
    ]:

        fig, axarr = plt.subplots(3, 6, figsize=(18, 9), squeeze=False, sharex='col')

        for row, (df_expt, label_expt) in enumerate(plot_block):

            # get an array of chamber IDs from dataframe
            chamber_groups = df_expt['chamber_group'].values

            # cmap = plt.get_cmap('tab20')
            # colors_chamber = [cmap(i)[0:3] for i in chamber_groups]  # strip alpha
            cmap = ['#44AA99', '#AA4499', '#999933', '#FF3D00']  # color for chambers 0, 1, 2, 3
            colors_chamber = [cmap[i] for i in chamber_groups]  # strip alpha

            for col, (slice, label_slice) in enumerate([
                ([0, 200], 'Hab 1'),
                ([200, 400], 'Hab 2'),
                ([400, 600], 'Hab 3'),
                ([600, 800], 'Hab 4'),
                ([800, 1000], 'Hab 5'),
                ([1000, 1050], 'Reactivity')]):

                npts = slice[1] - slice[0]
                assert npts in [200, 50]

                jump_data = df_to_arr_jumps(df_expt)[:, slice[0]:slice[1]]
                if convert_pm1:
                    jump_data = jump_data * 2 - 1  # convert to -1, 1

                cumulative_jumps = jump_data.cumsum(axis=1)  # shape is (n_flies, npts)
                cumulative_jumps_mean = cumulative_jumps.mean(axis=0)

                for k in range(cumulative_jumps.shape[0]):
                    axarr[row, col].plot(cumulative_jumps[k], alpha=0.1, c=colors_chamber[k])
                # plt.plot(cumulative_jumps.T, alpha=0.4, c=colors_chamber)

                # plot overall mean
                axarr[row, col].plot(cumulative_jumps_mean, color='black', linewidth=4, label='mean (all)', zorder=10)
                # get indices for chamber group, take mean, plot group mean
                for cg in range(4):
                    indices = np.where(chamber_groups == cg)[0]
                    cumulative_jumps_cg = cumulative_jumps[indices, :]
                    cumulative_jumps_cg_mean = cumulative_jumps_cg.mean(axis=0)
                    axarr[row, col].plot(cumulative_jumps_cg_mean, '-', color=cmap[cg], linewidth=4, label='chamber %d' % cg)

                # plot baselines
                axarr[row, col].plot([0, npts - 1], [1, npts], '--', color='red', linewidth=2, zorder=20, label=r'100% jump')
                if convert_pm1:
                    axarr[row, col].axhline(0, linestyle='--', color='k', linewidth=2, zorder=20, label='fair coin')
                    axarr[row, col].plot([0, npts - 1], [-1, -npts], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')
                else:
                    axarr[row, col].plot([0, npts - 1], [0, npts / 2.0], linestyle='--', color='k', linewidth=2, zorder=20,
                             label='fair coin')
                    axarr[row, col].plot([0, npts - 1], [0, 0], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')

                axarr[row, col].set_title('%s - %s' % (label_expt, label_slice))

                if row == 2 and col == 0:
                    axarr[row, col].set_xlabel('Stimulus index')
                if row == 1 and col == 0:
                    axarr[row, col].set_ylabel('Cumulative sum jumps%s' % (r' (converted to $\pm 1$)' if convert_pm1 else ''))
                if row == 0 and col == 0 and show_legend:
                    axarr[row, col].legend()
                if col in [1,2,3,4]:
                    axarr[row, col].yaxis.set_visible(False)
                    #for spine in ['top', 'right', 'left', 'bottom']:
                    #    axarr[row, col].spines[spine].set_visible(False)

                # zoom via x, y limits
                zoom = npts + 0.5  # default: npts + 0.5 - no zoom
                axarr[row, col].set_xlim(-1, zoom)
                if convert_pm1:
                    axarr[row, col].set_ylim(-zoom, zoom)
                else:
                    axarr[row, col].set_ylim(0, zoom)

                if col == 5:
                    ax2 = axarr[row, col].twinx()
                    ax2.set_ylim(0, 1)
                    ax2.set_ylabel('Final jump fraction')

        plt.suptitle('Cumulative jumps per fly (all ages): %s' % gene_bgr)
        plt.tight_layout()
        plt.show()


def make_synthetic_data_plot():
    nflies = 100
    npts = 50

    synthetic_jumps_pt5 = np.random.binomial(n=1, p=0.5, size=(nflies, npts))  # 10 flies, 1000 time points
    synthetic_jumps_pt25 = np.random.binomial(n=1, p=0.25, size=(nflies, npts))  # 10 flies, 1000 time points
    synthetic_jumps_pt75 = np.random.binomial(n=1, p=0.75, size=(nflies, npts))  # 10 flies, 1000 time points

    convert_pm1 = True  # if True, convert to 0, 1 ---> -1, 1

    colors = ['k', 'blue', 'red']

    # Create figure for 1 hab expt for 1 gene_bgr / day (e.g. GD, age 14)
    plt.figure(figsize=(6, 8))

    for k, (synthetic_jumps, label_expt) in enumerate([
        (synthetic_jumps_pt5, 'p=0.5'), (synthetic_jumps_pt25, 'p=0.25'), (synthetic_jumps_pt75, 'p=0.75')
    ]):

        jump_data = synthetic_jumps
        if convert_pm1:
            jump_data = jump_data * 2 - 1  # convert to -1, 1

        cumulative_jumps = jump_data.cumsum(axis=1)  # shape is (n_flies, npts)
        cumulative_jumps_mean = cumulative_jumps.mean(axis=0)

        # plot data curves + mean
        plt.plot(cumulative_jumps.T, alpha=0.2, c=colors[k])
        plt.plot(cumulative_jumps_mean, color=colors[k], linewidth=4, zorder=10, label=label_expt)

    # plot baselines
    plt.plot([0, npts - 1], [1, npts], '--', color='red', linewidth=2, zorder=20, label=r'100% jump')
    if convert_pm1:
        plt.axhline(0, linestyle='--', color='k', linewidth=2, zorder=20, label='fair coin')
        plt.plot([0, npts - 1], [-1, -npts], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')
    else:
        plt.plot([0, npts - 1], [0, npts / 2.0], linestyle='--', color='k', linewidth=2, zorder=20,
                 label='fair coin')
        plt.plot([0, npts - 1], [0, 0], '--', color='blue', linewidth=2, zorder=20, label=r'100% no-jump')

    plt.title('Cumulative jumps per fly: Synthetic data (bernoulli)')
    plt.xlabel('Stimulus index')
    plt.ylabel('Cumulative sum jumps%s' % (r' (converted to $\pm 1$)' if convert_pm1 else ''))
    plt.legend()

    # zoom via x, y limits
    zoom = npts + 0.5  # default: npts + 0.5 - no zoom
    plt.xlim(-1, zoom)
    if convert_pm1:
        plt.ylim(-zoom, zoom)
    else:
        plt.ylim(0, zoom)

    ax2 = plt.gca().twinx()
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Final jump fraction')

    plt.show()


def make_one_plot_per_individual_fly(df_full_mergetrials, convert_pm1=True, show_legend=True):
    """
    plot where the x axis is time index from 0 to 200 and y axis is cumulative number of jumps
    - repeat for each fly, each experiment
    - each line is one fly in one experiment (e.g. GD age 14, Hab 1 of 5)

    convert_pm1 = True  # if True, convert to 0, 1 ---> -1, 1
    """
    # TODO for each fly, make a multipanel figure with raw data top (6 rows, 5x hab + SRA heatmap)
    #  - add cross entropy score of model vs baselines (text or bars)
    #  - add cumsum panel         - data left, model right
    #  - add moving average panel - data left, model right
    #     in data plots, overlay the multistage mean and the population mean for that expt
    # TODO maybe this ^^^ should be done in the other script, plot_stan_results.py
    # - TODO 1) in the plot below, could add an overlay related to the population mean (for SRA and for all hab stages)
    # - TODO 2) make same plots but using the model mean and thus expected bernboulli variance?) for each time point
    df_merge_GD_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [7]})
    df_merge_GD_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [14]})
    df_merge_GD_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['GD'], 'age': [21]})

    df_merge_KK_7 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [7]})
    df_merge_KK_14 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [14]})
    df_merge_KK_21 = filter_df_by_filterdict(df_full_mergetrials, {'gene_bgr': ['KK'], 'age': [21]})

    for df_expt, label_expt in [
        (df_merge_GD_7, 'GD_a7'), (df_merge_GD_14, 'GD_a14'), (df_merge_GD_21, 'GD_a21'),
        (df_merge_KK_7, 'KK_a7'), (df_merge_KK_14, 'KK_a14'), (df_merge_KK_21, 'KK_a21')]:

        # get an array of chamber IDs from dataframe
        chamber_groups = df_expt['chamber_group'].values

        # cmap = plt.get_cmap('tab20')
        # colors_chamber = [cmap(i)[0:3] for i in chamber_groups]  # strip alpha
        cmap = ['#44AA99', '#AA4499', '#999933', '#FF3D00']  # color for chambers 0, 1, 2, 3
        colors_chamber = [cmap[i] for i in chamber_groups]  # strip alpha

        jump_data_full = df_to_arr_jumps(df_expt, jump_col='jumpdata_str')
        jump_data_full_pm1 = jump_data_full * 2 - 1  # convert to -1, 1

        # we will fill in these arrays for each expt day (e.g. KK, age 14)
        cumulative_jumps_M_T_hab_stage = np.zeros((jump_data_full.shape[0], 200, 5))
        cumulative_jumps_M_T_sra       = np.zeros((jump_data_full.shape[0], 50))

        movingavg_10slide10_M_T_hab_stage = np.zeros((jump_data_full.shape[0], 20, 5))  # note T -> T/10
        movingavg_10slide10_M_T_sra = np.zeros((jump_data_full.shape[0], 5))            # note T -> T/10

        for stage, (slice, label_slice) in enumerate(
                [([0, 200], 'Hab 1'),
                 ([200, 400], 'Hab 2'),
                 ([400, 600], 'Hab 3'),
                 ([600, 800], 'Hab 4'),
                 ([800, 1000], 'Hab 5'),
                 ([1000, 1050], 'Reactivity')]
        ):

            npts = slice[1] - slice[0]
            assert npts in [200, 50]

            if convert_pm1:
                jump_data_for_cumsum = jump_data_full_pm1[:, slice[0]:slice[1]]
            else:
                jump_data_for_cumsum = jump_data_full[:, slice[0]:slice[1]]
            jump_data_for_ma = jump_data_full[:, slice[0]:slice[1]]

            if stage < 5:
                cumulative_jumps_M_T_hab_stage[:, :npts, stage] = jump_data_for_cumsum.cumsum(axis=1)  # shape is (n_flies, 200, 5)
                # take moving average over 20 slices, by partitioning the 200 trials into separate blocks of 10
                for k in range(20):
                    # TODO 01 vs pm1 form...
                    movingavg_10slide10_M_T_hab_stage[:, k, stage] = np.sum(jump_data_for_ma[:, k*10:(k+1)*10], axis=1)

            else:
                cumulative_jumps_M_T_sra[:, :] = jump_data_for_cumsum.cumsum(axis=1)  # shape is (n_flies
                for k in range(5):
                    # TODO 01 vs pm1 form...
                    movingavg_10slide10_M_T_sra[:, k] = np.sum(jump_data_for_ma[:, k*10:(k+1)*10], axis=1)

        # For each fly on a given day in the dataset, plot the cumulative jumps over the 6 stages in a single figure
        for fly_idx in range(jump_data_full.shape[0]):
            plt.figure(figsize=(6, 8))
            for stage in range(5):
                cumulative_jumps = cumulative_jumps_M_T_hab_stage[fly_idx, :, stage]
                plt.plot(cumulative_jumps, alpha=0.6)
            # now plot the 50 sra jumps sum
            plt.plot(cumulative_jumps_M_T_sra[fly_idx, :], alpha=0.8, c='k')
            plt.axhline(0, linestyle='--', color='k', linewidth=2, zorder=20, label='fair coin')
            plt.plot([0, 49], [-1, -50], '-', color='blue', linewidth=0.5, zorder=20, label=r'100% no-jump')
            plt.plot([0, 49], [1, 50], '-', color='red', linewidth=0.5, zorder=20, label=r'100% jump')
            plt.legend()
            plt.title('Cumulative jumps for fly idx #%s: %s - %s' % (fly_idx, label_expt, 'all stages'))
            plt.show()

            # For each fly on a given day in the dataset, plot the moving average data over the 6 stages in a single figure
            plt.figure(figsize=(6, 8))
            for stage in range(5):
                plt.plot(movingavg_10slide10_M_T_hab_stage[fly_idx, :, stage], '--o', alpha=0.6)
            # now plot the 50 sra jumps sum
            plt.plot(movingavg_10slide10_M_T_sra[fly_idx, :], '-o', alpha=1.0, c='k')
            plt.axhline(5, linestyle='--', color='grey', linewidth=1, zorder=20, label='fair coin')
            #plt.plot([0, 49], [-1, -50], '-', color='blue', linewidth=0.5, zorder=20, label=r'100% no-jump')
            #plt.plot([0, 49], [1, 50], '-', color='red', linewidth=0.5, zorder=20, label=r'100% jump')
            # make ylabels 0 to 10 inclusive
            plt.yticks(np.arange(0, 11, 1))  # plt.yticks(np.arange(0, 11, 2))
            # make xticks 0 to 20 inclusive, label them 1 to 5 then 10, 15, 20
            xticks = [0, 1, 2, 3, 4, 9, 14, 19]
            plt.xticks(xticks,
                       labels=['%d' % (i+1) for i in xticks])
            plt.ylabel('number of jumps in sliding window (10)')
            plt.xlabel('time index (10 trials per index)')
            plt.legend()
            plt.title('Moving average (10-slide-10) for fly idx #%s: %s - %s' % (fly_idx, label_expt, 'all stages'))
            plt.show()
    return


if __name__ == '__main__':
    data_version = 'v3'  # try either v2 (given Jan 2024) or v3: (given April 2024)
    assert data_version == 'v3'  # generalize later

    if data_version == 'v2':
        heatmap_title = 'Merged jump data (%s - each row is 3 HAB + 1 SRA experiments)' % data_version
        heatmap_aspect = 0.1
    else:
        assert data_version == 'v3'
        heatmap_title = 'Merged jump data (%s - each row is 5 HAB + 1 SRA experiments)' % data_version
        heatmap_aspect = 0.75

    df_full = build_dataframe_from_data(data_version=data_version)
    df_full_mergetrials = build_mergetrials_dataframe(df_full, data_version=data_version)  # v2 is 350; v3 is 1050

    plt.figure(figsize=(8, 7))
    plt.imshow(df_to_arr_jumps(df_full_mergetrials), aspect=heatmap_aspect, interpolation='None')
    plt.title(heatmap_title, fontsize=12)
    plt.show()

    # synthetic version (Bernoulli events)
    make_synthetic_data_plot()

    # cumulative sum-of-jumps plot - v1 (on expt data)
    make_one_plot_per_expt_slice(df_full_mergetrials)

    # cumulative sum-of-jumps plot - v2 (on expt data)
    #make_one_plot_per_genebgr(df_full_mergetrials, show_legend=False)

    # cumulative sum-of-jumps plots + moving avg every fly - v3 (on expt data)
    #make_one_plot_per_individual_fly(df_full_mergetrials, convert_pm1=True, show_legend=True)
