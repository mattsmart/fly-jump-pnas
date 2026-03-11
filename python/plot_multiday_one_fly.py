import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from data_tools import df_to_arr_jumps
from functions_common import likelihood_func, moving_average, get_conf_movingavg
from plot_common import plot_posterior_likelihood_summary_over_days
from settings import *


def plot_multiday_one_fly(summary_csv, draws_over_age, genotype, fly_id, days_to_plot,
                          show_ma_each_hab_stage=False, cols_to_rows=False,
                          ma5_hab=False,
                          ma5_sra=False,
                          publication_style=False,  # Set default to False
                          slim_labeling=False,
                          ma_color_by_day=False,
                          show=True):
    """
    Plot multi-day jump data and model summaries for a single fly.

    This function generates a multi-panel figure comparing observed jump data and
    model predictions for a given fly across multiple experimental days. For each day,
    the figure is divided into three subplots:
      1. A heatmap showing the jump data (binary 1/0 for jumps) across trials.
      2. A plot of the moving average of jump probability, with additional annotations
         such as estimated jump probabilities and confidence intervals computed under
         a binomial model (see scipy.stats.binomtest documentation).
      3. A plot summarizing the posterior likelihood (model predictions) with a credible
         interval overlay.

    The layout of the figure can be adjusted with the `cols_to_rows` flag:
      - If False (default), each day is displayed in its own column using a 3-row layout.
      - If True, each day is displayed in its own row using a 1-row, 3-column layout.

    Alternatively, the publication_style flag can be used for a specific layout shown
    in the reference image.

    Parameters
    ----------
    summary_csv : pandas.DataFrame
        A CSV-loaded DataFrame containing summary statistics for each fly and day.
        It should include columns such as 'genotype', 'day', 'fly_id', 'jumpdata',
        and various inferred model parameters (e.g., 'alpha', 'beta', 'p0', etc.).
    draws_over_age : dict
        A dictionary containing posterior draws indexed by genotype and day. These
        draws are used to generate the posterior likelihood summary plot.
    genotype : str
        The genotype identifier for the fly. This is used to filter the summary CSV and
        to label the plots.
    fly_id : int or str
        The unique identifier for the fly. This is used to filter the summary CSV and
        to label the plots.
    days_to_plot : list
        A list of days (e.g., [1, 2, 3, ...]) for which the data and model predictions
        should be plotted.
    show_ma_each_hab_stage : bool, optional
        Flag to indicate whether to overlay the moving averages for each habituation stage
        on the moving average plot. Default is False.
    cols_to_rows : bool, optional
        Flag to control the layout of the figure:
          - False (default): Each day is displayed as a column (3 subplots in a column).
          - True: Each day is displayed as a row (3 subplots in a row).
    ma5_hab; ma5_sra : bool, optional
        Flag to use 5-point moving average instead of 10-point. Default is False.
    publication_style : bool, optional
        Flag to use the publication style layout as shown in the reference image.
        This overrides the cols_to_rows flag. Default is False.
    slim_labeling: bool, Default is False
        Hide detailed labels/legends (for small plots)
    show : bool, optional
        Flag to control whether to display the plot. Default is True.

    Returns
    -------
    None
        The function displays the plot and saves it as PNG and SVG files in the directory
        specified by DIR_OUTPUT.

    Notes
    -----
    - The function computes 95% confidence intervals for the moving average using a binomial
      model (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
    - It assumes that the jump data for each fly is stored in a specific format within the
      summary_csv, and that model parameters (e.g., 'alpha', 'beta', 'p0') and additional metrics
      (e.g., T1_hab_mag_rel_mean, T1_hab_time_half_rel_mean) are available.
    - The function uses nested gridspec objects from matplotlib to arrange subplots based on the
      specified layout.
    """
    # Set figure size depending on layout and number of days
    if publication_style:
        if len(days_to_plot) == 1:
            # Single age layout: One row with heatmap, one row with data and model side by side
            fig = plt.figure(figsize=(12, 8))  # Adjusted for better aspect ratio
            # Create a 2-row GridSpec with more space for bottom row
            outer_gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.22, 0.78], hspace=0.7)

            # Bottom row divided into two columns for data and model - MINIMAL SPACING
            bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1], wspace=-0.25)
            ax_data_pjump = fig.add_subplot(bottom_gs[0, 0])
            ax_model_pjump = fig.add_subplot(bottom_gs[0, 1])

            # Top row for heatmap - create AFTER bottom row to match exact width
            ax_jump_heatmap = fig.add_subplot(outer_gs[0])

            # Add a title for the figure - includes genotype
            plt.figtext(0.5, 0.92, f'Fly ID: #{fly_id} ({genotype})', fontsize=20, ha='center')
        else:
            # Multiple ages: Create a column of blocks, one for each age
            # Adjust height based on number of days
            fig = plt.figure(figsize=(12, 6 * len(days_to_plot)))
            # Outer gridspec: each day in its own row with MODERATE spacing between blocks
            outer_gs = gridspec.GridSpec(len(days_to_plot), 1, figure=fig, hspace=0.3)  # Increased from 0.15 to 0.3

            axes_by_day = {}  # We'll store axes by day for later use

            # Add overall title for the figure at the top
            plt.figtext(0.5, 0.98, f'Fly ID: #{fly_id} ({genotype})', fontsize=20, ha='center')

            for day_idx, day in enumerate(days_to_plot):
                # For each day, create a nested GridSpec with 2 rows
                day_gs = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                          subplot_spec=outer_gs[day_idx],
                                                          height_ratios=[0.22, 0.78],
                                                          hspace=0.7)

                # Bottom row first - divided for data and model with minimal spacing
                bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                             subplot_spec=day_gs[1],
                                                             wspace=-0.25)
                ax_data_pjump  = fig.add_subplot(bottom_gs[0, 0])
                ax_model_pjump = fig.add_subplot(bottom_gs[0, 1])

                # Top row for heatmap - create AFTER bottom row to match exact width
                ax_jump_heatmap = fig.add_subplot(day_gs[0])

                # Add age indicator above each heatmap
                ax_jump_heatmap.set_title(f'Age {day}', fontsize=16, pad=10)

                axes_by_day[day] = {
                    'heatmap': ax_jump_heatmap,
                    'data': ax_data_pjump,
                    'model': ax_model_pjump
                }

    elif cols_to_rows:
        # Original cols_to_rows code remains the same
        fig = plt.figure(figsize=(14, 2 * len(days_to_plot)))
        outer_gs = gridspec.GridSpec(len(days_to_plot), 1, figure=fig, hspace=0.3)
    else:
        # Original default layout code remains the same
        fig = plt.figure(figsize=(6 * len(days_to_plot), 10))
        outer_gs = gridspec.GridSpec(1, len(days_to_plot), figure=fig, wspace=0.3)

    if ma_color_by_day:
        day_palette = days_palettes['Deep']
        hab_ma_colors      = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}
        hab_ma_colors_fill = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}
    else:
        hab_ma_colors      = {7: 'tab:blue',   14: 'tab:blue',   21: 'tab:blue'}
        hab_ma_colors_fill = {7: 'tab:purple', 14: 'tab:purple', 21: 'tab:purple'}

    # Define the window size for moving averages based on ma5 flag
    window_size_hab = 5 if ma5_hab else 10
    window_size_sra = 5 if ma5_sra else 10

    for day_idx, day in enumerate(days_to_plot):
        if publication_style and len(days_to_plot) > 1:
            # For multiple days in publication style, we've already created the axes
            ax_jump_heatmap = axes_by_day[day]['heatmap']
            ax_data_pjump = axes_by_day[day]['data']
            ax_model_pjump = axes_by_day[day]['model']
        elif publication_style:
            # For a single day in publication style, we've already created the axes above
            pass
        elif cols_to_rows:
            # Original cols_to_rows inner gridspec setup
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, 3,
                subplot_spec=outer_gs[day_idx],
                wspace=0.4,
                width_ratios=[0.45, 0.35, 0.35]
            )
            ax_jump_heatmap = fig.add_subplot(inner_gs[0, 0])
            ax_data_pjump = fig.add_subplot(inner_gs[0, 1])
            ax_model_pjump = fig.add_subplot(inner_gs[0, 2])
        else:
            # Original default inner gridspec setup
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                3, 1,
                subplot_spec=outer_gs[0, day_idx],
                hspace=0.4,
                height_ratios=[0.15, 0.45, 0.45]
            )
            ax_jump_heatmap = fig.add_subplot(inner_gs[0])
            ax_data_pjump = fig.add_subplot(inner_gs[1])
            ax_model_pjump = fig.add_subplot(inner_gs[2])

        df_row_fly = summary_csv[(summary_csv['genotype'] == genotype) &
                                 (summary_csv['day'] == day) &
                                 (summary_csv['fly_id'] == fly_id)]
        assert df_row_fly.shape[0] == 1

        label_expt = '%s_%d' % (genotype, day)
        print('Working on day:', day)


        # =================================================================
        # extract the data for this fly on this day from the summary_csv

        jumpdata = df_to_arr_jumps(df_row_fly, jump_col='jumpdata')[0, :]

        alpha = df_row_fly['alpha'].values[0]
        beta = df_row_fly['beta'].values[0]
        p0 = df_row_fly['p0'].values[0]

        T1_hab_mag_rel = df_row_fly['T1_hab_mag_rel_mean'].values[0]  # NOTE: for each, have median, std, CI_lower/upper
        T1_hab_mag_abs = df_row_fly['T1_hab_mag_abs_mean'].values[0]
        T1_hab_time_half_rel = df_row_fly['T1_hab_time_half_rel_mean'].values[0]
        T1_hab_time_95_rel = df_row_fly['T1_hab_time_95_rel_mean'].values[0]
        T1_hab_time_half_abs = df_row_fly['T1_hab_time_half_abs_mean'].values[0]  # can be NaN
        # T5 ... (same as above)

        # =================================================================
        # compute jumpsum stats for this fly on this day

        # We will fill in these arrays for each expt day (e.g. KK, age 14)
        # Calculate how many blocks we'll need based on window size
        num_blocks_hab = 200 // window_size_hab
        num_blocks_sra = 50 // window_size_sra

        movingavg_slide_M_T_hab_stage = np.zeros((num_blocks_hab, 5))  # note T -> T/window_size
        movingavg_slide_M_T_hab_mean = np.zeros((num_blocks_hab))
        movingavg_slide_M_T_sra = np.zeros(num_blocks_sra)  # note T -> T/window_size

        jumpmean_est_M_T_hab_stage = np.zeros((200, 5))
        jumpmean_est_M_T_sra = np.zeros(50)

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

            jump_data_for_ma = jumpdata[slice[0]:slice[1]]

            if stage < 5:
                # for the jumpmean_est_ use 'for_ma' bc want 01 here not pm1 | we divide each column by num. stimuli T
                jumpmean_est_M_T_hab_stage[:npts, stage] = jump_data_for_ma.cumsum() / np.arange(1, npts + 1)
                # take moving average over blocks, by partitioning the trials into separate blocks
                for k in range(num_blocks_hab):
                    movingavg_slide_M_T_hab_stage[k, stage] = np.sum(
                        jump_data_for_ma[k * window_size_hab:(k + 1) * window_size_hab])
            else:
                # for the jumpmean_est_ use 'for_ma' bc want 01 here not pm1 | we divide each column by num. stimuli T
                jumpmean_est_M_T_sra[:] = jump_data_for_ma.cumsum() / np.arange(1, npts + 1)
                for k in range(50 // window_size_sra):
                    movingavg_slide_M_T_sra[k] = np.sum(jump_data_for_ma[k * window_size_sra:(k + 1) * window_size_sra])

        movingavg_slide_M_T_hab_mean[:] = np.mean(movingavg_slide_M_T_hab_stage[:, :], axis=1)

        # =================================================================
        # model timeseries (likelihood) for this fly on this day
        model_likelihood_hab = likelihood_func(np.arange(0, 200), alpha, beta, p0, 1.0)
        model_likelihood_sra = likelihood_func(np.arange(0, 50), alpha, beta, p0, 5.0)

        # in panel (0,0), plot a heatmap of the jump data using imshow style
        # axarr[0, 0].imshow(jump_data_full[fly_idx, :], aspect='auto', cmap='viridis')
        # get the 01-jumpdata for this fly

        # make a heatmap of size 6x200 in the style of plot_jumpmap_chunked_by_fly_and_age_tall
        # -  no data, end of SRA (-1) | no jump (0) | jump (1)
        jumpdata_6x200 = -1 * np.ones((6, 200))
        for stage in range(5):
            jumpdata_6x200[stage, :] = jumpdata[200 * stage: 200 * (stage + 1)]
        jumpdata_6x200[5, 0:50] = jumpdata[1000:]

        # heatmap panel settings
        fs_ylabels = 8  # figsize (8, 12) and 64 flies use fs=8
        HEATMAP_WIDTH = 5 * 200 + 50  # num_hab_cycles * n_pulse_hab + n_pulse_reactivity
        dataversion_xticks = [0, 200, 400, 600, 800, 1000]

        # cmap = mpl.colors.ListedColormap(['whitesmoke', 'darkblue', 'orange'])
        cmap = mpl.colors.ListedColormap([heatmap_nodata, heatmap_0, heatmap_1])
        heatmap_aspect = 'auto'  # 2.0  #'auto'  # auto will fill the space

        # =================================================================
        # Plot 1: heatmap of jump data
        # =================================================================
        im1 = ax_jump_heatmap.imshow(jumpdata_6x200, aspect=heatmap_aspect, interpolation='None', cmap=cmap, vmin=-1,
                                     vmax=1)
        ax_jump_heatmap.set_yticks([i for i in range(6)])
        ax_jump_heatmap.set_yticklabels(['H1', 'H2', 'H3', 'H4', 'H5', 'SRA'])
        for label in ax_jump_heatmap.get_yticklabels():
            label.set_fontsize(9.5)  # Change 10 to your desired font size

        # axarr[0, 0].set_ylabel('Jump data (1050)', fontsize=fs_ylabels, rotation=0, labelpad=6.0)
        # for each col...

        # change fontsize
        # ax0_wide_A.tick_params(axis='both', which='minor', labelsize=5)
        # ax0_wide_A.tick_params(axis='y',    which='major', labelsize=5)
        # change all spines
        # im1.set_extent([-0.5, num_ages + 0.5, -0.5, HEATMAP_WIDTH + 0.5])
        for axis in ['top', 'bottom', 'left', 'right']:
            # axarr[i, j].spines[axis].set_linewidth(0.2)
            ax_jump_heatmap.spines[axis].set_linewidth(0.05)
        # increase tick width
        ax_jump_heatmap.tick_params(width=0.75)
        # set title of subplot - not needed for publication style
        if day_idx == 0 and not publication_style:
            ax_jump_heatmap.set_title('Data: Jump data (1050 stimuli)', fontsize=12)

        # adjust position down slightly
        # pos = ax0_wide_A.get_position()
        # ax0_wide_A.set_position([pos.x0, pos.y0 - 0.1, pos.width, pos.height])
        # add a colorbar to the first heatmap (discrete)
        '''
        cbar_A = fig.colorbar(im1, ax=ax_jump_heatmap, orientation='vertical', shrink=1.0, pad=0.03)
        cbar_A.set_label('Jumps data', fontsize=10)'''
        # the following should be the x ticks: 1,2,3,4,5, 10, 25, 50, 75, 100, 125, 150, 175, 200
        heatmap_xticks_force = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
        ax_jump_heatmap.set_xticks([a - 1 for a in heatmap_xticks_force])
        ax_jump_heatmap.set_xticklabels([str(a) if a not in [2, 3, 4, 5] else '' for a in heatmap_xticks_force])#, fontsize=14)

        # =================================================================
        # Plot 2: moving average of jump data
        # =================================================================
        # Increase fontsize for tick labels
        ax_data_pjump.tick_params(axis='both', which='major', labelsize=14)

        # plot the more detailed moving average of the early trajectory with p_0 and p_inf axhlines
        # first plot the axhline envelope
        #  - upper: SRA average
        #  - lower: p_inf estimate from the tail of the data (average p50 to 200)
        p_upper_SRA = np.mean(movingavg_slide_M_T_sra[:]) / window_size_sra
        p_lower_inf_hab_mean = np.mean(jumpdata_6x200[0:5, 50:200])
        p_lower_inf_hab_1_to_5 = np.mean(jumpdata_6x200[0:5, 50:200], axis=1)
        p_hab_mean_timeseries_1pt = np.mean(jumpdata_6x200[0:5, :], axis=0)

        ax_data_pjump.axhline(p_upper_SRA, linestyle='--', color='k', linewidth=2, zorder=20,
                              label=r'$\langle \mathrm{SRA} \rangle$')
        ax_data_pjump.axhline(p_lower_inf_hab_mean, linestyle='--', color=hab_ma_colors[day], linewidth=2, zorder=20,
                              label=r'$\langle \mathrm{Hab}_{tail} \rangle$')

        # now plot the sra jumps sum - make sure x and y have the same dimensions
        sra_x = np.arange(window_size_sra / 2 + 0.5, 50 + 0.5, window_size_sra)
        ax_data_pjump.plot(sra_x[:len(movingavg_slide_M_T_sra)],
                           movingavg_slide_M_T_sra[:] / window_size_sra, '-o', alpha=1.0, c='k',
                           markersize=4)  # label='sra')
        # plot the moving avg for each hab stage
        if show_ma_each_hab_stage:
            for stage in range(5):
                hab_x = np.arange(window_size_hab / 2 + 0.5, 200 + 0.5, window_size_hab)
                ax_data_pjump.plot(hab_x[:len(movingavg_slide_M_T_hab_stage)],
                                   movingavg_slide_M_T_hab_stage[:, stage] / window_size_hab, '--', alpha=0.4, marker='s',
                                   markersize=2)

        # now plot the average over the 5 hab stages
        hab_mean_x = np.arange(window_size_hab / 2 + 0.5, 200 + 0.5, window_size_hab)
        ax_data_pjump.plot(hab_mean_x[:len(movingavg_slide_M_T_hab_mean)],
                           movingavg_slide_M_T_hab_mean[:] / window_size_hab, '-s', alpha=1.0, markersize=4,
                           c=hab_ma_colors[day],
                           label='hab mean')

        # Calculate the appropriate position for the sliding window MA
        t_arr_slide1 = np.arange(1 + (window_size_hab - 1) / 2, 201 - (window_size_hab - 1) / 2, 1)
        data_ma_slide1 = moving_average(p_hab_mean_timeseries_1pt, n=window_size_hab)

        # Make sure the arrays have the same length
        conf_interval_slide1 = get_conf_movingavg(data_ma_slide1, n=window_size_hab * 5, confidence_level=0.95,
                                                  method='exact')

        # Ensure arrays match in length before plotting
        min_len = min(len(t_arr_slide1), len(conf_interval_slide1))

        # use fill to plot conf interval with matching dimensions
        ax_data_pjump.fill_between(t_arr_slide1[:min_len],
                                   conf_interval_slide1[:min_len, 0],
                                   conf_interval_slide1[:min_len, 1],
                                   color=hab_ma_colors_fill[day], alpha=0.2,
                                   label=f'MA {window_size_hab}s1 95% CI')

        zoom_data_ma = False
        if zoom_data_ma:
            xticks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
            # ax_data_pjump.set_xticks(xticks, labels=['%d' % (i + 1) for i in xticks])
            ax_data_pjump.set_xticks(xticks)
            ax_data_pjump.set_xticklabels(['%d' % i for i in xticks])
            ax_data_pjump.set_xlim((-0.01, 51))
        else:
            # xticks = [10, 20, 30, 40, 50, 100, 150, 200]
            xticks = [10, 30, 50, 100, 150, 200]
            # ax_data_pjump.set_xticks(xticks, labels=['%d' % (i + 1) for i in xticks])
            ax_data_pjump.set_xticks(xticks)
            ax_data_pjump.set_xticklabels(['%d' % i for i in xticks])
            # ax_data_pjump.set_xlim((-0.01, 51))

        ax_data_pjump.set_ylim((-0.01, 1.01))

        if publication_style:
            ax_data_pjump.set_ylabel(r'$p_{jump}(k)$', fontsize=12)
            ax_data_pjump.set_xlabel('Stimulus index', fontsize=12)
            if day_idx == 0:
                ax_data_pjump.set_title(r'Data: Estimate $p_{\text{jump}}(k)$ by moving average', fontsize=14)
        else:
            ax_data_pjump.set_ylabel(r'estimate $p_{\text{jump}}(k)$', fontsize=14)
            if (day_idx == len(days_to_plot) - 1 and cols_to_rows) or (not cols_to_rows and day_idx == 0):
                ax_data_pjump.set_xlabel('Stimulus index', fontsize=14)
            if day_idx == 0:
                ax_data_pjump.set_title(r'Data: estimate $p_{\text{jump}}(k)$ by moving average',
                                        fontsize=14)  # $\langle j(t) \rangle$')

        # hide legend for now
        #if day_idx == 0:
        #    ax_data_pjump.legend(ncol=2, fontsize=10)

        # =================================================================
        # Plot 3: posterior likelihood with credible interval
        # =================================================================
        # Increase fontsize for tick labels
        ax_model_pjump.tick_params(axis='both', which='major', labelsize=14)

        # (2,1) now plot the model mean and variance
        # - model mean is <p(t)> = sum p_i / n
        # - model variance is <p(t)^2> - <p(t)>^2
        old_panel_style = False
        if old_panel_style:
            ax_model_pjump.plot(model_likelihood_sra, '--ok', alpha=1.0, label='model: SRA', markersize=3)
            ax_model_pjump.plot(model_likelihood_hab, '--o', alpha=1.0, label='model: Hab', markersize=3)
            ax_model_pjump.set_ylim((-0.01, 1.01))
            ax_model_pjump.set_title('Model: $p_{\mathrm{jump}}(k)$')
            ax_model_pjump.legend(fontsize=12)
        else:
            plot_posterior_likelihood_summary_over_days(
                draws_over_age, genotype, fly_id, days_to_show=[day],
                spaghetti=False, spaghetti_alpha=0.1, ci_level=0.95, n_samples=100,
                ax=ax_model_pjump)

            # Adjust the heatmap aspect ratio to make it wider
            if publication_style:
                ax_jump_heatmap.set_aspect(3)  # Increased aspect ratio to make the heatmap wider

                # Ensure the heatmap width exactly matches the combined bottom plots
                # This needs to be done after the plots are drawn
                fig.canvas.draw()

                # Get the position of the bottom plots
                pos_data = ax_data_pjump.get_position()
                pos_model = ax_model_pjump.get_position()

                # Get the position of the heatmap
                pos_heatmap = ax_jump_heatmap.get_position()

                # Set the heatmap position to match the combined width of bottom plots
                # Add a small margin on each side to ensure perfect alignment
                new_pos = [pos_data.x0 - 0.002, pos_heatmap.y0,
                           (pos_model.x1 - pos_data.x0) + 0.004, pos_heatmap.height]
                ax_jump_heatmap.set_position(new_pos)


            # When setting titles, adjust for the new layout
            if publication_style:
                # Set titles appropriate for the new layout
                ax_data_pjump.set_title(r'Data: Estimate $p_{\text{jump}}(k)$ by moving average', fontsize=13)
                ax_model_pjump.set_title("Model: Posterior predictive density", fontsize=13)

                ax_data_pjump.set_ylabel(r'$p_{jump}(k)$', fontsize=12)
                ax_data_pjump.set_xlabel('Stimulus index', fontsize=12)
                #ax_model_pjump.set_xlabel("Stimulus index", fontsize=12)
                #ax_model_pjump.set_ylabel(r"$p_{\text{jump}}(k)$", fontsize=12)

                # remove y ticks for model plot only
                ax_model_pjump.tick_params(axis='y', labelleft=False)  # Remove labels but keep ticks

                # Show parameter values more centered in the right plot
                # Position at x=0.65 (instead of 0.85) and y=0.75
                param_text = f"$\\alpha = {alpha:.2f}$\n$\\beta = {beta:.2f}$\n$p_0 = {p0:.2f}$"
                ax_model_pjump.text(0.4, 0.95, param_text, transform=ax_model_pjump.transAxes,
                                    fontsize=10, verticalalignment='top')

                # Make the bottom plots square-ish
                ax_data_pjump.set_box_aspect(0.7)
                ax_model_pjump.set_box_aspect(0.7)
            else:
                ax_model_pjump.set_ylabel(r"$p_{\text{jump}}(k)$", fontsize=14)
                # Original title setting logic remains
                if (day_idx == len(days_to_plot) - 1 and cols_to_rows) or (not cols_to_rows and day_idx == 0):
                    ax_model_pjump.set_xlabel("Stimulus index", fontsize=14)
                else:
                    # remove y ticks for right plots
                    for ax_to_mod in [ax_jump_heatmap, ax_data_pjump, ax_model_pjump]:
                        ax_to_mod.tick_params(axis='y', labelleft=False)  # Remove labels but keep ticks
                        ax_to_mod.set_ylabel('')
                if day_idx == 0:
                    ax_model_pjump.set_title(f"Posterior Density: Fly {fly_id} ({genotype})", fontsize=14)

        ax_model_pjump.legend(fontsize=12)

        # add text to the top right of plot relating to the \theta parameters if not in publication style
        if not publication_style and not slim_labeling:
            dx = -0.3
            text_kw = dict(fontsize=10, transform=ax_model_pjump.transAxes)
            ax_model_pjump.text(0.51 + dx, 0.90 + 0.02, r'$\alpha=%.2f$' % alpha, **text_kw)
            ax_model_pjump.text(0.51 + dx, 0.84 + 0.02, r'$\beta=%.2f$' % beta, **text_kw)
            ax_model_pjump.text(0.51 + dx, 0.78 + 0.02, r'$p_0=%.2f$' % p0, **text_kw)

            T1_hab_time_half_abs_str = 'inf' if np.isinf(T1_hab_time_half_abs) else '%.2f' % T1_hab_time_half_abs
            ax_model_pjump.text(0.71 + dx, 0.9 + 0.02, r'$S_1=%.2f$' % T1_hab_mag_rel, **text_kw)
            ax_model_pjump.text(0.71 + dx, 0.84 + 0.02, r'$S_2=%.2f$' % T1_hab_mag_abs, **text_kw)
            ax_model_pjump.text(0.71 + dx, 0.78 + 0.02, r'abs: $T_{1/2}=%s$' % T1_hab_time_half_abs_str, **text_kw)
            ax_model_pjump.text(0.71 + dx, 0.72 + 0.02, r'rel: $T_{1/2}=%.2f$' % T1_hab_time_half_rel, **text_kw)
            ax_model_pjump.text(0.71 + dx, 0.66 + 0.02, r'rel: $T_{sat}=%.2f$' % T1_hab_time_95_rel, **text_kw)

    if publication_style:
        # Don't use suptitle for publication style - we already added titles above
        pass
    elif cols_to_rows:
        plt.suptitle('Data vs. model for %s - fly ID: %s' % (label_expt, fly_id), fontsize=14, y=0.99)
    else:
        plt.suptitle('Data vs. model for %s - fly ID: %s' % (label_expt, fly_id), fontsize=14, y=0.93)

    # save the figure
    plt.savefig(DIR_OUTPUT + os.sep + 'indiv_%s_fly%d.png' % (label_expt, fly_id), dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'indiv_%s_fly%d.svg' % (label_expt, fly_id))
    if show:
        plt.show()

    return


if __name__ == '__main__':

    # load the parameters inferred from the fit
    # - note: the -habscores variant is generated by running data_format_add_score_columns.py
    df_summary_csv = pd.read_csv(DIR_FITS + os.sep + "fly-stability-days-detailed-3d-habscores.csv")

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

    # now get draws for specific fly
    genotype = 'KK'
    # KK:   repeatable (examples): 44, 43
    # KK:      erratic (examples): 105, 117, 53, 122

    # - note: publication_style built for single day_to_plot (e.g. like fig. 2, not "passport style" multi-day)
    # loop mode
    #for fly_id in range(1, 128 + 1):
    # choice mode
    for fly_id in [44, 105]:
        try:
            plot_multiday_one_fly(df_summary_csv, draws_over_age, genotype, fly_id, [7, 14, 21],
                                  cols_to_rows=False,
                                  ma5_hab=True,
                                  ma5_sra=False,
                                  publication_style=False,
                                  slim_labeling=True,
                                  ma_color_by_day=True,
                                  show=False)
        except AssertionError:
            print(f'Fly ID {fly_id} not found in summary CSV or data.')
            continue
