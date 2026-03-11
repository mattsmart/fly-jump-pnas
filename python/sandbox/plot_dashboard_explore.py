import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
#from torch.nn.functional import cross_entropy

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

from python.data_tools import filter_df_by_filterdict, df_to_arr_jumps
from python.fit_experimental_data import parse_fly_data, jump_prob, to_file
from python.plot_jump_data_vs_fit import plot_jumps, plot_full_data, simulate_from_fit
from python.functions_common import likelihood_func, moving_average, get_conf_movingavg
from python.plot_common import plot_posterior_likelihood_summary_over_days
from python.settings import DIR_DATA_EXPT, DIR_FITS, DIR_OUTPUT, DIR_STAN, days_palettes, heatmap_nodata, heatmap_0, heatmap_1

"""
Plot the results of a fit to the fly jump data.
- The fit is performed in the script `fit_experimental_data.py` and the results are saved in the directory `fits`.
- Main output file is `fly-stability-days-detailed.csv` which contains the inferred parameters for each fly.
- "Inferred parameters" refers to "Expectation over the posterior distribution of the parameters".

The script `plot_stan_results.py` loads the inferred parameters and plots the experimental data and the simulated data.
"""


def plot_3d_scatter_posterior_means(df, scatter_3d_shadow=True, quiver_traj_days=False):
    # 3d matplotlib scatterplot of the alpha, beta, p0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['alpha'], df['beta'], df['p0'])
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$p_0$')
    plt.title('Parameter estimates (from n=%d trajectories)' % len(df))
    plt.show()
    # repeat plot on the unconstrained scale
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['log_alpha'], df['log_beta'], df['logit_p0'])
    ax.set_xlabel(r'$\log(\alpha)$')
    ax.set_ylabel(r'$\log(\beta)$')
    ax.set_zlabel(r'$\mathrm{logit}(p_0)$')
    plt.title('[Unconstrained] Parameter estimates (from n=%d trajectories)' % len(df))
    plt.show()

    # 3d plotly scatterplot (can manipulate it, rotate + zoom etc.)
    fig = px.scatter_3d(df, x='alpha', y='beta', z='p0', color='genotype', symbol='day', opacity=0.8)
    # add a shadow on the alpha beta plane
    if scatter_3d_shadow:
        df['shadow_z'] = [df['p0'].min()] * len(df)
        shadow_trace = px.scatter_3d(df, x='alpha', y='beta', z='shadow_z', symbol='day')
        shadow_trace.update_traces(marker=dict(color='grey', opacity=0.5))
        for trace in shadow_trace.data:
            fig.add_trace(trace)
    fig.update_layout(title='Parameter estimates (from n=%d trajectories)' % len(df))
    fig.show()
    html_file = DIR_OUTPUT + os.sep + 'parameter_estimates_all.html'
    fig.write_html(html_file)
    ''' latex rendering is not working...
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    html_content = html_content.replace('<head>', f'<head>{mathjax_script}')
    with open(html_file, 'w', encoding='utf-8') as file:
        file.write(html_content)'''

    # repeat plot on the unconstrained scale
    fig = px.scatter_3d(df, x='log_alpha', y='log_beta', z='logit_p0', color='genotype', symbol='day')
    # add a shadow on the alpha beta plane
    if scatter_3d_shadow:
        #fig.add_trace(px.scatter_3d(df, x='log_alpha', y='log_beta', z=np.zeros(len(df)), color='genotype', symbol='day'))
        shadow_trace = px.scatter_3d(df, x='log_alpha', y='log_beta', z=[df['logit_p0'].min()]*len(df), symbol='day')
        shadow_trace.update_traces(marker=dict(color='grey', opacity=0.5))
        for trace in shadow_trace.data:
            fig.add_trace(trace)
    fig.update_layout(title='[Unconstrained] Parameter estimates (from n=%d trajectories)' % len(df))
    fig.show()
    html_file = DIR_OUTPUT + os.sep + 'parameter_estimates_all_unconstr.html'
    fig.write_html(html_file)

    # now do a specialized 3d scatterplot with only one genotype and ages different colors
    # TODO add an overlay where we show how some individuals moved from theta_7 -> theta_14 -> theta_21
    for genotype in ['KK', 'GD']:
        df_subset = filter_df_by_filterdict(df, {'genotype': [genotype]})
        df_subset['day'] = df_subset['day'].astype(str)  # create new df entry where day is a str not an int
        title = 'Parameter estimates for %s (from n=%d trajectories)' % (genotype, len(df_subset))
        # PLOT 1 - constrained =======================================================================================
        fig = px.scatter_3d(df_subset, x='alpha', y='beta', z='p0', color='day')
        fig.update_layout(title=title)
        if scatter_3d_shadow:
            df_subset['shadow_z'] = [df_subset['p0'].min()] * len(df_subset)
            shadow_trace = px.scatter_3d(df_subset, x='alpha', y='beta', z='shadow_z', color='day')
            shadow_trace.update_traces(marker=dict(color='grey', opacity=0.5))
            for trace in shadow_trace.data:
                fig.add_trace(trace)
        fig.show(); fig.write_html(DIR_OUTPUT + os.sep + 'parameter_estimates_gbr%s.html' % genotype)
        # PLOT 2 - now do 'unconstrained' version ====================================================================
        fig = px.scatter_3d(df_subset, x='log_alpha', y='log_beta', z='logit_p0', color='day')

        #fig = go.Figure()
        #px.scatter_3d(df_subset, x='log_alpha', y='log_beta', z='logit_p0', color='day')

        fig.update_layout(title='[Unconstrained] ' + title)
        if scatter_3d_shadow:
            df_subset['log_shadow_z'] = [df_subset['logit_p0'].min()] * len(df_subset)
            print('len(log_shadow_z df) vs len(log_alpha df) vs ', len(df_subset['log_shadow_z']), len(df_subset['log_alpha']))
            shadow_trace = px.scatter_3d(df_subset, x='log_alpha', y='log_beta', z='log_shadow_z', color='day')
            shadow_trace.update_traces(marker=dict(color='grey', opacity=0.5))
            for trace in shadow_trace.data:
                fig.add_trace(trace)

        # for fly ID [...11, 12] etc. in the df, plot the trajectory in 3d space using two arrows
        # - arrow one: day 7 -> day 14
        # - arrow two: day 14 -> day 21
        # Filter the dataframe for the specific fly ID
        if quiver_traj_days:
            for fly_id in [10, 11, 12, 13]:
                df_fly = df_subset[df_subset['fly'] == fly_id]

                # Ensure the dataframe is sorted by day explicitly using days 7, 14, 21
                df_fly['day'] = df_fly['day'].astype(int)
                df_fly = df_fly.sort_values(by='day')

                # Create a 3D scatter plot
                #fig = go.Figure()

                # Add arrows for the trajectory
                colors_days = ['blue', 'red', 'green']

                print(df_fly)
                for i in range(len(df_fly) - 1):
                    print(i)
                    print(i, df_fly.iloc[i]['day'])
                    print(colors_days[i], colors_days[i + 1])
                    traj_trace = go.Scatter3d(
                        # fig.add_trace(px.scatter_3d(
                        x=[df_fly.iloc[i]['log_alpha'], df_fly.iloc[i + 1]['log_alpha']],
                        y=[df_fly.iloc[i]['log_beta'], df_fly.iloc[i + 1]['log_beta']],
                        z=[df_fly.iloc[i]['logit_p0'], df_fly.iloc[i + 1]['logit_p0']],
                        mode='lines+markers',
                        marker=dict(size=5, color=[colors_days[i], colors_days[i + 1]]),
                        line=dict(color='red', width=2),
                        name='Trajectory %s to %s' % (df_fly.iloc[i]['day'], df_fly.iloc[i + 1]['day'])
                    )
                    #for trace in traj_trace.data:
                    fig.add_trace(traj_trace)

        fig.show(); fig.write_html(DIR_OUTPUT + os.sep + 'parameter_estimates_unconstr_gbr%s.html' % genotype)

    print('done plotting 3d scatters of posterior means')
    return


def plot_diag_ageshift_scatter(df):
    """
    Get day 7 and 14 and 21 data; make a 3x3 scatter plot of the inferred parameters for each fly
    - plot 1: alpha_7 vs alpha_14, alpha_14 vs alpha_21, alpha_7 vs alpha_21
    - plot 2: beta_7 vs beta_14, beta_14 vs beta_21, beta_7 vs beta_21
    - plot 3: p0_7 vs p0_14, p0_14 vs p0_21, p0_7 vs p0_21
    """
    for genotype in ['KK', 'GD']:
        df_genotype = filter_df_by_filterdict(df, {'genotype': [genotype]})

        # Get the data for each day
        df_day7 = filter_df_by_filterdict(df_genotype, {'day': [7]})
        df_day14 = filter_df_by_filterdict(df_genotype, {'day': [14]})
        df_day21 = filter_df_by_filterdict(df_genotype, {'day': [21]})

        # Make the scatter plots
        fig, axarr = plt.subplots(3, 3, figsize=(13, 13))

        ax_labels = [
            [r'$\alpha^7$', r'$\alpha^{14}$', r'$\alpha^{21}$'],
            [r'$\beta^7$',  r'$\beta^{14}$',  r'$\beta^{21}$'],
            [r'$p_0^7$',    r'$p_0^{14}$',    r'$p_0^{21}$']
        ]

        for i, kw in enumerate(['alpha', 'beta', 'p0']):
            for j, (df1, df2, label) in enumerate([
                (df_day7, df_day14,  'day7 vs day14'),
                (df_day14, df_day21, 'day14 vs day21'),
                (df_day7, df_day21,  'day7 vs day21')]):
                ax = axarr[i, j]
                sc = ax.scatter(df1[kw], df2[kw], s=6, c='k')
                if j < 2:
                    xlabel = ax_labels[i][j]
                    ylabel = ax_labels[i][j+1]
                else:
                    xlabel = ax_labels[i][0]
                    ylabel = ax_labels[i][2]
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if i == 0:
                    ax.set_title(label)  # only for first row
                # add a diagonal line to the plot, dashed black
                ax_min = min(np.min(df1[kw]), np.min(df2[kw]))
                ax_max = max(np.max(df1[kw]), np.max(df2[kw]))
                ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', lw=1)
        plt.suptitle('Inferred params age-shift scatter for %s' % genotype)
        plt.savefig(DIR_OUTPUT + os.sep + 'diag_ageshift_scatter_%s.png' % genotype, dpi=300)
        plt.show()


def plot_jumpmap_chunked_by_fly_and_age_tall(df_full, max_flies=16, figsize=(8, 4), show_yticks=False):
    gene_bgr_todo = ['GD', 'KK']
    vars_todo = [1, 2]

    fs_ylabels = 8  # figsize (8, 12) and 64 flies use fs=8
    HEATMAP_WIDTH = 5 * 200 + 50  # num_hab_cycles * n_pulse_hab + n_pulse_reactivity
    dataversion_xticks = [0, 200, 400, 600, 800, 1000]

    # cmap = mpl.colors.ListedColormap(['whitesmoke', 'pink', 'darkblue', 'orange'])  # no data (-1) | no jump (0) | jump (1)
    cmap = mpl.colors.ListedColormap([heatmap_nodata, heatmap_0, heatmap_1])  # no data (-1) | no jump (0) | jump (1)

    for gene_bgr in gene_bgr_todo:
        df_select_gene_bgr = filter_df_by_filterdict(df_full, dict(genotype=[gene_bgr]))

        for var_idx in vars_todo:
            df_select_gene_bgr_and_var = filter_df_by_filterdict(df_select_gene_bgr, dict(var=[var_idx]))
            fly_indices = list(df_select_gene_bgr_and_var['fly_id'].unique())
            print("fly_indices", fly_indices)

            unique_ages = list(df_select_gene_bgr_and_var['day'].unique())
            unique_ages.sort()
            print("unique_ages", unique_ages)
            num_ages = len(unique_ages)

            nrows, ncols = max_flies, 1
            assert nrows * ncols == max_flies

            fig, axarr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True)

            for k, fly_id in enumerate(fly_indices[0:max_flies]):

                i = k // ncols
                j = k % ncols
                #i = (fly_id - 1) // ncols
                #j = (fly_id - 1) % ncols

                df_select_fly = filter_df_by_filterdict(df_select_gene_bgr_and_var, dict(
                    fly_id=[fly_id]))  # df filtered now for gene_bgr -> var_idx -> idx_fly

                # we will assume fly age is in ascending order in the jumps array
                age_list = list(df_select_fly['day'])
                assert age_list == unique_ages[0: len(age_list)]

                print(i, j, k, 'fly_idx=%d | list_idx=%d | num_ages=%d' % (fly_id, k, len(age_list)))

                data_to_plot = np.zeros((num_ages, HEATMAP_WIDTH)) - 1  # -1 means "no data collected" (escape/dead)

                jumps = df_to_arr_jumps(df_select_fly, jump_col='jumpdata')
                M, n = jumps.shape
                print('Data shape', jumps.shape)
                # jumps_sums = np.sum(jumps, axis=0) / M
                # print('\tcolumn sums:', jumps_sums.shape)

                assert n == HEATMAP_WIDTH  # max timepoints sampled (jump or no jump)
                assert M <= num_ages  # max num. ages (days) sampled

                data_to_plot[0:M, :] = jumps

                # aspect = n/M
                # aspect = 25

                '''
                data_to_plot[2, :] = 17  # iomshow testing...
                data_to_plot[2, 1:3] = 0  # iomshow testing...
                data_to_plot[2, -3:-1] = 0  # iomshow testing...'''

                im1 = axarr[i, j].imshow(data_to_plot, aspect='auto', interpolation='None', cmap=cmap, vmin=-1,
                                         vmax=1)

                # for each row...
                if show_yticks:
                    yticks = np.arange(num_ages)[::3]
                    ytick_labels = unique_ages[::3]
                    if j == 0:
                        axarr[i, j].set_yticks(yticks + 0.5 * num_ages)
                        axarr[i, j].set_yticklabels(ytick_labels)
                        axarr[i, j].set_ylabel('%d' % fly_id, fontsize=fs_ylabels, rotation=0, labelpad=6.0)
                else:
                    if j == 0:
                        axarr[i, j].set_yticks([])
                        axarr[i, j].set_yticklabels([])
                        # move the ylabel a bit left, pad it
                        axarr[i, j].set_ylabel('%d' % fly_id, fontsize=fs_ylabels, rotation=20, labelpad=6.0)
                        # axarr[i, j].set_ylabel('%d' % fly_id, fontsize=fs_ylabels, rotation=0, loc='bottom', labelpad=8.0)  #def pad 4.0
                        # axarr[i, j].set_ylabel('%d' % fly_id, fontsize=fs_ylabels)  #def pad 4.0

                # for each col...
                xtick_labels = dataversion_xticks
                if i == nrows - 1:
                    axarr[i, j].set_xticks(dataversion_xticks)
                    axarr[i, j].set_xticklabels(xtick_labels)

                # change fontsize
                axarr[i, j].tick_params(axis='both', which='minor', labelsize=5)
                axarr[i, j].tick_params(axis='y', which='major', labelsize=5)

                # change all spines
                # im1.set_extent([-0.5, num_ages + 0.5, -0.5, HEATMAP_WIDTH + 0.5])
                for axis in ['top', 'bottom', 'left', 'right']:
                    # axarr[i, j].spines[axis].set_linewidth(0.2)
                    axarr[i, j].spines[axis].set_linewidth(0.05)
                # increase tick width
                axarr[i, j].tick_params(width=0.75)

            # plt.subplots_adjust(hspace=0.5)#), wspace=4)
            plt.xlabel(
                '%d ages: genotype=%s | var=%d | Up to %d flies shown' % (num_ages, gene_bgr, var_idx, max_flies))
            # plt.tight_layout()

            fbase = DIR_OUTPUT + os.sep + 'heatmap_chunk_by_fly_ages'
            fpath = fbase + '_tall_%s_var%d_nn%d' % (gene_bgr, var_idx, max_flies)
            # plt.tight_layout()
            plt.savefig(fpath + '.svg', bbox_inches="tight")
            plt.savefig(fpath + '.pdf', bbox_inches="tight")
            plt.savefig(fpath + '.jpg', dpi=300, bbox_inches="tight")
            plt.show()
    return


def plot_hab_scores(df):
    """
    Recall dynamic bernoulli model for jump events:
          p_jump(t) = p0 * sigma(beta (1 - q ** t) / (1-q) ),   where
              - q = exp(- alpha * delta_T)
              - sigma(x) = 1 / (1 + x ** 2) -- arbitrary monotonic 'link' function ensures that p_jump(t) is in [0, p0]
    Notes:
     - if alpha = 0 OR beta = 0, then p_jump(t) = p0 for all t  (i.e. no habituation, jumps are static coin flips ~ p0)
     - for alpha > 0, q -> 0 as t -> \infty,
        then (1)  sigma(\infty) = beta / (1-q)
             (2) p_jump(\infty) = p0 * sigma(\infty)
                                = p0 / (1 + [beta/ (1 - q)] ** 2)    for the choice sigma(x) = 1 / (1 + x^2)

    Here we define 4 "scores" (scalar maps from the 3D parameter space to R)
        - S_0 - x_stady_state "how full can the bucket get" -- x_ss = beta / (1 - exp(- alpha * T))
        - S_1 - hab magnitude (relative) -- 1 - p_jump(\infty) / p0            - function of alpha, beta
        - S_2 - hab magnitude (absolute) -- p0 - p_jump(\infty)                - function of p0, alpha, beta
        - S_3 - hab speed     (relative) -- time when p_jump(t) reaches p0 / 2 - function of alpha, beta
        - S_4 - hab speed     (absolute) -- time when p_jump(t) reaches 0.25   - function of p0, alpha, beta TODO

    These will depend on the choice of link function/static non-linearity sigma(x)

    Notes
    =======================================================================
         S_0 = x_ss
             = beta / (1 - q)

    - S_0 score in [0, \infty]
        - score = 0:         no habituation
        - score = 1:     'half' habituation
        - score = \infty:   max habituation

    Habituation score #1 - magnitude (relative) -- 1 - p_jump(\infty) / p0
    =======================================================================
        S_1 = 1 - p_jump(\infty) / p0
            = 1 - sigma(\infty)
            = beta^2 / (beta^2 + (1 - q)^2)

       S_1 score = 0 ->  no habituation (magnitude) -- e.g. beta = 0 OR alpha = 0
       S_1 score = 1 -> max habituation (magnitude) -- e.g. large beta  OR  small alpha

        Comment: the score is independent of p0, so it's a relative measure of habituation
                 (could do 2d plot, color by p0)

        Limiting cases
        - if alpha -> 0,      then S = 0       (i.e. no habituation)
        - if  beta -> 0,      then S = 0       (i.e. no habituation)
        - if alpha -> \infty, then S -> \infty (strong habituation)
        - if  beta -> \infty, then S -> \infty (strong habituation)

    *** In the case that p0 = 1, then S_1 is equal to the absolute habituation score S_2 ***

    Habituation score #2 - magnitude (absolute) -- p0 - p_jump(\infty)
    =======================================================================
         S_2 = p0 - p_jump(\infty)
             = p0 (1 - sigma(\infty))
             = p0 (1 - S1)

    - S_2 score in [0, 1]
        - score = 0: no habituation
        - score = 1: strong habituation

    Habituation score #3 - habituation speed A  -- time when p_jump(t) reaches p0 / 2
    =======================================================================
    - Note: this score could be undefined (NaN) if p_jump(t) never reaches p0 / 2
    - This score is independent of p0, so it's a "relative" measure of habituation speed (time to sigma = 0.5)

    If p_jump(t) = p0 / 2, then
        - p0 / 2 = p0 * sigma( x(t*))  <-->  1/2 = sigma(x(t*))
        - this implies x(t*) = 1 for the choice of sigma(x) = 1 / (1 + x^2)

    - x(t*) = beta (1 - q ** t*) / (1 - q) = 1
    - beta = (1 - q) / (1 - q ** t*)

    S_3 = t_half = log(1 - 2 * p0) / log(q)

    Habituation score #4 - habituation speed B  -- time when p_jump(t) reaches p0 - p_inf/2
    =======================================================================
    Note: his score could be undefined (NaN) if p_jump(t) = p0 for all t (i.e. alpha = 0 OR beta = 0)
    """

    df['log_x_ss'] = np.log(df['T1_x_ss_mean'])

    # Now merge some of the plots above into a single plot with 2 rows and 4 columns
    # =================================================================================
    # - 0,0 - alpha, beta scatter -> color by p0
    # - 0,1 - alpha, beta scatter -> color by x_ss
    # - 0,2 - alpha, beta scatter -> color by S1
    # - 0,3 - alpha, beta scatter -> color by S2
    # - 1,0 - same but log log
    # - 1,1 - same but log log
    # - 1,2 - same but log log
    # - 1,3 - same but log log
    # =================================================================================
    fig, axarr = plt.subplots(2, 6, figsize=(14, 5))

    # TODO add total num jumps too... need dataframe from expt for that, fly IDs etc.
    # TODO add contours
    row_info = [
        (df['p0'], 'p0', 'Spectral', r'$p_0$'),
        # (df['x_ss'], 'x_ss', 'Spectral', r'$x_{ss}$'),
        (df['log_x_ss'], 'log_x_ss', 'Spectral', r'$\log(x_{ss})$'),
        #(df['x_ss_subtract_x_200'], 'x_ss_subtract_x_200', 'Spectral', r'$x_{ss} - x_{200}$'),
        (df['T1_hab_mag_rel_mean'], 'hab_magnitude_rel', 'Spectral', r'$1 - p_{ss}/p_0$'),
        (df['T1_hab_mag_abs_mean'], 'hab_magnitude_abs', 'Spectral', r'$p_{ss} - p_0$'),
        (df['T1_hab_time_half_abs_mean'], 'hab_halftime (abs)', 'Spectral',
         r'$\frac{1}{\alpha}\log \left( \frac{x_{ss}}{x_{ss} - 1} \right)$'),
        (df['T1_hab_time_half_rel_mean'], 'hab_halftime (rel)', 'Spectral',
         r'$T_{1/2}$ (rel; mean)')
    ]
    for i, (df_data, df_key, cmap, subplot_label) in enumerate(row_info):
        ax = axarr[0, i]
        sc = ax.scatter(df['alpha'], df['beta'], c=df_data, s=10 * df['p0'], cmap=cmap)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.set_title(subplot_label)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(subplot_label)

        if df_key == 'T1_hab_time_half_abs_mean':
            # in this case, plot scatter in grey of points where x_ss < 1
            df_low_x_ss = df[df['x_ss'] < 1]
            ax.scatter(df_low_x_ss['alpha'], df_low_x_ss['beta'], s=10 * df_low_x_ss['p0'], alpha=0.5,
                       edgecolor='grey', facecolor='none', lw=0.75)

    # now repeat but row 2 is all log-log axis
    for i, (df_data, df_key, cmap, subplot_label) in enumerate(row_info):
        ax = axarr[1, i]
        sc = ax.scatter(df['log_alpha'], df['log_beta'], c=df_data, s=10 * df['p0'],
                        cmap=cmap)  # TODO np log of data or no?
        ax.set_xlabel(r'$\log(\alpha)$')
        ax.set_ylabel(r'$\log(\beta)$')
        ax.set_title(subplot_label)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(subplot_label)
        if df_key == 'hab_halftime':
            # in this case, plot scatter in grey of points where x_ss < 1
            df_low_x_ss = df[df['x_ss'] < 1]
            ax.scatter(df_low_x_ss['log_alpha'], df_low_x_ss['log_beta'], s=10 * df_low_x_ss['p0'], alpha=0.5,
                       edgecolor='grey', facecolor='none', lw=0.75)

    fig.subplots_adjust(wspace=0.9, hspace=0.7)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_scores_2x5.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_scores_2x5.svg')
    plt.show()

    # hab score S_0
    # =================================================================================
    # now scatter the points in df['x_ss'] on a 2d heatmap alpha, beta
    fig, ax = plt.subplots(figsize=(4, 3))

    # before scatter, first make heatmap and add the contour for x_ss = 1
    alpha = np.linspace(0, 0.7, 100)
    beta = np.linspace(0, 1.5, 100)
    alpha, beta = np.meshgrid(alpha, beta)
    x_ss_mesh = beta * np.exp(-alpha) / (1 - np.exp(-alpha))
    #c = ax.contourf(alpha, beta, x_ss_mesh, levels=100, cmap='Purples')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title(r'Bucket amount $x_{ss} = \beta / (1-e^{-\alpha \Delta})$')
    # plot the 1 contour line as dashed black line, label it x_ss=1
    cl = ax.contour(alpha, beta, x_ss_mesh, levels=[1], colors='black', linestyles='dashed', linewidths=2)
    ax.clabel(cl, fmt="%s", fontsize=8)

    sc = ax.scatter(df['alpha'], df['beta'], c=df['T1_x_ss_mean'], s=10 * df['p0'], cmap='Spectral')
    #ax.set_xlabel(r'$\alpha$')
    #ax.set_ylabel(r'$\beta$')
    #ax.set_title(r'Hab score $S_0$ - "how full is bucket"')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$x_{ss}$')
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_xss.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_xss.svg')
    plt.show()

    # now repeat plot on a log axis, including the colorbar score
    fig, ax = plt.subplots(figsize=(4, 3))

    sc = ax.scatter(df['log_alpha'], df['log_beta'], c=np.log(df['T1_x_ss_mean']), s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel(r'$\log(\alpha)$')
    ax.set_ylabel(r'$\log(\beta)$')
    ax.set_title(r'Bucket amount $x_{ss} = \beta / (1-e^{-\alpha \Delta})$')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$\log(x_{ss})$')
    # add the x_ss = 1.0 contour on the loglog plot
    # - recalculate x_ss_mesh for log-log plot
    log_alpha = np.linspace(np.log(0.01), np.log(0.7), 100)
    log_beta = np.linspace(np.log(0.01), np.log(1.5), 100)
    log_alpha, log_beta = np.meshgrid(log_alpha, log_beta)
    x_ss_mesh_log = np.exp(log_beta) * np.exp(-np.exp(log_alpha)) / (1 - np.exp(-np.exp(log_alpha)))
    cl = ax.contour(log_alpha, log_beta, x_ss_mesh_log, levels=[1], colors='black', linestyles='dashed', linewidths=2)
    ax.clabel(cl, fmt="%s", fontsize=8)

    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_xss_loglog.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_xss_loglog.svg')
    plt.show()

    # hab score S_1 - relative magnitude
    # =================================================================================
    # now scatter the points in df['hab_score'] on a 2d heatmap of the function sigma(\infty)
    # (i.e. the habituation score #1 - magnitude (relative))''''
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['alpha'], df['beta'], c=df['T1_hab_mag_rel_mean'], s=10*df['p0'], cmap='Spectral')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_title(r'Hab score $S_1$ - (rel.) magnitude')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$S_1$')
    fig.tight_layout()
    plt.show()

    # now don't plot the data,just plot a 2d heatmap of the function sigma(\infty)
    # (i.e. the habituation score #1 - magnitude (relative))
    # define a manual cmap for the heatmap - purples from 0 to 1.0
    nvals = 5  # 5 or None -- number of values in the colormap
    cmap = plt.cm.get_cmap('Purples', nvals)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    custom_cmap_purples = cmap.from_list('Custom cmap (Purples_r)', cmaplist, cmap.N)
    # define the bins and normalize
    norm = plt.Normalize(vmin=0, vmax=1)

    # make cbar have 5 categories
    # cmap = plt.cm.get_cmap('Purples_r', 5)

    alpha = np.linspace(0, 0.7, 100)
    beta = np.linspace(0, 1.5, 100)
    p0 = np.linspace(0.01, 1.0, 100)

    alpha, beta = np.meshgrid(alpha, beta)
    hab_score_S1_mesh = 1 - 1 / (1 + (beta * np.exp(-alpha)) ** 2 / (1 - np.exp(-alpha)) ** 2)

    fig, ax = plt.subplots(figsize=(4, 3))
    c = ax.contourf(alpha, beta, hab_score_S1_mesh,
                    levels=100, cmap=custom_cmap_purples, norm=norm, vmin=0.0, vmax=1.0)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_title(r'Hab magnitude (relative)')
    # plot the 0.5 contour line as dashed black line, label it \rho=0.5
    cl = ax.contour(alpha, beta, hab_score_S1_mesh, levels=[0.5], colors='black', linestyles='dashed',
                    linewidths=2)
    # ax.clabel(cl, fmt="%s", fontsize=8)
    ax.clabel(cl, fontsize=10, inline=True, fmt=r'$\rho=0.5$', inline_spacing=15, manual=[(0.5, 1.1)])

    # set bar label to S_1
    cbar = fig.colorbar(c, ax=ax, label=r'$S_1$')
    c.set_clim(0, 1)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    plt.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_magnitude_rel_noscatter.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_magnitude_rel_noscatter.svg')

    # colour the data scatterplot using spectral cmap on the value of df p0
    sc_data = ax.scatter(df['alpha'], df['beta'], c=df['p0'], cmap='Spectral', s=4)
    # ax.scatter(df['alpha'], df['beta'], c='black', alpha=0.5, s=4)
    # ax.scatter(df['alpha'], df['beta'], c='white', alpha=0.5, s=1)
    # add extra colorbar on bottom
    cbar = fig.colorbar(sc_data, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label(r'$p_0$')

    plt.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_magnitude_rel.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_magnitude_rel.svg')
    plt.show()

    # plot the habituation scores for each fly
    # - hab score T_1 - time to hit x(t)=1 <-> p_jump(t) = p0 / 2  [note: it can be NaN if x(t) never hits 1]
    # =================================================================================
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['alpha'], df['beta'], c=df['T1_hab_time_half_abs_mean'], s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    num_nan = np.sum(np.isnan(df['T1_hab_time_half_abs_mean']))
    ax.set_title(r'Habituation half-time $p(T_1)=p_0/2$' + '\n(%d NaN of %d)' % (num_nan, len(df)))
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$T_1$')
    # plot the points which are nan in a different color - grey
    is_nan = np.isnan(df['T1_hab_time_half_abs_mean'])
    sc_nan = ax.scatter(df['alpha'][is_nan], df['beta'][is_nan],
                        c='grey', s=10 * df['p0'][is_nan], alpha=0.5, zorder=1)
    # add a dashed black contour where T_1 = 200
    alpha = np.linspace(0, 0.7, 200)
    beta = np.linspace(0, 1.5, 200)
    alpha, beta = np.meshgrid(alpha, beta)
    x_ss = beta * np.exp(-alpha) / (1 - np.exp(-alpha))
    hab_score_T1_mesh = (1 / alpha) * np.log(x_ss / (x_ss - 1))
    cl = ax.contour(alpha, beta, hab_score_T1_mesh, levels=[1, 5, 10, 50],
                    colors='black', linestyles='dashed', linewidths=1)
    ax.clabel(cl, fmt="%s", fontsize=8)
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_halftime_abs.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_halftime_abs.svg')
    plt.show()

    # repeat now as a log log plot
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['log_alpha'], df['log_beta'], c=np.log(df['T1_hab_time_half_abs_mean']), s=10 * df['p0'], cmap='Spectral')
    #sc = ax.scatter(df['log_alpha'], df['log_beta'], c=df['hab_halftime'], s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel(r'$\log(\alpha)$')
    ax.set_ylabel(r'$\log(\beta)$')
    num_nan = np.sum(np.isnan(df['T1_hab_time_half_abs_mean']))
    ax.set_title(r'Habituation half-time $p(T_1)=p_0/2$' + '\n(%d NaN of %d)' % (num_nan, len(df)))
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$\log(T_1)$')
    #cbar.set_label(r'$T_1$')
    # plot the points which are nan in a different color - grey
    is_nan = np.isnan(df['T1_hab_time_half_abs_mean'])
    sc_nan = ax.scatter(df['log_alpha'][is_nan], df['log_beta'][is_nan],
                        s=10 * df['p0'][is_nan], alpha=0.5,
                        facecolor='none', edgecolor='grey', zorder=1, lw=0.75)
    # add a dashed black contour where T_1 = 200
    #log_alpha = np.linspace(np.log(0.01), np.log(0.7), 200)
    log_alpha = np.linspace(np.log(0.01), np.log(5.2), 400)
    log_beta = np.linspace(np.log(0.01), np.log(1.5), 400)
    log_alpha, log_beta = np.meshgrid(log_alpha, log_beta)
    x_ss = np.exp(log_beta) * np.exp(-np.exp(log_alpha)) / (1 - np.exp(-np.exp(log_alpha)))
    hab_score_T1_mesh = (1 / np.exp(log_alpha)) * np.log(x_ss / (x_ss - 1))
    cl = ax.contour(log_alpha, log_beta, hab_score_T1_mesh, levels=[1, 5, 10, 50],
                    colors='black', linestyles='dashed', linewidths=1, zorder=30)
    #manual_positions = [(0.1, 0.5), (0.4, 1.0), (0.6, 1.5)]
    manual_positions = [(-4.0, 0), (-4.0, -1.5), (-4.0, -2.5), (-4.0, -3.5)]
    ax.clabel(cl, fmt="%s", fontsize=8, manual=manual_positions)
    # add a grey contour where x_ss = 1; label is r'$x_{ss}=1$'
    cl = ax.contour(log_alpha, log_beta, x_ss, levels=[1],
                    colors='grey', linestyles='dashed', linewidths=1,
                    zorder=1)
    #ax.clabel(cl, fmt=r"$x_{ss}=1$", fontsize=8)
    ax.clabel(cl, fmt=r"", fontsize=8)
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_halftime_abs_loglog.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_halftime_abs_loglog.svg')
    plt.show()

    # make a plot where x = hab magnitude rel, y = hab magnitude abs, color = p0
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['T1_hab_mag_rel_mean'], df['T1_hab_mag_abs_mean'], c=df['p0'], s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel(r'$S_1 = 1-p_{ss}/p_0$')
    ax.set_ylabel(r'$S_2 = p_0 S_1$')
    ax.set_title(r'Hab magnitude (relative vs. absolute)')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$p_0$')
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_S2.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_S2.svg')
    plt.show()

    # repeat the plot above BUT color points by T_1; if NaN make them grey
    fig, ax = plt.subplots(figsize=(4, 3))
    # scatter all non-NaN points first using spectral cmap
    sc = ax.scatter(df['T1_hab_mag_rel_mean'], df['T1_hab_mag_abs_mean'], c=np.log(df['T1_hab_time_half_abs_mean']), s=10 * df['p0'], cmap='Spectral')
    # which points have NaN for T_1? scatter them grey
    is_nan = np.isnan(df['T1_hab_time_half_abs_mean'])
    sc_nan = ax.scatter(df['T1_hab_mag_rel_mean'][is_nan], df['T1_hab_mag_abs_mean'][is_nan], s=10 * df['p0'][is_nan],
                        facecolor='none', edgecolor='grey', alpha=0.5, zorder=1, lw=0.75)
    ax.set_xlabel(r'$S_1 = 1-p_{ss}/p_0$')
    ax.set_ylabel(r'$S_2 = p_0 \,S_1$')
    ax.set_title(r'Hab magnitude (relative vs. absolute)')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$\log(T_{1/2})$ (abs)')
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_S2_cT1.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_S2_cT1.svg')
    plt.show()

    # make a plot where x = hab magnitude rel, y = hab time
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['T1_hab_mag_rel_mean'], df['T1_hab_time_half_abs_mean'], c=df['p0'], s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel(r'$S_1$')
    ax.set_ylabel(r'$T_{1/2}$ (abs)')
    ax.set_title(r'Hab score $S_1$ vs $T_{1/2}$ abs.')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$p_0$')
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_Thalf_abs.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S1_vs_Thalf_abs.svg')
    plt.show()

    # make a plot where x = hab magnitude abs., y = hab time
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(df['T1_hab_mag_abs_mean'], df['T1_hab_time_half_abs_mean'], c=df['p0'], s=10 * df['p0'], cmap='Spectral')
    ax.set_xlabel(r'$S_2$')
    ax.set_ylabel(r'$T_{1/2}$ (abs)')
    ax.set_title(r'Hab score $S_2$ vs $T_{1/2}$ (abs)')
    cbar = fig.colorbar(sc, ax=ax)  # add cbar
    cbar.set_label(r'$p_0$')
    fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S2_vs_Thalf_abs.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_S2_vs_Thalf_abs.svg')
    plt.show()

    # repeat the 2 plots above to make a 2x3 grid of plots
    # - 0,0 - S1 vs S1 (SRA) - alpha
    # - 0,1 - S1 vs S1 (SRA) - beta
    # - 0,2 - S1 vs S1 (SRA) - p0
    # - 1,0 - S2 vs S2 (SRA) - alpha
    # - 1,1 - S2 vs S2 (SRA) - beta
    # - 1,2 - S2 vs S2 (SRA) - p0
    fig, axarr = plt.subplots(2, 3, figsize=(12, 6))
    for i, col in enumerate(['alpha', 'beta', 'p0']):
        # make a plot where x = hab magnitude rel., y = same but for SRA
        # - make 3 copies, one for each color {alpha, beta, p0}
        for row, hab_score in enumerate(['T1_hab_mag_rel_mean', 'T1_hab_mag_abs_mean']):
            sc = axarr[row, i].scatter(df[hab_score], df['T5' + hab_score[2:]], c=df[col], s=10 * df['p0'], cmap='Spectral')
            axarr[row, i].set_xlabel(r'$S_{%d, T=1}$' % (row+1))
            axarr[row, i].set_ylabel(r'$S_{%d, T=5}$ (SRA)' % (row+1))
            axarr[row, i].set_title(r'Hab magnitude $S_%d$ vs $S_%d$ (SRA)' % (row+1, row+1))
            cbar = fig.colorbar(sc, ax=axarr[row, i])
            cbar.set_label(col)
            # add a 1:1 line (dashed, black)
            # - make the extent equal to the largest extent of the data | keep plot square
            data_max = np.max([df[hab_score].max(), df['T5' + hab_score[2:]].max()])
            extent_xy = min(1.0, data_max * 1.1)
            axarr[row, i].plot([0, extent_xy], [0, extent_xy], '--k', lw=1)
            fig.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_T1_vs_T5(SRA)_multicbar.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_T1_vs_T5(SRA)_multicbar.svg')
    plt.show()

    # make same 2x3 plot as above, but now compute the S1, S2 scores after 50 stimuli instead of at steady-state
    fig, axarr = plt.subplots(2, 3, figsize=(12, 6))
    for i, col in enumerate(['alpha', 'beta', 'p0']):
        # make a plot where x = hab magnitude rel., y = same but for SRA
        # - make 3 copies, one for each color {alpha, beta, p0}
        for row, hab_score in enumerate(['T1_hab_mag_rel_mean', 'T1_hab_mag_abs_mean']):
            if row == 0:
                den_term_T1 = df['beta'] * np.exp(-df['alpha'] * 1) * (1 - np.exp(-df['alpha'] * 1 * 50)) / (1 - np.exp(-df['alpha'] * 1))
                den_term_T5 = df['beta'] * np.exp(-df['alpha'] * 5) * (1 - np.exp(-df['alpha'] * 5 * 50)) / (1 - np.exp(-df['alpha'] * 5))
                df[hab_score + '_50']          = 1 - 1 / (1 + den_term_T1 ** 2)
                df['T5' + hab_score[2:] + '_50'] = 1 - 1 / (1 + den_term_T5 ** 2)
            else:
                df[hab_score + '_50']          = df['p0'] * df['T1_hab_mag_rel_mean_50']
                df['T5' + hab_score[2:] + '_50'] = df['p0'] * df['T5_hab_mag_rel_mean_50']
            sc = axarr[row, i].scatter(df[hab_score + '_50'], df['T5' + hab_score[2:] + '_50'],
                                       c=df[col], s=10 * df['p0'], cmap='Spectral')
            axarr[row, i].set_xlabel(r'$S_{%d, T=1}^{(50)}$' % (row+1))
            axarr[row, i].set_ylabel(r'$S_{%d, T=5}^{(50)}$ (SRA)' % (row+1))
            axarr[row, i].set_title(r'Hab magnitude $S_%d$ vs $S_%d$ (SRA)' % (row+1, row+1))
            cbar = fig.colorbar(sc, ax=axarr[row, i])
            cbar.set_label(col)
            # add a 1:1 line (dashed, black)
            # - make the extent equal to the largest extent of the data | keep plot square
            data_max = np.max([df[hab_score + '_50'].max(), df['T5' + hab_score[2:] + '_50'].max()])
            extent_xy = min(1.0, data_max * 1.1)
            axarr[row, i].plot([0, extent_xy], [0, extent_xy], '--k', lw=1)
            fig.tight_layout()
    plt.suptitle('Hab vs. SRA dropoff after 50 stimuli')
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_T1_vs_T5(SRA)_50notInf_multicbar.png', dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_T1_vs_T5(SRA)_50notInf_multicbar.svg')
    plt.show()

    # 2025, Feb. - make scatter where y=magnitude of dropoff, x=half-time of dropoff, color = p0
    df_KK_14 = filter_df_by_filterdict(df, {'genotype': ['KK'], 'day': [14]})
    df_choice = df_KK_14
    y_choices = [
        ('T1_hab_mag_rel_', r'$S_1$ (rel. magnitude)', r'KK_a14: Hab score $T_{1/2}$ (rel) vs $S_1$ (rel)'),
        ('T1_hab_mag_abs_', r'$S_2$ (abs. magnitude)', r'KK_a14: Hab score $T_{1/2}$ (rel) vs $S_2$ (abs)'),
    ]
    for idx in range(2):
        yvals_str, ylabel, title = y_choices[idx]
        print('TODO annotate uncertainty in the data')


        fig, ax = plt.subplots(figsize=(4, 3))
        sc = ax.scatter(df_choice['T1_hab_time_half_rel_mean'], df_choice[yvals_str + 'mean'], c=df_choice['p0'],
                        s=10 * df_choice['p0'],
                        cmap='Spectral_r',
                        #edgecolor='white', linewidth=0.25,
                        edgecolor='k', linewidth=0.25,
                        zorder=5)
        # add cross-style errorbars
        ax.errorbar(df_choice['T1_hab_time_half_rel_mean'], df_choice[yvals_str + 'mean'],
                    xerr=[df_choice['T1_hab_time_half_rel_mean'] - df_choice['T1_hab_time_half_rel_CI_lower'],
                          df_choice['T1_hab_time_half_rel_CI_upper'] - df_choice['T1_hab_time_half_rel_mean']],  # Asymmetric x error
                    yerr=[df_choice[yvals_str + 'mean'] - df_choice[yvals_str + 'CI_lower'],
                          df_choice[yvals_str + 'CI_upper'] - df_choice[yvals_str + 'mean']],  # Asymmetric y error
                    fmt='o', ecolor='black',
                    markersize=0,
                    capsize=1.5, elinewidth=0.5, alpha=0.5, zorder=2)

        ax.set_xlabel(r'$T_{1/2}$ (trials to half-saturate)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        cbar = fig.colorbar(sc, ax=ax)  # add cbar
        cbar.set_label(r'$p_0$')
        fig.tight_layout()

        plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_Thalf_rel_vs_S%d_KK14.png' % (idx+1), dpi=300)
        plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_Thalf_rel_vs_S%d_KK14.svg' % (idx+1))

        plt.xlim(-0.5, 25)
        plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_Thalf_rel_vs_S%d_KK14_zoom.png' % (idx + 1), dpi=300)
        plt.savefig(DIR_OUTPUT + os.sep + 'hab_score_Thalf_rel_vs_S%d_KK14_zoom.svg' % (idx + 1))

        plt.show()

    print('done plotting habituation scores')
    return


def plot_likelihood_timeseries(df):

    # 1) make an example plot first - handpick the params
    alpha = 0.1
    beta = 0.2
    p0 = 0.6

    trange_hab = np.arange(0, 200)
    trange_reactivity = np.arange(0, 50)
    likelihood_pjump_hab = likelihood_func(trange_hab,        alpha, beta, p0,  1.0)
    likelihood_pjump_sra = likelihood_func(trange_reactivity, alpha, beta, p0, 5.0)

    plt.figure(figsize=(4, 3))
    plt.title(r'Example $p_\mathrm{jump}$ timeseries for $(\alpha, \beta, p_0)=(%.2f, %.2f, %.2f)$' % (alpha, beta, p0))
    plt.plot(trange_hab, likelihood_pjump_hab, '--ob', label=r'$\Delta T = 1s$')
    plt.plot(trange_reactivity, likelihood_pjump_sra, '--ok', label=r'$\Delta T = 5s$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_\mathrm{jump}$')
    plt.legend()
    plt.xlim(0, 30)
    plt.show()

    # 2) plot examples for handful of flies (e.g. KK, age 14) - hab data (T=1s)
    # =================================================================================
    gbr = 'KK'
    day = 14

    plt.figure(figsize=(4, 3))
    df_subset = filter_df_by_filterdict(df, {'genotype': [gbr], 'day': [day]})
    print(df_subset.head())
    # plot the likelihood timeseries for each fly
    global_mean_sim = np.zeros(200)
    for i, row in df_subset.iterrows():
        alpha = row['alpha']
        beta = row['beta']
        p0 = row['p0']  # row['p0']  ||  set to 1.0 to see relative dropoff (ignore p0)
        likelihood_pjump_hab = likelihood_func(trange_hab, alpha, beta, p0, 1.0)
        global_mean_sim += likelihood_pjump_hab
        #plt.plot(trange_hab, likelihood_pjump_hab, '-o', markersize=1, alpha=0.3, lw=0.5)  #label='fly %d' % row['fly'],
        plt.plot(trange_hab, likelihood_pjump_hab, '-', alpha=0.5, lw=0.5)

    # now add a curve corresponding to their mean
    global_mean_sim /= len(df_subset)
    plt.plot(trange_hab, global_mean_sim, '-k', label='model: mean', lw=3, zorder=20)
    plt.title('Data for %s, day %d (T=1s) vs. inferred $p_\mathrm{jump}$ timeseries' % (gbr, day))

    # 3) overlay data mean(s) on the plot; extract from 'jumpdata' column of the df
    jumpdata_arr = df_to_arr_jumps(df_subset, jump_col='jumpdata')
    # plot mean over the 5 hab trials as sep curves
    global_mean = np.zeros(200)

    # stage to color:
    # 5 shades of blue
    # blue 1 = 0-200
    # blue 2 = 200-400
    # blue 3 = 400-600
    # blue 4 = 600-800
    # blue 5 = 800-1000
    stage_to_color = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']

    for k in range(5):
        hab_200_k = jumpdata_arr[:, 200*k:200*(k+1)]
        hab_200_k_mean = np.mean(hab_200_k, axis=0)
        if k == 0:
            label = 'data: 5x Hab means'
            # label='data: Hab%d' % (k+1)
        else:
            label = None
        #plt.plot(trange_hab, hab_200_k_mean, '--b', label=label, lw=1)
        plt.plot(trange_hab, hab_200_k_mean, '-b', label=label, lw=1, alpha=0.5)
        #plt.plot(trange_hab, hab_200_k_mean, '--', c=stage_to_color[k], label=label, lw=1)

        global_mean += hab_200_k_mean
    # plot data: global mean over 5 hav stages
    global_mean /= 5
    #plt.plot(trange_hab, global_mean, '--k', label='data: Sum', lw=1, zorder=19)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_\mathrm{jump}$')
    #plt.xlim(0, 200)
    plt.xlim(0, 35) #plt.xlim(0, 15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'likelihood_timeseries_%s_day%d.png' % (gbr, day), dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'likelihood_timeseries_%s_day%d.svg' % (gbr, day))
    plt.show()

    # repeat 2) and 3) plot above -- but now for the last 50 trials where T=5s (SRA)
    # =================================================================================
    plt.figure(figsize=(4, 3))
    # - plot the likelihood timeseries for each fly
    global_mean_sim = np.zeros(50)
    for i, row in df_subset.iterrows():
        alpha = row['alpha']
        beta = row['beta']
        p0 = row['p0']  # row['p0']  ||  set to 1.0 to see relative dropoff (ignore p0)
        likelihood_pjump_sra = likelihood_func(trange_reactivity, alpha, beta, p0, 5.0)
        global_mean_sim += likelihood_pjump_sra
        #plt.plot(trange_reactivity, likelihood_pjump_sra, '-o', markersize=1, alpha=0.3, lw=0.5)  #label='fly %d' % row['fly'],
        plt.plot(trange_reactivity, likelihood_pjump_sra, '-', alpha=0.5, lw=0.5)

    # now add a curve corresponding to their mean
    global_mean_sim /= len(df_subset)
    plt.plot(trange_reactivity, global_mean_sim, '-k', label='model: mean', lw=3, zorder=20)
    plt.title('Data for %s, day %d (T=5s SRA) vs. inferred $p_\mathrm{jump}$ timeseries' % (gbr, day))

    # 3) overlay data mean on the plot; extract from 'jumpdata' column of the df
    jumpdata_arr = df_to_arr_jumps(df_subset, jump_col='jumpdata')
    sra_50_allflies = jumpdata_arr[:, -50:]
    sra_50_mean = np.mean(sra_50_allflies, axis=0)
    label = 'data: SRA mean'
    #plt.plot(trange_reactivity, sra_50_k_mean, '--b', label=label, lw=1)
    plt.plot(trange_reactivity, sra_50_mean, '-b', label=label, lw=1, alpha=0.5)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$p_\mathrm{jump}$')
    plt.xlim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT + os.sep + 'likelihood_timeseries_SRA_%s_day%d.png' % (gbr, day), dpi=300)
    plt.savefig(DIR_OUTPUT + os.sep + 'likelihood_timeseries_SRA_%s_day%d.svg' % (gbr, day))
    plt.show()

    print('done plotting likelihood timeseries')
    return


def score_cross_entropy(indiv_observations, model_prob, return_arr=False):
    """
    Calculate a scalar score for an individual model (one fly on one day)
      indiv_observations - binary observation timeseries 1:T
      model_prob         - model probability timeseries  1:T

    For each fly, we fit on the following info:
    - single day {7, 14, or 21} across 6 stages (hab1-5, reactivity)

    Score:
    """
    # compute cross entropy
    # - cross entropy = - sum_t (y_t log(p_t) + (1 - y_t) log(1 - p_t))
    tols = 1e-9
    model_prob = model_prob.astype(np.float64)

    model_prob = np.clip(model_prob, tols, 1 - tols)             # clip model prob to avoid log(0)
    term_0 = (1 - indiv_observations) * np.log(1 - model_prob)   # (1-x) * log (1-p)
    term_1 = indiv_observations * np.log(model_prob)             #     x * log p
    if return_arr:
        cross_entropy = -(term_0 + term_1)
    else:
        # compress to scalar via mean
        cross_entropy = -np.mean(term_0 + term_1, axis=0)

    return cross_entropy


def model_performance_population(df):
    print('model_performance_population(...) - WIP')

    nexpt = len(df)
    # get all jumpdata from df and convert to np
    jumpdata_arr = df_to_arr_jumps(df, jump_col='jumpdata')
    jumpdata_arr = jumpdata_arr.astype(np.float64)  # ensure precision for calculating cross-entropy
    assert jumpdata_arr.shape[0] == nexpt   # expected shape is len(df) x 1050
    assert jumpdata_arr.shape[1] == 1050    # expected shape is len(df) x 1050

    score_model_global = 0
    score_model_per_expt = np.zeros(nexpt)
    score_model_per_expt_per_stage = np.zeros((nexpt, 6))
    # five baselines: 0.1, 0.5, 0.9, p ~ num_jumps (each stage), p ~ num_jumps (SRA only)
    score_dummy_per_expt_per_stage = np.zeros((nexpt, 6, 5))

    # if True, then fore the 5x hab stages just look at the first T trials (e.g. T=50)
    flag_focus_first_T_trials = True
    first_T_trials = 30

    # for each expt (e.g. KK, age 14, fly 74), calculate the score of the model against baselines
    stage_to_label = ['hab1', 'hab2', 'hab3', 'hab4', 'hab5', 'sra']

    for k, row in df.iterrows():
        # get the data for the fly
        genotype = row['genotype']
        day = row['day']
        fly_id = row['fly']
        jumpdata = jumpdata_arr[k, :]
        alpha = row['alpha']
        beta = row['beta']
        p0 = row['p0']
        print('fly %s, genotype %s, day %d, score: ...' % (fly_id, genotype, day))
        # simulate the model
        model_prob_hab = likelihood_func(np.arange(0, 200), alpha, beta, p0, 1.0)
        model_prob_sra = likelihood_func(np.arange(0, 50), alpha, beta, p0, 5.0)
        # calculate the score for the model
        for stage in range(6):
            if stage == 5:
                data_target = jumpdata[1000:]
                model_pred = model_prob_sra
            else:
                data_target = jumpdata[200 * stage: 200 * (stage + 1)]
                model_pred = model_prob_hab
                if flag_focus_first_T_trials:
                    data_target = data_target[:first_T_trials]
                    model_pred = model_pred[:first_T_trials]

            # calculate the score for the model
            stage_score_model             = score_cross_entropy(data_target, model_pred)
            # calculate the score on several baselines
            nn = data_target.shape[0]
            pguess_stage = np.sum(data_target)     / nn
            pguess_SRA   = np.sum(jumpdata[1000:]) / 50  # note these are the same for the SRA stage

            stage_score_dummy_coin0       = score_cross_entropy(data_target, 0.1 * np.ones(nn))
            stage_score_dummy_coinfair    = score_cross_entropy(data_target, 0.5 * np.ones(nn))
            stage_score_dummy_coin1       = score_cross_entropy(data_target, 0.9 * np.ones(nn))
            stage_score_dummy_coin_pguess_per_stage = score_cross_entropy(data_target, pguess_stage * np.ones(nn))
            stage_score_dummy_coin_pguess_SRA       = score_cross_entropy(data_target, pguess_SRA   * np.ones(nn))
            # store the scores
            score_model_per_expt_per_stage[k, stage] = stage_score_model
            score_dummy_per_expt_per_stage[k, stage, 0] = stage_score_dummy_coin0
            score_dummy_per_expt_per_stage[k, stage, 1] = stage_score_dummy_coinfair
            score_dummy_per_expt_per_stage[k, stage, 2] = stage_score_dummy_coin1
            score_dummy_per_expt_per_stage[k, stage, 3] = stage_score_dummy_coin_pguess_per_stage
            score_dummy_per_expt_per_stage[k, stage, 4] = stage_score_dummy_coin_pguess_SRA

            print('\t %s %d/6 - njumps of nn - %d of %d' % (stage_to_label[stage], stage+1, np.sum(data_target), nn))
            print('\t %s %d/6 - stage_score_model: %.2f' % (stage_to_label[stage], stage+1, stage_score_model))
            print('\t %s %d/6 - score_dummy_per_expt_per_stage' % (stage_to_label[stage], stage+1), score_dummy_per_expt_per_stage[k, stage, :])
            print('\t %s %d/6 - pguess_SRA' % (stage_to_label[stage], stage+1), pguess_SRA)

        # sum scores for that fly on given day
        #expt_score = (hab_score_1 + hab_score_2 + hab_score_3 + hab_score_4 + hab_score_5 + sra_score) / 6.0
        #print(hab_score_1, hab_score_2, hab_score_3, hab_score_4, hab_score_5, sra_score)
        # sum the scores
        #print('\tscore: %.2f' % (expt_score))
        #score_per_fly[k] = expt_score

    score_model_per_expt[:] = np.mean(score_model_per_expt_per_stage, axis=1)
    score_dummy_per_expt = np.mean(score_dummy_per_expt_per_stage, axis=1)
    print('score_dummy_per_expt.shape', score_dummy_per_expt.shape)

    label_pguess = r'$\hat p = \sum x_i / n$'

    # calculate bins
    bin_edges = np.histogram_bin_edges(score_model_per_expt_per_stage.flatten(), bins=20)
    bins = bin_edges  # before was doing 20 everywhere

    for idx in range(6):
        plt.hist(score_model_per_expt_per_stage[:, idx], bins=bins, alpha=0.5,    label='%s - model' % stage_to_label[idx])
        plt.hist(score_dummy_per_expt_per_stage[:, idx, 3], bins=bins, alpha=0.5, label='guess: coin-per-stage %s' % (label_pguess))
        plt.axvline(-np.log(0.5), color='black', linestyle='dashed', lw=1,        label='guess: fair coin')
        plt.ylabel('frequency')
        plt.xlabel('cross-entropy score')
        plt.legend()
        plt.title('Histogram of model scores for the 6 expt stages (M=%d)\n(each score is 200 or 50 trials)' % (nexpt))
        plt.show()

    # now repeat the 6 plots above as a 2x3 subplots
    fig, axarr = plt.subplots(2, 3, figsize=(12, 6))
    for i in range(6):
        row, col = i // 3, i % 3
        ax = axarr[row, col]
        ax.hist(score_model_per_expt_per_stage[:, i], bins=bins, alpha=0.5,    label='dynamic model')
        ax.hist(score_dummy_per_expt_per_stage[:, i, 3], bins=bins, alpha=0.5, label='guess: coin-per-stage %s' % label_pguess)
        ax.hist(score_dummy_per_expt_per_stage[:, i, 4], bins=bins, alpha=0.5, label='guess: coin-SRA  %s' % label_pguess)
        ax.axvline(-np.log(0.5), color='black', linestyle='dashed', lw=1,    label='guess: fair coin')
        ax.set_ylabel('frequency')
        ax.set_xlabel('cross-entropy score')
        ax.set_title(stage_to_label[i])
        if i == 0:
            ax.legend()
    plt.suptitle('Histogram of model scores for the 6 expt stages (M=%d)\n(each score is 200 or 50 trials, truncate_hab=%s)'
                 % (nexpt, flag_focus_first_T_trials))
    fig.tight_layout()
    plt.show()

    plt.hist(score_model_per_expt,       bins=bins, alpha=0.5, zorder=20, label='dynamic model')
    plt.hist(score_dummy_per_expt[:, 3], bins=bins, alpha=0.5, zorder=30, label='guess: coin-per-stage')
    plt.hist(score_dummy_per_expt[:, 4], bins=bins, alpha=0.5, zorder=10, label='guess: coin-SRA')
    plt.axvline(-np.log(0.5), color='black', linestyle='dashed', lw=1, label='model: fair coin')
    plt.ylabel('frequency')
    plt.xlabel('cross-entropy score')
    plt.legend()
    plt.title('Histogram of model scores for each expt (M=%d)\n(each score is 1050 trials, truncate_hab=%s)'
              % (nexpt, flag_focus_first_T_trials))
    plt.show()

    plt.hist(score_dummy_per_expt[:, 0], bins=bins, alpha=0.6, label='coin 0.1')
    plt.hist(score_dummy_per_expt[:, 1], bins=bins, alpha=0.6, label='coin 0.5')
    plt.hist(score_dummy_per_expt[:, 2], bins=bins, alpha=0.6, label='coin 0.9')
    plt.hist(score_dummy_per_expt[:, 3], bins=bins, alpha=0.6, label='coin pguess-per-stage')
    plt.hist(score_dummy_per_expt[:, 4], bins=bins, alpha=0.6, label='coin pguess-SRA')
    plt.axvline(-np.log(0.5), color='black', linestyle='dashed', lw=1, label='model: fair coin')
    plt.ylabel('frequency')
    plt.xlabel('cross-entropy score')
    plt.legend()
    plt.title('Histogram of baseline scores for each expt (M=%d)\n(each score is 1050 trials, truncate_hab=%s)'
              % (nexpt, flag_focus_first_T_trials))
    plt.show()

    print('TODO model_performance_population')
    # TODO could also add a column to the df with the scores for each fly on given day (e.g. KK age 14 fly 17...)
    return


def plot_model_performance_exemplars(df, convert_pm1=True, show_legend=True):
    """
    plot 1:
    - x axis is time index from 0 to 200 and y axis is cumulative number of jumps
    plot 2:
    - x axis is time index from 0 to 200 and y axis is moving average of jumps (10 trials)

    Notes
    - repeat for each fly, each experiment
    - each line is one fly in one experiment (e.g. GD age 14, Hab 1 of 5)

    convert_pm1 = True  # if True, convert to 0, 1 ---> -1, 1
    """
    # TODO for each fly, make a multipanel figure with raw data top (6 rows, 5x hab + SRA heatmap)
    #  - add cross entropy score of model vs baselines (text or bars)
    #  - add cumsum panel         - data left, model right
    #  - add moving average panel - data left, model right
    #     in data plots, overlay the multistage mean and the population mean for that expt
    #  - instead of -1+1 cumsum, do an empirical mean and variance <j(t)> vs model <p(t)>
    # TODO maybe this ^^^ should be done in the other script, plot_stan_results.py
    # - TODO 1) in the plot below, could add an overlay related to the population mean (for SRA and for all hab stages)
    # - TODO 2) make same plots but using the model mean and thus expected bernboulli variance?) for each time point
    df_merge_GD_7 = filter_df_by_filterdict(df, {'genotype': ['GD'], 'day': [7]})
    df_merge_GD_14 = filter_df_by_filterdict(df, {'genotype': ['GD'], 'day': [14]})
    df_merge_GD_21 = filter_df_by_filterdict(df, {'genotype': ['GD'], 'day': [21]})

    df_merge_KK_7 = filter_df_by_filterdict(df, {'genotype': ['KK'], 'day': [7]})
    df_merge_KK_14 = filter_df_by_filterdict(df, {'genotype': ['KK'], 'day': [14]})
    df_merge_KK_21 = filter_df_by_filterdict(df, {'genotype': ['KK'], 'day': [21]})

    for df_expt, label_expt in [
        (df_merge_KK_7, 'KK_a7'), (df_merge_KK_14, 'KK_a14'), (df_merge_KK_21, 'KK_a21'),
        (df_merge_GD_7, 'GD_a7'), (df_merge_GD_14, 'GD_a14'), (df_merge_GD_21, 'GD_a21')]:

        # get an array of chamber IDs from dataframe
        #chamber_groups = df_expt['chamber_group'].values

        # cmap = plt.get_cmap('tab20')
        # colors_chamber = [cmap(i)[0:3] for i in chamber_groups]  # strip alpha
        cmap = ['#44AA99', '#AA4499', '#999933', '#FF3D00']  # color for chambers 0, 1, 2, 3
        #colors_chamber = [cmap[i] for i in chamber_groups]  # strip alpha (currently unused)

        jump_data_full = df_to_arr_jumps(df_expt, jump_col='jumpdata')
        jump_data_full_pm1 = jump_data_full * 2 - 1  # convert to -1, 1

        fly_col    = df_expt['fly'].values
        fly_id_col = df_expt['fly_id'].values  # this is the ID we preserve across days
        alpha_col  = df_expt['alpha'].values
        beta_col   = df_expt['beta'].values
        p0_col     = df_expt['p0'].values

        val_genotype = df_expt['genotype'].values[0]
        val_day      = df_expt['day'].values[0]
        assert val_genotype == label_expt[0:2]
        assert val_day      == int(label_expt[4:])

        T1_hab_mag_rel_col = df_expt['T1_hab_mag_rel_mean'].values  # NOTE: for each, have median, std, CI_lower/upper
        T1_hab_mag_abs_col = df_expt['T1_hab_mag_abs_mean'].values
        T1_hab_time_half_rel_col = df_expt['T1_hab_time_half_rel_mean'].values
        T1_hab_time_95_rel_col = df_expt['T1_hab_time_95_rel_mean'].values
        T1_hab_time_half_abs_col = df_expt['T1_hab_time_half_abs_mean'].values   # can be NaN
        #T5 ... (same as above)

        # we will fill in these arrays for each expt day (e.g. KK, age 14)
        cumulative_jumps_M_T_hab_stage = np.zeros((jump_data_full.shape[0], 200, 5))
        cumulative_jumps_M_T_sra       = np.zeros((jump_data_full.shape[0], 50))

        movingavg_10slide10_M_T_hab_stage = np.zeros((jump_data_full.shape[0], 20, 5))  # note T -> T/10
        movingavg_10slide10_M_T_hab_mean  = np.zeros((jump_data_full.shape[0], 20))
        movingavg_10slide10_M_T_sra       = np.zeros((jump_data_full.shape[0], 5))      # note T -> T/10

        jumpmean_est_M_T_hab_stage = np.zeros((jump_data_full.shape[0], 200, 5))
        jumpmean_est_M_T_sra       = np.zeros((jump_data_full.shape[0], 50))

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
                cumulative_jumps_M_T_hab_stage[:, :npts, stage] = jump_data_for_cumsum.cumsum(axis=1)
                # for the jumpmean_est_ use 'for_ma' bc want 01 here not pm1 | we divide each column by num. stimuli T
                jumpmean_est_M_T_hab_stage[:, :npts, stage]     = jump_data_for_ma.cumsum(axis=1) / np.arange(1, npts+1)
                # take moving average over 20 slices, by partitioning the 200 trials into separate blocks of 10
                for k in range(20):
                    movingavg_10slide10_M_T_hab_stage[:, k, stage] = np.sum(jump_data_for_ma[:, k*10:(k+1)*10], axis=1)
            else:
                cumulative_jumps_M_T_sra[:, :] = jump_data_for_cumsum.cumsum(axis=1)
                # for the jumpmean_est_ use 'for_ma' bc want 01 here not pm1 | we divide each column by num. stimuli T
                jumpmean_est_M_T_sra[:, :]     = jump_data_for_ma.cumsum(axis=1) / np.arange(1, npts+1)
                for k in range(5):
                    movingavg_10slide10_M_T_sra[:, k] = np.sum(jump_data_for_ma[:, k*10:(k+1)*10], axis=1)

        movingavg_10slide10_M_T_hab_mean[:, :] = np.mean(movingavg_10slide10_M_T_hab_stage[:, :, :], axis=2)

        # For each fly on a given day in the dataset, plot the cumulative jumps over the 6 stages in a single figure
        stage_to_label = ['hab1', 'hab2', 'hab3', 'hab4', 'hab5', 'sra']
        for fly_idx in range(jump_data_full.shape[0]):

            fly_id_true = fly_id_col[fly_idx]

            # model info for this fly on this day
            alpha = alpha_col[fly_idx]
            beta = beta_col[fly_idx]
            p0 = p0_col[fly_idx]

            # also extract mean scores pre-computed using script: data_format_add_score_columns.py
            T1_hab_mag_rel = T1_hab_mag_rel_col[fly_idx]
            T1_hab_mag_abs = T1_hab_mag_abs_col[fly_idx]
            T1_hab_time_half_rel = T1_hab_time_half_rel_col[fly_idx]
            T1_hab_time_95_rel = T1_hab_time_95_rel_col[fly_idx]
            T1_hab_time_half_abs = T1_hab_time_half_abs_col[fly_idx]

            # model timeseries (likelihood) for this fly on this day
            model_likelihood_hab = likelihood_func(np.arange(0, 200), alpha, beta, p0, 1.0)
            model_likelihood_sra = likelihood_func(np.arange(0, 50),  alpha, beta, p0, 5.0)

            # create a 5x2 single figure using gridspec
            # - 0, span - data, heatmap
            # - 1, span - model, cross-entropy heatmap
            # note: the panels below have been re-arranged
            # - 2,0 - data, cumsum plot
            # - 2,1 - model, cumsum plot
            # - 3,0 - data, <j(t)> empirical estimate plot
            # - 3,1 - model, model p(t) and var plot
            # - 4,0 - data, moving average plot
            # - 4,1 - model, moving average plot

            # create a figure and a gridspec object
            fig = plt.figure(figsize=(12, 18))
            gs = gridspec.GridSpec(5, 2, figure=fig,
                                   hspace=0.4,
                                   height_ratios=[0.25, 0.25, 1.0, 1.0, 1.0])

            # add subplots
            ax0_wide_A = fig.add_subplot(gs[0, :])  # first two plots should span all columns of the gridspec
            ax0_wide_B = fig.add_subplot(gs[1, :])  # first two plots should span all columns of the gridspec

            # ax0_wide_Bcbar = fig.add_subplot(gs[2, :])  # first two plots should span all columns of the gridspec
            # left: data est pcoin; right: model likelihood curves
            ax_data_pjump   = fig.add_subplot(gs[2, 0])
            ax_model_p_of_t = fig.add_subplot(gs[2, 1])
            # moving average panels
            ax_data_MA  = fig.add_subplot(gs[3, 0])
            ax_model_MA = fig.add_subplot(gs[3, 1])
            # cumsum panels
            ax_data_cumsum  = fig.add_subplot(gs[4, 0])
            ax_model_cumsum = fig.add_subplot(gs[4, 1])

            ###plt.tight_layout()
            ###plt.show()

            ###fig, axarr = plt.subplots(4, 2, figsize=(12, 18))

            # in panel (0,0), plot a heatmap of the jump data using imshow style
            #axarr[0, 0].imshow(jump_data_full[fly_idx, :], aspect='auto', cmap='viridis')
            # get the 01-jumpdata for this fly
            jumpdata = jump_data_full[fly_idx, :]
            # make a heatmap of size 6x200 in the style of plot_jumpmap_chunked_by_fly_and_age_tall
            # -  no data, end of SRA (-1) | no jump (0) | jump (1)
            jumpdata_6x200 = -1 * np.ones((6, 200))
            for stage in range(5):
                jumpdata_6x200[stage, :] = jumpdata[200*stage : 200*(stage+1)]
            jumpdata_6x200[5, 0:50] = jumpdata[1000:]

             # heatmap panel settings
            fs_ylabels = 8  # figsize (8, 12) and 64 flies use fs=8
            HEATMAP_WIDTH = 5 * 200 + 50  # num_hab_cycles * n_pulse_hab + n_pulse_reactivity
            dataversion_xticks = [0, 200, 400, 600, 800, 1000]

            #cmap = mpl.colors.ListedColormap(['whitesmoke', 'darkblue', 'orange'])
            cmap = mpl.colors.ListedColormap([heatmap_nodata, heatmap_0, heatmap_1])
            heatmap_aspect = 'auto' #2.0  #'auto'  # auto will fill the space

            im1 = ax0_wide_A.imshow(jumpdata_6x200, aspect=heatmap_aspect, interpolation='None', cmap=cmap, vmin=-1, vmax=1)
            ax0_wide_A.set_yticks([i for i in range(6)])
            ax0_wide_A.set_yticklabels(['H1', 'H2', 'H3', 'H4', 'H5', 'SRA'])
            #axarr[0, 0].set_ylabel('Jump data (1050)', fontsize=fs_ylabels, rotation=0, labelpad=6.0)
            # for each col...
            # TODO add xtick where predicted hab time is
            print('TODO add xtick where predicted hab time is based on model')

            # change fontsize
            #ax0_wide_A.tick_params(axis='both', which='minor', labelsize=5)
            #ax0_wide_A.tick_params(axis='y',    which='major', labelsize=5)
            # change all spines
            # im1.set_extent([-0.5, num_ages + 0.5, -0.5, HEATMAP_WIDTH + 0.5])
            for axis in ['top', 'bottom', 'left', 'right']:
                # axarr[i, j].spines[axis].set_linewidth(0.2)
                ax0_wide_A.spines[axis].set_linewidth(0.05)

            # increase tick width
            ax0_wide_A.tick_params(width=0.75)
            # set title of subplot
            ax0_wide_A.set_title('Data: Jump data (1050 stimuli)', fontsize=12)
            # adjust position down slightly
            #pos = ax0_wide_A.get_position()
            #ax0_wide_A.set_position([pos.x0, pos.y0 - 0.1, pos.width, pos.height])
            # add a colorbar to the first heatmap (discrete)
            cbar_A = fig.colorbar(im1, ax=ax0_wide_A, orientation='vertical', shrink=1.0, pad=0.03)
            cbar_A.set_label('Jumps data', fontsize=10)
            # the following should be the x ticks: 1,2,3,4,5, 10, 25, 50, 75, 100, 125, 150, 175, 200
            heatmap_xticks_force = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
            ax0_wide_A.set_xticks([a-1 for a in heatmap_xticks_force])
            ax0_wide_A.set_xticklabels([str(a) if a not in [2, 3, 4, 5] else '' for a in heatmap_xticks_force])

            # (0, 1) - wide - heatmap of cross entropy scores for the posterior mean model
            # - get the model scores for this fly
            cross_entropy_scores_6x200 = np.full((6, 200), np.nan)
            # calculate cross entropy for each stage
            for stage in range(5):
                cross_entropy_scores_6x200[stage, :] = score_cross_entropy(jumpdata[200*stage:200*(stage+1)],
                                                                           model_likelihood_hab, return_arr=True)
            # calculate cross entropy for the SRA stage
            cross_entropy_scores_6x200[5, 0:50] = score_cross_entropy(jumpdata[-50:],
                                                                      model_likelihood_sra, return_arr=True)
            # cmap for imshow
            cmap_CE = plt.get_cmap('PuBuGn')  # want to truncate to avoid the whites
            cmap_CE = mpl.colors.ListedColormap(cmap_CE(np.linspace(0.1, 1.0, 100)))
            im2 = ax0_wide_B.imshow(cross_entropy_scores_6x200, aspect=heatmap_aspect, interpolation='None', cmap=cmap_CE)
            # vmin=-1, vmax=1)
            ax0_wide_B.set_yticks([i for i in range(6)])
            ax0_wide_B.set_yticklabels(['H1', 'H2', 'H3', 'H4', 'H5', 'SRA'])
            # change all spines
            # im1.set_extent([-0.5, num_ages + 0.5, -0.5, HEATMAP_WIDTH + 0.5])
            for axis in ['top', 'bottom', 'left', 'right']:
                # axarr[i, j].spines[axis].set_linewidth(0.2)
                ax0_wide_B.spines[axis].set_linewidth(0.05)
            # increase tick width
            ax0_wide_B.tick_params(width=0.75)
            # add a colorbar to the right of the heatmap
            cbar_B = fig.colorbar(im2, ax=ax0_wide_B, orientation='vertical', shrink=1.0, pad=0.03)  # pad=0.0,
            cbar_B.set_label('Cross-entropy', fontsize=10)

            ax0_wide_B.set_xticks([a-1 for a in heatmap_xticks_force])
            ax0_wide_B.set_xticklabels([str(a) if a not in [2, 3, 4, 5] else '' for a in heatmap_xticks_force])

            # set title of subplot
            mean_CE = np.nanmean(cross_entropy_scores_6x200)
            ax0_wide_B.set_title('Model: Cross-entropy cost per timepoint (1050 stimuli) - Mean: %.3f' % mean_CE,
                                 fontsize=12)

            # (1,0) plot the cumulative jumps for each fly on a given day in the dataset
            for stage in range(5):
                cumulative_jumps = cumulative_jumps_M_T_hab_stage[fly_idx, :, stage]
                ax_data_cumsum.plot(cumulative_jumps, alpha=0.6)#, label='data')
            # now plot the 50 sra jumps sum
            ax_data_cumsum.plot(cumulative_jumps_M_T_sra[fly_idx, :], alpha=0.8, c='k', label='data')
            ax_data_cumsum.axhline(0.5, linestyle='-', color='grey', linewidth=0.5, zorder=20, label='fair coin')
            ax_data_cumsum.plot([0, 49], [-1, -50], '-', color='blue', linewidth=0.5, zorder=20, label=r'100% no-jump')
            ax_data_cumsum.plot([0, 49], [1, 50], '-', color='red', linewidth=0.5, zorder=20, label=r'100% jump')
            ax_data_cumsum.set_ylim((-200, 200))
            ax_data_cumsum.legend()
            ax_data_cumsum.set_title(r'Data: Cumulative jumps ($\pm 1$)')

            # (1,1) plot the cumulative jumps for each fly (based on the model)
            # - use model_likelihood_hab
            # - use model_likelihood_sra
            model_cumulative_jumps_hab = (model_likelihood_hab * 2 - 1).cumsum()
            model_cumulative_jumps_sra = (model_likelihood_sra * 2 - 1).cumsum()
            # - now plot the 50 sra jumps sum
            ax_model_cumsum.plot(model_cumulative_jumps_sra, alpha=1.0, c='k', label='model: SRA')
            ax_model_cumsum.plot(model_cumulative_jumps_hab, alpha=1.0,        label='model: Hab')
            ax_model_cumsum.axhline(0.5, linestyle='-',      color='grey', linewidth=0.5, zorder=20)
            ax_model_cumsum.plot([0, 49], [-1, -50], '-', color='blue', linewidth=0.5, zorder=20)
            ax_model_cumsum.plot([0, 49], [1,   50], '-', color='red',  linewidth=0.5, zorder=20)
            ax_model_cumsum.set_ylim((-200, 200))
            ax_model_cumsum.legend()
            ax_model_cumsum.set_title(r'Model: Cumulative jumps ($\pm 1$)')

            # (2,0) plot the more detailed moving average of the early trajectory with p_0 and p_inf axhlines
            # first plot the axhline envelope
            #  - upper: SRA average
            #  - lower: p_inf estimate from the tail of the data (average p50 to 200)
            p_upper_SRA = np.mean(movingavg_10slide10_M_T_sra[fly_idx, :]) / 10
            p_lower_inf_hab_mean      = np.mean(jumpdata_6x200[0:5, 50:200])
            p_lower_inf_hab_1_to_5    = np.mean(jumpdata_6x200[0:5, 50:200], axis=1)
            p_hab_mean_timeseries_1pt = np.mean(jumpdata_6x200[0:5, :], axis=0)

            ax_data_pjump.axhline(p_upper_SRA, linestyle='--', color='k', linewidth=2, zorder=20,
                                  label=r'$\langle \mathrm{SRA} \rangle$')
            ax_data_pjump.axhline(p_lower_inf_hab_mean, linestyle='--', color='tab:blue', linewidth=2, zorder=20,
                                  label=r'$\langle \mathrm{Hab}_{tail} \rangle$')
            #print('p_inf:', p_inf_hab_mean, p_inf_hab_1_to_5)
            #print(p_inf_hab_1_to_5.shape)

            # now plot the 50 sra jumps sum
            ax_data_pjump.plot(np.arange(5+0.5, 55+0.5, 10),
                               movingavg_10slide10_M_T_sra[fly_idx, :] / 10.0, '-o', alpha=1.0, c='k', markersize=4) #label='sra')
            # plot the moving avg for each hab stage
            '''
            for stage in range(5):
                ax_data_pjump.plot(np.arange(5+0.5, 205+0.5, 10),
                                   movingavg_10slide10_M_T_hab_stage[fly_idx, :, stage] / 10.0, '--', alpha=0.4, marker='s', markersize=2)
            '''
            # now plot the average over the 5 hab stages
            ax_data_pjump.plot(np.arange(5+0.5, 205+0.5, 10),
                               movingavg_10slide10_M_T_hab_mean[fly_idx, :] / 10.0, '-s', alpha=1.0, markersize=4, c='tab:blue',
                               label='hab mean')
            # plot 1-point-moving-avg over the 5 hab stages (call it 1-slide-1)
            t_arr_1s1 = np.arange(1, 201)
            '''
            ax_data_pjump.plot(t_arr_1s1, p_hab_mean_timeseries_1pt, '-s',
                               alpha=0.25, markersize=2, linewidth=0.5,
                               c='tab:red', label='hab mean (1s1)')
            '''
            # calc 95% confidence interval assuming binomial model using scipy
            #   - see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
            conf_interval_1s1 = get_conf_movingavg(p_hab_mean_timeseries_1pt, n=5, confidence_level=0.95, method='exact')
            # use fill to plot conf interval
            ax_data_pjump.fill_between(t_arr_1s1, conf_interval_1s1[:, 0], conf_interval_1s1[:, 1],
                                       color='tab:red', alpha=0.1, label='MA 1s1 95% CI')

            # now do 2-slide-1
            t_arr_2s1 = np.arange(1 + 0.5, 201 + 0.5 - 1)
            data_ma_2s1 = moving_average(p_hab_mean_timeseries_1pt, n=2)  # first point occurs at "1.5"
            conf_interval_2s1 = get_conf_movingavg(data_ma_2s1, n=10, confidence_level=0.95, method='exact')
            '''
            ax_data_pjump.plot(np.arange(1 + 0.5, 201 + 0.5 - 1),
                               data_ma_2s1, '-s',
                               alpha=0.5, markersize=2, linewidth=0.5,
                               c='tab:green', label='hab mean (2s1)')
            '''
            # use fill to plot conf interval
            ax_data_pjump.fill_between(t_arr_2s1, conf_interval_2s1[:, 0], conf_interval_2s1[:, 1],
                                       color='tab:green', alpha=0.15, label='MA 2s1 95% CI')

            t_arr_10s1 = np.arange(1 + 4 + 0.5, 201 + 0.5 - 5, 1)
            data_ma_10s1 = moving_average(p_hab_mean_timeseries_1pt, n=10)  # first point occurs at "5.5"
            conf_interval_10s1 = get_conf_movingavg(data_ma_10s1, n=50, confidence_level=0.95, method='exact')
            '''
            ax_data_pjump.plot(t_arr_10s1,
                               data_ma_10s1, '-s',
                               alpha=0.75, markersize=2, linewidth=0.5,
                               c='tab:purple')#, label='hab mean (10s1)')
            '''
            # use fill to plot conf interval
            ax_data_pjump.fill_between(t_arr_10s1, conf_interval_10s1[:, 0], conf_interval_10s1[:, 1],
                                       color='tab:purple', alpha=0.2, label='MA 10s1 95% CI')

            # now plot the fair coin mean
            ###ax_data_pjump.axhline(0.5, linestyle='-', color='grey', linewidth=0.5, zorder=20)

            # now plot the SRA mean +- binomial std which is np.sqrt(10*0.5*(1-0.5)))
            #SRA_mean_binomial_std = np.sqrt(p_upper_SRA * (1 - p_upper_SRA))
            # use matplotlib fill between
            #ax_data_MA.fill_between(np.arange(0, 55), p_upper_SRA - SRA_mean_binomial_std, SRA_mean + SRA_mean_binomial_std,
            #                        color='k', alpha=0.05, zorder=20, label=r'sra mean $\pm\, \sigma_{binomial}$')

            zoom_data_ma = False
            if zoom_data_ma:
                xticks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
                #ax_data_pjump.set_xticks(xticks, labels=['%d' % (i + 1) for i in xticks])
                ax_data_pjump.set_xticks(xticks, labels=['%d' % i for i in xticks])
                ax_data_pjump.set_xlim((-0.01, 51))
            else:
                xticks = [10, 20, 30, 40, 50, 100, 150, 200]
                # ax_data_pjump.set_xticks(xticks, labels=['%d' % (i + 1) for i in xticks])
                ax_data_pjump.set_xticks(xticks, labels=['%d' % i for i in xticks])
                #ax_data_pjump.set_xlim((-0.01, 51))

            ax_data_pjump.set_ylim((-0.01, 1.01))

            ax_data_pjump.set_ylabel(r'estimate $p_{jump}(t)$')
            ax_data_pjump.set_xlabel('time index')
            ax_data_pjump.legend(ncol=2)
            ax_data_pjump.set_title(r'Data: estimate $p_{jump}(t)$ by moving average') # $\langle j(t) \rangle$')

            # (2,1) now plot the model mean and variance
            # - model mean is <p(t)> = sum p_i / n
            # - model variance is <p(t)^2> - <p(t)>^2
            old_panel_style = False
            if old_panel_style:
                ax_model_p_of_t.plot(model_likelihood_sra, '--ok', alpha=1.0, label='model: SRA', markersize=3)
                ax_model_p_of_t.plot(model_likelihood_hab, '--o', alpha=1.0, label='model: Hab', markersize=3)
                ###ax_model_p_of_t.axhline(0.5, linestyle='-', color='grey', linewidth=0.5, zorder=20)
                ax_model_p_of_t.set_ylim((-0.01, 1.01))

                ax_model_p_of_t.set_title('Model: $p_{\mathrm{jump}}(t)$')
                ax_model_p_of_t.legend(fontsize=12)

            else:
                plot_posterior_likelihood_summary_over_days(
                    draws_over_age, val_genotype, fly_id_true, days_to_show=[val_day],
                    spaghetti=False, spaghetti_alpha=0.1, ci_level=0.95, n_samples=100, ax=ax_model_p_of_t)

            # add text to the top right of plot relating to the \theta parameters
            text_kw = dict(fontsize=10, transform=ax_model_p_of_t.transAxes)
            ax_model_p_of_t.text(0.51, 0.90 + 0.02, r'$\alpha=%.2f$' % alpha, **text_kw)
            ax_model_p_of_t.text(0.51, 0.85 + 0.02, r'$\beta=%.2f$' % beta, **text_kw)
            ax_model_p_of_t.text(0.51, 0.80 + 0.02, r'$p_0=%.2f$' % p0, **text_kw)

            T1_hab_time_half_abs_str = 'inf' if np.isinf(T1_hab_time_half_abs) else '%.2f' % T1_hab_time_half_abs
            ax_model_p_of_t.text(0.71, 0.9 + 0.02, r'$S_1=%.2f$'    % T1_hab_mag_rel, **text_kw)
            ax_model_p_of_t.text(0.71, 0.85 + 0.02, r'$S_2=%.2f$'   % T1_hab_mag_abs, **text_kw)
            ax_model_p_of_t.text(0.71, 0.80 + 0.02, r'abs: $T_{1/2}=%s$'  % T1_hab_time_half_abs_str, **text_kw)
            ax_model_p_of_t.text(0.71, 0.75 + 0.02, r'rel: $T_{1/2}=%.2f$' % T1_hab_time_half_rel,     **text_kw)
            ax_model_p_of_t.text(0.71, 0.70 + 0.02, r'rel: $T_{sat}=%.2f$' % T1_hab_time_95_rel,       **text_kw)


            # (3,0) plot the moving avg for each fly on a given day in the dataset
            for stage in range(5):
                ax_data_MA.plot(movingavg_10slide10_M_T_hab_stage[fly_idx, :, stage], '--', alpha=0.4, marker='s', markersize=3)
            # now plot the 50 sra jumps sum
            ax_data_MA.plot(movingavg_10slide10_M_T_sra[fly_idx, :], '-o', alpha=1.0, c='k', label='sra')
            SRA_mean = np.mean(movingavg_10slide10_M_T_sra[fly_idx, :])
            ax_data_MA.axhline(SRA_mean, linestyle='--', color='k', linewidth=1, zorder=20)
            # now plot the fair coin mean
            ###ax_data_MA.axhline(5, linestyle='-', color='grey', linewidth=0.5, zorder=20)  # , label='fair coin')
            # now plot the SRA mean +- binomial std which is np.sqrt(10*0.5*(1-0.5)))
            SRA_p_est             = SRA_mean / 10
            SRA_mean_binomial_std = np.sqrt(10 * SRA_p_est * (1 - SRA_p_est))
            # use matplotlib fill between
            """
            ax_data_MA.fill_between(np.arange(0, 5), SRA_mean - SRA_mean_binomial_std, SRA_mean + SRA_mean_binomial_std,
                                    color='k', alpha=0.05, zorder=20, label=r'sra mean $\pm\, \sigma_{binomial}$')
            """
            # now plot the average over the 5 hab stages
            ax_data_MA.plot(movingavg_10slide10_M_T_hab_mean[fly_idx, :], '-s', alpha=1.0, c='tab:blue', label='hab mean')

            ax_data_MA.axhline(p_lower_inf_hab_mean*10, linestyle='--', color='tab:blue',
                               linewidth=2, zorder=20,
                               label=r'$\langle \mathrm{Hab}_{tail} \rangle$')

            ax_data_MA.legend()
            ax_data_MA.set_ylim((-0.1, 10.1))
            ax_data_MA.set_title('Data: Moving average (10-slide-10) - show hab 1-5')
            ax_data_MA.set_ylabel('number of jumps in sliding window (10)')
            ax_data_MA.set_xlabel('time index (10 trials per marker)')

            # (3,1) moving average for the model
            movingavg_10slide10_model_hab = np.zeros(20)
            movingavg_10slide10_model_sra = np.zeros(5)
            for k in range(20):
                movingavg_10slide10_model_hab[k] = 10 * np.mean(model_likelihood_hab[k*10:(k+1)*10])
            for k in range(5):
                movingavg_10slide10_model_sra[k] = 10 * np.mean(model_likelihood_sra[k*10:(k+1)*10])

            ax_model_MA.plot(movingavg_10slide10_model_hab, '--o', alpha=1.0)
            # now plot the 50 sra jumps sum
            ax_model_MA.plot(movingavg_10slide10_model_sra, '-ok', alpha=1.0)
            ###ax_model_MA.axhline(5, linestyle='-', color='grey', linewidth=0.5, zorder=20)
            #ax_model_MA.legend()
            ax_model_MA.set_ylim((-0.1, 10.1))
            ax_model_MA.set_title('Model: Moving average')

            # make ylabels 0 to 10 inclusive
            ax_data_MA.set_yticks(np.arange(0,  11, 1))  # plt.yticks(np.arange(0, 11, 2))
            ax_model_MA.set_yticks(np.arange(0, 11, 1))  # plt.yticks(np.arange(0, 11, 2))
            # make xticks 0 to 20 inclusive, label them 1 to 5 then 10, 15, 20
            xticks = [0, 1, 2, 3, 4, 9, 14, 19]
            ax_data_MA.set_xticks(xticks, labels=['%d' % (10*(i + 1)) for i in xticks])
            ax_model_MA.set_xticks(xticks, labels=['%d' % (10*(i + 1)) for i in xticks])

            plt.suptitle('Data vs. model for %s - fly ID: %s' % (label_expt, fly_id_true), fontsize=14, y=0.93)
            #plt.tight_layout()  -- sketchy with gridspec, dont use

            plt.savefig(DIR_OUTPUT + os.sep + 'indiv_%s_fly%d.png' % (label_expt, fly_id_true), dpi=300)
            plt.savefig(DIR_OUTPUT + os.sep + 'indiv_%s_fly%d.svg' % (label_expt, fly_id_true))
            #plt.show()

    return


def plot_fly_dashboard_across_days(df_means_csv, fly_id, genotype, draws_over_age, fill_kde=True):
    """
    Multi-day dashboard for a single fly showing:
    1. Posterior pairwise KDE plots across 3 days.
    2. Heatmaps of jump data across 3 days.
    3. Placeholder plots for future use.

    Args:
        draws_over_age: Dictionary of posterior samples {genotype: {day: {fly_id: samples}}}
        fly_id: The fly ID to plot (e.g., 'fly_id_1')
        genotype: The genotype to plot (e.g., 'GD' or 'KK')
        jump_data_over_age: Dictionary of jump data {genotype: {day: {fly_id: jump_array}}}
        fill_kde: Whether to fill the KDE plots.
    """

    # Define plot layout (3 rows, 3 columns)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    colors = {7: 'blue', 14: 'green', 21: 'red'}

    ### --- Row 1: Posterior KDE Plots for Days 7, 14, 21 --- ###
    for idx, day in enumerate([7, 14, 21]):
        ax = fig.add_subplot(gs[0, idx])
        try:
            samples = draws_over_age[genotype][day][fly_id]
            alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

            # Pairwise KDE plot
            df_samples = pd.DataFrame({'alpha': alpha, 'beta': beta, 'p0': p0})
            sns.kdeplot(data=df_samples, x='alpha', y='beta', fill=fill_kde, ax=ax, cmap="Blues")

            ax.set_title(f"Posterior KDE (Day {day})", fontsize=10)
            ax.set_xlabel("α")
            ax.set_ylabel("β")
        except KeyError:
            ax.text(0.5, 0.5, f"No Data (Day {day})", ha='center', va='center')
            ax.axis('off')

    ### --- Row 2: Heatmaps of Jumps for Days 7, 14, 21 --- ###
    for idx, day in enumerate([7, 14, 21]):
        df_subset = filter_df_by_filterdict(df_means_csv, {'genotype': [genotype], 'day': [day]})
        jump_data_day = df_to_arr_jumps(df_subset, jump_col='jumpdata')

        ax = fig.add_subplot(gs[1, idx])
        try:
            #jump_data = jump_data_over_age[day][fly_id]
            jump_data = jump_data_day[fly_id, :]

            # Convert jump data to a heatmap-friendly format (6 stages)
            jump_heatmap = -1 * np.ones((6, 200))
            for stage in range(5):
                jump_heatmap[stage, :] = jump_data[stage*200:(stage+1)*200]
            jump_heatmap[5, 0:50] = jump_data[1000:]

            cmap_heatmap = plt.get_cmap('coolwarm')
            im = ax.imshow(jump_heatmap, aspect='auto', cmap=cmap_heatmap, vmin=-1, vmax=1)

            ax.set_title(f"Jump Heatmap (Day {day})", fontsize=10)
            ax.set_yticks(range(6))
            ax.set_yticklabels(['H1', 'H2', 'H3', 'H4', 'H5', 'SRA'])
            ax.set_xlabel("Stimulus #")
        except KeyError:
            ax.text(0.5, 0.5, f"No Data (Day {day})", ha='center', va='center')
            ax.axis('off')

    ### --- Row 3: Placeholder Panels --- ###
    for idx in range(3):
        ax = fig.add_subplot(gs[2, idx])
        ax.text(0.5, 0.5, "Placeholder", ha='center', va='center', fontsize=12, alpha=0.5)
        ax.axis('off')
        ax.set_title(f"Future Panel {idx + 1}")

    ### --- Final Layout --- ###
    plt.suptitle(f"Fly {fly_id} ({genotype}): Multi-Day Dashboard", fontsize=16, y=0.92)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    # load the parameters inferred from the fit
    # note: the -habscores variant is generated by running data_format_add_score_columns.py
    df_means_csv = pd.read_csv(DIR_FITS + os.sep + "fly-stability-days-detailed-habscores.csv")
    #print(df_means_csv.head())

    # Load posterior draws for multiple days/genotypes
    # - access posterior for a particular fly id using draws_KK_day14[f"fly_id_{id_number}"]
    draws_GD_day7  = np.load(DIR_FITS + os.sep + "GD_day7_3d_draws.npz", allow_pickle=True)
    draws_GD_day14 = np.load(DIR_FITS + os.sep + "GD_day14_3d_draws.npz", allow_pickle=True)
    draws_GD_day21 = np.load(DIR_FITS + os.sep + "GD_day21_3d_draws.npz", allow_pickle=True)

    draws_KK_day7  = np.load(DIR_FITS + os.sep + "KK_day7_3d_draws.npz", allow_pickle=True)
    draws_KK_day14 = np.load(DIR_FITS + os.sep + "KK_day14_3d_draws.npz", allow_pickle=True)
    draws_KK_day21 = np.load(DIR_FITS + os.sep + "KK_day21_3d_draws.npz", allow_pickle=True)

    draws_over_age = {
        'KK': {7: draws_KK_day7, 14: draws_KK_day14, 21: draws_KK_day21},
        'GD': {7: draws_GD_day7, 14: draws_GD_day14, 21: draws_GD_day21}
    }

    # Select which plot groups to make
    # ========================================================
    # scatter plots of posterior means from the fit
    flag_param_scatters               = False
    # scatter plot of param atr age A vs age B for same fly
    flag_diag_ageshift_scatter        = False
    # tall " ID + age " heatmaps + heatmaps of experimental data against simulated data (Fig. supplement)
    flag_jump_heatmaps                = False
    # scatter plots of parameter combos (e.g. hab. magnitude vs. hab. speed)
    flag_hab_scores                   = False
    # plot the likelihood of the data under the model as a function of time
    flag_plot_likelihood_timeseries   = False
    # cross-entropy scores
    flag_model_performance_population = False
    # 'dashboard' plot for each fly
    flag_model_performance_exemplars  = True
    # 'dashboard' plot for each fly over the three ages
    # TODO
    flag_model_performance_exemplars_multiday = False
    # TODO
    flag_posteror_checks_univariate   = False
    # ========================================================

    if flag_param_scatters:
        plot_3d_scatter_posterior_means(df_means_csv, scatter_3d_shadow=True)  # scatter plot of the inferred parameters

    if flag_diag_ageshift_scatter:
        print('plot_diag_ageshift_scatter...')
        plot_diag_ageshift_scatter(df_means_csv)

    if flag_jump_heatmaps:

        # plot heatmaps of experimental data against simulated data
        genotype = 'KK'
        day = 7
        # experimental data (currently sorted, undo this)
        data_jumps_arr, data_plot = plot_full_data(genotype=genotype, day=day, filtered=True, detailed_format=True, use_mpl=True)
        data_plot.show()
        # synthetic data (one trajectory is sampled from the posterior mean for each fly)
        sim_df, sim_jumps_arr, sim_plot = simulate_from_fit(genotype=genotype, day=day, use_mpl=True)
        sim_plot.show()

        # make 'tall' heatmaps where each block is 3 x 1050 (e.g. fly 17, GD age 7/14/21, all data)
        # - need to make 2 per genetic background (one per var), since 64 rows max
        plot_jumpmap_chunked_by_fly_and_age_tall(df_means_csv, max_flies=64, figsize=(8, 12), show_yticks=False)

    if flag_hab_scores:
        plot_hab_scores(df_means_csv)

    if flag_plot_likelihood_timeseries:
        plot_likelihood_timeseries(df_means_csv)

    if flag_model_performance_population:
        # measure model performance on training set by comparing cross-entropy of model vs. baselines (e.g. coin flips)
        model_performance_population(df_means_csv)

    if flag_model_performance_exemplars:
        # TODO - WIP
        # for now we will compare the performance on just 5 individuals on specific day
        print('TODO flag_model_performance_exemplars')
        #genotype = 'KK'
        #day = 14
        # load the df with the habscores appended as extra rows; alternatively we could simply recompute them here
        plot_model_performance_exemplars(df_means_csv)  # if no indivs are specified, all the df rows will be plotted

    if flag_model_performance_exemplars_multiday:
        # TODO - WIP
        fly_id = 40
        genotype = 'KK'
        plot_fly_dashboard_across_days(df_means_csv, fly_id, genotype, draws_over_age, fill_kde=True)

    if flag_posteror_checks_univariate:
        # TODO - WIP
        # use the functions from functions_posterior_checks.py to check the univariate posteriors
        from functions_posterior_check import plot_univariate_posterior
        print('TODO flag_posteror_checks_univariate')
