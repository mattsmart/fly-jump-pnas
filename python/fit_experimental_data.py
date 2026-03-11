# $ cd fly-jump/python
# $ python3
# >>> exec(open('fit-experimental-data.py').read())

import cmdstanpy as csp
import numpy as np
import os
import pandas as pd
import plotnine as pn
import time
import logging
import warnings

from data_tools import filter_df_by_filterdict
from functions_common import logit, jump_prob
from settings import DIR_DATA_EXPT, DIR_FITS, DIR_OUTPUT, DIR_STAN, heatmap_0, heatmap_1, OMIT_FLY_IDS


# default: both 2
FIT_KWARGS_DEFAULT = dict(
    chains=2, parallel_chains=2,
    iter_warmup=400, iter_sampling=400, max_treedepth=10,
    adapt_delta=0.8,
    show_console=False,
    refresh=1,
)

FIT_KWARGS_HEAVY = dict(
    chains=8, parallel_chains=8,
    iter_warmup=1000, iter_sampling=1000, max_treedepth=10,
    adapt_delta=0.9,  # read default: 0.8, but try 0.9 or 0.95 for harder models
    show_console=False,
    refresh=1,
)

FIT_KWARGS_SELECT = FIT_KWARGS_HEAVY

# reduce griping
def reduce_griping():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", module=r"plotnine\..*")
    csp.utils.get_logger().setLevel(logging.ERROR)


def to_file(genotype, day, filtered=False, detailed_format=False):
    sep = os.sep
    detailed_str = '_with_details' if detailed_format else ''
    file_format  =           'csv' if detailed_format else 'txt'
    if filtered:
        return f"..{sep}data{sep}experiment{sep}{genotype}_{day}days{detailed_str}_filtered.{file_format}"
    else:
        return f"..{sep}data{sep}experiment{sep}{genotype}_{day}days{detailed_str}_unfiltered.{file_format}"


def parse_fly_data(filename, detailed_format=False, omit_fly_ids=None):
    """
    Reads the fly jumping experiment data file (e.g. genotype GD, age 7 -> GD_7days_filtered.txt)
    and produces a dictionary for Stan with the following keys:
    - num_experiments: Number of unique flies (rows of the data matrix).
    - num_trials_per_experiment: Number of trials concatenated per experiment.
    - jump: A 2D array where each row corresponds to concatenated binary (0/1) values
      from all stages (1-5 and R) for a single experiment.

    If omit_fly_ids is provided, skips rows with matching fly_id.
    """
    if omit_fly_ids is None:
        omit_fly_ids = set()
    else:
        omit_fly_ids = set(str(fid) for fid in omit_fly_ids)

    fly_data = {}

    jump_matrix = []
    # extra data to pass around in same structure as jump_matrix
    list_fly_id = []
    list_chamber_id = []
    list_fly_id_64 = []
    list_var = []
    list_exp_name = []
    list_date = []
    list_jumpdata = []  # store as list of str instead of np array

    if detailed_format:
        with open(filename, 'r') as file:
            # assert header row has expected format (basic version control)
            header = next(file).strip()
            assert header == 'genotype,day,fly_id,chamber_id,fly_id_64,var,exp_name,date,flies_remaining,jumpdata'
            # for the data lines, extract the jumpdata column
            for idx, line in enumerate(file):
                genotype, day, fly_id, chamber_id, fly_id_64, var, exp_name, date, _, jumpdata_hab1to5_plus_sra = line.strip().split(',')
                if fly_id in omit_fly_ids:
                    continue
                jump_matrix.append([int(bit) for bit in jumpdata_hab1to5_plus_sra])
                list_fly_id.append(fly_id)
                list_chamber_id.append(chamber_id)
                list_fly_id_64.append(fly_id_64)
                list_var.append(var)
                list_exp_name.append(exp_name)
                list_date.append(date)
                list_jumpdata.append(jumpdata_hab1to5_plus_sra)

    else:
        print('TODO in parse_fly_data(..., detailed_format=False), why are we sorting the stages hab1-5?')
        with open(filename, 'r') as file:
            for line in file:
                fly_id, stage, values = line.strip().split(',', 2)
                if fly_id not in fly_data:
                    fly_data[fly_id] = []
                fly_data[fly_id].append(values)

            for fly_id, stages in sorted(fly_data.items()):

                # note: we originally performed "sorting" in the first term
                sorted_stages = sorted(stages[:-1]) + [stages[-1]]
                concatenated_values = ''.join(sorted_stages)
                jump_matrix.append([int(bit) for bit in concatenated_values])

    num_experiments = len(jump_matrix)
    num_trials_per_experiment = len(jump_matrix[0])
    jump_array = np.array(jump_matrix, dtype=int)

    return {
        'num_experiments': num_experiments,
        'num_trials_per_experiment': num_trials_per_experiment,
        'jump': jump_array,
        'fly_id': list_fly_id,
        'chamber_id': list_chamber_id,
        'fly_id_64': list_fly_id_64,
        'var': list_var,
        'exp_name': list_exp_name,
        'date': list_date,
        'jumpdata_str': list_jumpdata,
    }


def fit(model, inits_fun, genotype, day, show_progress=False, filtered=False, detailed_format=False):
    file_name = to_file(genotype, day, filtered, detailed_format=detailed_format)
    omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
    data_dict = parse_fly_data(file_name, detailed_format=detailed_format, omit_fly_ids=omit_fly_ids)
    # take relevant keys only; extra keys are present in the detailed format (tracking fly ID and other metadata)
    subdict_for_stan = {key: data_dict[key] for key in ['num_experiments', 'num_trials_per_experiment', 'jump']}
    N = subdict_for_stan['num_experiments']
    fit = model.sample(
        data=subdict_for_stan,
        inits=inits_fun(N),
        show_progress=show_progress,
        **FIT_KWARGS_SELECT
    )
    return fit


def fit_2d_alpha_global(model, inits_fun, alpha_global, genotype, day, show_progress=False, filtered=False, detailed_format=False):
    """Modified fit function for 2D model that includes alpha_global in data"""
    file_name = to_file(genotype, day, filtered, detailed_format=detailed_format)
    omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
    data_dict = parse_fly_data(file_name, detailed_format=detailed_format, omit_fly_ids=omit_fly_ids)
    # take relevant keys and add alpha_global
    subdict_for_stan = {key: data_dict[key] for key in ['num_experiments', 'num_trials_per_experiment', 'jump']}
    subdict_for_stan['alpha_global'] = alpha_global
    N = subdict_for_stan['num_experiments']
    fit = model.sample(
        data=subdict_for_stan,
        inits=inits_fun(N),
        show_progress=show_progress,
        **FIT_KWARGS_SELECT,
    )
    return fit


def sample_correlation(a_draws, b_draws, c_draws):
    a_bar = a_draws.mean(axis=0)
    b_bar = b_draws.mean(axis=0)
    c_bar = c_draws.mean(axis=0)
    return np.corrcoef([a_bar, b_bar, c_bar])


def scatterplot(cons, a_draws, b_draws, p_draws):
    a_hat = a_draws.mean(axis=0)
    b_hat = b_draws.mean(axis=0)
    p_hat = p_draws.mean(axis=0)
    d = {'alpha': a_hat, 'beta': b_hat, 'p0': p_hat}
    df = pd.DataFrame(d)
    unique_pairs = [('alpha', 'beta'), ('alpha', 'p0'), ('beta', 'p0')]
    pairs = []
    for x_var, y_var in unique_pairs:
        temp_df = df[[x_var, y_var]].copy()
        temp_df = temp_df.rename(columns={x_var: 'x', y_var: 'y'})
        temp_df['x_label'] = x_var
        temp_df['y_label'] = y_var
        pairs.append(temp_df)
        pairs_df = pd.concat(pairs, ignore_index=True)
        scatter_plot = (
            pn.ggplot(pairs_df, pn.aes(x='x', y='y'))
            + pn.geom_point()
            + pn.facet_wrap('~x_label + y_label', scales='free')
            + pn.labs(x='top variable', y='bottom variable', title=cons)
            + pn.theme(subplots_adjust={'wspace': 0.3, 'hspace': 0.3})
            )
        melted_df = df.melt(var_name='variable', value_name='value')
        hist_plot = (
            pn.ggplot(melted_df, pn.aes(x='value'))
            + pn.geom_histogram(bins=6, color='black', fill='lightblue')
            + pn.facet_wrap('~variable', scales='free')
            + pn.labs(x='value', y='number of flies', title=cons)
            + pn.theme(subplots_adjust={'wspace': 0.3, 'hspace': 0.3})
        )
    return scatter_plot, hist_plot


def make_model(directory_name, file_name):
    file_path = directory_name + file_name
    m = csp.CmdStanModel(stan_file=file_path)
    return m


def pooled(directory_name):
    file_name = 'habituation-pool.stan'
    model =  make_model(directory_name, file_name)
    def init_fun(N = 0):
        return {
            'alpha': 0.1,
            'beta': 0.1,
            'p0': 0.5
        }
    return model, init_fun


def no_pooled(directory_name):
    """Fits the full 3d model with no pooling"""
    file_name = 'habituation-no-pool.stan'
    model =  make_model(directory_name, file_name)
    def init_fun(N = 0):
        return {
            'alpha': N * [0.1],
            'beta': N * [0.1],
            'p0': N * [0.5]
        }
    return model, init_fun


def partially_pooled(directory_name):
    file_name = 'habituation-partial-pool.stan'
    model =  make_model(directory_name, file_name)
    def init_fun(N = 0):
        return {
            'log_alpha': N * [np.log(0.1)],
            'log_beta': N * [np.log(0.1)],
            'logit_p0': N * [logit(0.5)],
            'mu_alpha': np.log(0.1),
            'mu_beta': np.log(0.1),
            'mu_p0': logit(0.6),
            'sigma_alpha': 1.0,
            'sigma_beta': 1.0,
            'sigma_p0': 1.0
        }

    return model, init_fun


def partially_pooled_multi(directory_name):
    file_name = 'habituation-partial-pool-multi.stan'
    model = make_model(directory_name, file_name)

    def init_fun(N=0):
        return {
            'log_alpha': N * [np.log(0.1)],
            'log_beta': N * [np.log(0.1)],
            'logit_p0': N * [logit(0.5)],
            'mu_alpha': np.log(0.1),
            'mu_beta': np.log(0.1),
            'mu_p0': logit(0.6),
            'sigma_alpha': 1.0,
            'sigma_beta': 1.0,
            'sigma_p0': 1.0,
            'Omega': np.eye(3)
        }

    return model, init_fun


def no_pooled_1d(directory_name):
    file_name = 'habituation-no-pool-1d-p0.stan'
    model = make_model(directory_name, file_name)
    def init_fun(N=0):
        return {
            'p0': N * [0.5]
        }
    return model, init_fun


def no_pooled_2d(directory_name):
    file_name = 'habituation-no-pool-2d-p0-beta.stan'
    model = make_model(directory_name, file_name)
    def init_fun(N=0):
        return {
            'beta': N * [0.1],
            'p0': N * [0.5]
        }
    return model, init_fun


def run_pooled():
    print("\n\nPOOLED (MODEL 1)")
    print("===================================================================")
    d = DIR_STAN + os.sep + 'dynamics' + os.sep
    model, init_fun = pooled(d)
    genotypes=['GD', 'KK']
    days=[7, 14, 21]
    for genotype in genotypes:
        for day in days:
            print(f"\n{genotype=} {day=}")
            sample = fit(model, init_fun, genotype, day, show_progress=False)
            print(sample.summary(sig_figs=2))


def run_no_pooled():
    print("\n\nNO POOLED (MODEL 2)  [warning: allow 10m/dataset to fit]")
    print("===================================================================")
    d = DIR_STAN + os.sep + 'dynamics' + os.sep

    model, init_fun = no_pooled(d)
    genotypes=['KK']  # ['GD', 'KK']
    days=[7, 14, 21]
    samples = {}
    for genotype in genotypes:
        for day in days:
            gd = f"{genotype}_{day}"
            print(f"\n{genotype=} {day=}")
            print("------------------------------------------------------------")
            sample = fit(model, init_fun, genotype, day, show_progress=False)
            samples[gd] = sample

            alpha_draws = sample.stan_variable('alpha')
            beta_draws = sample.stan_variable('beta')
            p0_draws = sample.stan_variable('p0')
            corr_constrained = sample_correlation(alpha_draws, beta_draws, p0_draws)

            log_alpha_draws = np.log(alpha_draws)
            log_beta_draws = np.log(beta_draws)
            logit_p0_draws = logit(p0_draws)
            corr_unconstrained = sample_correlation(log_alpha_draws, log_beta_draws, logit_p0_draws)

            print(sample.summary(sig_figs=2))
            print(f"Posterior estimate correlation (constrained):\n {corr_constrained}")
            print(f"\nPosterior estimate correlation (unconstrained):\n {corr_unconstrained}")

            scat, hist         = scatterplot("CONSTRAINED", alpha_draws, beta_draws, p0_draws)
            scat_unc, hist_unc = scatterplot("UNCONSTRAINED", log_alpha_draws, log_beta_draws, logit_p0_draws)
            scat.save(gd + '_no_pool_scat.pdf', width=7.5, height=2.5)
            hist.save(gd + '_no_pool_hist.pdf', width=7.5, height=2.5)
            scat_unc.save(gd + '_no_pool_unc_scat.pdf', width=9, height=3)
            hist_unc.save(gd + '_no_pool_unc_hist.pdf', width=9, height=3)
    return samples


def run_partial_pooled_multi():
    """
    Unused function to fit the multivariate partial pooling model (MODEL 4).
    """
    print("\n\nMULTIVARIATE PARTIAL POOLING (MODEL 4)  [warning: 10m/dataset fit]")
    print("===================================================================")
    d = DIR_STAN + os.sep + 'dynamics' + os.sep
    model, init_fun = partially_pooled_multi(d)
    genotype = 'KK'
    day = 14
    gd = f"{genotype}_{day}"
    print(f"\n{genotype=} {day=}")
    print("------------------------------------------------------------")
    sample = fit(model, init_fun, genotype, day, show_progress=False)
    samples[gd] = sample

    mu_draws = sample.stan_variable('mu')
    Sigma_draws = sample.stan_variable('Sigma')

    #sim_jumps = np.zeros((128, 1050))
    #for n in range(128):
    #    sim_jumps[n, :] = sim_trial(mu_draws[n], Sigma_draws[n])
    #return sim_data

    return


def filter_to_common_flies(filenames, output_filenames):
    # Filter data to include only flies that are present in all files.
    fly_data = [pd.read_csv(file, header=None, names=["fly_id", "trial", "response"]) for file in filenames]
    fly_ids = [set(df["fly_id"]) for df in fly_data]

    # Find the intersection of fly IDs present in all files
    common_flies = set.intersection(*fly_ids)

    # Filter each file for the common flies and write to new files
    for df, output_file in zip(fly_data, output_filenames):
        filtered_df = df[df["fly_id"].isin(common_flies)]
        filtered_df.to_csv(output_file, index=False, header=False)


def run_fly_stability(out_file='fly-stability-days.csv'):
    """
    Steps:
        1. Filter the data to include only flies that are present in all files.
        2. Fit the no-pooled model to the filtered data.
        3. Calculate the correlation between the parameters for each genotype and day.
        4. Save the correlation results to a CSV file.
    """
    print("\n\nSTABILITY OF FLY ESTIMATES OVER DAYS (both genotype)")
    print("===================================================================")
    data_dir = DIR_DATA_EXPT + os.sep

    input_files_gd = [
        data_dir + "GD_7days_unfiltered.txt",
        data_dir + "GD_14days_unfiltered.txt",
        data_dir + "GD_21days_unfiltered.txt"
    ]
    output_files_gd = [
        data_dir + "GD_7days_filtered.txt",
        data_dir + "GD_14days_filtered.txt",
        data_dir + "GD_21days_filtered.txt"
    ]
    filter_to_common_flies(input_files_gd, output_files_gd)

    input_files_kk = [
        data_dir + "KK_7days_unfiltered.txt",
        data_dir + "KK_14days_unfiltered.txt",
        data_dir + "KK_21days_unfiltered.txt"
    ]
    output_files_kk = [
        data_dir + "KK_7days_filtered.txt",
        data_dir + "KK_14days_filtered.txt",
        data_dir + "KK_21days_filtered.txt"
    ]
    filter_to_common_flies(input_files_kk, output_files_kk)

    model_dir = DIR_STAN + os.sep + 'dynamics' + os.sep
    model, init_fun = no_pooled(model_dir)
    data = []
    genotypes = ['GD', 'KK']
    days = [7, 14, 21]
    for genotype in genotypes:
        for day in days:
            genotype_day = genotype + "_" + str(day)
            file = data_dir + genotype_day + "days_filtered.txt",
            print(f"\n{genotype=} {day=}")
            print("------------------------------------------------------------")
            sample = fit(model, init_fun, genotype, day, show_progress=False, filtered=True)
            alpha_draws = sample.stan_variable('alpha')
            beta_draws = sample.stan_variable('beta')
            p0_draws = sample.stan_variable('p0')
            alpha_hat = alpha_draws.mean(axis=0)
            beta_hat = beta_draws.mean(axis=0)
            p0_hat = p0_draws.mean(axis=0)
            num_flies = len(alpha_hat)
            for fly in range(num_flies):
                data.append({
                    'genotype': genotype,
                    'day': day,
                    'fly' : fly,
                    'alpha': alpha_hat[fly],
                    'beta': beta_hat[fly],
                    'p0': p0_hat[fly],
                    'log_alpha': np.log(alpha_hat[fly]),
                    'log_beta': np.log(beta_hat[fly]),
                    # ORIGINAL LINE: 'logit_p0': logit(beta_hat[fly])   --> critical typo!
                    'logit_p0': logit(p0_hat[fly])
                })
    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)


def run_fly_stability_detailed_csv(out_file=DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv'):
    """
    Fit the 3D model (alpha, beta, p0) for all genotype×day combinations.

    Steps:
        1. Fit the no-pooled 3D model to the filtered data (excluding OMIT_FLY_IDS).
        2. Save posterior draws to {genotype}_day{day}_3d_draws.npz files.
        3. Save parameter estimates and metadata to detailed CSV.
        4. Print Stan diagnostics for each fit.

    Returns:
        None (saves files to DIR_FITS)
    """
    print("\n\nSTABILITY OF FLY ESTIMATES OVER DAYS (both genotype)")
    print("===================================================================")
    model_dir = DIR_STAN + os.sep + 'dynamics' + os.sep
    model, init_fun = no_pooled(model_dir)
    data = []
    genotypes = ['GD', 'KK']
    days = [7, 14, 21]
    for genotype in genotypes:
        for day in days:

            file_name = to_file(genotype, day, filtered=True, detailed_format=True)
            omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
            parsed_data_dict = parse_fly_data(file_name, detailed_format=True, omit_fly_ids=omit_fly_ids)

            print(f"\n{genotype=} {day=}")
            print("------------------------------------------------------------")

            # Fit models for each fly using Stan, and record the runtime
            start_time = time.time()
            sample = fit(model, init_fun, genotype, day, show_progress=False, filtered=True, detailed_format=True)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Fit completed in {elapsed_time/60:.2f} minutes")

            # Check Stan diagnostics
            print("\nStan Diagnostics:")
            print(sample.diagnose())

            # Extract the posterior draws for alpha, beta, and p0
            alpha_draws = sample.stan_variable('alpha')  # np.arr shape: fit.iter_sampling x num_flies
            beta_draws = sample.stan_variable('beta')
            p0_draws = sample.stan_variable('p0')

            # Create a dictionary to store parameter triplets for all flies in this condition
            # - will have one *.npz per condition (gene_bgr, day)
            # - e.g. KK_day7_draws.npz will have all posterior draws for each fly
            # API to load posterior draws for GD on day 7
            #   draws = np.load('fits/GD_day7_draws.npz')
            #   fly_triplets = draws["fly_3"]  # Shape: (n_draws, 3)
            draws_dict = {}

            alpha_hat = alpha_draws.mean(axis=0)
            beta_hat = beta_draws.mean(axis=0)
            p0_hat = p0_draws.mean(axis=0)

            num_flies = len(alpha_hat)
            # each index has a corresponding row in jumpdata which was concatenated from the raw data; match fly_id
            for idx in range(num_flies):

                fly_id = parsed_data_dict['fly_id'][idx]

                # 1) prep the data row for the summary CSV
                data.append({
                    'genotype': genotype,
                    'day': day,
                    'fly': idx,
                    'fly_id': fly_id,
                    'alpha': alpha_hat[idx],
                    'beta': beta_hat[idx],
                    'p0': p0_hat[idx],
                    'log_alpha': np.log(alpha_hat[idx]),
                    'log_beta': np.log(beta_hat[idx]),
                    'logit_p0': logit(p0_hat[idx]),
                    'chamber_id': parsed_data_dict['chamber_id'][idx],
                    'fly_id_64': parsed_data_dict['fly_id_64'][idx],
                    'var': parsed_data_dict['var'][idx],
                    'exp_name': parsed_data_dict['exp_name'][idx],
                    'date': parsed_data_dict['date'][idx],
                    'jumpdata': parsed_data_dict['jumpdata_str'][idx],
                })

                # 2) prep denser fit data for the npz blob
                param_triplets = np.stack([alpha_draws[:, idx], beta_draws[:, idx], p0_draws[:, idx]], axis=1)
                # - store triplets (arr: num_draws x 3) with a single key
                draws_dict[f"fly_id_{fly_id}"] = param_triplets

            # Save draws to a genotype-day specific file
            file_name = DIR_FITS + os.sep + f"{genotype}_day{day}_3d_draws.npz"
            np.savez_compressed(file_name, **draws_dict)

    df = pd.DataFrame(data)
    df.to_csv(out_file, index=False)

    print("\n\n3D MODEL FITTING COMPLETE!")
    print("===================================================================")
    print("Output files:")
    print(f"  - {out_file}")
    print(f"  - {{genotype}}_day{{day}}_3d_draws.npz (6 files)")
    print("\nNext steps:")
    print("  1. Run post-processing: python data_format_add_score_columns.py 3d")
    print("  2. Fit 2D/1D models: run_model_comparison(fit_3d=False, fit_2d=True, fit_1d=True)")


def calculate_day_correlations(unconstrained=True,
                               in_file=DIR_FITS+os.sep+'fly-stability-days.csv',
                               out_file=DIR_FITS+os.sep+'fly-stability-days-corr.csv'):
    """
    Reads the fly data from a CSV file and calculates correlations for alpha, beta, and p0
    across day pairs (7 vs 14, 7 vs 21, and 14 vs 21), faceted by genotype.

    Parameters:
        unconstrained (bool): whether to report correlations for transformed parameters
        in_file (str): Path to the CSV file containing the fly data.
        out_file (str): Path to the CSV file to which correlations are written.

    Returns:
        pd.DataFrame: A DataFrame with columns ['genotype', 'parameter', 'dayA', 'dayB', 'correlation'].
    """
    df = pd.read_csv(in_file)
    if unconstrained:
        df['alpha'] = np.log(df['alpha'])
        df['beta'] = np.log(df['beta'])
        df['p0'] = logit(df['p0'])

    parameters = ['alpha', 'beta', 'p0']
    days = [7, 14, 21]
    genotypes = df['genotype'].unique()  # Extract unique genotypes from the data
    correlation_results = []
    for genotype in genotypes:
        for param in parameters:
            for i, dayA in enumerate(days):
                for dayB in days[i + 1:]:  # Only compute correlations for dayA < dayB
                    dayA_data = df[(df['genotype'] == genotype) & (df['day'] == int(dayA))][param].reset_index(drop=True)
                    dayB_data = df[(df['genotype'] == genotype) & (df['day'] == int(dayB))][param].reset_index(drop=True)
                    if len(dayA_data) == len(dayB_data):  # Ensure the data is aligned
                        correlation = dayA_data.corr(dayB_data)
                        correlation_results.append({
                            'genotype': genotype,
                            'parameter': param,
                            'dayA': dayA,
                            'dayB': dayB,
                            'correlation': correlation
                        })
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df.to_csv(out_file, index=False)
    return correlation_df


def compute_alpha_global_from_3d_fits():
    """
    Compute alpha_global from existing 3D model fits.
    Returns a dict keyed by (genotype, day) with mean alpha for that condition.

    Returns:
        dict: {(genotype, day): alpha_global_value}
    """
    print("\nComputing alpha_global per (genotype, day) from 3D model fits...")
    print("------------------------------------------------------------")

    alpha_global_dict = {}
    genotypes = ['GD', 'KK']
    days = [7, 14, 21]

    for genotype in genotypes:
        for day in days:
            npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_3d_draws.npz"
            if not os.path.exists(npz_file):
                # Try old naming convention (without _3d suffix)
                npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_draws.npz"
                if not os.path.exists(npz_file):
                    raise FileNotFoundError(
                        f"3D fit file not found: {npz_file}\n"
                        f"Please run run_model_comparison(fit_3d=True) first to generate 3D fits."
                    )

            alpha_samples_condition = []
            draws = np.load(npz_file)
            omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])

            # Each fly's data is stored as (n_draws, 3) where columns are [alpha, beta, p0]
            for key in draws.files:
                # Extract fly_id from key like "fly_id_31"
                fly_id = int(key.split('_')[-1])
                if fly_id in omit_fly_ids:
                    continue

                fly_params = draws[key]  # shape: (n_draws, 3)
                alpha_samples = fly_params[:, 0]  # first column is alpha
                alpha_samples_condition.append(alpha_samples)

            # Pool alpha samples for this condition and compute mean
            alpha_pooled = np.concatenate(alpha_samples_condition)
            alpha_global = np.mean(alpha_pooled)
            alpha_global_dict[(genotype, day)] = alpha_global

            print(f"  {genotype} day {day}: alpha_global = {alpha_global:.6f} "
                  f"(n={len(alpha_pooled)} samples, range=[{np.min(alpha_pooled):.4f}, {np.max(alpha_pooled):.4f}])")

    return alpha_global_dict


def run_model_comparison(fit_3d=False, fit_2d=True, fit_1d=True):
    """
    Fit 1D, 2D, and 3D models for model comparison.

    Parameters:
        fit_3d (bool): Whether to fit 3D model (if False, uses existing fits)
        fit_2d (bool): Whether to fit 2D model
        fit_1d (bool): Whether to fit 1D model

    Outputs:
        - CSV summaries: fly-stability-days-detailed-{model}.csv
        - NPZ draws: {genotype}_day{day}_{model}_draws.npz

    Notes:
        - for 2D model, uses fixed alpha_global = 0.2 for all conditions (in 3D model, this is a free parameter)
    """
    print("\n\nMODEL COMPARISON: 1D, 2D, 3D")
    print("===================================================================")

    model_dir = DIR_STAN + os.sep + 'dynamics' + os.sep
    genotypes = ['GD', 'KK']
    days = [7, 14, 21]

    # ========== 3D MODEL ==========
    if fit_3d:
        print("\n>>> Fitting 3D model (p0, alpha, beta)")
        print("===================================================================")
        model_3d, init_fun_3d = no_pooled(model_dir)
        data_3d = []

        for genotype in genotypes:
            for day in days:
                file_name = to_file(genotype, day, filtered=True, detailed_format=True)
                omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
                parsed_data_dict = parse_fly_data(file_name, detailed_format=True, omit_fly_ids=omit_fly_ids)

                print(f"\n{genotype=} {day=}")
                print("------------------------------------------------------------")

                start_time = time.time()
                sample = fit(model_3d, init_fun_3d, genotype, day, show_progress=False, filtered=True, detailed_format=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Fit completed in {elapsed_time/60:.2f} minutes")

                # Check Stan diagnostics
                print("\nStan Diagnostics:")
                print(sample.diagnose())

                # Extract posterior draws
                alpha_draws = sample.stan_variable('alpha')
                beta_draws = sample.stan_variable('beta')
                p0_draws = sample.stan_variable('p0')

                alpha_hat = alpha_draws.mean(axis=0)
                beta_hat = beta_draws.mean(axis=0)
                p0_hat = p0_draws.mean(axis=0)

                draws_dict = {}
                num_flies = len(alpha_hat)

                for idx in range(num_flies):
                    fly_id = parsed_data_dict['fly_id'][idx]

                    # CSV data
                    data_3d.append({
                        'genotype': genotype,
                        'day': day,
                        'fly': idx,
                        'fly_id': fly_id,
                        'alpha': alpha_hat[idx],
                        'beta': beta_hat[idx],
                        'p0': p0_hat[idx],
                        'log_alpha': np.log(alpha_hat[idx]),
                        'log_beta': np.log(beta_hat[idx]),
                        'logit_p0': logit(p0_hat[idx]),
                        'chamber_id': parsed_data_dict['chamber_id'][idx],
                        'fly_id_64': parsed_data_dict['fly_id_64'][idx],
                        'var': parsed_data_dict['var'][idx],
                        'exp_name': parsed_data_dict['exp_name'][idx],
                        'date': parsed_data_dict['date'][idx],
                        'jumpdata': parsed_data_dict['jumpdata_str'][idx],
                    })

                    # NPZ data: (alpha, beta, p0) triplets
                    param_triplets = np.stack([alpha_draws[:, idx], beta_draws[:, idx], p0_draws[:, idx]], axis=1)
                    draws_dict[f"fly_id_{fly_id}"] = param_triplets

                # Save draws
                npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_3d_draws.npz"
                np.savez_compressed(npz_file, **draws_dict)

        # Save CSV summary
        df_3d = pd.DataFrame(data_3d)
        df_3d.to_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-3d.csv', index=False)
        print("\n3D model fitting complete!")

    # Use fixed alpha_global = 0.2 for all conditions (cleaner model comparison)
    # (Previously computed from 3D fits, but using fixed value is more principled)
    ALPHA_GLOBAL_FIXED = 0.2

    # ========== 2D MODEL ==========
    if fit_2d:
        print("\n>>> Fitting 2D model (p0, beta) with FIXED alpha_global = 0.2")
        print("===================================================================")
        model_2d, init_fun_2d = no_pooled_2d(model_dir)
        data_2d = []

        for genotype in genotypes:
            for day in days:
                alpha_global = ALPHA_GLOBAL_FIXED
                file_name = to_file(genotype, day, filtered=True, detailed_format=True)
                omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
                parsed_data_dict = parse_fly_data(file_name, detailed_format=True, omit_fly_ids=omit_fly_ids)

                print(f"\n{genotype=} {day=} alpha_global={alpha_global:.6f}")
                print("------------------------------------------------------------")

                start_time = time.time()
                sample = fit_2d_alpha_global(model_2d, init_fun_2d, alpha_global, genotype, day,
                                             show_progress=False, filtered=True, detailed_format=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Fit completed in {elapsed_time/60:.2f} minutes")

                # Check Stan diagnostics
                print("\nStan Diagnostics:")
                print(sample.diagnose())

                # Extract posterior draws
                beta_draws = sample.stan_variable('beta')
                p0_draws = sample.stan_variable('p0')

                beta_hat = beta_draws.mean(axis=0)
                p0_hat = p0_draws.mean(axis=0)

                draws_dict = {}
                num_flies = len(beta_hat)

                for idx in range(num_flies):
                    fly_id = parsed_data_dict['fly_id'][idx]

                    # CSV data
                    data_2d.append({
                        'genotype': genotype,
                        'day': day,
                        'fly': idx,
                        'fly_id': fly_id,
                        'alpha': alpha_global,  # fixed value
                        'beta': beta_hat[idx],
                        'p0': p0_hat[idx],
                        'log_alpha': np.log(alpha_global),
                        'log_beta': np.log(beta_hat[idx]),
                        'logit_p0': logit(p0_hat[idx]),
                        'chamber_id': parsed_data_dict['chamber_id'][idx],
                        'fly_id_64': parsed_data_dict['fly_id_64'][idx],
                        'var': parsed_data_dict['var'][idx],
                        'exp_name': parsed_data_dict['exp_name'][idx],
                        'date': parsed_data_dict['date'][idx],
                        'jumpdata': parsed_data_dict['jumpdata_str'][idx],
                    })

                    # NPZ data: (beta, p0) pairs
                    param_pairs = np.stack([beta_draws[:, idx], p0_draws[:, idx]], axis=1)
                    draws_dict[f"fly_id_{fly_id}"] = param_pairs

                # Save draws
                npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_2d_draws.npz"
                np.savez_compressed(npz_file, **draws_dict)

        # Save CSV summary
        df_2d = pd.DataFrame(data_2d)
        df_2d.to_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-2d.csv', index=False)
        print("\n2D model fitting complete!")

    # ========== 1D MODEL ==========
    if fit_1d:
        print("\n>>> Fitting 1D model (p0 only)")
        print("===================================================================")
        model_1d, init_fun_1d = no_pooled_1d(model_dir)
        data_1d = []

        for genotype in genotypes:
            for day in days:
                file_name = to_file(genotype, day, filtered=True, detailed_format=True)
                omit_fly_ids = OMIT_FLY_IDS.get(genotype, [])
                parsed_data_dict = parse_fly_data(file_name, detailed_format=True, omit_fly_ids=omit_fly_ids)

                print(f"\n{genotype=} {day=}")
                print("------------------------------------------------------------")

                start_time = time.time()
                sample = fit(model_1d, init_fun_1d, genotype, day, show_progress=False, filtered=True, detailed_format=True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Fit completed in {elapsed_time/60:.2f} minutes")

                # Check Stan diagnostics
                print("\nStan Diagnostics:")
                print(sample.diagnose())

                # Extract posterior draws
                p0_draws = sample.stan_variable('p0')
                p0_hat = p0_draws.mean(axis=0)

                draws_dict = {}
                num_flies = len(p0_hat)

                for idx in range(num_flies):
                    fly_id = parsed_data_dict['fly_id'][idx]

                    # CSV data
                    data_1d.append({
                        'genotype': genotype,
                        'day': day,
                        'fly': idx,
                        'fly_id': fly_id,
                        'alpha': np.nan,  # not fitted
                        'beta': np.nan,   # not fitted
                        'p0': p0_hat[idx],
                        'log_alpha': np.nan,
                        'log_beta': np.nan,
                        'logit_p0': logit(p0_hat[idx]),
                        'chamber_id': parsed_data_dict['chamber_id'][idx],
                        'fly_id_64': parsed_data_dict['fly_id_64'][idx],
                        'var': parsed_data_dict['var'][idx],
                        'exp_name': parsed_data_dict['exp_name'][idx],
                        'date': parsed_data_dict['date'][idx],
                        'jumpdata': parsed_data_dict['jumpdata_str'][idx],
                    })

                    # NPZ data: just p0 values (as 2D array for consistency)
                    draws_dict[f"fly_id_{fly_id}"] = p0_draws[:, idx:idx+1]  # shape: (n_draws, 1)

                # Save draws
                npz_file = DIR_FITS + os.sep + f"{genotype}_day{day}_1d_draws.npz"
                np.savez_compressed(npz_file, **draws_dict)

        # Save CSV summary
        df_1d = pd.DataFrame(data_1d)
        df_1d.to_csv(DIR_FITS + os.sep + 'fly-stability-days-detailed-1d.csv', index=False)
        print("\n1D model fitting complete!")

    print("\n\nMODEL COMPARISON COMPLETE!")
    print("===================================================================")
    print("Output files:")
    if fit_3d:
        print("  - fly-stability-days-detailed-3d.csv")
        print("  - {genotype}_day{day}_3d_draws.npz")
    if fit_2d:
        print("  - fly-stability-days-detailed-2d.csv")
        print("  - {genotype}_day{day}_2d_draws.npz")
    if fit_1d:
        print("  - fly-stability-days-detailed-1d.csv")
        print("  - {genotype}_day{day}_1d_draws.npz")


if __name__ == '__main__':

    # ==========================================
    reduce_griping()

    fit_diagnostic = False
    fit_main_3d = True
    fit_1d_and_2d = False

    # 0) Perform inference for one of: the pooled / partially-pooled / no pooled models
    # - quick diagnostic runs
    if fit_diagnostic:
        run_pooled()
        #run_partial_pooled_multi()
        samples = run_no_pooled()

    # 1) Perform inference for no-pooled model ('full model' - fit all 3 params per fly)
    if fit_main_3d:
        # 1) run fit
        estimates_df = run_fly_stability_detailed_csv()

        # 2) Calculate correlation between the parameters
        corr_df = calculate_day_correlations()

    # 3) Perform model comparison by fitting 1D and 2D models (fewer params per fly)
    if fit_1d_and_2d:
        run_model_comparison(fit_3d=False, fit_2d=True, fit_1d=True)
