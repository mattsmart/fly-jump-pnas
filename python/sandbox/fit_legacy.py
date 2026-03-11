import cmdstanpy as csp
import pandas as pd
import numpy as np
import plotnine as pn


"""
Currently un-used; consider removing.
"""


def parse_data(csvfile):
    df = pd.read_csv(csvfile, header=None)
    fly_ids = df[0].values
    experiment_numbers = df[1].values
    response = df[2].apply(lambda x: list(map(int, x))).values
    response = np.vstack(response)
    response += 1
    return { 'num_experiments': len(fly_ids),
             'obs_per_experiment': 200,
             'num_flies': np.max(fly_ids),
             'experiment_numbers': experiment_numbers,
             'fly': fly_ids,
             'response': response }


def plot_cumulative(data_file, subsample_size=None):
    data = parse_data(data_file)
    response = data['response']
    indices = np.arange(response.shape[0])
    if subsample_size is not None:
        indices = np.random.choice(indices, subsample_size, replace=False)
    response = response[indices, :]
    n_experiment, experiment_len = np.shape(response)
    time = np.zeros(n_experiment * experiment_len)
    experiment = np.zeros(n_experiment * experiment_len)
    i = 0
    for m in range(n_experiment):
        response[m, 0] = response[m, 0] - 1
        for n in range(1, experiment_len):
            response[m, n] += (response[m, n - 1] - 1)
            experiment[i] = m
            time[i] = n
            i += 1
    flat_cum_jumps = np.array(response.flatten(), dtype=float)
    flat_cum_jumps += np.random.normal(0, 0.25, flat_cum_jumps.shape)
    df = pd.DataFrame({
        'Time': np.tile(np.arange(1, experiment_len + 1), n_experiment),
        'Cumulative Jumps': flat_cum_jumps,
        'Series': np.repeat(np.arange(n_experiment), experiment_len)
        })
    plot = (pn.ggplot(df, pn.aes(x='Time', y='Cumulative Jumps', group='Series'))
            + pn.geom_line(alpha=0.2)
            + pn.labs(x="time", y="cumulative jump (per experiment)"))
    plot.show()
    return plot

def plot_proportions(data_file):
    data = parse_data(data_file)
    response = data['response']
    response -= 1  # back to binary
    column_averages = response.mean(axis=0)
    df = pd.DataFrame({
        'Time': np.arange(1, response.shape[1] + 1),  # Time steps (1 to number of columns)
        'Proportion Jump': column_averages
        })
    df['Moving Average'] = df['Proportion Jump'].rolling(window=10, center=True).mean()
    alpha = 0.2456  # values are posterior means of Bayesian GLM with expectation given below
    beta = -0.7220
    gamma = -0.1218
    df['Theoretical Line'] = alpha + (1 - alpha) * np.exp(beta + gamma * (df['Time'] - 1))

    plot = (pn.ggplot(df, pn.aes(x='Time'))
        + pn.geom_line(pn.aes(y='Proportion Jump', color='"observed"'))
        # + pn.geom_line(pn.aes(y='Moving Average', color='"moving avg"'))
        + pn.geom_line(pn.aes(y='Theoretical Line', color='"estimated"'))
        + pn.labs(title="Fly jumps", x="time", y="proportion jump")
        + pn.geom_line(pn.aes(y='Theoretical Line', color='"estimate"'))
        + pn.labs(x="time", y="population jump proportion")
        + pn.scale_x_continuous(limits=(0, 200),
                                    breaks=25 * np.arange(9))
        + pn.scale_y_continuous(limits=(0, 1),
                                        breaks=(0.1 * np.arange(0, 11)))
        + pn.scale_color_manual(values={"observed": "black",
                                        "moving avg": "blue",
                                        "estimated": "red"},
                             name="Legend")
        )
    return plot


def fit_hmm_mle(data_file, stan_file, summarize=True):
    data = parse_data(data_file)
    m = csp.CmdStanModel(stan_file=stan_file)
    mle = m.optimize(data = data)

    if summarize:
        print(f"KEY  \t VALUE")
        for key, value in mle.optimized_params_dict.items():
            print(f"{key}:\t {value:0.2f}")
    return mle
    
def fit_hmm_mcmc(data_file, stan_file, summarize=True):
    data = parse_data(data_file)
    m = csp.CmdStanModel(stan_file=stan_file)
    draws = m.sample(chains=1, data=data)
    if summarize:
        print(draws.summary())
    return draws

def fit_glm_mcmc(data_file, stan_file, summarize=True):
    data = parse_data(data_file)
    data['response'] -= 1
    m = csp.CmdStanModel(stan_file=stan_file)
    draws = m.sample(data=data, inits=0, iter_warmup=1000, iter_sampling=1000, max_treedepth=10,
                         show_console=True, parallel_chains=4, refresh=10)
    if summarize:
        print(draws.summary())
    return draws

def plot_histogram(draws, v):
    v_draws = draws.stan_variable(v)
    means = v_draws.mean(axis=0)
    df = pd.DataFrame({v: means})
    plot = (pn.ggplot(df, pn.aes(x=v))
                + pn.geom_histogram(bins=20, color='white'))
    return plot

def plot_scatter(draws, v1, v2):
    v1_draws = draws.stan_variable(v1)
    v2_draws = draws.stan_variable(v2)
    means1 = v1_draws.mean(axis=0)
    means2 = v2_draws.mean(axis=0)
    df = pd.DataFrame({v1: means1, v2: means2})
    plot = (pn.ggplot(df, pn.aes(x=v1, y=v2))
                + pn.geom_point())
    return plot


# CONFIG
np.set_printoptions(threshold=np.inf)
pd.options.display.max_rows = None
data_file = '../data/KK_14days_responses_filtered.txt'


# EXPLORATORY PLOTS
plot_cumulative = plot_cumulative(data_file=data_file)
plot_cumulative.save('fly-jump-cumulative.pdf')
plot_cumulative.show()


plot_proportion = plot_proportions(data_file=data_file)
plot_proportion.save("fly-jump-population-rate.pdf")
plot_proportion.show()

# GLM MCMC POSTERIOR
draws = fit_glm_mcmc(data_file=data_file,
                     stan_file='../stan/fly-jump-glm-hier.stan')

for v in ('alpha', 'beta', 'gamma'):
    plot = plot_histogram(draws, v)
    plot.show()
    plot.save('fly-jump-marginal-posterior-' + v + '.pdf')

for (v1, v2) in (('alpha', 'beta'), ('alpha', 'gamma'), ('beta', 'gamma')):
    plot = plot_scatter(draws, v1, v2)
    plot.show()
    plot.save('fly-jump-scatter-posterior-' + v1 + '-' + v2 + '.pdf')


# OLD HMM FIT CODE

# mle = fit_hmm_mle(data_file=data_file,
#     stan_file='../stan/fly-jump-pooled.stan')

# draws = fit_hmm_mcmc(data_file=data_file,
#                      stan_file='../stan/fly-jump-pooled.stan')


