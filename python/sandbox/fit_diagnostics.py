import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import cmdstanpy as csp
from plotnine.options import aspect_ratio


""" 
Overview:

Select a point in parameter space (alpha, beta, p0). 
 - simulate a habituation experiment with 1050 trials
 - fit the simulated data with the same model
 - plot the posterior draws for the parameters --> do they contain the original generating parameters?
"""


def jump_probabilities(alpha, beta, p0):
    """
    This function calculates the probability of jumping over 1050 trials in a habituation experiment

    The assumed structure of th 1050 "trials":
        5x of: 200 shadows every 1s + 120 sec break (no stimuli)
        1x of: 50 shadows every 5s

    Assumptions:
    - the breaks fully reset the habituation process
    - the stimulus is a delta function with unit area

    At the n-th trial the fly jumps with probability p(n) or not with probability 1 - p(n)

    This functions returns the probability p(n) for n=0, 1,..., 1050
    """

    def sigma_of_x(x):
        """
        Want a function that maps a non-negative x to
            1 (for x = 0)
            0 (for x -> inf)
        """
        return 1 / (1 + x ** 2)

    # structure of the experimental stimulus timeseries (shadows) - this is fixed across the current dataset
    pulse_area = 1.0               # "shadow dose"
    num_trials_habituation = 200
    num_cycles_hab = 5
    num_trials_reactivity = 50
    period_hab = 1
    period_reactivity = 5

    # construct timeseries p[n] for n = 0, 1, ..., 1050
    q_hab = np.exp(- alpha * period_hab)
    q_reactivity = np.exp(- alpha * period_reactivity)
    Q = beta * pulse_area

    # timeseries for *** one *** habituation block
    arr_n = np.arange(0, num_trials_habituation)
    arr_x_hab_high = Q * (1 - q_hab ** arr_n) / (1 - q_hab)
    arr_x_hab_low = arr_x_hab_high * q_hab  # account for decay during the inter-stimulus break
    arr_p_hab = p0 * sigma_of_x(arr_x_hab_low)

    # timeseries for *** one *** reactivity block
    arr_n = np.arange(0, num_trials_reactivity)
    arr_x_reactivity_high = Q * (1 - q_reactivity ** arr_n) / (1 - q_reactivity)
    arr_x_reactivity_low = arr_x_reactivity_high * q_reactivity  # account for decay during the inter-stimulus break
    arr_p_reactivity = p0 * sigma_of_x(arr_x_reactivity_low)

    # combine the timeseries into a single array: p_of_n
    p_of_n = np.concatenate([arr_p_hab for _ in range(num_cycles_hab)] + [arr_p_reactivity])

    return p_of_n


def plot_posterior_draws(alpha, beta, p0, fit, num_flies, param_limits=None):

    # Extract the posterior draws for alpha, beta, and p0
    alpha_draws = fit.stan_variable('alpha')  # np.arr shape: fit.iter_sampling x num_flies
    beta_draws = fit.stan_variable('beta')
    p0_draws = fit.stan_variable('p0')
    print(alpha_draws.shape, beta_draws.shape, p0_draws.shape)

    num_draws, _ = alpha_draws.shape
    assert alpha_draws.shape[1] == num_flies

    mean_fit = np.mean(alpha_draws, axis=0), np.mean(beta_draws, axis=0), np.mean(p0_draws, axis=0)
    print('samples alpha mean', mean_fit[0].shape)

    for k in range(num_flies):

        df = pd.DataFrame({'alpha': alpha_draws[:, k], 'beta': beta_draws[:, k], 'p0': p0_draws[:, k]})

        # Generate Pairwise KDE Plot
        g = sns.pairplot(
            df,
            kind='kde',
            diag_kind='kde',
            corner=True,
            # palette='blue',
            plot_kws={'alpha': 0.4, 'fill': True}
        )

        # Annotate Posterior Means for Each Day
        for i, param_y in enumerate(['alpha', 'beta', 'p0']):
            for j, param_x in enumerate(['alpha', 'beta', 'p0']):

                ax = g.axes[i, j]

                if j < i:
                    mean_x = mean_fit[0][k] if param_x == 'alpha' else mean_fit[1][k] if param_x == 'beta' else mean_fit[2][k]
                    mean_y = mean_fit[0][k] if param_y == 'alpha' else mean_fit[1][k] if param_y == 'beta' else mean_fit[2][k]

                    true_x = alpha if param_x == 'alpha' else beta if param_x == 'beta' else p0
                    true_y = alpha if param_y == 'alpha' else beta if param_y == 'beta' else p0


                    ax.scatter(
                        [mean_x], [mean_y],
                        edgecolor='black',
                        s=80,
                        label=f'Mean (fit)',
                        zorder=20
                    )

                    ax.scatter(
                        [true_x], [true_y],
                        edgecolor='black',
                        s=80,
                        marker='*',
                        label=r'True $\theta$',
                        zorder=20
                    )

                    # Apply parameter-specific axis limits if provided
                    if param_limits[param_x] is not None:
                        ax.set_xlim(param_limits[param_x])
                    if param_limits[param_y] is not None:
                        ax.set_ylim(param_limits[param_y])

                    if i == 1 and j == 0:
                        ax.legend()

                if i==j:
                    mean_x = mean_fit[0][k] if param_x == 'alpha' else mean_fit[1][k] if param_x == 'beta' else \
                    mean_fit[2][k]
                    true_x = alpha if param_x == 'alpha' else beta if param_x == 'beta' else p0

                    ax.axvline(mean_x, linestyle='-', color='orange', linewidth=2)
                    ax.axvline(true_x, linestyle='--', color='black', linewidth=2)

        g.fig.subplots_adjust(wspace=0.3, hspace=0.2)  # Adjust horizontal/vertical spacing (def: 0.2)

        # Create annotation text for mean values
        annotation_text = "\n".join([
            f"Fit: α={mean_fit[0][k]:.3f}, β={mean_fit[1][k]:.3f}, p₀={mean_fit[2][k]:.3f}",
            f"True: α={alpha:.2f}, β={beta:.2f}, p₀={p0:.2f}"]
        )

        # Add annotation text to the top-right corner
        plt.gcf().text(0.55, 0.75, f"{annotation_text}",
                       fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Final touches
        plt.suptitle(f"Posterior KDE: Can we recover true parameters?\n(expt %d of %d)" % (k+1, num_flies), y=0.99)

        plt.show()


if __name__ == '__main__':
    print("1) Simulating data")
    alpha = 0.1  #0.01
    beta = 0.5   #0.01
    p0 = 0.6

    p_of_n = jump_probabilities(alpha, beta, p0)
    num_flies = 5  # when fitting many individuals, this corresp. to number of individuals
    num_trials_per_experiment = len(p_of_n)

    jump = np.zeros((num_flies, num_trials_per_experiment), dtype=int)
    for n in range(num_flies):
        jump[n] = np.random.binomial(1, p_of_n)


    fig, axarr = plt.subplots(2,1)
    axarr[0].imshow(jump, aspect=20, interpolation='none')
    axarr[0].set_title(r'Simulated data (same $\theta$, sample %d times)' % num_flies)

    print('p_of_n', len(p_of_n))
    print('num_flies', num_flies)
    print('num_trials_per_experiment', num_trials_per_experiment)

    axarr[1].plot(p_of_n[800:], '--ok')
    axarr[1].set_title(r'Jump probability for 1050 trials in the habituation experiment' + '\n'
              + r'$\alpha=%.2f$, $\beta=%.2f$, $p_0=%.2f$' % (alpha, beta, p0))
    axarr[1].set_ylim(0, 1)
    axarr[1].grid(alpha=0.5)

    plt.show()

    data = {
        'num_experiments': num_flies,
        'num_trials_per_experiment': num_trials_per_experiment,
        'jump': jump
    }

    # NOTE: these are taken from no-pool fn in fit_experimental_data.py; how does this interact with the prior?
    def init_fun(N=0):
        return {
            'alpha': N * [0.1],
            'beta':  N * [0.1],
            'p0':    N * [0.5]
        }

    print("2) Compiling Stan model...")
    #model = csp.CmdStanModel(stan_file='..' + os.sep + 'stan' + os.sep + 'dynamics' + os.sep + 'habituation-pool.stan')
    model = csp.CmdStanModel(stan_file='..' + os.sep + 'stan' + os.sep + 'dynamics' + os.sep + 'habituation-no-pool.stan')

    print('3) Performing fit...')
    fit = model.sample(
        data=data,
        #iter_warmup=600, iter_sampling=600,
        iter_warmup=400, iter_sampling=400,
        inits=init_fun(num_flies),  # when fitting many individuals, this corresp. to number of individuals
        chains=2, parallel_chains=2, max_treedepth=10,
        show_console=True, refresh=1
    )
    print(fit.summary(sig_figs=2))
    

    print('4) Plotting results (inferred mean likelihood curves)...')
    plt.plot(p_of_n[800:], '--ok', markersize=10, zorder=1)

    text_list = []
    for k in range(num_flies):
        alpha_fit_k = np.mean(fit.stan_variable('alpha')[:, k])
        beta_fit_k = np.mean(fit.stan_variable('beta')[:, k])
        p0_fit_k = np.mean(fit.stan_variable('p0')[:, k])

        p_of_n_fit = jump_probabilities(alpha_fit_k, beta_fit_k, p0_fit_k)

        fit_label = f"Fit {k + 1}: α={alpha_fit_k:.2f}, β={beta_fit_k:.2f}, p₀={p0_fit_k:.2f}"
        plt.plot(p_of_n_fit[800:], '--o', alpha=0.5, zorder=20, markersize=4, label=fit_label)

    plt.title(r'Jump probability for T=1s, 5s trials (recover true likelihood)' + '\n'
              + r'True: $\alpha=%.2f$, $\beta=%.2f$, $p_0=%.2f$' % (alpha, beta, p0))
    plt.xlabel('Trial number')
    plt.ylabel(r'$p(k)$')
    plt.ylim(0, 1)
    plt.grid(alpha=0.5)
    plt.legend(loc=9, fontsize=9)
    plt.tight_layout()
    plt.show()

    # make plot of posterior draws - scatter true params vs. sample mean
    print('4) Plotting results (posterior)...')
    param_limits_manual = dict(alpha=None, beta=None, p0=[0.0, 1.0])
    plot_posterior_draws(alpha, beta, p0, fit, num_flies,
                         param_limits=param_limits_manual)
