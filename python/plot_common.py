import numpy as np
import matplotlib.pyplot as plt

from functions_common import likelihood_func
from settings import days_palettes



def plot_posterior_likelihood_summary_over_days(draws_over_age, genotype, fly_id, days_to_show=[7, 14, 21],
                                                spaghetti=False, spaghetti_alpha=0.1, ci_level=0.95, n_samples=100,
                                                ax=None, show=False, save_path=None):
    """
    Plots a summary of likelihood time series across multiple days for a given fly,
    with individual traces and mean ± CI bands.

    Args:
        draws_over_age: Dictionary {genotype: {day: {fly_id: samples}}}
        genotype: Genotype of the fly ('GD' or 'KK').
        fly_id: Fly ID to plot.
        days_to_show: List of days to include in the plot (default: [7, 14, 21]).
        alpha: Transparency for individual traces.
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI).
        n_samples: Number of posterior samples to use for plotting.
        save_path: If provided, save figure as PNG and SVG (without extension)

    Function originally from: plot_posterior_draws.py
    """
    day_palette = days_palettes['Deep']
    colors = {7: day_palette[0], 14: day_palette[1], 21: day_palette[2]}

    if ax is None:
        fresh_figure = True
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    else:
        fresh_figure = False

    for day in days_to_show:
        print(draws_over_age[genotype][day]['fly_id_%d' % fly_id])
        try:
            # Extract posterior draws for the given fly on the specified day
            draws = draws_over_age[genotype][day]['fly_id_%d' % fly_id]

            # Subsample posterior draws if necessary
            if n_samples is None:
                n_samples = draws.shape[0]
            n_samples = min(n_samples, draws.shape[0])
            sample_indices = np.random.choice(draws.shape[0], size=n_samples, replace=False)
            sampled_draws = draws[sample_indices]

            # Time ranges for habituation and SRA phases
            t_hab = np.arange(0, 200)
            t_sra = np.arange(0, 50)
            t_merge = np.arange(0, 250)

            # Storage for likelihood samples
            likelihood_samples = []

            # Plot individual posterior sample likelihoods
            for sample in sampled_draws:
                alpha_draw, beta_draw, p0_draw = sample
                #alpha_param = np.exp(sample[0])  # Inverse log transform
                #beta_param = np.exp(sample[1])
                #p0_param = 1 / (1 + np.exp(-sample[2]))  # Inverse logit

                # Generate likelihood time series
                likelihood_hab = likelihood_func(t_hab, alpha_draw, beta_draw, p0_draw, 1.0)
                likelihood_sra = likelihood_func(t_sra, alpha_draw, beta_draw, p0_draw, 5.0)

                # Merge habituation and SRA phases
                likelihood_merge = np.concatenate((likelihood_hab, likelihood_sra))
                likelihood_samples.append(likelihood_merge)

                # Plot individual traces
                if spaghetti:
                    ax.plot(t_merge, likelihood_merge, color=colors[day], alpha=spaghetti_alpha, zorder=5)

            # Convert to array for summary statisticss
            likelihood_samples = np.array(likelihood_samples)

            # Compute mean and CI bands
            mean_likelihood = np.mean(likelihood_samples, axis=0)
            lower_ci = np.percentile(likelihood_samples, (1 - ci_level) / 2 * 100, axis=0)
            upper_ci = np.percentile(likelihood_samples, (1 + ci_level) / 2 * 100, axis=0)

            # Plot mean likelihood with CI bands
            ax.plot(t_merge[0:200], mean_likelihood[0:200], color=colors[day], linewidth=2, label=f'Day {day} Mean', zorder=20)
            ax.plot(t_merge[200:],  mean_likelihood[200:],  color=colors[day], linewidth=2, zorder=20)#, label=f'sra - Day {day} Mean')

            ax.fill_between(t_merge[0:200], lower_ci[0:200], upper_ci[0:200], color=colors[day], alpha=0.3, label=f'Day {day} {int(ci_level * 100)}% CI')
            ax.fill_between(t_merge[200:],  lower_ci[200:],  upper_ci[200:],  color=colors[day], alpha=0.3) #label=f'sra - Day {day} {int(ci_level * 100)}% CI')

        except KeyError:
            print(f"No data available for Fly {fly_id}, Genotype {genotype}, Day {day}")
            continue

    # Plot settings
    ax.set_ylim(0, 1)
    if fresh_figure:
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Likelihood p(t)")
        ax.set_title(f"Posterior Likelihood Summary for Fly {fly_id} ({genotype})")
        ax.legend()

    # Save figure if save_path provided
    if save_path is not None and fresh_figure:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', bbox_inches='tight')
        print(f"Saved: {save_path}.png and .svg")

    if show:
        plt.show()

    return ax
