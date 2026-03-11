import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import seaborn as sns
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.special import logit


# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

from data_format_add_score_columns import (compute_hab_magnitude_rel, compute_hab_time_half_rel,
                                           helper_summarize_univariate_samples)
from functions_common import likelihood_func
from settings import DIR_FITS, DIR_OUTPUT, day_palette


"""
Currently un-used

Using posterior draws from e.g. KK age 14, extract the mean and covariance + cluster these across flies using K-means
"""


def extract_posterior_features_with_covariance(draws_day):
    features = []

    for fly_id, samples in draws_day.items():
        alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

        # Compute mean vector
        mean_vector = np.mean(samples, axis=0)

        # Compute the 3x3 covariance matrix
        cov_matrix = np.cov(samples.T)

        # Flatten the covariance matrix (6 unique values)
        cov_flat = cov_matrix[np.triu_indices(3)]

        # Combine mean and covariance into the feature vector
        features.append({
            'fly_id': fly_id,
            'alpha_mean': mean_vector[0],
            'beta_mean': mean_vector[1],
            'p0_mean': mean_vector[2],
            'var_alpha': cov_matrix[0, 0],
            'var_beta': cov_matrix[1, 1],
            'var_p0': cov_matrix[2, 2],
            'cov_alpha_beta': cov_matrix[0, 1],
            'cov_alpha_p0': cov_matrix[0, 2],
            'cov_beta_p0': cov_matrix[1, 2]
        })

    return pd.DataFrame(features)


def extract_transformed_posterior_features(draws_day, epsilon=1e-6):
    features = []

    for fly_id, samples in draws_day.items():
        alpha, beta, p0 = samples[:, 0], samples[:, 1], samples[:, 2]

        # Apply log transformation to alpha and beta
        log_alpha = np.log(alpha + epsilon)  # Avoid log(0)
        log_beta = np.log(beta + epsilon)

        # Apply logit transformation to p0
        logit_p0 = logit(np.clip(p0, epsilon, 1 - epsilon))  # Avoid logit(0) and logit(1)

        # Stack transformed samples for covariance
        transformed_samples = np.stack([log_alpha, log_beta, logit_p0], axis=1)

        # Compute mean vector
        mean_vector = np.mean(transformed_samples, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(transformed_samples.T)

        # Flatten the covariance matrix (upper triangle)
        cov_flat = cov_matrix[np.triu_indices(3)]

        # Store mean and covariance features
        features.append({
            'fly_id': fly_id,
            'log_alpha_mean': mean_vector[0],
            'log_beta_mean': mean_vector[1],
            'logit_p0_mean': mean_vector[2],
            'var_log_alpha': cov_matrix[0, 0],
            'var_log_beta': cov_matrix[1, 1],
            'var_logit_p0': cov_matrix[2, 2],
            'cov_log_alpha_beta': cov_matrix[0, 1],
            'cov_log_alpha_logit_p0': cov_matrix[0, 2],
            'cov_log_beta_logit_p0': cov_matrix[1, 2]
        })

    return pd.DataFrame(features)


def cluster_posteriors(feature_df, n_clusters=3):
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df.drop(columns=['fly_id']))

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    feature_df['cluster'] = kmeans.fit_predict(features_scaled)

    return feature_df


def visualize_clusters(feature_df, logmode=True):

    if logmode:
        vars = ['log_alpha_mean', 'log_beta_mean', 'logit_p0_mean']
    else:
        vars = ['alpha_mean', 'beta_mean', 'p0_mean']

    sns.pairplot(
        feature_df,
        vars=vars,
        hue='cluster',
        palette='Set2',
        diag_kind='kde'
    )
    plt.suptitle("Clustered posteriors (mean, covariance) across flies (logmode:%s)" % logmode, y=1.0)
    plt.tight_layout()
    plt.show()


def plot_clustered_time_series(clustered_df, alpha=0.3):
    """
    Plots all fly time series overlaid, colored by their cluster assignment.

    Args:
        clustered_df: DataFrame with cluster assignments and parameters.
        foo: Function that converts (alpha, beta, p0) → time series p(t).
        n_timepoints: Length of the time series.
        alpha: Transparency for overlapping lines.
    """
    # Define a color palette for the clusters
    cluster_colors = sns.color_palette("Set2", n_colors=clustered_df['cluster'].nunique())

    plt.figure(figsize=(12, 6))

    # Prepare storage for cluster means
    cluster_time_series = {cluster_id: [] for cluster_id in clustered_df['cluster'].unique()}

    # Loop through each fly
    for _, row in clustered_df.iterrows():
        # Extract alpha, beta, p0
        alpha_param = row['alpha_mean']
        beta_param = row['beta_mean']
        p0_param = row['p0_mean']

        # Generate time series p(t)
        t_sra = np.arange(0, 50)
        likelihood_sra = likelihood_func(t_sra, alpha_param, beta_param, p0_param, 5.0)

        t_hab = np.arange(0, 200)
        likelihood_hab = likelihood_func(t_hab, alpha_param, beta_param, p0_param, 1.0)

        t_merge = np.arange(0, 250)
        # concatenate likelihood_sra to likelihood_hab
        likelihood_merge = np.concatenate((likelihood_hab, likelihood_sra))

        # Plot the time series, colored by cluster
        plt.plot(
            t_merge,
            likelihood_merge,
            color=cluster_colors[row['cluster']],
            alpha=alpha
        )

        # Store for mean calculation
        cluster_time_series[row['cluster']].append(likelihood_merge)

    # Plot the mean time series for each cluster as a thicker line
    for cluster_id, series_list in cluster_time_series.items():
        # Compute the mean across all flies in the cluster
        cluster_mean_series = np.mean(series_list, axis=0)

        # Plot the mean as a thicker line
        plt.plot(
            t_merge,
            cluster_mean_series,
            color=cluster_colors[cluster_id],
            linewidth=3,
            label=f'Cluster {cluster_id} Mean'
        )

    # Final touches
    plt.xlabel("Time (t)")
    plt.ylabel("Likelihood p(t)")
    plt.title("Time Series Likelihoods Colored by Cluster with Mean Overlay")

    # Custom legend for clusters
    handles = [plt.Line2D([0], [0], color=cluster_colors[i], lw=2, label=f'Cluster {i}')
               for i in clustered_df['cluster'].unique()]
    plt.legend(handles=handles, title="Cluster ID", loc="upper right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Load posterior draws for multiple days/genotypes
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

    posterior_draws_to_cluster = draws_KK_day14

    print('Working on feature extraction and clustering...')
    # first do directly un the raw means (alpha, beta, p0)
    df_features = extract_posterior_features_with_covariance(posterior_draws_to_cluster)
    clustered_df = cluster_posteriors(df_features, n_clusters=1)
    visualize_clusters(clustered_df, logmode=False)
    clustered_df = cluster_posteriors(df_features, n_clusters=3)
    visualize_clusters(clustered_df, logmode=False)

    # Assuming foo(alpha, beta, p0) → p(t) time series
    plot_clustered_time_series(clustered_df)

    # now repeat with the transformed means (log, log, logit)
    df_features_log_logit = extract_transformed_posterior_features(posterior_draws_to_cluster)
    clustered_logmode_df = cluster_posteriors(df_features_log_logit, n_clusters=1)
    visualize_clusters(clustered_logmode_df, logmode=True)
    clustered_logmode_df = cluster_posteriors(df_features_log_logit, n_clusters=3)
    visualize_clusters(clustered_logmode_df, logmode=True)
