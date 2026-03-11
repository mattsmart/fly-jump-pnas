#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, cm, patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2

import os
import sys

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

from data_tools import df_to_arr_jumps, get_TTC_canonical, habituation_time_fisherexact
from data_format_add_score_columns import (
    compute_p_ss, compute_hab_time_half_rel, compute_hab_time_95_rel,
    compute_hab_magnitude_abs, compute_hab_magnitude_rel)
from functions_common import likelihood_func_vec
from settings import DIR_OUTPUT, DIR_DATA_EXPT, DIR_FITS, OMIT_FLY_IDS

##############################
# Toggle flags:
MEDIAN_MODE = False       # if True, use median instead of mean for central value calculations.
SHOW_CI = True            # if True, overlay 95% CI ellipsoid/ellipse.
COLOR_BY = "box"          # choose grouping for colors: "chamber" or "box".
##############################

# --- Highlight Fly IDs ---
HIGHLIGHT_FLY_IDS = {
    'KK': [],
    'GD': []
}

CHAMBER_IDX_TO_BOX = {i: ((i - 1) // 8) for i in range(1, 33)}

def central_value(x):
    return np.median(x) if MEDIAN_MODE else np.mean(x)

################################
# Load Data
################################
df_summary_csv = pd.read_csv(os.path.join(DIR_FITS, "fly-stability-days-detailed-habscores.csv"))
print(df_summary_csv.head())

draws_GD_day7 = np.load(os.path.join(DIR_FITS, "GD_day7_3d_draws.npz"), allow_pickle=True)
draws_GD_day14 = np.load(os.path.join(DIR_FITS, "GD_day14_3d_draws.npz"), allow_pickle=True)
draws_GD_day21 = np.load(os.path.join(DIR_FITS, "GD_day21_3d_draws.npz"), allow_pickle=True)

draws_KK_day7 = np.load(os.path.join(DIR_FITS, "KK_day7_3d_draws.npz"), allow_pickle=True)
draws_KK_day14 = np.load(os.path.join(DIR_FITS, "KK_day14_3d_draws.npz"), allow_pickle=True)
draws_KK_day21 = np.load(os.path.join(DIR_FITS, "KK_day21_3d_draws.npz"), allow_pickle=True)

draws_over_age = {
    'KK': {7: draws_KK_day7, 14: draws_KK_day14, 21: draws_KK_day21},
    'GD': {7: draws_GD_day7, 14: draws_GD_day14, 21: draws_GD_day21}
}

genotype = 'KK'
age = 14

df_summary_csv_genotype = df_summary_csv[df_summary_csv['genotype'] == genotype]
df_summary_csv_genotype_day = df_summary_csv_genotype[df_summary_csv_genotype['day'] == age]

draws = draws_over_age[genotype][age]
num_draws = draws[list(draws.keys())[0]].shape[0]

num_flies = 128
available_fly_keys = [key for key in draws.keys() if key.startswith("fly_id_")]
omit_ids = OMIT_FLY_IDS.get(genotype, [])
available_fly_keys = [k for k in available_fly_keys if int(k.split('_')[-1]) not in omit_ids]
available_fly_keys.sort(key=lambda x: int(x.split('_')[-1]))
available_fly_ids = [int(k.split('_')[-1]) for k in available_fly_keys]
chambers_for_avail_fly_ids = df_summary_csv_genotype_day[
    df_summary_csv_genotype_day['fly_id'].isin(available_fly_ids)
]['chamber_id'].values
boxes_for_avail_fly_ids = [CHAMBER_IDX_TO_BOX[chamber_id] for chamber_id in chambers_for_avail_fly_ids]
fly_to_box = {fly_key: box for fly_key, box in zip(available_fly_keys, boxes_for_avail_fly_ids)}
fly_to_chamber = {fly_key: chamber for fly_key, chamber in zip(available_fly_keys, chambers_for_avail_fly_ids)}

num_flies = min(num_flies, len(available_fly_keys))
default_xticklabels = [k.split('_')[-1] for k in available_fly_keys]

################################
# Set up colormap for scatter
################################
scatter_cmap = plt.get_cmap('Spectral')
all_p0 = []
for key in available_fly_keys:
    samples = draws[key]
    all_p0.extend(samples[:, 2])
vmin = np.min(all_p0)
vmax = np.max(all_p0)
print('p0: vmin/vmax (all draws) - ', vmin, vmax)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

metric_names = ['hab_tail', 'sra_mean', 'hab_halftime_rel', 'hab_saturation_time_rel',
                'hab_ttc', 'hab_fisherexact', 'hab_magnitude_rel', 'hab_magnitude_abs',
                'alpha', 'beta', 'p0']
metrics_to_skip = ['hab_fisherexact']

data_for_out_csv = {
    a: {'data': np.zeros((num_flies, num_draws)),
        'fly_id': available_fly_keys}
    for a in metric_names
}

################################
# Transform functions
################################
def log_forward(x):
    return np.log(x)
def log_inv(z):
    return np.exp(z)

def logit_forward(x):
    return np.log(x/(1.0 - x))
def logit_inv(z):
    return 1.0/(1.0 + np.exp(-z))

# For Case 1: [alpha, beta, p0]
forward_case1 = [log_forward, log_forward, logit_forward]
inv_case1     = [log_inv,     log_inv,     logit_inv]

# For Case 2: [mean_SRA, hab_tail, hab_halftime_rel]
forward_case2 = [logit_forward, logit_forward, log_forward]
inv_case2     = [logit_inv,     logit_inv,     log_inv]

################################
# compute_case2_measures
################################
def compute_case2_measures(samples):
    alpha, beta, p0 = samples[:,0], samples[:,1], samples[:,2]
    sra_curves = likelihood_func_vec(np.arange(50), alpha, beta, p0, pulse_period=5)
    sra_draws = np.mean(sra_curves, axis=1)
    hab_tail_draws = compute_p_ss(alpha, beta, p0, T=1)
    hab_halftime_draws = compute_hab_time_half_rel(alpha, beta, T=1)
    return np.column_stack([sra_draws, hab_tail_draws, hab_halftime_draws])

################################
# Plotting Ellipses/Ellipsoids (Fixed version)
################################
def plot_3d_ellipsoid(ax, mu_t, cov_t, n_std, forward=None, inv=None,
                      facecolor='none', edgecolor='k', alpha=0.2, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov_t)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rx, ry, rz = n_std * np.sqrt(eigvals)
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    n_u, n_v = x.shape
    for i in range(n_u):
        for j in range(n_v):
            pt = np.array([x[i,j], y[i,j], z[i,j]])
            pt_trans = mu_t + eigvecs @ pt
            pt_orig = np.array([inv[k](pt_trans[k]) for k in range(3)])
            x[i,j], y[i,j], z[i,j] = pt_orig
    ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolor=facecolor,
                    edgecolor=edgecolor, alpha=alpha, **kwargs)

def plot_ellipse(ax, mu_t, cov_t, n_std, dims, forward=None, inv=None,
                 facecolor='none', edgecolor='k', alpha=0.2, **kwargs):
    """
    Generate a local ellipse around (0,0), rotate it, then shift by mu_t.
    Finally, apply inverse transforms if in "constrained" mode.
    """
    eigvals, eigvecs = np.linalg.eigh(cov_t)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)

    theta = np.linspace(0, 2*np.pi, 100)
    # local coords (centered at 0,0)
    xs = (width/2)*np.cos(theta)
    ys = (height/2)*np.sin(theta)

    # rotate
    R = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle))]
    ])
    pts = np.vstack((xs, ys))  # shape (2, 100)
    pts = R @ pts
    # shift by mu_t
    pts[0,:] += mu_t[0]
    pts[1,:] += mu_t[1]

    # If length(inv) == 3 => we are in constrained mode (Case 1 or Case 2).
    # If length(inv) == len(dims) => unconstrained mode => pass identity transforms.
    if len(inv) == 3:
        # For each dimension in dims, apply the appropriate inverse transform.
        for local_idx, global_dim in enumerate(dims):
            pts[local_idx,:] = [inv[global_dim](val) for val in pts[local_idx,:]]
    else:
        # length == len(dims) => unconstrained => no transform or identity
        for local_idx in range(len(dims)):
            pts[local_idx,:] = [inv[local_idx](val) for val in pts[local_idx,:]]

    ellipse_patch = patches.Polygon(np.column_stack((pts[0], pts[1])), closed=True,
                                    facecolor=facecolor, edgecolor=edgecolor,
                                    alpha=alpha, **kwargs)
    ax.add_patch(ellipse_patch)

################################
# Scale factors for 95% CI
################################
scale_3d = np.sqrt(chi2.ppf(0.95, df=3))
scale_2d = np.sqrt(chi2.ppf(0.95, df=2))

################################
# Plotting Functions for Constrained Space
################################
def get_fly_color(fid):
    if COLOR_BY == "chamber":
        grp = fly_to_chamber[fid]
        cmap_group = plt.get_cmap("hsv", 32)
        return cmap_group(grp)
    else:
        grp = fly_to_box[fid]
        cmap_group = plt.get_cmap("tab10", 4)
        return cmap_group(grp)

def plot_case1_3d(fly_summaries, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_orig[0], mu_orig[1], mu_orig[2], color=c, s=40)
        if SHOW_CI:
            plot_3d_ellipsoid(ax, mu_t, cov_t, n_std=scale_3d,
                              forward=forward_case1, inv=inv_case1,
                              facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_zlabel("p0")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_case1_2d(fly_summaries, dims, title):
    fig, ax = plt.subplots()
    label_dict = {0: "alpha", 1: "beta", 2: "p0"}
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_orig[dims[0]], mu_orig[dims[1]], color=c, s=40)
        if SHOW_CI:
            sub_mu = mu_t[list(dims)]
            sub_cov = cov_t[np.ix_(dims, dims)]
            plot_ellipse(ax, sub_mu, sub_cov, n_std=scale_2d, dims=dims,
                         forward=forward_case1, inv=inv_case1,
                         facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel(label_dict[dims[0]])
    ax.set_ylabel(label_dict[dims[1]])
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_case2_3d(fly_summaries, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_orig[0], mu_orig[1], mu_orig[2], color=c, s=40)
        if SHOW_CI:
            plot_3d_ellipsoid(ax, mu_t, cov_t, n_std=scale_3d,
                              forward=forward_case2, inv=inv_case2,
                              facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel("mean_SRA")
    ax.set_ylabel("hab_tail")
    ax.set_zlabel("hab_halftime_rel")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_case2_2d(fly_summaries, dims, title, flip_xy=False):
    fig, ax = plt.subplots()
    label_dict = {0: "mean_SRA", 1: "hab_tail", 2: "hab_halftime_rel"}
    if flip_xy:
        dims = (dims[1], dims[0])
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_orig[dims[0]], mu_orig[dims[1]], color=c, s=40)
        if SHOW_CI:
            sub_mu = mu_t[list(dims)]
            sub_cov = cov_t[np.ix_(dims, dims)]
            plot_ellipse(ax, sub_mu, sub_cov, n_std=scale_2d, dims=dims,
                         forward=forward_case2, inv=inv_case2,
                         facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel(label_dict[dims[0]])
    ax.set_ylabel(label_dict[dims[1]])
    ax.set_title(title)
    fig.tight_layout()
    return fig

################################
# Unconstrained Plot Functions
################################
identity = lambda x: x

def ellipsoid_surface(mu, cov, n_std, num_points=50):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rx, ry, rz = n_std * np.sqrt(eigvals)
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, num_points),
                       np.linspace(0, np.pi, num_points))
    x = rx * np.cos(u) * np.sin(v) + mu[0]
    y = ry * np.sin(u) * np.sin(v) + mu[1]
    z = rz * np.cos(v) + mu[2]
    return x, y, z

def plot_case1_3d_unconstrained(fly_summaries, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_t[0], mu_t[1], mu_t[2], color=c, s=40)
        if SHOW_CI:
            X, Y, Z = ellipsoid_surface(mu_t, cov_t, n_std=scale_3d)
            ax.plot_surface(X, Y, Z, rstride=4, cstride=4,
                            facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel("log(alpha)")
    ax.set_ylabel("log(beta)")
    ax.set_zlabel("logit(p0)")
    ax.set_title(title + " (unconstrained)")
    fig.tight_layout()
    return fig

def plot_case1_2d_unconstrained(fly_summaries, dims, title):
    fig, ax = plt.subplots()
    label_dict = {0:"log(alpha)", 1:"log(beta)", 2:"logit(p0)"}
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_t[dims[0]], mu_t[dims[1]], color=c, s=40)
        if SHOW_CI:
            sub_mu = mu_t[list(dims)]
            sub_cov = cov_t[np.ix_(dims, dims)]
            plot_ellipse(ax, sub_mu, sub_cov, n_std=scale_2d, dims=dims,
                         forward=[identity]*len(dims), inv=[identity]*len(dims),
                         facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel(label_dict[dims[0]])
    ax.set_ylabel(label_dict[dims[1]])
    ax.set_title(title + " (unconstrained)")
    fig.tight_layout()
    return fig

def plot_case2_3d_unconstrained(fly_summaries, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_t[0], mu_t[1], mu_t[2], color=c, s=40)
        if SHOW_CI:
            X, Y, Z = ellipsoid_surface(mu_t, cov_t, n_std=scale_3d)
            ax.plot_surface(X, Y, Z, rstride=4, cstride=4,
                            facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel("logit(mean_SRA)")
    ax.set_ylabel("logit(hab_tail)")
    ax.set_zlabel("log(hab_halftime_rel)")
    ax.set_title(title + " (unconstrained)")
    fig.tight_layout()
    return fig

def plot_case2_2d_unconstrained(fly_summaries, dims, title, flip_xy=False):
    fig, ax = plt.subplots()
    label_dict = {0:"logit(mean_SRA)", 1:"logit(hab_tail)", 2:"log(hab_halftime_rel)"}
    if flip_xy:
        dims = (dims[1], dims[0])
    for fid, (mu_t, cov_t, mu_orig) in fly_summaries.items():
        c = get_fly_color(fid)
        ax.scatter(mu_t[dims[0]], mu_t[dims[1]], color=c, s=40)
        if SHOW_CI:
            sub_mu = mu_t[list(dims)]
            sub_cov = cov_t[np.ix_(dims, dims)]
            plot_ellipse(ax, sub_mu, sub_cov, n_std=scale_2d, dims=dims,
                         forward=[identity]*len(dims), inv=[identity]*len(dims),
                         facecolor=c, edgecolor='k', alpha=0.2)
    ax.set_xlabel(label_dict[dims[0]])
    ax.set_ylabel(label_dict[dims[1]])
    ax.set_title(title + " (unconstrained)")
    fig.tight_layout()
    return fig

################################
# Compute summaries in transformed space
################################
fly_summaries_case1 = {}
for fid in available_fly_keys:
    samples = draws[fid]  # shape: (n_draws, 3) => [alpha, beta, p0]
    trans_draws = np.column_stack([forward_case1[i](samples[:, i]) for i in range(3)])
    mu_t = np.mean(trans_draws, axis=0)
    cov_t = np.cov(trans_draws, rowvar=False)
    mu_orig = np.array([inv_case1[i](mu_t[i]) for i in range(3)])
    fly_summaries_case1[fid] = (mu_t, cov_t, mu_orig)

def compute_case2_measures(samples):
    alpha, beta, p0 = samples[:,0], samples[:,1], samples[:,2]
    sra_curves = likelihood_func_vec(np.arange(50), alpha, beta, p0, pulse_period=5)
    sra_draws = np.mean(sra_curves, axis=1)
    hab_tail_draws = compute_p_ss(alpha, beta, p0, T=1)
    hab_halftime_draws = compute_hab_time_half_rel(alpha, beta, T=1)
    return np.column_stack([sra_draws, hab_tail_draws, hab_halftime_draws])

fly_summaries_case2 = {}
for fid in available_fly_keys:
    samples = draws[fid]
    draws_case2 = compute_case2_measures(samples)  # shape: (n_draws, 3)
    trans_draws = np.column_stack([forward_case2[i](draws_case2[:, i]) for i in range(3)])
    mu_t = np.mean(trans_draws, axis=0)
    cov_t = np.cov(trans_draws, rowvar=False)
    mu_orig = np.array([inv_case2[i](mu_t[i]) for i in range(3)])
    fly_summaries_case2[fid] = (mu_t, cov_t, mu_orig)

################################
# Produce paired plots for each case: constrained vs. unconstrained
################################
# Case 1
fig1_case1_3d_constrained = plot_case1_3d(fly_summaries_case1, f"Case 1 (α, β, p₀) for {genotype} day {age}")
fig1_case1_3d_unconstrained = plot_case1_3d_unconstrained(fly_summaries_case1, f"Case 1 (transformed) for {genotype} day {age}")
fig1_case1_2d_xy_constrained = plot_case1_2d(fly_summaries_case1, dims=(0,1),
                                             title=f"Case 1 (α vs β) for {genotype} day {age}")
fig1_case1_2d_xy_unconstrained = plot_case1_2d_unconstrained(fly_summaries_case1, dims=(0,1),
                                             title=f"Case 1 (transformed: log(α) vs log(β)) for {genotype} day {age}")
fig1_case1_2d_xz_constrained = plot_case1_2d(fly_summaries_case1, dims=(0,2),
                                             title=f"Case 1 (α vs p₀) for {genotype} day {age}")
fig1_case1_2d_xz_unconstrained = plot_case1_2d_unconstrained(fly_summaries_case1, dims=(0,2),
                                             title=f"Case 1 (transformed: log(α) vs logit(p₀)) for {genotype} day {age}")
fig1_case1_2d_yz_constrained = plot_case1_2d(fly_summaries_case1, dims=(1,2),
                                             title=f"Case 1 (β vs p₀) for {genotype} day {age}")
fig1_case1_2d_yz_unconstrained = plot_case1_2d_unconstrained(fly_summaries_case1, dims=(1,2),
                                             title=f"Case 1 (transformed: log(β) vs logit(p₀)) for {genotype} day {age}")

# Case 2
fig1_case2_3d_constrained = plot_case2_3d(fly_summaries_case2,
    f"Case 2 (mean_SRA, hab_tail, hab_halftime_rel) for {genotype} day {age}")
fig1_case2_3d_unconstrained = plot_case2_3d_unconstrained(fly_summaries_case2,
    f"Case 2 (transformed) for {genotype} day {age}")
fig1_case2_2d_xy_constrained = plot_case2_2d(fly_summaries_case2, dims=(0,1),
    title=f"Case 2 (mean_SRA vs hab_tail) for {genotype} day {age}", flip_xy=True)
fig1_case2_2d_xy_unconstrained = plot_case2_2d_unconstrained(fly_summaries_case2, dims=(0,1),
    title=f"Case 2 (transformed: logit(mean_SRA) vs logit(hab_tail)) for {genotype} day {age}", flip_xy=True)
fig1_case2_2d_xz_constrained = plot_case2_2d(fly_summaries_case2, dims=(0,2),
    title=f"Case 2 (mean_SRA vs hab_halftime_rel) for {genotype} day {age}", flip_xy=True)
fig1_case2_2d_xz_unconstrained = plot_case2_2d_unconstrained(fly_summaries_case2, dims=(0,2),
    title=f"Case 2 (transformed: logit(mean_SRA) vs log(hab_halftime_rel)) for {genotype} day {age}", flip_xy=True)
fig1_case2_2d_yz_constrained = plot_case2_2d(fly_summaries_case2, dims=(1,2),
    title=f"Case 2 (hab_tail vs hab_halftime_rel) for {genotype} day {age}")
fig1_case2_2d_yz_unconstrained = plot_case2_2d_unconstrained(fly_summaries_case2, dims=(1,2),
    title=f"Case 2 (transformed: logit(hab_tail) vs log(hab_halftime_rel)) for {genotype} day {age}")

# Save a few unconstrained plots (example)
fig1_case1_3d_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case1_3d_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case1_2d_xy_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case1_xy_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case1_2d_xz_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case1_xz_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case1_2d_yz_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case1_yz_unconstrained_{genotype}_{age}.png"), dpi=300)

fig1_case2_3d_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case2_3d_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case2_2d_xy_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case2_xy_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case2_2d_xz_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case2_xz_unconstrained_{genotype}_{age}.png"), dpi=300)
fig1_case2_2d_yz_unconstrained.savefig(os.path.join(DIR_OUTPUT, f"case2_yz_unconstrained_{genotype}_{age}.png"), dpi=300)

plt.show()
