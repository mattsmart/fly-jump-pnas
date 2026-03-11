import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.lines import Line2D
from seaborn import light_palette, rugplot

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

"""
This script extends the analysis of python/plot_multiday_correlations.py
"""

from settings import DIR_FITS, DIR_OUTPUT, days_palettes, OMIT_FLY_IDS

# Import your utility functions
from plot_multiday_stats_utils import (
    # Core data processing
    filter_complete_flies,
    extract_parameters_by_age,
    calculate_fly_distances,

    # Plotting functions
    plot_parameter_comparisons_across_ages,
    plot_parameter_comparisons_with_null,
    plot_path_efficiency,
    plot_transition_heatmaps,
    plot_stability_trajectories,
    plot_distance_comparison_scatters,
    plot_parameter_age_distributions,
    plot_parameter_normalized_trajectories,
    plot_parameter_rank_stability,
    #plot_parameter_transition_matrix,
    plot_transition_with_null_comparison,
)

from plot_multiday_utils import (
    process_phenotype_data,
    plot_parameter_consistency,
    examine_null_correlation_distributions
)

# --- User‐configurable ---
CHAMBER_IDX_TO_BOX = {i: ((i - 1) // 8) for i in range(1, 33)}
genotype = 'KK'
ages = [7, 14, 21]
HIGHLIGHT_ONLY = True  # if True, grey‐out the rest
age_styles = {7: {'marker': 'o'},
              14: {'marker': 's'},
              21: {'marker': '^'}}
MEDIAN_MODE = False
SHOW_CI = True

FLAG_SKIP_SLOW_TRANSITION_TEST = True
FLAG_SKIP_SLOW_KDE = True

# --- Generalization: select X and Y variables ---
X_VAR = 'p0'        # e.g. p0,   sra_mean, mag_abs, mag_rel, k_star
Y_VAR = 'p_ss'      # e.g. p_ss, sra_mean, or p_ss_normed

# Which paths to highlight?
HIGHLIGHT_FLY_IDS = [44, 105]

# Define target trajectory codes
TARGET_PATHS = []

# Load and preprocess summary data
df_summary = pd.read_csv(os.path.join(
    DIR_FITS, "fly-stability-days-detailed-habscores.csv"))

# Load posterior draws
draws_over_age = {
    g: {d: np.load(os.path.join(DIR_FITS, f"{g}_day{d}_3d_draws.npz"), allow_pickle=True)
        for d in ages}
    for g in ['KK', 'GD']
}

# Process phenotype data
df_pheno, df_paths, all_draws = process_phenotype_data(
    df_summary, genotype, ages, draws_over_age,
    X_VAR, Y_VAR, OMIT_FLY_IDS
)

# Define target paths
fly_paths = dict(zip(df_paths['fly_id'], df_paths['path_code']))
to_highlight = {f for f, p in fly_paths.items() if p in TARGET_PATHS}

# Create output directory if it doesn't exist
os.makedirs(DIR_OUTPUT, exist_ok=True)

# =====================================================================
# 1) Parameter comparisons across ages
fig = plot_parameter_comparisons_across_ages(
    df_pheno, [X_VAR, Y_VAR],
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    show_marginals=True
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY_across_ages.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY_across_ages.svg'),
            bbox_inches='tight')
plt.show()

# =====================================================================
# 2) Parameter comparisons with null model
fig_null, null_fig, null_results = plot_parameter_comparisons_with_null(
    df_pheno, [X_VAR, Y_VAR],
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    n_permutations=1000,
    random_seed=42,
    create_null_example_fig=True
)

plt.figure(fig_null.number)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY.svg'),
            bbox_inches='tight')

plt.figure(null_fig.number)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY_sepnull.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_stability_XY_sepnull.svg'),
            bbox_inches='tight')
plt.show()

# =====================================================================
# 2B) scrutinize the null model

# Analyze the null correlation distributions
print("Analyzing null correlation distributions...")
results_df, null_distributions, null_diag_fig = examine_null_correlation_distributions(
    df_pheno,
    [X_VAR, Y_VAR],
    ages=ages,
    n_permutations=1000
)

# Save the diagnostic figure
plt.figure(null_diag_fig.number)
plt.savefig(os.path.join(DIR_OUTPUT, 'null_correlation_diagnostics.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# Print summary table
print("\nNull Correlation Summary:")
print(results_df[['Parameter', 'Age1', 'Age2', 'Actual_r', 'Null_Mean', 'Null_Std', 'N_Flies', 'Value_Range']])

# =====================================================================
# 3) Plot transition heatmaps
fig_heat = plot_transition_heatmaps(df_paths, [X_VAR, Y_VAR])
plt.savefig(os.path.join(DIR_OUTPUT, 'trajectory_heatmaps.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# =====================================================================
# 4) Path efficiency plot
fig_eff, path_data = plot_path_efficiency(
    df_pheno, [X_VAR, Y_VAR],
    highlight_fly_ids=HIGHLIGHT_FLY_IDS
)
plt.savefig(os.path.join(DIR_OUTPUT, 'path_efficiency_scatter.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# =====================================================================
# 5) Original stability plot with lines and CI
fig_stab = plot_stability_trajectories(
    df_pheno, X_VAR, Y_VAR,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    to_highlight=to_highlight,
    highlight_only=HIGHLIGHT_ONLY,
    show_ci=SHOW_CI,
    age_styles=age_styles
)
plt.savefig(os.path.join(DIR_OUTPUT, f"stability_{genotype}_{X_VAR}_{Y_VAR}.png"),
            dpi=300, bbox_inches='tight')
plt.show()

# =====================================================================
# 6) Distance scatter comparisons (if needed)
distances = calculate_fly_distances(df_pheno, [X_VAR, Y_VAR], ages=ages)
fig, _ = plot_distance_comparison_scatters(
    df_pheno, [X_VAR, Y_VAR],
    distances=distances,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS
)
plt.savefig(os.path.join(DIR_OUTPUT, 'distance_AtoBtoC_comparisons.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# =====================================================================
# 7) Additional plots from the original script (as needed)
# For example, if you still want to plot these:

# Bar-plot of trajectory patterns
if hasattr(df_paths, 'path_code'):
    topk = df_paths.path_code.value_counts().nlargest(40)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=topk.index, y=topk.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("# flies")
    plt.title("Top trajectory patterns")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUTPUT, 'trajectory_patterns.png'), dpi=300)
    plt.show()

# =====================================================================
# 8) Llongitudinal distribution plots

# Dotplot version with highlighted flies tracked
fig_dotplot = plot_parameter_age_distributions(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    #track_flies='highlighted',
    track_flies='all',
    plot_type='dotplot',
    figsize=(12, 6)
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_age_distributions_dotplot.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# Create histogram version with all flies tracked
fig_hist = plot_parameter_age_distributions(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='all',
    plot_type='histogram',
    figsize=(12, 6)
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_age_distributions_histogram.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# Create violin version with a random subset of flies tracked
fig_violin = plot_parameter_age_distributions(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='random_subset',
    random_subset_size=5,
    plot_type='violin',
    figsize=(12, 6)
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_age_distributions_violin.png'),
           dpi=300, bbox_inches='tight')
plt.show()


# ==================================================================
# 9)
# Create Z-score normalized trajectories plot
fig_z = plot_parameter_normalized_trajectories(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='all',
    normalize_method='z_score'
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_zscore.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# Create percentile-rank version
fig_pct = plot_parameter_normalized_trajectories(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='random_subset',
    random_subset_size=15,
    normalize_method='percentile'
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_percentile.png'),
           dpi=300, bbox_inches='tight')
plt.show()


# =====================================================================
# 10) stability of position/rank across days, for each parameter

# Call the Parameter Consistency Plot function
print("Creating parameter consistency plots...")

# For p0 parameter
fig_consistency_p0 = plot_parameter_consistency(
    df_pheno,
    param_name=X_VAR,
    age_pairs=[(7, 14), (14, 21), (7, 21)],
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    quantile_bands=True,
    band_quantiles=[0.25, 0.5, 0.75],
    figsize=(15, 5)
)
plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_consistency_{X_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_consistency_{X_VAR}.svg'),
           bbox_inches='tight')
plt.show()

# For p_ss parameter
fig_consistency_pss = plot_parameter_consistency(
    df_pheno,
    param_name=Y_VAR,
    age_pairs=[(7, 14), (14, 21), (7, 21)],
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    quantile_bands=True,
    band_quantiles=[0.25, 0.5, 0.75],
    figsize=(15, 5)
)
plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_consistency_{Y_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'parameter_consistency_{Y_VAR}.svg'),
           bbox_inches='tight')
plt.show()

# =====================================================================
# Call the Rank Stability Plot function
print("Creating rank stability plots...")

# For p0 parameter with percentile ranking
fig_rank_p0 = plot_parameter_rank_stability(
    df_pheno,
    param_name=X_VAR,
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    percentiles=True,
    figsize=(10, 7)
)
plt.savefig(os.path.join(DIR_OUTPUT, f'rank_stability_percentile_{X_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'rank_stability_percentile_{X_VAR}.svg'),
           bbox_inches='tight')
plt.show()

# For p_ss parameter with percentile ranking
fig_rank_pss = plot_parameter_rank_stability(
    df_pheno,
    param_name=Y_VAR,
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    percentiles=True,
    figsize=(10, 7)
)
plt.savefig(os.path.join(DIR_OUTPUT, f'rank_stability_percentile_{Y_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'rank_stability_percentile_{Y_VAR}.svg'),
           bbox_inches='tight')
plt.show()

# Also create versions with integer ranking for comparison
fig_rank_p0_int = plot_parameter_rank_stability(
    df_pheno,
    param_name=X_VAR,
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    percentiles=False,
    figsize=(10, 7)
)
plt.savefig(os.path.join(DIR_OUTPUT, f'rank_stability_integer_{X_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# =====================================================================
# Create transition matrices with corrected null model
print("Creating transition matrices with corrected null model...")

# For p0 parameter (regular order)
fig_trans_null_p0, data_p0 = plot_transition_with_null_comparison(
    df_pheno,
    param_name=X_VAR,
    ages=ages,
    n_quantiles=4,
    figsize=(18, 15),
    n_permutations=100
)
plt.savefig(os.path.join(DIR_OUTPUT, f'transition_matrix_with_null_{X_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'transition_matrix_with_null_{X_VAR}.svg'),
           bbox_inches='tight')
plt.show()

# For p0 parameter (with swapped columns)
fig_trans_null_p0_swapped, data_p0_swapped = plot_transition_with_null_comparison(
    df_pheno,
    param_name=X_VAR,
    ages=ages,
    n_quantiles=4,
    figsize=(18, 15),
    n_permutations=100,
    custom_column_order=[0, 2, 1, 3]  # Swap columns 2 and 3
)
plt.savefig(os.path.join(DIR_OUTPUT, f'transition_matrix_with_null_{X_VAR}_swapped.png'),
           dpi=300, bbox_inches='tight')
plt.show()

# For p_ss parameter
fig_trans_null_pss, data_pss = plot_transition_with_null_comparison(
    df_pheno,
    param_name=Y_VAR,
    ages=ages,
    n_quantiles=4,
    figsize=(18, 15),
    n_permutations=100
)
plt.savefig(os.path.join(DIR_OUTPUT, f'transition_matrix_with_null_{Y_VAR}.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, f'transition_matrix_with_null_{Y_VAR}.svg'),
           bbox_inches='tight')
plt.show()


# =====================================================================
# Optional: Combined parameter stability analysis with normalized trajectories
print("Creating normalized trajectory plots...")

# Z-score normalized trajectories
fig_z = plot_parameter_normalized_trajectories(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='all',
    normalize_method='z_score',
    figsize=(12, 5)
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_zscore.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_zscore.svg'),
           bbox_inches='tight')
plt.show()

# Percentile-rank version
fig_pct = plot_parameter_normalized_trajectories(
    df_pheno,
    param_names=[X_VAR, Y_VAR],
    ages=ages,
    highlight_fly_ids=HIGHLIGHT_FLY_IDS,
    track_flies='all',
    normalize_method='percentile',
    figsize=(12, 5)
)
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_percentile.png'),
           dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(DIR_OUTPUT, 'parameter_normalized_trajectories_percentile.svg'),
           bbox_inches='tight')
plt.show()

print("Stability analysis completed!")

# =====================================================================
# =====================================================================
# =====================================================================
# 1) Stability plot with CI and highlighting
plt.figure(figsize=(4,4))
for fid, grp in df_pheno.groupby('fly_id'):
    grp = grp.sort_values('day')
    if len(grp)<3: continue
    is_target = (fid in HIGHLIGHT_FLY_IDS) or (fid in to_highlight)
    if HIGHLIGHT_ONLY and not is_target:
        col, alpha, lw, ms, s, zz = 'lightgrey', .4, .7, 3, 25, 7
    else:
        col, alpha, lw, ms, s, zz = f"C{fid%10}", 1, 2, 6, 50, 10
    # CI bars
    if is_target and SHOW_CI:
        for _, r in grp.iterrows():
            plt.errorbar(
                r[X_VAR], r[Y_VAR],
                xerr=[[r[X_VAR]-r[f"{X_VAR}_ci_low"]], [r[f"{X_VAR}_ci_high"]-r[X_VAR]]],
                yerr=[[r[Y_VAR]-r[f"{Y_VAR}_ci_low"]], [r[f"{Y_VAR}_ci_high"]-r[Y_VAR]]],
                fmt='none', ecolor=col, alpha=alpha, capsize=3, zorder=zz
            )
    # points & lines
    for _, r in grp.iterrows():
        plt.scatter(r[X_VAR], r[Y_VAR], color=col,
                    marker=age_styles[r.day]['marker'], s=s, lw=lw, alpha=alpha, zorder=zz)
    plt.plot(grp[X_VAR], grp[Y_VAR], color=col, lw=lw, alpha=alpha, zorder=zz,
             label=f"Fly {fid}" if fid in HIGHLIGHT_FLY_IDS else None)
plt.xlabel(X_VAR); plt.ylabel(Y_VAR)
plt.title(f"Stability ({genotype}): {X_VAR} vs {Y_VAR}")
if X_VAR == 'p0' and Y_VAR == 'p_ss':
    plt.plot([0,1],[0,1],'--',color='grey',alpha=.5)
plt.grid(alpha=0.2)
# legend
age_handles = [mlines.Line2D([],[],color='grey',marker=age_styles[d]['marker'], linestyle='None',markersize=8,label=str(d)) for d in ages]
ax=plt.gca(); h,l = ax.get_legend_handles_labels(); ax.legend(h+age_handles, l+[h.get_label() for h in age_handles])
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT,f"stability_{genotype}_{X_VAR}_{Y_VAR}.png"),dpi=300)
plt.show()

# 2) Bar‐plot of trajectory patterns
topk = df_paths.path_code.value_counts().nlargest(40)
plt.figure(figsize=(8,4))
sns.barplot(x=topk.index, y=topk.values, palette="viridis")
plt.xticks(rotation=45)
plt.ylabel("# flies")
plt.title("Top trajectory patterns")
plt.tight_layout()
plt.show()

# ========================================================================
# 3) Heatmaps of transitions
order = ['-','0','+']
fig, axes = plt.subplots(1,2,figsize=(6,2.5))
for ax, var in zip(axes, [X_VAR, Y_VAR]):
    step1, step2 = f"{var}_step1", f"{var}_step2"
    pivot = pd.crosstab(df_paths[step1], df_paths[step2]).reindex(index=order,columns=order,fill_value=0)
    # percent
    pct = pivot / pivot.values.sum() * 100
    annot = pct.round(1).astype(str) + '%'
    sns.heatmap(pct, annot=annot, fmt='', cmap='Blues', ax=ax, cbar=False, linewidths=0.5)
    ax.invert_yaxis()
    ax.set_xlabel('14→21'); ax.set_ylabel('7→14')
    ax.set_title(f"{var} transitions")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, 'trajectory_heatmaps.png'), dpi=300)
plt.show()

if not FLAG_SKIP_SLOW_TRANSITION_TEST:

    # ——— 3b/3c/3d - Fully build null heatmaps and empirical p‑values ———
    print('working on bootstrapping for paths heatmap...')
    """
    Logic here:
      - pick a random fly at 7, 14, and 21 (instead of same fly), 
      - then for 7->14 ask: 'does shift in parameter mean fall outside the 95CI?' 
            - this will return a -, 0, + code 
            - then repeat for 14-> 21
    """
    N_perm  = 5000  # default: 5000
    fly_ids = df_pheno['fly_id'].unique()
    n_flies = len(fly_ids)

    order     = ['-','0','+']
    code_map  = {'-': 0, '0': 1, '+': 2}

    # Pre‐cache fly_ids available at each age
    ids_by_age = {
        age: df_pheno[df_pheno.day == age]['fly_id'].unique()
        for age in ages
    }

    def make_step(from_row, to_row, var):
        lo = from_row[f"{var}_ci_low"]
        hi = from_row[f"{var}_ci_high"]
        mu_next = to_row[var]
        return '+' if mu_next > hi else ('-' if mu_next < lo else '0')

    # split your df_pheno once into three day‐indexed DataFrames, indexed by fly_id
    df7  = df_pheno[df_pheno.day==ages[0]].set_index('fly_id')
    df14 = df_pheno[df_pheno.day==ages[1]].set_index('fly_id')
    df21 = df_pheno[df_pheno.day==ages[2]].set_index('fly_id')

    null_pct_sims = { 'X': np.zeros((N_perm,3,3)), 'Y': np.zeros((N_perm,3,3)) }

    null_records = []
    for rep in range(N_perm):
        if rep % 100 == 0:
            print(f"  {rep}/{N_perm} ({rep/N_perm:.1%})")
        # 1) sample one fly at each age (no replacement *across ages*)
        pick7 = np.random.permutation(ids_by_age[7])[:n_flies]
        pick14 = np.random.permutation(ids_by_age[14])[:n_flies]
        pick21 = np.random.permutation(ids_by_age[21])[:n_flies]

        # 2) grab the rows in one go
        d7_vals  = df7.loc[pick7]
        d14_vals = df14.loc[pick14]
        d21_vals = df21.loc[pick21]

        # 3) for X_VAR: step1 (7→14) and step2 (14→21)
        # compare day‑14 mean to day‑7 CI, etc.
        m7   = d7_vals [X_VAR].values
        lo7  = d7_vals [f"{X_VAR}_ci_low"].values
        hi7  = d7_vals [f"{X_VAR}_ci_high"].values
        m14  = d14_vals[X_VAR].values
        lo14 = d14_vals[f"{X_VAR}_ci_low"].values
        hi14 = d14_vals[f"{X_VAR}_ci_high"].values
        m21  = d21_vals[X_VAR].values

        # vectorized step calls
        s1x = np.where(m14 > hi7, '+',
              np.where(m14 < lo7, '-', '0'))
        s2x = np.where(m21 > hi14,'+',
              np.where(m21 < lo14,'-','0'))
        # map to ints 0,1,2
        i1 = np.vectorize(code_map.__getitem__)(s1x)
        i2 = np.vectorize(code_map.__getitem__)(s2x)
        Hx, _, _ = np.histogram2d(i1, i2, bins=3, range=[[0,3],[0,3]])
        null_pct_sims['X'][rep] = Hx / n_flies * 100

        # 4) same for Y_VAR
        m7   = d7_vals [Y_VAR].values
        lo7  = d7_vals [f"{Y_VAR}_ci_low"].values
        hi7  = d7_vals [f"{Y_VAR}_ci_high"].values
        m14  = d14_vals[Y_VAR].values
        lo14 = d14_vals[f"{Y_VAR}_ci_low"].values
        hi14 = d14_vals[f"{Y_VAR}_ci_high"].values
        m21  = d21_vals[Y_VAR].values

        s1y = np.where(m14 > hi7, '+',
              np.where(m14 < lo7, '-', '0'))
        s2y = np.where(m21 > hi14,'+',
              np.where(m21 < lo14,'-','0'))
        j1 = np.vectorize(code_map.__getitem__)(s1y)
        j2 = np.vectorize(code_map.__getitem__)(s2y)
        Hy, _, _ = np.histogram2d(j1, j2, bins=3, range=[[0,3],[0,3]])
        null_pct_sims['Y'][rep] = Hy / n_flies * 100

    null_df = pd.DataFrame(null_records)

    # now you can compute, for example, the mean null heatmap:
    mean_null_pct = {
        k: null_pct_sims[k].mean(axis=0)
        for k in ['X','Y']
    }
    # ...and likewise your 2.5 and 97.5 percentiles per cell:
    ci_low_null = {
        k: np.percentile(null_pct_sims[k], 2.5, axis=0)
        for k in ['X','Y']
    }
    ci_hi_null = {
        k: np.percentile(null_pct_sims[k], 97.5, axis=0)
        for k in ['X','Y']
    }

    # observed counts & %
    obs_ct = {
      'X': pd.crosstab(df_paths[f"{X_VAR}_step1"], df_paths[f"{X_VAR}_step2"])
               .reindex(index=order, columns=order, fill_value=0),
      'Y': pd.crosstab(df_paths[f"{Y_VAR}_step1"], df_paths[f"{Y_VAR}_step2"])
               .reindex(index=order, columns=order, fill_value=0)
    }
    obs_pct = {k: obs_ct[k].values / n_flies * 100 for k in obs_ct}

    # difference
    diff_pct = {k: obs_pct[k] - mean_null_pct[k] for k in ['X','Y']}

    # two‑sided empirical p‑values
    pval = {}
    for k in ['X','Y']:
        sims = null_pct_sims[k]
        mnull = mean_null_pct[k]
        pmat = np.zeros((3,3))
        for i in range(3):
          for j in range(3):
            dev_null = np.abs(sims[:,i,j] - mnull[i,j])
            dev_obs  = abs(obs_pct[k][i,j] - mnull[i,j])
            pmat[i,j] = np.mean(dev_null >= dev_obs)
        pval[k] = pmat

    # ——— now plot all 4 panels in one 4×2 figure ———
    panels = [
        ("Observed %",    obs_pct,        "Blues",  False),
        ("Null mean %",   mean_null_pct,  "Greys",  True),
        ("Difference %",  diff_pct,       "RdBu_r", True),
        ("2‑sided p‑val", {k: pval[k]*100 for k in pval}, "Reds", False),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(8, 3*len(panels)), sharex=True, sharey=True)

    for row_idx, (title, data_dict, cmap_name, annotate_ci) in enumerate(panels):
        for col_idx, var in enumerate(['X','Y']):
            ax = axes[row_idx, col_idx]
            m = data_dict[var]

            # build the base annotation
            if title.endswith("%"):
                annot = np.vectorize(lambda v: f"{v:.1f}%")(m)
            else:
                annot = np.vectorize(lambda v: f"{v/100:.3f}")(m)

            sns.heatmap(
                m,
                annot=annot, fmt="",
                cmap=cmap_name,
                center=(0 if "Difference" in title else None),
                cbar_kws={'label': title},
                linewidths=0.5, linecolor='white',
                annot_kws={'fontsize':12},
                xticklabels=order,
                yticklabels=order,
                ax=ax
            )
            ax.invert_yaxis()
            ax.set_xlabel("14→21")
            ax.set_ylabel("7→14")
            ax.set_title(f"{title} ({X_VAR if var=='X' else Y_VAR})")
            ax.set_yticklabels(order, rotation=0)
            ax.set_xticklabels(order, rotation=0)

            if annotate_ci:
                # get the QuadMesh facecolors for luminance checks
                quadmesh = ax.collections[0]
                fc = quadmesh.get_facecolors().reshape(m.shape[0], m.shape[1], 4)
                lo = ci_low_null[var]
                hi = ci_hi_null[var]

                for i in range(m.shape[0]):
                    for j in range(m.shape[1]):
                        rgba = fc[i, j]
                        lum = 0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2]
                        text_color = 'white' if lum < 0.5 else 'black'
                        ax.text(
                            j+0.5, i+0.2,
                            f"{lo[i,j]:.1f}–{hi[i,j]:.1f}",
                            ha='center', va='bottom',
                            fontsize=8,
                            color=text_color
                        )

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUTPUT, 'stability_paths_heatmaps_4x2.svg'))
    plt.show()

if not FLAG_SKIP_SLOW_KDE:

    # … 4) KDE of marginals
    print('working on KDE…')

    # … user‑configurable flags, near the top of your script …
    SHOW_HIST   = True              # overlay histogram under each KDE
    HIST_BINS   = 30                # number of bins in that histogram
    SHOW_RUG    = False             # add a rug plot of individual draws
    KDE_BW_ADJ  = 1.0               # adjust bandwidth: <1 = more wiggly, >1 = smoother
    KDE_ORIENTATION = 'horizontal'  # 'horizontal' or 'vertical'

    if KDE_ORIENTATION == 'horizontal':
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    day_palette = days_palettes['Deep']
    colors = {d: day_palette[i] for i, d in enumerate(ages)}

    for ax, (var, linestyle) in zip(axes.flatten(), [(X_VAR, '-'), (Y_VAR, '--')]):
        for d in ages:
            draws = all_draws[d]['x'] if var == X_VAR else all_draws[d]['y']

            # 1) optional histogram
            if SHOW_HIST:
                ax.hist(
                    draws,
                    bins=HIST_BINS,
                    density=True,
                    color=colors[d],
                    alpha=0.25,
                    label=None
                )

            # 2) the KDE itself
            sns.kdeplot(
                draws,
                ax=ax,
                label=f"Day {d}",
                color=colors[d],
                linestyle=linestyle,
                clip=(0, 1),
                bw_adjust=KDE_BW_ADJ
            )

            # 3) optional rug
            if SHOW_RUG:
                sns.rugplot(
                    draws,
                    ax=ax,
                    color=colors[d],
                    alpha=0.5,
                    height=0.05
                )

        ax.set_title(f"{var} marginal density")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(title="Age")

    fig.tight_layout()
    plt.savefig(
        os.path.join(DIR_OUTPUT, f"posterior_density_{X_VAR}_{Y_VAR}_{genotype}.png"),
        dpi=300
    )
    plt.show()

    # 5) 2D KDE contour per age with subplots and discrete shading
    print('working on KDE...')
    fig, axes = plt.subplots(1, len(ages), figsize=(5*len(ages), 5), sharex=True, sharey=True)
    for ax, d in zip(axes, ages):
        # build a light-to-strong colormap for this age's base color
        base_col = colors[d]
        cmap_fill = light_palette(base_col, n_colors=6, reverse=False, as_cmap=True)
        # plot filled contours for discrete density levels
        sns.kdeplot(
            x=all_draws[d]['x'],
            y=all_draws[d]['y'],
            levels=5,
            fill=True,
            cmap=cmap_fill,
            thresh=0.05,
            clip=((0,1),(0,1)),
            ax=ax
        )
        # outline contours
        sns.kdeplot(
            x=all_draws[d]['x'],
            y=all_draws[d]['y'],
            levels=5,
            color=base_col,
            linewidths=1,
            clip=((0,1),(0,1)),
            ax=ax
        )
        ax.set_title(f"Day {d}")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel(X_VAR)
        ax.set_ylabel(Y_VAR)

        if Y_VAR == 'p_ss' and X_VAR == 'p0':
            # add a diagonal line for p_ss = p0
            ax.plot([0, 1], [0, 1], '--', color='grey', alpha=0.5)

    fig.suptitle("2D posterior densities by age", y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(DIR_OUTPUT, f"posterior_density2d_by_age_{genotype}.png"), dpi=300)
    plt.show()
