import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import chisquare
from statsmodels.stats.anova import anova_lm

import os
import sys

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

from python.data_tools import df_to_arr_jumps, get_TTC_canonical, habituation_time_fisherexact
from python.data_format_add_score_columns import (
    compute_p_ss, compute_hab_time_half_rel, compute_hab_time_95_rel,
    compute_hab_magnitude_abs, compute_hab_magnitude_rel)
from python.functions_common import likelihood_func_vec
from python.settings import DIR_OUTPUT, DIR_DATA_EXPT, DIR_FITS, OMIT_FLY_IDS

##############################
# Settings / Toggle Options
##############################
# filter for a given genotype and day
genotype = 'KK'
age = 14

scalar_measure_choice = 'ttc5_mean'
assert scalar_measure_choice in ['sum_sra', 'sum_1050', 'sum_habtail_mean', 'ttc5_mean']

# ----- CONFIGURATION -----
# Number of permutations for the permutation test
#N_PERMUTATIONS = 10000
N_PERMUTATIONS = 1000
# --------------------------

# --- New Options: Highlight Fly IDs ---
# These lists use the numeric fly ID (extracted from the key "fly_id_X").
HIGHLIGHT_FLY_IDS = {
    # 'KK': [91, 44],         # For genotype 'KK', highlight these fly numbers.
    'KK': [],
    'GD': []
}

CHAMBER_IDX_TO_BOX = {i: ((i - 1) // 8) for i in range(1, 33)}

# --- Helper function: central_value ---
# (This is used in the transformation scripts; not needed for the following tests.)
MEDIAN_MODE = False
def central_value(x):
    return np.median(x) if MEDIAN_MODE else np.mean(x)

##############################
# Data Loading and Filtering
##############################
# Load the parameters inferred from the fit
df_summary_csv = pd.read_csv(os.path.join(DIR_FITS, "fly-stability-days-detailed-habscores.csv"))
print("Summary CSV head:")
print(df_summary_csv.head())

# (In our case, we assume the CSV includes columns: fly_id, genotype, day, chamber_id, and sum_jumps)
df_summary_csv_genotype = df_summary_csv[df_summary_csv['genotype'] == genotype]
df_summary_csv_genotype_day = df_summary_csv_genotype[df_summary_csv_genotype['day'] == age]
# create a copy instead of a view
df_summary_csv_genotype_day = df_summary_csv_genotype_day.copy()

# (If you have multiple entries per fly and want to aggregate, you could do that here.
# For simplicity we assume each row corresponds to a single fly.)
print("Filtered data (first few rows):")
print(df_summary_csv_genotype_day.head())

# get num rows in the df
nflies = df_summary_csv_genotype_day.shape[0]
print("Number of flies: %d" % nflies)

# add a column called sum_sra which sums the number of jumps in trials 1000:1050; need to extract jumpstring as an array
# 1) first extract jumpdata as array
jumps_all = df_to_arr_jumps(df_summary_csv_genotype_day, jump_col='jumpdata')[:, :]
assert jumps_all.shape == (nflies, 1050)
jumpsum_sra = np.sum(jumps_all[:, 1000:1050], axis=1)
jumpsum_1050 = np.sum(jumps_all[:, 0:], axis=1)
jumpsum_habtail_mean = np.mean(
    [np.sum(jumps_all[:,  50:200], axis=1),
        np.sum(jumps_all[:, 250:400], axis=1),
        np.sum(jumps_all[:, 450:600], axis=1),
        np.sum(jumps_all[:, 650:800], axis=1),
        np.sum(jumps_all[:, 850:1000], axis=1)],
    axis=0)  # constructed arr is 5 x nflies, we average over first axis to 1d arr of len nflies
jumps_to_ttc5_mean = np.mean(
    np.array([
        [get_TTC_canonical(''.join(str(x) for x in jumps_all[a,   0:200]), num_non_jump=5) for a in range(nflies)],
        [get_TTC_canonical(''.join(str(x) for x in jumps_all[a, 200:400]), num_non_jump=5) for a in range(nflies)],
        [get_TTC_canonical(''.join(str(x) for x in jumps_all[a, 400:600]), num_non_jump=5) for a in range(nflies)],
        [get_TTC_canonical(''.join(str(x) for x in jumps_all[a, 600:800]), num_non_jump=5) for a in range(nflies)],
        [get_TTC_canonical(''.join(str(x) for x in jumps_all[a, 800:1000]), num_non_jump=5) for a in range(nflies)]
    ]), axis=0)  # constructed arr is 5 x nflies, we average over first axis to 1d arr of len nflies

# store each scalar measure as a column in the dataframe (we use only one for the analysis)
df_summary_csv_genotype_day['sum_sra'] = jumpsum_sra
df_summary_csv_genotype_day['sum_1050'] = jumpsum_1050
df_summary_csv_genotype_day['sum_habtail_mean'] = jumpsum_habtail_mean
df_summary_csv_genotype_day['ttc5_mean'] = jumps_to_ttc5_mean

##############################
# Hypothesis Testing for Chamber Bias
##############################
# For this example, we test whether the chambers are statistically identical in terms of the sum_jumps statistic.
# There are 32 chambers. The idea is to test the null hypothesis that all chambers have the same jump rate.

# Aggregate data by chamber: count the number of flies and compute the sum of jumps per chamber.
# - count the number of flies per chamber and compute the mean sum_jumps per chamber.
chamber_stats = df_summary_csv_genotype_day.groupby("chamber_id")[scalar_measure_choice].agg(["count", "sum", "mean"]).reset_index()
print("\nChamber statistics:")
print(chamber_stats)

# Load data – assume columns: fly_id, chamber_id (1–32), sum_jumps
print("Data preview:")
print(df_summary_csv_genotype_day.head())

# ---- Test 1: Global Chi-Squared Test ----
# Under the null, assume that each fly has the same expected jump sum.
global_mean = df_summary_csv_genotype_day[scalar_measure_choice].mean()
# Expected sum per chamber = (global_mean * number of flies in chamber)
chamber_stats["expected"] = chamber_stats["count"] * global_mean

# Chi-squared test: statistic = sum((observed - expected)^2 / expected)
observed = chamber_stats["sum"].values
expected = chamber_stats["expected"].values

chi2_stat, p_value_chi2 = chisquare(f_obs=observed, f_exp=expected)
print("\nChi-squared test:")
print("Chi2 statistic = {:.3f}, p-value = {:.4f}".format(chi2_stat, p_value_chi2))


# Suppose you have 'chamber_stats' DataFrame with columns: chamber_id, sum (observed), expected
fig, ax = plt.subplots(figsize=(10,5))
xvals = chamber_stats["chamber_id"].values
width = 0.4

ax.bar(xvals - width/2, chamber_stats["sum"], width=width, color="steelblue", label=r"Observed $\sum_i z_i$")
ax.bar(xvals + width/2, chamber_stats["expected"], width=width, color="orange", label=r"Expected value (pop. average * nflies)")

ax.set_xlabel("Chamber ID")
ax.set_ylabel("%s" % scalar_measure_choice)
ax.set_title("Observed vs. Expected Sum of z=%s by Chamber" % scalar_measure_choice)
ax.legend()
plt.tight_layout()
plt.show()


# ---- Test 2: One-Way ANOVA ----
# Here we test if the mean scalar_measure_choice differs by chamber.
model = smf.ols("%s ~ C(chamber_id)" % scalar_measure_choice, data=df_summary_csv_genotype_day).fit()
anova_results = anova_lm(model, typ=2)
print("\nANOVA results:")
print(anova_results)
print('The p-value for the chamber effect is given in the ANOVA table')

# Plot chamber_id on x-axis, sum_jumps on y-axis, with confidence intervals
# (This uses the built-in confidence intervals from seaborn's barplot.)
plt.figure(figsize=(12,5))
sns.barplot(data=df_summary_csv_genotype_day, x="chamber_id", y=scalar_measure_choice, errorbar=('ci', 95), color="skyblue")
# Overlay the individual data points as a scatter plot
sns.stripplot(data=df_summary_csv_genotype_day, x="chamber_id", y=scalar_measure_choice, color="black", size=5, alpha=0.7)
plt.title("Mean %s by Chamber (with 95%% CI)" % scalar_measure_choice)
plt.xlabel("Chamber ID")
plt.ylabel("Mean %s" % scalar_measure_choice)
plt.tight_layout()
plt.show()

# ---- Test 3: Permutation Test ----
# We want to test if the observed between-chamber differences are larger than expected by chance.
# We'll use the F-statistic from one-way ANOVA as our test statistic.
def compute_f_stat(data, group_col, response_col):
    model = smf.ols(f"{response_col} ~ C({group_col})", data=data).fit()
    anova_res = anova_lm(model, typ=2)
    return anova_res["F"].iloc[0]

# Compute the observed F statistic.
observed_F = compute_f_stat(df_summary_csv_genotype_day, group_col="chamber_id", response_col=scalar_measure_choice)
print("\nObserved F statistic: {:.3f}".format(observed_F))

# Now, permute the chamber labels among flies and compute the F statistic each time.
F_stats = []
print('\nWorking on permutations (N=%d)...' % N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    df_perm = df_summary_csv_genotype_day.copy()
    df_perm["chamber_id"] = np.random.permutation(df_perm["chamber_id"].values)
    F_stats.append(compute_f_stat(df_perm, group_col="chamber_id", response_col=scalar_measure_choice))
F_stats = np.array(F_stats)
# p-value: proportion of permutation F-statistics greater than or equal to the observed F.
p_value_perm = np.mean(F_stats >= observed_F)
print("\nPermutation test:")
print("Empirical p-value = {:.4f}".format(p_value_perm))

# ---- Plotting the permutation null distribution ----
plt.hist(F_stats, bins=50, color="skyblue", edgecolor="k")
plt.axvline(observed_F, color="red", linestyle="--", linewidth=2, label=f"Observed F = {observed_F:.2f}")
plt.text(observed_F + 0.1, plt.ylim()[1]*0.8,
         f"Empirical p = {p_value_perm:.4f}",
         color="red", fontsize=12)
plt.xlabel("F statistic")
plt.ylabel("Frequency")
plt.title("Permutation Test Null Distribution (z=%s)" % scalar_measure_choice)
plt.legend()
plt.tight_layout()
plt.show()
