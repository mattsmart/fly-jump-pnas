import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work

# try to import UMAP; if it’s not installed, we’ll skip that part
try:
    from umap import UMAP
    has_umap = True
except ImportError:
    has_umap = False
    print("UMAP not installed; skipping UMAP plot.")

from data_tools import df_to_arr_jumps
from data_format_add_score_columns import compute_p_ss
from settings import DIR_OUTPUT, DIR_FITS

# User‐config
genotype      = 'KK'
ages          = [7, 14, 21]
OMIT_FLY_IDS  = {'KK': [31, 121, 80, 76], 'GD': []}
CHAMBER_TO_BOX = {i: (i-1)//8 for i in range(1,33)}

# Load the summary CSV
df_summary = pd.read_csv(os.path.join(DIR_FITS, "fly-stability-days-detailed-habscores.csv"))
df_summary = df_summary[(df_summary.genotype == genotype) &
                        (df_summary.day.isin(ages))].copy()

# drop flies to omit
omit = set(OMIT_FLY_IDS.get(genotype, []))
df_summary = df_summary[~df_summary.fly_id.isin(omit)].reset_index(drop=True)


# Helper to extract p0 and p_ss for one fly at one day
def summarize_fly(row, draws_dict):
    fid = int(row.fly_id)
    samples = draws_dict[f"fly_id_{fid}"]  # shape: (n_draws, 3)
    α, β, p0_draws = samples.T
    p0   = p0_draws.mean()
    p_ss = compute_p_ss(α, β, p0_draws, T=1).mean()
    return p0, p_ss

# Load all the posterior draws
draws_over_age = {
    'KK': {},
    'GD': {}
}
for g in ['KK','GD']:
    for d in ages:
        draws_over_age[g][d] = np.load(
            os.path.join(DIR_FITS, f"{g}_day{d}_3d_draws.npz"),
            allow_pickle=True
        )

# Build a long table: one row per fly per day
records = []
for day in ages:
    draws_day = draws_over_age[genotype][day]
    df_day   = df_summary[df_summary.day==day]
    for _, r in df_day.iterrows():
        p0, p_ss = summarize_fly(r, draws_day)
        records.append({
            'fly_id': int(r.fly_id),
            'day':    day,
            'p0':     p0,
            'p_ss':   p_ss
        })
df_pheno = pd.DataFrame(records)

# Pivot to wide: one row per fly_id, six columns
df_wide = (
    df_pheno
    .pivot(index='fly_id', columns='day', values=['p0','p_ss'])
)
df_wide.columns = [f"{metric}_{day}" for metric,day in df_wide.columns]
df_wide = df_wide.dropna()   # drop flies missing any day

# Standardize the 6‑dim vectors
X      = df_wide.values
X_std  = StandardScaler().fit_transform(X)

# Prepare colors by “box” membership
fly2chamber = df_summary.groupby('fly_id')['chamber_id'].first().to_dict()
fly2box     = {f: CHAMBER_TO_BOX[c] for f,c in fly2chamber.items()}
boxes       = [fly2box[f] for f in df_wide.index]
cmap        = plt.get_cmap('tab10', 4)
colors      = [cmap(b) for b in boxes]

# -- PCA plot --
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_std)

plt.figure(figsize=(6,6))
plt.scatter(pcs[:,0], pcs[:,1], c=colors, s=50, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"PCA of (p₀,p_ss) @ {ages} days for {genotype}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(DIR_OUTPUT, f"multiday_pca_{genotype}.png"), dpi=300)
plt.show()

# -- UMAP plot (if available) --
if has_umap:
    reducer   = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=0)
    embedding = reducer.fit_transform(X_std)

    plt.figure(figsize=(6,6))
    plt.scatter(embedding[:,0], embedding[:,1], c=colors, s=50, alpha=0.8)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP of (p₀,p_ss) @ {ages} days for {genotype}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_OUTPUT, f"multiday_umap_{genotype}.png"), dpi=300)
    plt.show()
else:
    print("UMAP not available; install umap-learn to enable.")

# Print out your final data matrix shape
print("Phenotype matrix (rows=flies, cols=[p0, p_ss]×days):")
print(df_wide.shape)
print(df_wide.head())
