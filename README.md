## Fly jumping

This repository contains data, models, and analysis code for "Dynamical modeling of individual sensory reactivity and habituation learning" (Boon et al., 2026).

#### File organization

* The experimental data is in `data/`.

* The Bayesian statistical models are coded in Stan in repo `stan/`.

* The code to fit models is in `python/fit_experimental_data.py/`.

* Python scripts for analysis and plotting are in `python/`.

* R script and appropriately formatted data for analysis and plotting are in `R/`. 

### Reproducing results

### 0. Testing Stan installation (optional but recommended)
Before running the full analysis pipeline, you can verify your Stan setup works correctly:
```
$ cd fly-jump/python
$ python test_stan_fit.py
```
This runs a single fit (KK day 7) and compares with pre-computed results. High correlations (>0.90) between new and old parameter estimates indicate the setup is working correctly. Note: Exact reproduction is not expected due to MCMC stochasticity.

### 1. `stan` fits
- Precomputed fits are provided in the `fits` directory:
  - `fits/fly-stability-days-detailed-3d.csv`
  - `fits/%s_day%d_%s_draws.npz` (for each genotype `[KK, GD]` and day `[7,14,21]` and model variant `[1d,2d,3d]`)
- These can be used to reproduce figures without needing to run the Stan models again
- Fits can be regenerated (provided `cmdstanpy` is installed, see notes below), as follows
```
$ cd fly-jump/python
$ python fit_experimental_data.py
```

### 2. Data processing after fit
- To process the fit summaries for plotting (e.g. pre-compute means), run `python/data_format_add_score_columns.py` 
- This will create `fits/fly-stability-days-detailed-3d-habscores.csv`

### 3. Figure reproduction (after stan fits)
After completing the fits and post-processing, manuscript figures can be reproduced by running the 
following scripts in `python/`:
- Figures 1F and S2:     `plot_jump_data_vs_fit.py`
- Figures 2C,D and 4E,F: `plot_multiday_one_fly.py`
- Figures 3A-C and S5:   `plot_metrics_by_fly_id.py`
- Figures 4G,H:          `plot_multiday_correlations.py`
- Figure S4:             `plot_posterior_mean_distributions.py`

For the model comparison figures (Figures S3B and S7C), the 1D and 2D models need to be fit separately:
- run `fit_experimental_data.py` with main option `fit_1d_and_2d = True` to generate the respective fits
- generate the main figure panel Fig. 3D with `plot_fig3_model_validation_scatters.py`  
- generate the supplementary plots with `plot_fig3_model_1d2d3d_scatters.py`

### 4. Plotting of populational data and analyses in R(Studio)
- To compute the empirical statistics, perform mean pairwise squared differences and repeatability analyses, and produce manuscript figures 1E, 4C,D,I and  S1A-B, S6, S7A, and S8A-C, run `R/R_script.R`
- The required CSV files are provided as three separate ZIP archives. Please extract these ZIP folders so that the contained CSV files are restored before running the script.
- Further instructions are provided at the start of the script

### Notes on excluded flies (technical issues during collection)

Four KK flies were excluded from analysis due to clear technical artifacts
(extreme cut-power or delay-jump values, or presumed death during the assay),
leaving 119 flies with complete datasets for follow-up analysis.

- KK_a7, fly_id 31  (expt1908mb, chamber 31, system A2B; extreme high cut-power; present in python raw dataset with the same ID)
- KK_a7, fly_id 121 (expt2002mb:2007mbSRA, chamber 25, system A2B; died during experimentation; already been removed from our raw dataset)
- KK_a14, fly_id 80 (expt2022mb, chamber 10, system A1B; extreme low delay-jump; present in python raw dataset with the same ID)
- KK_a14, fly_id 76 (expt2022mb, chamber 14, system A1B; extreme low cut-power; present in python raw dataset with the same ID)

## Installation 

### Python packages

- dependencies are listed in `requirements.txt` (for Python >=3.10)
- install via `pip install -r requirements.txt`

### Installing `cmdstanpy` package

See: https://cmdstanpy.readthedocs.io/en/v1.2.0/installation.html

CmdStanPy documentation: https://cmdstanpy.readthedocs.io/en/v1.2.0

**Installation on Windows**  
On Windows, CmdStan can be installed by running  
`cmdstanpy.install_cmdstan(compile=True)`   
from the Python environment after installing the CmdStanPy package via `pip`.

In case you run into compiler issues when fitting models with CmdStanPy, 
first try following the Windows and troubleshooting sections here:  
https://mc-stan.org/docs/cmdstan-guide/installation.html.   
Then verify that you can run the Bernoulli test example described there (from terminal). 

### Installing R and RStudio

See: https://rstudio-education.github.io/hopr/starting.html
