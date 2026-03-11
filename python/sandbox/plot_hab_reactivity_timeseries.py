# hab_reactivity_timeseries.py
# Simulate & plot time series for the 3D model under two assays:
#  - Habituation assay: T = 1
#  - Reactivity assay:  T = 5
# Parameters are (alpha, beta, p0). The script also provides the
# equivalent (alpha, p_ss, p0) parameterization:
#   beta = (exp(alpha*T) - 1) * sqrt(p0/p_ss - 1)
#
# Model:
#   x[k] = beta * exp(-alpha*T) * (1 - exp(-alpha*k*T)) / (1 - exp(-alpha*T))
#   p[k] = p0 / (1 + x[k]^2)

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# Adds fly-jump/python to sys.path and change working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)  # Change to python/ directory for relative paths to work


from settings import DIR_OUTPUT


def x_series(alpha, beta, T, K):
    k = np.arange(K)
    den = 1.0 - np.exp(-alpha * T)
    if np.isclose(den, 0.0):
        # limit alpha->0: x[k] -> beta * k  (unit impulses, no leak)
        return beta * k
    num = beta * np.exp(-alpha * T) * (1.0 - np.exp(-alpha * k * T))
    return num / den

def p_series_alpha_beta_p0(alpha, beta, p0, T, K):
    x = x_series(alpha, beta, T, K)
    return p0 / (1.0 + x**2)

def beta_from_alpha_p0_pss(alpha, p0, p_ss, T):
    if not (0.0 < p_ss <= p0):
        raise ValueError("Require 0 < p_ss <= p0.")
    x_ss = np.sqrt(p0 / p_ss - 1.0)
    return x_ss * (np.exp(alpha * T) - 1.0)

def p_series_alpha_pss_p0(alpha, p_ss, p0, T, K):
    beta = beta_from_alpha_p0_pss(alpha, p0, p_ss, T)
    return p_series_alpha_beta_p0(alpha, beta, p0, T, K)

def plot_family(param_sets, T, K=200, title=""):
    plt.figure()
    for (alpha, beta, p0, label) in param_sets:
        p = p_series_alpha_beta_p0(alpha, beta, p0, T, K)
        plt.plot(np.arange(K), p, label=label)
    plt.xlabel("trial k")
    plt.ylabel("jump probability p[k]")
    plt.title(f"{title} (T={T})")
    plt.legend()
    plt.tight_layout()

def verify_equivalence(examples, T, K=80, tol=1e-12):
    ok_all = True
    print(f"Equivalence check for T={T}: (alpha,beta,p0)  <->  (alpha,p_ss,p0)")
    for (alpha, beta, p0) in examples:
        den = 1.0 - np.exp(-alpha * T)
        if np.isclose(den, 0.0):
            print(f"  Skipping (alpha≈0) example: {(alpha,beta,p0)}")
            continue
        x_ss = beta * np.exp(-alpha * T) / den
        p_ss = p0 / (1.0 + x_ss**2)
        p1 = p_series_alpha_beta_p0(alpha, beta, p0, T, K)
        p2 = p_series_alpha_pss_p0(alpha, p_ss, p0, T, K)
        mad = float(np.max(np.abs(p1 - p2)))
        ok = mad < tol
        ok_all = ok_all and ok
        print(f"  {alpha:.3f}, {beta:.3f}, {p0:.3f} -> p_ss={p_ss:.6f}, max|Δ|={mad:.3e}, ok={ok}")
    return ok_all

if __name__ == "__main__":
    # Example families to plot (edit as you like)
    alpha_global = 0.2
    param_sets = [
        (alpha_global, 0.1, 0.65, "α=%.2f, β=0.1, p0=0.65" % alpha_global),
        (alpha_global, 0.25, 0.55, "α=%.2f, β=0.25, p0=0.55" % alpha_global),
        (alpha_global, 0.5, 0.35, "α=%.2f, β=0.5, p0=0.35" % alpha_global),
        (0.10, 1.0, 0.50, "α=0.10, β=1.0, p0=0.50"),
    ]

    # Plot families
    plot_family(param_sets, T=1, K=200, title="Habituation assay family")
    plot_family(param_sets, T=5, K=200, title="Reactivity assay family")

    # Verify equivalence for a few examples
    examples = [(0.18, 1.2, 0.35), (0.25, 0.9, 0.40), (0.12, 1.8, 0.30)]
    verify_equivalence(examples, T=1)
    verify_equivalence(examples, T=5)

    # Save figures
    plt.figure(1); plt.savefig(DIR_OUTPUT + os.sep + "habituation_family_T1.png", dpi=160)
    plt.figure(2); plt.savefig(DIR_OUTPUT + os.sep + "reactivity_family_T5.png", dpi=160)
    print("\nSaved: habituation_family_T1.png, reactivity_family_T5.png")
