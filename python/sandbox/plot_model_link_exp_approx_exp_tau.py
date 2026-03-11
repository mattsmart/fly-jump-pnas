import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
This derivation is for the exponential link function
    
    p_jump[k] = p0 * exp(-x[k])
    
Naively, x(t) has an exponential dependence on time, making p_jump double-exponential. 
However, we can still fit an exponential to its decay profile -- this is defined by the expreesion "tau_improved"
    
    p_jump[k] ~ p_inf + (p0 - p_inf) * exp(-k/tau_p)
    
    where tau_p = -1/(alpha*T) * ln[1 + (1/x_star) * ln(e^{-x_star} + (1-e^{-x_star})/e)] 
    
      
"""

# Parameters
alpha = 0.1     # decay rate per second

T = 1           # inter-stimulus interval (s)
beta = 0.1      # increment per stimulus
p0 = 1.0        # maximal jump probability when x = 0
k_max = 50      # number of trials to simulate
k_vals = np.arange(0, k_max+1)

# Compute decay factor and steady state for x
r = np.exp(-alpha * T)
x_star = (beta * r) / (1 - r)  # steady state of x
x = x_star * (1 - np.exp(-alpha * T * k_vals))  # evolution of x[k]

# Define the exponential link function
def f_exp(x):
    return p0 * np.exp(-x)

# Compute observable jump probability for exponential link
p_exp = f_exp(x)

# Compute asymptotic value for the exponential link
p_inf_exp = f_exp(x_star)

# --- Fit an exponential model to the normalized decay ---
def exp_decay(k, tau):
    return np.exp(-k / tau)

def normalized_response(p, p_inf):
    return (p - p_inf) / (p0 - p_inf)

R_exp = normalized_response(p_exp, p_inf_exp)
popt_exp, _ = curve_fit(exp_decay, k_vals, R_exp, p0=[1/(alpha*T)])
tau_exp_fit = popt_exp[0]
R_exp_fit = exp_decay(k_vals, tau_exp_fit)

# --- Compute analytic tau using the exact expression ---
#tau_exp_exact = (1 - np.exp(-alpha * T)) / (alpha * T * beta * np.exp(-alpha * T))

# -----------------------------
# Analytic Effective Time Constant Estimates
# -----------------------------
# Simple (first-order) analytic tau:
tau_simple = 1 / (x_star * alpha * T)

# Improved analytic tau via the fractional-drop definition:
#   tau_p = -1/(alpha*T)*ln[1 + (1/x_star)*ln(e^{-x_star} + (1-e^{-x_star})/e)]
tau_improved = -1/(alpha * T) * np.log(1 + (1/x_star) * np.log(np.exp(-x_star) + (1 - np.exp(-x_star)) / np.e))
R_exp_improved = exp_decay(k_vals, tau_improved)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(16, 10))

# Panel 1: Internal State Evolution
plt.subplot(2, 2, 1)
plt.plot(k_vals, x, 'bo-', label=r'$x[k]$')
plt.axhline(x_star, color='gray', linestyle='--', label=r'Steady state $x^*$')
plt.axvline(1/(alpha*T), color='red', linestyle='--', label=f'1/(αT) = {1/(alpha*T):.1f} trials')
plt.xlabel('Trial k')
plt.ylabel('Internal state x[k]')
plt.title('Evolution of Internal State')
plt.legend()

# Panel 2: Observable Jump Probability (Exp Link)
plt.subplot(2, 2, 2)
plt.plot(k_vals, p_exp, 'r.-', label=r'$p_0\,e^{-x}$')
plt.xlabel('Trial k')
plt.ylabel('Jump probability p[k]')
plt.title('Observable Response (Exp Link)')
plt.legend()

# Panel 3: Normalized Decay and Exponential Fit
plt.subplot(2, 1, 2)
plt.plot(k_vals, R_exp, 'r.-', label='Normalized R(k) (Data)')
plt.plot(k_vals, R_exp_fit, 'r--', label=f'Exp Fit: τ_fit = {tau_exp_fit:.2f}')
plt.plot(k_vals, R_exp_improved, 'r-', label=f': τ_fit_imp = {tau_improved:.2f}')

plt.xlabel('Trial k')
plt.ylabel('Normalized Response R(k)')
plt.title('Exponential Fit to Normalized Decay (Exp Link)')
plt.text(0.55 * k_max, 0.85, f'Simple τ = {tau_simple:.2f}', color='red', fontsize=12)
plt.text(0.55 * k_max, 0.75, f'Improved τ = {tau_improved:.2f}', color='darkred', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()
