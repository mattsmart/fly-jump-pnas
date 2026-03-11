"""
Plots in this script are for model diagnostics, meant to be independent of the data.

Basically, explain the model behavior and parameters in terms of interpretable quantities.

1) Habituation magnitude
2) Habituation time

Let R[k] = (p[k] - p_inf) / (p0 - p_inf) be the normalized response for the quadratic link.

k_1/2 - number of stimuli for R[k] = 1/2;  always defined
k_z - number of stimuli for R[k] = z;  always defined

    - if z = 0.1, then k_z is the number of stimuli until you are 90% of the way to steady state
"""

cmap_choice = 'Spectral'

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0  # inter-stimulus interval (set to 1 for simplicity)

# Define functions to compute x*, k_{1/2} and k_{0.05} for the quadratic link
def x_star(alpha, beta):
    # x^* = beta * e^(-alpha T) / (1 - e^(-alpha T))
    return (beta * np.exp(-alpha * T)) / (1 - np.exp(-alpha * T))

def k_half(alpha, beta):
    # k_{1/2} = -1/(alpha*T)*ln(1 - 1/sqrt((x^*)^2+2))
    xs = x_star(alpha, beta)
    return -1/(alpha * T) * np.log(1 - 1/np.sqrt(xs**2 + 2))

def k_005(alpha, beta):
    # k_{0.05} = -1/(alpha*T)*ln(1 - sqrt(0.95/(1+0.05*(x^*)^2)))
    xs = x_star(alpha, beta)
    return -1/(alpha * T) * np.log(1 - np.sqrt(0.95/(1 + 0.05 * xs**2)))

# -----------------------------------
# Create grids for linear and log scales.
# -----------------------------------
# For linear scale:
alpha_lin = np.linspace(0.01, 1.0, 100)
beta_lin  = np.linspace(0.01, 1.0, 100)
ALPHA_lin, BETA_lin = np.meshgrid(alpha_lin, beta_lin)

# For log scale:
alpha_log = np.logspace(-2, 0, 100)   # from 0.01 to 1.0
beta_log  = np.logspace(-2, 0, 100)
ALPHA_log, BETA_log = np.meshgrid(alpha_log, beta_log)

# Compute k_{1/2} and k_{0.05} for the linear grid:
Khalf_lin = k_half(ALPHA_lin, BETA_lin)
K005_lin  = k_005(ALPHA_lin, BETA_lin)

# Compute for the log grid:
Khalf_log = k_half(ALPHA_log, BETA_log)
K005_log  = k_005(ALPHA_log, BETA_log)

# Define additional contour levels to plot (constant k values)
extra_levels = [10, 50, 500]

# -----------------------------------
# Plotting
# -----------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Top left: k_{1/2} (linear axes)
c1 = axs[0, 0].contourf(alpha_lin, beta_lin, Khalf_lin, 20, cmap=cmap_choice)
axs[0, 0].set_title(r'Half-time: $k_{1/2}$ (linear axes)')
axs[0, 0].set_xlabel(r'$\alpha$')
axs[0, 0].set_ylabel(r'$\beta$')
# Add diagonal dashed line (alpha = beta)
axs[0, 0].plot([alpha_lin[0], alpha_lin[-1]], [beta_lin[0], beta_lin[-1]], 'k--', linewidth=2)
# Add extra contour lines for k = 10,50,500:
CS1 = axs[0, 0].contour(alpha_lin, beta_lin, Khalf_lin, levels=extra_levels, colors='white', linestyles='dashed')
axs[0, 0].clabel(CS1, fmt='%1.0f', colors='white')
fig.colorbar(c1, ax=axs[0, 0])

# Bottom left: k_{0.05} (linear axes)
c2 = axs[1, 0].contourf(alpha_lin, beta_lin, K005_lin, 20, cmap=cmap_choice)
axs[1, 0].set_title(r'Saturation time: $k_{0.05}$ (linear axes)')
axs[1, 0].set_xlabel(r'$\alpha$')
axs[1, 0].set_ylabel(r'$\beta$')
# Add diagonal dashed line
axs[1, 0].plot([alpha_lin[0], alpha_lin[-1]], [beta_lin[0], beta_lin[-1]], 'k--', linewidth=2)
# Add extra contour lines for k = 10,50,500:
CS2 = axs[1, 0].contour(alpha_lin, beta_lin, K005_lin, levels=extra_levels, colors='white', linestyles='dashed')
axs[1, 0].clabel(CS2, fmt='%1.0f', colors='white')
fig.colorbar(c2, ax=axs[1, 0])

# Top right: k_{1/2} (log axes)
c3 = axs[0, 1].contourf(alpha_log, beta_log, Khalf_log, 20, cmap=cmap_choice)
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')
axs[0, 1].set_title(r'Half-time: $k_{1/2}$ (log axes)')
axs[0, 1].set_xlabel(r'$\alpha$ (log scale)')
axs[0, 1].set_ylabel(r'$\beta$ (log scale)')
# Add diagonal dashed line on log scale
axs[0, 1].plot([alpha_log[0], alpha_log[-1]], [beta_log[0], beta_log[-1]], 'k--', linewidth=2)
# Add extra contour lines for k = 10,50,500:
CS3 = axs[0, 1].contour(alpha_log, beta_log, Khalf_log, levels=extra_levels, colors='white', linestyles='dashed')
axs[0, 1].clabel(CS3, fmt='%1.0f', colors='white')
fig.colorbar(c3, ax=axs[0, 1])

# Bottom right: k_{0.05} (log axes)
c4 = axs[1, 1].contourf(alpha_log, beta_log, K005_log, 20, cmap=cmap_choice)
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].set_title(r'Saturation time: $k_{0.05}$ (log axes)')
axs[1, 1].set_xlabel(r'$\alpha$ (log scale)')
axs[1, 1].set_ylabel(r'$\beta$ (log scale)')
# Add diagonal dashed line on log scale
axs[1, 1].plot([alpha_log[0], alpha_log[-1]], [beta_log[0], beta_log[-1]], 'k--', linewidth=2)
# Add extra contour lines for k = 10,50,500:
CS4 = axs[1, 1].contour(alpha_log, beta_log, K005_log, levels=extra_levels, colors='white', linestyles='dashed')
axs[1, 1].clabel(CS4, fmt='%1.0f', colors='white')
fig.colorbar(c4, ax=axs[1, 1])

plt.tight_layout()
plt.show()
