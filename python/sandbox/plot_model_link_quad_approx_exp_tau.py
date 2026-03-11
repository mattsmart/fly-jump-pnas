import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
Script shows analytic time-scales for the quadratic link function

Let p[k] = p0 / (1 + x[k]^2) be the jump probability for the quadratic link

Let R(k) = (p[k] - p_inf)/(p0 - p_inf) be the normalized response

timescale 1: time-to-saturate k_z
  - let z = 0.05 <--> 9% % saturation of normalized response
  - then
    k_z = -1/(alpha T) * ln(1 - sqrt((1 - z)/(1 + z(x_star)^2)))

timescale 2: effective time constant tau_quad
  - also derived from the normalized response R(k) for the quadratic link
  - solve for R(k_1/2) = 0.5
  - then 
        tau_quad = k_1/2 / ln(2)
                 = - (1/(alpha T ln(2))) * ln( 1 - 1/np.sqrt((x_star)**2 + 2) )
    
"""

# Parameters
alpha = 0.01         # decay rate per second
T = 1                # inter-stimulus interval (s)
beta = 0.07          # increment per stimulus (try beta < alpha for the "slow accumulation" case)
p0 = 1.0             # maximal jump probability when x=0
k_max = 200          # number of trials to simulate
k_vals = np.arange(0, k_max+1)

# -----------------------------
# Internal State Dynamics
# -----------------------------
# Compute the decay factor and steady state for x:
r = np.exp(-alpha * T)
x_star = (beta * r) / (1 - r)   # steady state: x* = beta * e^(-alpha T) / (1 - e^(-alpha T))
# Evolution of x[k]:
x = x_star * (1 - np.exp(-alpha * T * k_vals))

# -----------------------------
# Quadratic Link Function
# -----------------------------
def f_quadratic(x):
    return p0 / (1 + x**2)

# Compute observable jump probability for quadratic link:
p_quad = f_quadratic(x)

# Compute asymptotic value for the quadratic link:
p_inf_quad = f_quadratic(x_star)

# -----------------------------
# Normalized Response for Quadratic Link
# -----------------------------
# By definition,
# R(k) = (p[k] - p_inf)/(p0 - p_inf)
R_quad = (p_quad - p_inf_quad) / (p0 - p_inf_quad)

# (It can be shown that, for this model, an algebraic simplification yields:
#   R(k) = [1 - (1 - e^{-alpha T k})^2] / [1 + (x^*)^2 (1 - e^{-alpha T k})^2],
# but here we compute it numerically.)

# -----------------------------
# Analytic Effective Time Constant for Quadratic Link
# -----------------------------
# We define the effective half-life k_{1/2} by R(k_{1/2}) = 1/2.
# For the quadratic link, one can derive (by algebra) that if we set
#   y = 1 - e^{-alpha T k},
# then one finds
#   R(k) = [1 - y^2] / [1 + (x^*)^2 y^2].
# Setting R(k)=1/2 gives:
#   [1 - y^2] / [1 + (x^*)^2 y^2] = 1/2.
# Multiply both sides by the denominator:
#   2(1 - y^2) = 1 + (x^*)^2 y^2  --> 2 - 2y^2 = 1 + (x^*)^2 y^2.
# Rearranging:
#   1 = y^2[(x^*)^2 + 2].
# Thus,
#   y = 1/sqrt((x^*)^2 + 2).
# But y = 1 - e^{-alpha T k_{1/2}}, so
#   e^{-alpha T k_{1/2}} = 1 - 1/sqrt((x^*)^2 + 2).
# Taking logarithms:
#   -alpha T k_{1/2} = ln(1 - 1/sqrt((x^*)^2+2)),
# so
#   k_{1/2} = - (1/(alpha T)) * ln(1 - 1/np.sqrt((x_star)**2 + 2)).

# -----------------------------
# Exponential Fit to Normalized Decay (for comparison)
# -----------------------------
def exp_decay(k, tau):
    return np.exp(-k/tau)

def unnormalized_response(R, p_inf):
    return p_inf + R * (p0 - p_inf)

# Then we define the effective time constant as:
tau_quad_analytic = (-1/(alpha * T) * np.log(1 - 1/np.sqrt(x_star**2 + 2))) / np.log(2)
R_quad_analytic = exp_decay(k_vals, tau_quad_analytic)


# Fit an exponential to the normalized response R_quad
popt_quad, _ = curve_fit(exp_decay, k_vals, R_quad, p0=[10])
tau_quad_fit = popt_quad[0]
R_quad_fit = exp_decay(k_vals, tau_quad_fit)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8, 5))

# Panel 1: Internal state evolution x[k]
plt.subplot(2, 2, 1)
plt.plot(k_vals, x, 'bo-', label=r'$x[k]$')
plt.axhline(x_star, color='gray', linestyle='--', label=r'Steady state $x^*$')
plt.xlabel('Trial k')
plt.ylabel('Internal state x[k]')
plt.title('Evolution of Internal State')
plt.legend()

# Panel 2: Observable jump probability for quadratic link
plt.subplot(2, 2, 2)
plt.plot(k_vals, p_quad, 'g.-', label=r'$p_0/(1+x^2)$')
plt.plot(k_vals, unnormalized_response(R_quad_analytic, p_inf_quad), 'g-', label=r'$\tau_{quad}$ fit')
plt.xlabel('Trial k')
plt.ylabel('Jump probability p[k]')
plt.title('Observable Response (Quadratic Link)')
plt.legend()

# Panel 3: Normalized decay R(k) for quadratic link and exponential fit
plt.subplot(2, 1, 2)
plt.plot(k_vals, R_quad, 'g.-', label='Normalized R(k) (Data)')
plt.plot(k_vals, R_quad_fit, 'g--', label=f'Exp Fit: tau_fit = {tau_quad_fit:.2f}')
plt.plot(k_vals, R_quad_analytic, 'g-', label=f'Exp Analytic: tau_quad = {tau_quad_analytic:.2f}')
plt.xlabel('Trial k')
plt.ylabel('Normalized Response R(k)')
plt.title('Exponential Fit to Normalized Decay (Quadratic Link)')
# Annotate analytic tau from half-life derivation
#plt.text(0.55*k_max, 0.85, f'Analytic tau (quad) = {tau_quad_analytic:.2f}', color='darkgreen', fontsize=12)
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Analytic time-to-saturation for a general fraction z:
z = 0.05  # For 90% saturation (normalized response 0.1)
# Here, x_star is our steady state value from the internal model.
k_z = -1/(alpha * T) * np.log(1 - np.sqrt((1 - z)/(1 + z*(x_star)**2)))

print("Time-to-%d%% saturation (z=%.2f) = %.2f" % ((100-100*z), z, k_z))

# Plotting the normalized response and marking k_z:
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(k_vals, R_quad, 'g.-', label='Normalized Response R(k)')
plt.axhline(z, color='red', linestyle='--', label=f'R = {z}')
plt.axvline(k_z, color='red', linestyle='--', label=r'$k_{z} \approx %.2f$ for $z=%.2f \rightarrow %d$' % (k_z, z, int(100 - 100*z))
            +  '%')
#plt.axvline(k_z, color='red', linestyle='--', label=r'for $z=%.2f$ %d%% saturation' % (z, (100 - 100*z)))
plt.axvline(tau_quad_analytic, color='green', linestyle='--', label=r'$\tau_{quad} \approx$' + f'{tau_quad_analytic:.2f}')
plt.axhline(1/np.exp(1), color='green', linestyle='--', label=r'hit $1/e \approx %.1f$' % (100/np.exp(1)) + '%')
plt.xlabel('Trial k')
plt.ylabel('Normalized Response R(k)')
plt.title('Time-to-Saturation for Quadratic Link')
plt.legend()
plt.show()
