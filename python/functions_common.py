import numpy as np
from scipy.stats import binomtest


def logit(x):
    return np.log(x / (1 - x))


def sigma_func(x):
    return 1 / (1 + x ** 2)


def jump_phase_prob(alpha, beta, p0, period, idxs):
    # idxs start at 1
    q = np.exp(-period * alpha)
    Q = beta
    return p0 * sigma_func(Q / (1.0 - q) * (1.0 - q**idxs) * q)


def jump_prob(alpha, beta, p0):
    total_trials = 1050
    idxs_hab = np.arange(0, 200)
    idxs_react = np.arange(0, 50)

    p = np.zeros(total_trials)
    # Habituation blocks (5 groups of 200 trials)
    p[0:200] = jump_phase_prob(alpha, beta, p0, 1.0, idxs_hab)
    for start in [200, 400, 600, 800]:
        p[start:start+200] = p[0:200]
    # Reactivity block (last 50 trials)
    p[1000:1050] = jump_phase_prob(alpha, beta, p0, 5.0, idxs_react)
    return p


def likelihood_func(time_indices, alpha, beta, p0, pulse_period):
    q = np.exp(-alpha * pulse_period)
    pjump = p0 * sigma_func(beta * (1 - q ** time_indices) / (1 - q) * q)
    return pjump


def likelihood_func_vec(time_indices, alpha, beta, p0, pulse_period):
    q = np.exp(-alpha * pulse_period)
    pjump = np.zeros((len(alpha), len(time_indices)))
    for idx in range(len(alpha)):
        pjump[idx, :] = p0[idx] * sigma_func(beta[idx] * (1 - q[idx] ** time_indices) / (1 - q[idx]) * q[idx])
    return pjump


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_conf_movingavg(timeseries_p_est, n=5, confidence_level=0.95, method='exact'):
    conf_interval = np.zeros((len(timeseries_p_est), 2))  # lower and upper CI
    for k in range(len(timeseries_p_est)):
        result = binomtest(k=int(timeseries_p_est[k] * n), n=n)
        conf_interval[k, :] = result.proportion_ci(confidence_level=confidence_level, method=method)
        # print(p_hab_mean_timeseries_1pt[k], result.statistic, result.proportion_ci())
    return conf_interval
