import numpy as np
import pandas as pd
import plotnine as pn

def sigma_x(x):
    return np.exp(-x)

def jump_phase_prob(alpha, beta, p0, period, idxs):
    q = np.exp(-period * alpha)
    Q = beta
    return p0 * sigma_x(Q / (1 - q) * q * (1 - q**idxs))

def jump_prob(alpha, beta, p0, idxs_hab, idxs_react):
    p = np.zeros(350)
    phase = jump_phase_prob(alpha, beta, p0, 1, idxs_hab)
    p[0:100] = phase
    p[100:200] = phase
    p[200:300] = phase
    p[300:350] = jump_phase_prob(alpha, beta, p0, 5, idxs_react)
    return p

idxs_hab = np.linspace(0, 99, 100)
idxs_react = np.linspace(0, 49, 50)

param_sets = [
    {"alpha": 0.1, "beta": 0.2, "p0": 0.6},
    {"alpha": 0.05, "beta": 0.1, "p0": 0.6},
#    {"alpha": 0.127, "beta": 0.182, "p0": 0.596},
#    {"alpha": 0.066, "beta": 0.10, "p0": 0.600},
#    {"alpha": 0.379, "beta": 0.167, "p0": 0.460},
#    {"alpha": 0.038, "beta": 0.021, "p0": 0.500},
]

rows = []
for i, params in enumerate(param_sets):
    p = jump_prob(params["alpha"], params["beta"], params["p0"], idxs_hab, idxs_react)
    for t in range(350):
        rows.append({
            "trial": t,
            "prob": p[t],
            "label": f'α={params["alpha"]}, β={params["beta"]}, p₀={params["p0"]}'
        })
df = pd.DataFrame(rows)
plot = (
    pn.ggplot(df, pn.aes(x="trial", y="prob", color="label"))
    + pn.geom_line()
    + pn.labs(x="Trial", y="Jump Probability", color="Parameters")
    + pn.theme_minimal()
)
plot.show()
plot.save('sim-solutions-3.jpeg')
