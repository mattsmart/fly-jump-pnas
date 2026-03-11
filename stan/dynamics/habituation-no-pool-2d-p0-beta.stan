// 2D model: Fit p0 and beta with alpha fixed globally
// Habituation dynamics with global habituation rate (alpha_global)

functions {
  vector sigma_x(vector x) {
    return inv(1 + square(x));
  }

  vector jump_phase_prob(real alpha, real beta, real p0, real period, vector idxs) {
    real q = exp(-period * alpha);
    real Q = beta;
    return p0 * sigma_x(Q / (1 - q) * (1 - pow(q, idxs)) * q);
  }

  vector jump_prob(real alpha, real beta, real p0, vector idxs_hab,
                   vector idxs_react, int total_trials) {
    vector[total_trials] p;
    p[1:200] = jump_phase_prob(alpha, beta, p0, 1, idxs_hab);
    for (j in {200, 400, 600, 800}) {
      p[j + 1:j + 200] = p[1:200];
    }
    p[1001:1050] = jump_phase_prob(alpha, beta, p0, 5, idxs_react);
    return p;
  }
}

data {
  real<lower=0> alpha_global;  // fixed habituation rate from 3D model
  int<lower=0> num_experiments, num_trials_per_experiment;
  array[num_experiments, num_trials_per_experiment] int<lower=0, upper=1> jump;
}

transformed data {
  vector[200] idxs_hab = linspaced_vector(200, 0, 199);
  vector[50] idxs_react = linspaced_vector(50, 0, 49);
  int total_trials = 1050;
}

parameters {
  vector<lower=0>[num_experiments] beta;
  vector<lower=0, upper=1>[num_experiments] p0;
}

model {
  // priors (same as 3D model)
  beta ~ exponential(3);  // central 95% approx (0.01, 1.2)
  p0 ~ beta(3, 3);        // central 95% approx (0.15, 0.85)

  // likelihood using fixed alpha_global
  for (n in 1:num_experiments) {
    vector[1050] p = jump_prob(alpha_global, beta[n], p0[n],
                               idxs_hab, idxs_react, total_trials);
    jump[n] ~ bernoulli(p);
  }
}
