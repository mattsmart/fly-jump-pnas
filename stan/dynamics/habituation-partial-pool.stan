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
  int<lower=0> num_experiments, num_trials_per_experiment;
  array[num_experiments, num_trials_per_experiment] int<lower=0, upper=1> jump;
}
transformed data {
  vector[200] idxs_hab = linspaced_vector(200, 0, 199);
  vector[50] idxs_react = linspaced_vector(50, 0, 49);
  int total_trials = 1050;
}

parameters {
  real mu_alpha, mu_beta, mu_p0; real<lower=0> sigma_alpha, sigma_beta, sigma_p0;
  vector<offset=mu_alpha, multiplier=sigma_alpha>[num_experiments] log_alpha;
  vector<offset=mu_beta, multiplier=sigma_beta>[num_experiments] log_beta;
  vector<offset=mu_p0, multiplier=sigma_p0>[num_experiments] logit_p0;
}
transformed parameters {
  vector[num_experiments] alpha = exp(log_alpha);
  vector[num_experiments] beta = exp(log_beta);
  vector[num_experiments] p0 = inv_logit(logit_p0);
}
model {
  // hyperpriors
  { mu_alpha, mu_beta, mu_p0 } ~ normal(0, 2);            // 95%: (-4, 4)
  { sigma_alpha, sigma_beta, sigma_p0} ~ lognormal(0, 0.5);  // 95% (0.4, 2.7)

  // priors
  log_alpha ~ normal(mu_alpha, sigma_alpha);
  log_beta ~ normal(mu_beta, sigma_beta);
  logit_p0 ~ normal(mu_p0, sigma_p0);

  // likelihood
  for (n in 1:num_experiments) {
    jump[n] ~ bernoulli(jump_prob(alpha[n], beta[n], p0[n],
                                  idxs_hab, idxs_react, total_trials));
  }
}
