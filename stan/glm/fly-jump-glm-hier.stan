data {
  int<lower=0> num_experiments;
  int<lower=0> obs_per_experiment;
  int<lower=0> num_flies;
  array[num_experiments] int<lower=1, upper=num_flies> fly;
  array[num_experiments, obs_per_experiment] int<lower=0, upper=1> response;
}
parameters {
  real<lower=0> sigma_alpha;
  real<offset=-0.7> mu_alpha;
  vector<offset=mu_alpha, multiplier=sigma_alpha>[num_flies] alpha;  // acclimated jump log odds

  real<lower=0> sigma_beta;
  real<offset=-0.3> mu_beta;
  vector<offset=mu_beta, multiplier=sigma_beta>[num_flies] beta;     // unacclimated log prob boost

  real<lower=0> sigma_gamma;
  real<offset=-2.1> mu_gamma;
  vector<offset=mu_gamma, multiplier=sigma_gamma>[num_flies] gamma;  // acclimation log scale decline
}
transformed parameters {
  vector<lower=0, upper=1>[num_flies] alpha_il = inv_logit(alpha_std); // logit(alpha) ~ normal(mu_alpha, sigma_alpha)
  vector<upper=0>[num_flies] beta_ne = -exp(beta_std);  // -beta ~ lognormal(mu_beta, sigma_beta)
  vector<upper=0>[num_flies] gamma_ne = -exp(gamma_std);  // -gamma ~ lognormal(mu_gamma, sigma_gamma)
}
model {
  sigma_alpha ~ exponential(1);
  sigma_beta ~ exponential(1);
  sigma_gamma ~ exponential(1);
  mu_alpha ~ normal(-0.7, 1);
  mu_beta ~ normal(-0.3, 1);
  mu_gamma ~ normal(-2.1, 1);
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  gamma ~ normal(mu_gamma, sigma_gamma);

  for (t in 1:obs_per_experiment) {
    vector[num_experiments] p = alpha_il[fly] + (1 - alpha_il[fly]) .* exp(beta_ne[fly] + gamma_ne[fly] * (t - 1));
    response[ , t] ~ bernoulli(p);
  }
}
