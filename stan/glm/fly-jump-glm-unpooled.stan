data {
  int<lower=0> num_experiments;
  int<lower=0> obs_per_experiment;
  int<lower=0> num_flies;
  array[num_experiments] int<lower=1, upper=num_flies> fly;
  array[num_experiments, obs_per_experiment] int<lower=0, upper=1> response;
}
parameters {
  vector<offset=-0.7>[num_flies] alpha_std;     // acclimated jump log odds
  vector<offset=-0.3>[num_flies] beta_std;      // unacclimated log prob boost
  vector<offset=-2.1>[num_flies] gamma_std;     // acclimation log scale decline
}
transformed parameters {
  vector<lower=0, upper=1>[num_flies] alpha = Phi_approx(alpha_std);
  vector<upper=0>[num_flies] beta = -exp(beta_std);
  vector<upper=0>[num_flies] gamma = -exp(gamma_std);
}
model {
  alpha_std ~ normal(-0.7, 1);
  beta_std ~ normal(-0.3, 1);
  gamma_std ~ normal(-2.1, 1);

  for (t in 1:obs_per_experiment) {
    vector[num_experiments] p = alpha[fly] + (1 - alpha[fly]) .* exp(beta[fly] + gamma[fly] * (t - 1));
    response[ , t] ~ bernoulli(p);
  }
}
