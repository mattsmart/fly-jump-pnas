data {
  int<lower=0> num_experiments;
  int<lower=0> obs_per_experiment;
  array[num_experiments, obs_per_experiment] int<lower=0, upper=1> response;
}
parameters {
  real alpha_std;     // acclimated jump log odds
  real beta_std;      // unacclimated log prob boost
  real gamma_std;     // acclimation log scale decline
}
transformed parameters {
  real<lower=0, upper=1> alpha = Phi_approx(alpha_std);
  real<upper=0> beta = -exp(beta_std);
  real<upper=0> gamma = -exp(gamma_std);
}
model {
  alpha_std ~ normal(-0.7, 1);
  beta_std ~ normal(-0.3, 1);
  gamma_std ~ normal(-2.1, 1);
  
  for (t in 1:obs_per_experiment) {
    real p = alpha + (1 - alpha) * exp(beta + gamma * (t - 1));
    response[ , t] ~ bernoulli(p);
  }
}
