// 1D model: Fit only p0 (biased coin, no habituation dynamics)
// Each fly has a constant jump probability across all 1050 trials

data {
  int<lower=0> num_experiments, num_trials_per_experiment;
  array[num_experiments, num_trials_per_experiment] int<lower=0, upper=1> jump;
}

parameters {
  vector<lower=0, upper=1>[num_experiments] p0;
}

model {
  // prior
  p0 ~ beta(3, 3);  // central 95% approx (0.15, 0.85)

  // likelihood - constant probability for all trials
  for (n in 1:num_experiments) {
    jump[n] ~ bernoulli(p0[n]);
  }
}
