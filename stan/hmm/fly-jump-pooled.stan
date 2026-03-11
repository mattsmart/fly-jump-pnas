data {
  int<lower=0> num_experiments;
  int<lower=0> obs_per_experiment;
  array[num_experiments, obs_per_experiment] int<lower=1, upper=2> response;
}
transformed data {
  int<lower=1> num_outcomes = 2;
  int<lower=1> num_states = 2;
  vector<lower=0>[2] alpha = [5.0, 5.0]';
}
parameters {
  // two states, two outputs
  simplex[num_states] init_prob, trans_prob1, trans_prob2;
  array[num_states] simplex[num_outcomes] emit_prob;
}
model {
  matrix[num_states, num_states] trans_prob = [trans_prob1', trans_prob2'];
  for (n in 1:num_experiments) {
    matrix[num_outcomes, obs_per_experiment] log_omega;
    for (k in 1:num_outcomes) {
      for (j in 1:obs_per_experiment) {
        log_omega[k, j] = log(emit_prob[k, response[n, j]]);
      }
    }
    target += hmm_marginal(log_omega, trans_prob, init_prob);
  }

  init_prob ~ dirichlet(alpha);
  trans_prob1 ~ dirichlet(alpha);
  trans_prob2 ~ dirichlet(alpha);
  for (k in 1:num_states) {
    emit_prob[k] ~ dirichlet(alpha);
  }
}
