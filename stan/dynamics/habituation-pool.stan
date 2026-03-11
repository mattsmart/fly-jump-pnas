functions {
  /**
   * Return an inverse quadratic complementary sigmoid of `x`.  The
   * choice here is somewhat arbitrary, but must be (a) continuously
   * differentiable, (b) downward monotonic, and (c) onto (0, 1],
   * which implies sigma_x(0) = 1 and sigma_x(x) -> 0 as x -> inf.
   *
   * @param x activity
   * @return probability
   */
  vector inverse_quadratic_decay(vector x) {
    return inv(1 + square(x));
  }
}
data {
  int<lower=0> num_experiments, num_trials_per_experiment;
  array[num_experiments, num_trials_per_experiment] int<lower=0, upper=1> jump;
}
transformed data {
  array[200] int<lower=0> hab_count = rep_array(0, 200);
  for (i in 1:200) {
    for (j in {0, 200, 400, 600, 800}) {
      hab_count[i] += sum(jump[, j + i]);
    }
  }

  array[50] int<lower=0> react_count;
  for (i in 1:50) {
    react_count[i] = sum(jump[, 1000 + i]);
  }
}
parameters {
  real<lower=0> alpha, beta;
  real<lower=0, upper=1> p0;
}
model {
  real q_hab = exp(-alpha);
  real q_react = exp(-5 * alpha);
  real Q = beta;
  vector[200] idx200 = linspaced_vector(200, 0, 199);
  vector[200] p_hab = p0 * inverse_quadratic_decay(Q / (1 - q_hab) * (1 - pow(q_hab, idx200)) * q_hab);
  vector[50] idx50 = linspaced_vector(50, 0, 49);
  vector[50] p_react = p0 * inverse_quadratic_decay(Q / (1 - q_react) * (1 - pow(q_react, idx50)) * q_react);

  // priors
  alpha ~ exponential(2);
  beta ~ exponential(2);
  p0 ~ beta(10, 10);
  
  // likelihood
  hab_count ~ binomial(5 * num_experiments, p_hab);
  react_count ~ binomial(num_experiments, p_react);
}
