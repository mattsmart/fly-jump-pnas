functions {
  /**
   * Return a quadratic sigmoid of `x`.  The choice here is somewhat
   * arbitrary, but must be (a) continuously differentiable, (b)
   * downward monotonic, and (c) onto (0, 1], which implies sigma_x(0) = 1 and
   * sigma_x(x) -> 0 as x -> inf.
   *
   * @param x activity
   * @return probability
   */
  real inverse_quadratic_decay(real x) {
    return inv(1 + square(x));
  }

  real exponential_decay(real x) {
    return exp(-x);
  }
  
  /**
   * Retrn the sequence of jump probabilities with parameters alpha,
   * beta, and p0.  The experimental design is baked in.
   *
   * @param alpha rate of habituation
   * @param beta intensity of pulse
   * @param p0 base probability
   * @return vector of fly jumping probabilities.
   */
  vector jump_prob(real alpha, real beta, real p0) {
    real pulse_area = 1.0;    int num_trials_hab = 200;
    int num_cycles_hab = 5;   int num_trials_react = 50;
    real period_hab = 1;      real period_react = 5;
    
    int total_trials = num_cycles_hab * num_trials_hab + num_trials_react;
    vector[total_trials] p_of_n;

    real q_hab = exp(-alpha * period_hab);
    real q_react = exp(-alpha * period_react);
    real Q = beta * pulse_area;

    int idx = 1;
    for (cycle in 1:num_cycles_hab) {
      for (i in 1:num_trials_hab) {
        real x_hab = Q * (1 - pow(q_hab, i)) / (1 - q_hab);
        p_of_n[idx] = p0 * inverse_quadratic_decay(x_hab);
        idx += 1;
      }
    }
    for (i in 1:num_trials_react) {
      real x_react = Q * (1 - pow(q_react, i)) / (1 - q_react);
      p_of_n[idx] = p0 * inverse_quadratic_decay(x_react);
      idx += 1;
    }
    return p_of_n;
  }
}
data {
  int<lower=0> num_experiments, num_trials_per_experiment;
  array[num_experiments, num_trials_per_experiment] int<lower=0, upper=1> jump;
}
parameters {
  real<lower=0> alpha, beta;
  real<lower=0, upper=1> p0;
}
model {
  // priors
  alpha ~ exponential(2);
  beta ~ exponential(2);
  p0 ~ beta(10, 10);
  // likelihood
  for (n in 1:num_experiments) {
    jump[n] ~ bernoulli(jump_prob(alpha, beta, p0));
  }
}
