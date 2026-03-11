functions {
  /**
   * The link function here returns the jump probability for a given
   * input value `x` (volume of water in the leaky bucket), which is
   * `1 / (1 + x**2)`.  The choice of link is not motivated by theory
   * other than it must be (a) a bijection from `(0, inf)` to `(0,
   * 1]`, (b) continuously differentiable, and (c) downward monotonic.
   * Together these imply `sigma_x(0) = 1` and `sigma_x(x) -> 0` as `x
   * -> inf`.  There's a natural connection to complementary cdfs for
   * a random variable `X`, `1 - F_X(x) = Pr[X > x]`.

The other obvious choices would be `sigma_x(x) =
   * exp(-x)`, which is the complementary cumulative distribution
   * function for `exponential(1)`.  The link `sigma_x(x) = 2 - 2 *
   * inv_logit(x)` is the complementary cdf for a half standad
   * logistic.  
   *
   * @param x compartment value
   * @return probability
   */
  real sigma_x(real x) {
    return inv(1 + square(x));
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
        p_of_n[idx] = p0 * sigma_x(x_hab);
        idx += 1;
      }
    }
    for (i in 1:num_trials_react) {
      real x_react = Q * (1 - pow(q_react, i)) / (1 - q_react);
      p_of_n[idx] = p0 * sigma_x(x_react);
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
  vector<lower=0>[num_experiments] alpha, beta;
  vector<lower=0, upper=1>[num_experiments] p0;
}
model {
  // priors
  alpha ~ exponential(2);
  beta ~ exponential(2);
  p0 ~ beta(10, 10);
  // likelihood
  for (n in 1:num_experiments) {
    jump[n] ~ bernoulli(jump_prob(alpha[n], beta[n], p0[n]));
  }
}
