
data {
  int<lower=0> N;
  vector[N] energies;
  real mu_lower;
  real mu_upper;
  real sigma_lower;
  real sigma_upper;
}
parameters {
  real<lower=mu_lower, upper=mu_upper> mu;
  real<lower=sigma_lower, upper=sigma_upper> sigma;
}

model {
  energies ~ normal(mu, sigma);
}
