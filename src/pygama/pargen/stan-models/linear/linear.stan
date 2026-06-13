
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] x_std;
    vector[N] y;
    vector[N] y_std;

    real a_mean;
    real a_std;
    real b_mean;
    real b_std;
}

parameters {
    real a;
    real b;
    vector[N] x_true;  // latent true x values
}

model {
    // priors
    a ~ normal(a_mean, a_std);
    b ~ normal(b_mean, b_std);
    x ~ normal(x_true, x_std);

    // likelihood
    y ~ normal(a * x_true + b, y_std);
}
