data {
  int<lower=1> N;
  int<lower=2> B;                      // number of blocks
  vector[N] y;                         // log(SWD) pairwise point estimates
  array[N] int<lower=1, upper=B> i;    // block index i
  array[N] int<lower=1, upper=B> j;    // block index j (j > i)
}

parameters {
  real mu_d;
  real log_d1;

  // Monotone decrement
  vector[B-1] delta;

  // Monotone decrement hyperparameters
  real<lower=0> mu_delta;
  real<lower=0> sigma_delta;

  // Observation noise on y = log(SWD_hat)
  real<lower=0> sigma;
}

transformed parameters {
  vector[B] log_d;
  vector<lower=0>[B] d;
  
  log_d[1] = log_d1;
  for (k in 2:B) {
    log_d[k] = log_d[k-1] - delta[k-1];
  }
  d = exp(log_d);
}

model {
  mu_d ~ normal(1.5, 1.8);
  log_d1 ~ normal(mu_d, 1.0);

  mu_delta  ~ normal(0.8, 0.2);
  sigma_delta ~ lognormal(log(0.2), 0.5);
  delta    ~ normal(mu_delta, sigma_delta);

  sigma ~ normal(0, 1);

  for (n in 1:N) {
    real dij = fmax(abs(d[i[n]] - d[j[n]]), 1e-12);
    real mu  = log(dij);
    y[n] ~ normal(mu, sigma);
  }
}

generated quantities {
  vector[N] y_fit;
  vector[N] y_rep;
  vector[N] log_lik;

  real<lower=0> d1    = exp(log_d1);

  for (n in 1:N) {
    real dij = fmax(abs(d[i[n]] - d[j[n]]), 1e-12);
    real mu  = log(dij);

    y_fit[n]   = mu;
    y_rep[n]   = normal_rng(mu, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma);
  }
}
