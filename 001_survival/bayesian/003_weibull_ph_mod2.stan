
data {
  // dimensions
  int<lower=0> N;             // number of observations
  int<lower=1> M;             // number of predictors
  
  // observations
  matrix[N, M] X;             // predictors for observation n
  vector[N] y;                // time for observation n
  vector[N] event;            // event status (1:event, 0:censor) for obs n
}

parameters {
  vector[M] beta;
  real intercept;  
  real<lower=0> shape;
  
}

model {
  
  vector[N] mu = intercept + X * beta;
  
  // priors
  shape ~ exponential(0.001);
  intercept ~ normal(0, 10);
  beta ~ normal(0, 10);
  

  // likelihood
  for (n in 1:N) {
      if (event[n]==1)
        target += weibull_lpdf(y[n] | shape, exp(-mu[n]/shape));
      else
        target += weibull_lccdf(y[n] | shape, exp(-mu[n]/shape));
  }
}

generated quantities{
  vector[10] y_pred0;
  vector[10] y_pred1;
  
  // for posterior predictive checks you want to simulate your 
  // original data set ie. 1 to N
  // here i just want to look at whether I can generate data
  
  for (i in 1:10) {
    y_pred0[i] = weibull_rng(shape, exp(-intercept/shape));
    y_pred1[i] = weibull_rng(shape, exp(-(intercept+beta[1])/shape));
  }
  
}


