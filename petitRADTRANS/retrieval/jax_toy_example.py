import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp

from jaxns import DefaultNestedSampler
from jaxns import Model
from jaxns import Prior
from jaxns import summary, plot_diagnostics, plot_cornerplot, save_results, load_results, save_pytree
tfpd = tfp.distributions

from jax._src.scipy.linalg import solve_triangular

def log_normal(x, mean, cov):
    model = jnp.ones(len(x))
    for i,gen in enumerate(x):
        model = model.at[i].set(gen)
    L = jnp.linalg.cholesky(cov)
    dx = model - mean
    dx = solve_triangular(L, dx, lower=True)
    return -0.5 * model.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * dx @ dx

# define our data and prior
ndims = 16
prior_mu = 15 * jnp.ones(ndims)
prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

data_mu = jnp.ones(ndims)*15
data_cov = jnp.diag(jnp.ones(ndims)) ** 2
data_cov = jnp.where(data_cov == 0., 0.99, data_cov)

true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

J = jnp.linalg.solve(data_cov + prior_cov, prior_cov)
post_mu = prior_mu + J.T @ (data_mu - prior_mu)
post_cov = prior_cov - J.T @ (prior_cov + data_cov) @ J

#print("True logZ={}".format(true_logZ))
#print("True post_mu={}".format(post_mu))
#print("True post_cov={}".format(post_cov))



#Uniform prior
def prior_model():
    x = []
    for i in range(ndims):
        gen = yield Prior(tfpd.Uniform(low=0, high= 30), name=f'theta_{i}')
        x.append(gen)
    return x

# The likelihood is a callable that will take
def log_likelihood(x):
    return log_normal(x, data_mu, data_cov)


model = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)

import jax
from time import time
start = time()
# Create the nested sampler class. In this case without any tuning.

ns = DefaultNestedSampler(
    model=model,
    max_samples=1e8,
    num_live_points = 400,
    parameter_estimation = True,
    verbose=True)

termination_reason, state = jax.jit(ns)(random.PRNGKey(42654))
results = ns.to_results(termination_reason=termination_reason, state=state)
save_results(results,'jaxns_toy_model_results.npz')

end = time()
print(f"JAXNS sampling runtime = {end-start} s")
"""
JAXNS sampling runtime = 10.343982696533203 s
--------
Termination Conditions:
Small remaining evidence
--------
likelihood evals: 440244
samples: 27784
phantom samples: 26128
likelihood evals / sample: 15.8
phantom fraction (%): 94.0%
--------
logZ=-47.5 +- 1.6
H=-57.31
ESS=104


PyMultinest
 ln(ev)=  -53.959502857582486      +/-  0.40153220726369360     
 Total Likelihood Evaluations:       128182
 Sampling finished. Exiting MultiNest
  analysing data from pmn_test.txt
PMN runtime = 48.168988943099976 s
evidence: -54.0 +- 0.4

"""
summary(results)