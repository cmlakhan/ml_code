## https://mlisi.xyz/post/simulating-correlated-variables-with-the-cholesky-factorization/
## Theoretical Background https://mlisi.xyz/post/bayesian-multilevel-models-r-stan/
##

!pip install --upgrade -q jax jaxlib

!pip install numpyro

!pip install --upgrade https://github.com/arviz-devs/arviz/archive/master.zip

# Make sure the Colab Runtime is set to Accelerator: TPU.
import requests
import os
if 'TPU_DRIVER_MODE' not in globals():
  url = 'http://' + os.environ['COLAB_TPU_ADDR'].split(':')[0] + ':8475/requestversion/tpu_driver_nightly'
  resp = requests.post(url)
  TPU_DRIVER_MODE = 1

# Colab runtime set to TPU accel
import requests
import os

# TPU driver as backend for JAX
from jax.config import config
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']
print(config.FLAGS.jax_backend_target)




import os
import arviz as az; az.style.use('arviz-darkgrid')


from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import seaborn as sns


import jax.numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.examples.datasets import UCBADMIT, load_dataset




plt.style.use('bmh')
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats('svg')

assert numpyro.__version__.startswith('0.2.4')







def glmm(dept, male, applications, admit=None):
    v_mu = numpyro.sample('v_mu', dist.Normal(0, np.array([4., 1.])))

    sigma = numpyro.sample('sigma', dist.HalfNormal(np.ones(2)))
    L_Rho = numpyro.sample('L_Rho', dist.LKJCholesky(2, concentration=2))
    scale_tril = sigma[..., np.newaxis] * L_Rho
    # non-centered parameterization
    num_dept = len(onp.unique(dept))
    z = numpyro.sample('z', dist.Normal(np.zeros((num_dept, 2)), 1))
    v = np.dot(scale_tril, z.T).T

    logits = v_mu[0] + v[dept, 0] + (v_mu[1] + v[dept, 1]) * male
    if admit is None:
        # we use a Delta site to record probs for predictive distribution
        probs = expit(logits)
        numpyro.sample('probs', dist.Delta(probs), obs=probs)
    numpyro.sample('admit', dist.Binomial(applications, logits=logits), obs=admit)




_, fetch_train = load_dataset(UCBADMIT, split='train', shuffle=False)
dept, male, applications, admit = fetch_train()
rng_key, rng_key_predict = random.split(random.PRNGKey(1))


kernel = NUTS(glmm)
mcmc = MCMC(kernel, args.num_warmup, args.num_samples, args.num_chains,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)

mcmc.run(rng_key, dept, male, applications, admit)

zs = mcmc.get_samples()

pred_probs = Predictive(glmm, zs)(rng_key_predict, dept, male, applications)['probs']

fig, ax = plt.subplots(1, 1)

ax.plot(range(1, 13), admit / applications, "o", ms=7, label="actual rate")
ax.errorbar(range(1, 13), np.mean(pred_probs, 0), np.std(pred_probs, 0),
            fmt="o", c="k", mfc="none", ms=7, elinewidth=1, label=r"mean $\pm$ std")
ax.plot(range(1, 13), np.percentile(pred_probs, 5, 0), "k+")
ax.plot(range(1, 13), np.percentile(pred_probs, 95, 0), "k+")
ax.set(xlabel="cases", ylabel="admit rate", title="Posterior Predictive Check with 90% CI")
ax.legend()

plt.savefig("ucbadmit_plot.pdf")
plt.tight_layout()


