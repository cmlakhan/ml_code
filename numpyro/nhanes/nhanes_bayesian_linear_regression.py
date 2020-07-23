## Useful tutorial https://colcarroll.github.io/ppl-api/

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


import numpyro
import jax
jax.lib.xla_bridge.get_backend().platform

import os
import arviz as az; az.style.use('arviz-darkgrid')


from IPython.display import set_matplotlib_formats
import jax.numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import seaborn as sns

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive


plt.style.use('bmh')
if "NUMPYRO_SPHINXBUILD" in os.environ:
    set_matplotlib_formats('svg')

assert numpyro.__version__.startswith('0.2.4')


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer




def model(X,ndims,ndata,y_obs=None):
    w = numpyro.sample('w', dist.Normal(np.zeros(ndims), np.ones(ndims)))
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    y = numpyro.sample('y', dist.Normal(np.matmul(X, w), sigma * np.ones(ndata)), obs=y_obs)




DATASET_URL = 'https://www.dropbox.com/s/h88yeq7n721436d/nhanes_a1c.tsv?dl=1'
dset = pd.read_csv(DATASET_URL, sep='\t')
dset.head()

dset=dset.astype({'id':'object','gender': 'object', 'ethnicity':'object'})

dset.dtypes



X = dset.drop(['id','year','a1c'], axis=1)
y = dset['a1c']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first'))])


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_features
categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


X_train_processed= preprocessor.fit_transform(X_train)
X_test_processed= preprocessor.transform(X_test)

X_train_processed=np.asarray(X_train_processed)
X_test_processed=np.asarray(X_test_processed)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)



rng_key, rng_key_predict = random.split(random.PRNGKey(0))

num_warmup, num_samples, num_chains = 9000, 12000,4

ndims = X_train_processed.shape[1]
ndata = X_train_processed.shape[0]

# Run NUTS.
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup, num_samples, num_chains,  progress_bar=True)
mcmc.run(rng_key_, X=X_train_processed,y_obs=y_train,ndims=ndims,ndata=ndata)
mcmc.print_summary()
samples_3 = mcmc.get_samples()


predictive = Predictive(model, samples_3)


predictions_3 = Predictive(model_se, samples_3)(rng_key_,
                                                X=X_test_processed,
                                                ndims=X_test_processed.shape[1],
                                                ndata=X_test_processed.shape[0])['y']



residuals_4 = y_test - predictions_3

residuals_mean = np.mean(residuals_4, axis=0)
residuals_hpdi = hpdi(residuals_4, 0.9)

err = residuals_hpdi[1] - residuals_mean
fig, ax = plt.subplots(nrows=1, ncols=1)


# Plot Residuals
ax.errorbar(residuals_mean, y_test, xerr=err,
            marker='o', ms=5, mew=4, ls='none', alpha=0.8)

