## Code for running elastic net on the UCI Communities and Crime dataset
## https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
## Analysis adapted from https://medium.com/@jayeshbahire/lasso-ridge-and-elastic-net-regularization-4807897cb722

import numpy as np
import dask.dataframe as dd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


import feather

import qgrid


from dask_ml.preprocessing import PolynomialFeatures

from dask_ml.preprocessing import RobustScaler

from sklearn.linear_model import SGDRegressor

from sklearn.pipeline import Pipeline

from dask_ml.preprocessing import OneHotEncoder

import dask_ml.model_selection as dcv

#%matplotlib inline

from dask.distributed import Client
import joblib

client = Client()  # Connect to a Dask Cluster



#X_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_train_filtered.feather')
#Y_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_train_filtered.feather')

#X_train.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_train_filtered.parquet.gzip', compression='gzip')
#Y_train.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_train_filtered.parquet.gzip', compression='gzip')

X_train = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_train_filtered.parquet.gzip')
Y_train = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_train_filtered.parquet.gzip')



#X_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_test_filtered.feather')
#Y_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_test_filtered.feather')


#X_test.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_test_filtered.parquet.gzip', compression='gzip')
#Y_test.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_test_filtered.parquet.gzip', compression='gzip')


X_test = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_test_filtered.parquet.gzip')
Y_test = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_test_filtered.parquet.gzip')



#X_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_rural_filtered.feather')
#Y_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_rural_filtered.feather')


#X_rural.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_rural_filtered.parquet.gzip', compression='gzip')
#Y_rural.to_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_rural_filtered.parquet.gzip', compression='gzip')

X_rural = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_rural_filtered.parquet.gzip')
Y_rural = dd.read_parquet('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_rural_filtered.parquet.gzip')


X_train.head()


sc = RobustScaler(quantile_range=(10.0, 90.0))
polynomial = PolynomialFeatures(degree=2,interaction_only=True)
data_pipe_line = Pipeline([('scale', sc), ('polynomial_features',polynomial)])



X_train_polynomial = data_pipe_line.fit_transform(X_train)

X_train_polynomial = X_train_polynomial.repartition(npartitions=20)




param_dist = {"alpha": np.logspace(-10, 1,10000)}



enet_poly = SGDRegressor(max_iter=10000,
                         penalty = 'elasticnet',
                         tol = 1e-4,
                         l1_ratio=.5,
                         warm_start=True,
                         random_state=1)


rs_polynomial = dcv.SuccessiveHalvingSearchCV(enet_poly,
                  parameters=param_dist)

rs_polynomial.fit(X_train_polynomial, Y_train)



enet_linear = ElasticNet(max_iter=10000,normalize=False, l1_ratio=.5, warm_start=True, precompute=True)
rs_linear = GridSearchCV(enet_linear,
                  param_grid=param_dist,
                  n_jobs=-1,
                  verbose=1,
                  cv=10)

rs_linear.fit(X_train, Y_train)




print(rs_polynomial.best_score_)

print(rs_linear.best_score_)
