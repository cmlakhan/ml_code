## Code for running elastic net on the UCI Communities and Crime dataset
## https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
## Analysis adapted from https://medium.com/@jayeshbahire/lasso-ridge-and-elastic-net-regularization-4807897cb722

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import time

import lightgbm as lgb

from xgboost import XGBRegressor

import numpy as np

import GPy


import qgrid

from scipy.stats import uniform
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import KFold

import sklearn

from sklearn.impute import KNNImputer


from sklearn.linear_model import ElasticNetCV

import feather


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import f_regression, mutual_info_regression

from joblib import Memory
from sklearn import feature_selection



from sklearn.feature_selection import SelectFwe, SelectPercentile, SelectFdr, SelectKBest

from sklearn.feature_selection import f_regression





#%matplotlib inline




X_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_train_filtered.feather')
Y_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_train_filtered.feather')


X_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_test_filtered.feather')
Y_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_test_filtered.feather')



X_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_rural_filtered.feather')
Y_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_rural_filtered.feather')

Y_train=Y_train[['DIABETES_CrudePrev']]
Y_test=Y_test[['DIABETES_CrudePrev']]
Y_rural=Y_rural[['DIABETES_CrudePrev']]



percent = .15
num_features = int(percent * X_train.shape[1])
var_sel = VarianceThreshold()
selector = SelectKBest(f_regression, k=num_features)
sc = RobustScaler(quantile_range=(10.0, 90.0))
polynomial = PolynomialFeatures(degree=2,interaction_only=True)
data_pipe_line = Pipeline([('variance',var_sel),('filter', selector), ('scale', sc), ('polynomial_features',polynomial)])
X_train_polynomial = data_pipe_line.fit_transform(X_train, Y_train)
X_train_polynomial.shape




param_dist = {"alpha": np.logspace(-10, 1,10000)}



enet_poly = ElasticNet(max_iter=10000,
                       normalize=False,
                       l1_ratio=.5,
                       warm_start=True,
                       precompute='auto',
                       random_state=1)


rs_polynomial = GridSearchCV(enet_poly,
                  param_grid=param_dist,
                  n_jobs=-1,
                  verbose=1,
                  cv=10)

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
