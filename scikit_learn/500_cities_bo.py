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


import GPyOpt

from GPyOpt.methods import BayesianOptimization



#%matplotlib inline

def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                ElasticNet(alpha=parameters[0],
                              l1_ratio=.5,
                           normalize=False,
                           max_iter=1000000),
                X_train, Y_train, scoring='r2', cv = 10,
        n_jobs=-1).mean()
    score = np.array(score)
    return score



X_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_train_filtered.feather')
Y_train = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_train_filtered.feather')


X_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_test_filtered.feather')
Y_test = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_test_filtered.feather')



X_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/X_rural_filtered.feather')
Y_rural = pd.read_feather('/home/ubuntu/efs/t2d_ses_project/v2/data/data_v2/Y_rural_filtered.feather')




Y_train=Y_train[['DIABETES_CrudePrev']]
Y_test=Y_test[['DIABETES_CrudePrev']]
Y_rural=Y_rural[['DIABETES_CrudePrev']]




percent = .30
num_features = int(percent * X_train.shape[1])
print(num_features)


var_sel = VarianceThreshold()
selector = SelectKBest(f_regression, k=num_features)
sc = RobustScaler(quantile_range=(10.0, 90.0))
polynomial = PolynomialFeatures(degree=2,interaction_only=True)
data_pipe_line = Pipeline([('variance',var_sel),('filter', selector), ('scale', sc), ('polynomial_features',polynomial)])
X_train_polynomial = data_pipe_line.fit_transform(X_train, Y_train)
X_train_polynomial.shape






bds = [{'name': 'alpha', 'type': 'continuous', 'domain': (1e-10, 1)}]



optimizer = BayesianOptimization(f=cv_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 verbosity=True)


optimizer.run_optimization(max_iter=300)



