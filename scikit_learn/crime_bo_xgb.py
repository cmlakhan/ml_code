
## Code for running elastic net on the UCI Communities and Crime dataset
## https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
## Analysis adapted from https://medium.com/@jayeshbahire/lasso-ridge-and-elastic-net-regularization-4807897cb722

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import time

import lightgbm as lgb

from xgboost import XGBRegressor

import numpy as np

#from sklearn.pipeline import Pipeline
import GPy

import GPyOpt

from GPyOpt.methods import BayesianOptimization


import qgrid

from scipy.stats import uniform

import sklearn

from sklearn.impute import KNNImputer

#%matplotlib inline




# Optimization objective
def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                XGBRegressor(learning_rate=parameters[0],
                              gamma=int(parameters[1]),
                              max_depth=int(parameters[2]),
                              n_estimators=int(parameters[3]),
                              min_child_weight = parameters[4]),
                X, Y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score


def search(pipeline, parameters, X_train, y_train, X_test, y_test, optimizer='grid_search', n_iter=None):
      start = time.time()

      if optimizer == 'grid_search':
            grid_obj = GridSearchCV(estimator=pipeline,
                                    param_grid=parameters,
                                    cv=10,
                                    refit=True,
                                    return_train_score=False,
                                    scoring='accuracy',
                                    )
            grid_obj.fit(X_train, y_train, )

      elif optimizer == 'random_search':
            grid_obj = RandomizedSearchCV(estimator=pipeline,
                                          param_distributions=parameters,
                                          cv=5,
                                          n_iter=n_iter,
                                          refit=True,
                                          return_train_score=False,
                                          scoring='accuracy',
                                          random_state=1)
            grid_obj.fit(X_train, y_train, )

      else:
            print('enter search method')
            return

      estimator = grid_obj.best_estimator_
      cvs = cross_val_score(estimator, X_train, y_train, cv=5)
      results = pd.DataFrame(grid_obj.cv_results_)

      print("##### Results")
      print("Score best parameters: ", grid_obj.best_score_)
      print("Best parameters: ", grid_obj.best_params_)
      print("Cross-validation Score: ", cvs.mean())
      print("Test Score: ", estimator.score(X_test, y_test))
      print("Time elapsed: ", time.time() - start)
      print("Parameter combinations evaluated: ", results.shape[0])

      return results, estimator


crime = pd.read_csv('CommViolPredUnnormalizedData.txt',
                    header=None,
                    index_col=False,
                    na_values='?',
                    names=['communityname',
                          'state',
                          'countyCode',
                          'communityCode',
                          'fold',
                          'population',
                          'householdsize',
                          'racepctBlack',
                          'racePctWhite',
                          'racePctAsian',
                          'racePctHisp',
                          'agePct12t21',
                          'agePct12t29'
                          'agePct16t24',
                          'agePct65up',
                          'numbUrban',
                          'pctUrban',
                          'medIncome',
                          'pctWWage',
                          'pctWFarmSelf',
                          'pctWInvInc',
                          'pctWSocSec',
                          'pctWPubAsst',
                          'pctWRetire',
                          'medFamInc',
                          'perCapInc',
                          'whitePerCap',
                          'blackPerCap',
                          'indianPerCap',
                          'AsianPerCap',
                          'OtherPerCap',
                          'HispPerCap',
                          'NumUnderPov',
                          'PctPopUnderPov',
                          'PctLess9thGrade',
                          'PctNotHSGrad',
                          'PctBSorMore',
                          'PctUnemployed',
                          'PctEmploy',
                          'PctEmplManu',
                          'PctEmplProfServ',
                          'PctOccupManu',
                          'PctOccupMgmtProf',
                          'MalePctDivorce',
                          'MalePctNevMarr',
                          'FemalePctDiv',
                          'TotalPctDiv',
                          'PersPerFam',
                          'PctFam2Par',
                          'PctKids2Par',
                          'PctYoungKids2Par',
                          'PctTeen2Par',
                          'PctWorkMomYoungKids',
                          'PctWorkMom',
                          'NumKidsBornNeverMar',
                          'PctKidsBornNeverMar',
                          'NumImmig',
                          'PctImmigRecent',
                          'PctImmigRec5',
                          'PctImmigRec8',
                          'PctImmigRec10',
                          'PctRecentImmig',
                          'PctRecImmig5',
                          'PctRecImmig8',
                          'PctRecImmig10',
                          'PctSpeakEnglOnly',
                          'PctNotSpeakEnglWell',
                          'PctLargHouseFam',
                          'PctLargHouseOccup',
                          'PersPerOccupHous',
                          'PersPerOwnOccHous',
                          'PersPerRentOccHous',
                          'PctPersOwnOccup',
                          'PctPersDenseHous',
                          'PctHousLess3BR',
                          'MedNumBR',
                          'HousVacant',
                          'PctHousOccup',
                          'PctHousOwnOcc',
                          'PctVacantBoarded',
                          'PctVacMore6Mos',
                          'MedYrHousBuilt',
                          'PctHousNoPhone',
                          'PctWOFullPlumb',
                          'OwnOccLowQuart',
                          'OwnOccMedVal',
                          'OwnOccHiQuart',
                          'OwnOccQrange',
                          'RentLowQ',
                          'RentMedian',
                          'RentHighQ',
                          'RentQrange',
                          'MedRent',
                          'MedRentPctHousInc',
                          'MedOwnCostPctInc',
                          'MedOwnCostPctIncNoMtg',
                          'NumInShelters',
                          'NumStreet',
                          'PctForeignBorn',
                          'PctBornSameState',
                          'PctSameHouse85',
                          'PctSameCity85',
                          'PctSameState85',
                          'LemasSwornFT',
                          'LemasSwFTPerPop',
                          'LemasSwFTFieldOps',
                          'LemasSwFTFieldPerPop',
                          'LemasTotalReq',
                          'LemasTotReqPerPop',
                          'PolicReqPerOffic',
                          'PolicPerPop',
                          'RacialMatchCommPol',
                          'PctPolicWhite',
                          'PctPolicBlack',
                          'PctPolicHisp',
                          'PctPolicAsian',
                          'PctPolicMinor',
                          'OfficAssgnDrugUnits',
                          'NumKindsDrugsSeiz',
                          'PolicAveOTWorked',
                          'LandArea',
                          'PopDens',
                          'PctUsePubTrans',
                          'PolicCars',
                          'PolicOperBudg',
                          'LemasPctPolicOnPatr',
                          'LemasGangUnitDeploy',
                          'LemasPctOfficDrugUn',
                          'PolicBudgPerPop',
                          'murders',
                          'murdPerPop',
                          'rapes',
                          'rapesPerPop',
                          'robberies',
                          'robbbPerPop',
                          'assaults',
                          'assaultPerPop',
                          'burglaries',
                          'burglPerPop',
                          'larcenies',
                          'larcPerPop',
                          'autoTheft',
                          'autoTheftPerPop',
                          'arsons',
                          'arsonsPerPop',
                          'ViolentCrimesPerPop',
                          'nonViolPerPop'])


crime.head()

count_null = crime.isnull().sum(axis = 0)
qgrid.show_grid(count_null)


crime.columns[crime.isnull().mean() < 0.5]
crime_filtered = crime[crime.columns[crime.isnull().mean() < 0.5]]
count_null = crime_filtered.isnull().sum(axis = 0)
qgrid.show_grid(count_null)


crime_filtered=crime_filtered.dropna(subset = ['larcPerPop'])



X=crime_filtered.loc[:, 'population':'PolicBudgPerPop']
imputer = KNNImputer(n_neighbors=10,weights='distance')
X = imputer.fit_transform(X)

Y = crime_filtered.loc[:, 'larcPerPop']



## Split into a Training and Test Set

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=.3,random_state=42)

train_x.shape
test_x.shape
train_y.shape
test_y.shape





param_dist = {"learning_rate": uniform(0, 1),
              "gamma": uniform(0, 5),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10)}



bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (1, 50)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (1, 300)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 10)}]



xgb = XGBRegressor()

baseline = cross_val_score(xgb, train_x, train_y, scoring='r2', cv=10).mean()


rs = RandomizedSearchCV(xgb, param_distributions=param_dist,
                        scoring='r2', n_iter=100, n_jobs=-1,verbose=2, cv=10 )


rs.fit(train_x, train_y)


ncore=10
batch=10

optimizer = BayesianOptimization(f=cv_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 evaluator_type='local_penalization',
                                 batch_size = batch,
                                 num_cores=ncore,
                                 verbosity=True)


# Only 20 iterations because we have 5 initial random points
optimizer.run_optimization(max_iter=10)


y_rs = np.maximum.accumulate(rs.cv_results_['mean_test_score'])
y_bo = np.maximum.accumulate(-optimizer.Y).ravel()

print(f'Baseline neg. MSE = {baseline:.2f}')
print(f'Random search neg. MSE = {y_rs[-1]:.2f}')
print(f'Bayesian optimization neg. MSE = {y_bo[-1]:.2f}')

plt.plot(y_rs, 'ro-', label='Random search')
plt.plot(y_bo, 'bo-', label='Bayesian optimization')
plt.xlabel('Iteration')
plt.ylabel('Neg. MSE')
plt.ylim(-5000, -3000)
plt.title('Value of the best sampled CV score')
plt.legend()
