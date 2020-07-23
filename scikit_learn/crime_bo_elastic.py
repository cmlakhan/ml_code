
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

from sklearn.linear_model import ElasticNet

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
from sklearn.utils.fixes import loguniform


import sklearn

from sklearn.impute import KNNImputer

#%matplotlib inline

def cv_score(parameters):
    parameters = parameters[0]
    score = cross_val_score(
                ElasticNet(alpha=parameters[0],
                              l1_ratio=parameters[1],
                           normalize=True,
                           max_iter=10000),
                train_x, train_y, scoring='r2', cv = 10,
        n_jobs=-1).mean()
    score = np.array(score)
    return score




def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


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

count_null = crime.isnull().sum(axis = 0)
crime.columns[crime.isnull().mean() < 0.5]
crime_filtered = crime[crime.columns[crime.isnull().mean() < 0.5]]
count_null = crime_filtered.isnull().sum(axis = 0)
crime_filtered=crime_filtered.dropna(subset = ['larcPerPop'])



X=crime_filtered.loc[:, 'population':'PolicBudgPerPop']
imputer = KNNImputer(n_neighbors=10,weights='distance')
X = imputer.fit_transform(X)

Y = crime_filtered.loc[:, 'larcPerPop']

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=.3,random_state=42)



bds = [{'name': 'alpha', 'type': 'continuous', 'domain': (1e-10, 10)},
        {'name': 'l1_ratio', 'type': 'continuous', 'domain': (0, 1)}]

param_dist = {"alpha": loguniform(1e-10, 10e0),
              "l1_ratio": uniform(0, 1)}


enet = ElasticNet(max_iter=10000,normalize=True)



baseline = cross_val_score(enet, train_x, train_y, scoring='r2', cv=10).mean()


rs = RandomizedSearchCV(enet,
                        param_distributions=param_dist,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=2,
                        cv=10,
                        n_iter=100)

rs.fit(train_x, train_y)



optimizer = BayesianOptimization(f=cv_score,
                                 domain=bds,
                                 model_type='GP',
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.05,
                                 exact_feval=True,
                                 maximize=True,
                                 verbosity=True)


optimizer.run_optimization(max_iter=100)






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


report(rs.cv_results_)
