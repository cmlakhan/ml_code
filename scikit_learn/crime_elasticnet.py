
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import qgrid

#%matplotlib inline


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


X=crime_filtered.loc[:, 'population':'PolicBudgPerPop']
Y=crime_filtered.loc[:, 'larcPerPop']

X.head()

Y.head()