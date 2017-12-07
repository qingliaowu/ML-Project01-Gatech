#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:00:55 2017

@author: cc
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
patients_features = pd.read_csv("../data/patients_3.csv")


target_col = 'DISCHARGE_LOCATION'
data_cols = list(patients_features)
cat_cols = ['ICU_day', 'GENDER', 'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY', 'ICD9_CODE', 'FIRST_CAREUNIT', 'LAST_CAREUNIT']
num_cols = [e for e in data_cols if e not in cat_cols]
id_col = ['SUBJECT_ID','HADM_ID_x']

data_params = dict()
data_params['cat_cols'] = cat_cols
data_params['num_cols'] = num_cols
data_params['num_cols'].remove('SUBJECT_ID')
data_params['num_cols'].remove('HADM_ID')
data_params['num_cols'].remove('DISCHARGE_LOCATION')
data_params['num_cols'].remove('ADMITTIME')
data_params['num_cols'].remove('LOS')
data_params['tgt_col'] = target_col
data_params['ID_col'] = id_col

patients_features = patients_features.fillna(0)
patients_features[cat_cols] = patients_features[cat_cols].astype(str)

# transform the categorical features
for col in data_params['cat_cols']:
    print ("Label encoding  %s" % (col))
    LBL = LabelEncoder()
    LBL.fit(patients_features[col])
    patients_features[col]=LBL.transform(patients_features[col])
    
# transform the predicting target
LBL = LabelEncoder()
LBL.fit(patients_features['DISCHARGE_LOCATION'])
tgt_cls = dict(zip(patients_features['DISCHARGE_LOCATION'].unique()
               , LBL.transform(patients_features['DISCHARGE_LOCATION'].unique())))

patients_features['DISCHARGE_LOCATION']=LBL.transform(patients_features['DISCHARGE_LOCATION'])
    
full_cols = data_params['cat_cols'] + data_params['num_cols']
X = patients_features[full_cols].values
y = patients_features[data_params['tgt_col']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

np.savez("../data/train.npz", X_train=X_train, y_train=y_train)
np.savez("../data/test.npz", X_test=X_test, y_test=y_test)

