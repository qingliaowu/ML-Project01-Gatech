#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:55:32 2017

@author: cc
"""

import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, confusion_matrix

from random_forest import randomForest

def multiclass_roc_auc_score(truth, pred, average="weighted"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return make_scorer(roc_auc_score(truth, pred, average=average), greater_is_better=True)



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
data_params['num_cols'].remove('DRG_TYPE')

data_params['cat_cols'].remove('ICU_day')
data_params['cat_cols'].remove('ADMISSION_TYPE')
data_params['cat_cols'].remove('INSURANCE')
data_params['cat_cols'].remove('ICD9_CODE')
data_params['cat_cols'].remove('FIRST_CAREUNIT')
data_params['cat_cols'].remove('LAST_CAREUNIT')

data_params['tgt_col'] = target_col
data_params['ID_col'] = id_col

patients_features = patients_features.fillna(0)
patients_features[cat_cols] = patients_features[cat_cols].astype(str)

# transform the categorical features
for col in data_params['cat_cols']:
    print ("Label encoding  %s" % (col))
    LBL = LabelEncoder()
    LBL.fit(patients_features[col])
    pickle.dump(LBL, open("trained_LBL_" + col + ".sav", 'wb'))
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

#Random Forest
print("Running Random Forest Classifier...")



params_rf = {"n_estimators": [10, 30, 50, 70, 90, 100, 120, 160, 200, 240], 
             "min_samples_split": [2, 4, 6, 8], 
             "min_samples_leaf": [1, 2, 3, 4]}
rf = randomForest(params_rf, X_train, y_train)
best_rf = rf.get_best_model()

best_rf.fit(X_train, y_train)
pickle.dump(best_rf, open("trained_best_model.sav", 'wb'))

#y_train_pred_rf = best_rf.predict(X_train)
#y_test_pred_rf = best_rf.predict(X_test)
#
#AUC_train_rf = multiclass_roc_auc_score(y_train, y_train_pred_rf)
#AUC_test_rf = multiclass_roc_auc_score(y_test, y_test_pred_rf)
#
#print("AUC for training set is: " + str(AUC_train_rf))
#print("AUC for test set is: " + str(AUC_test_rf))



