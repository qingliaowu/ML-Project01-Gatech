#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:42:15 2017

@author: cc
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

admission_data = pd.read_csv('ADMISSIONS.csv')

admission_data['DIAGNOSIS'] = admission_data['DIAGNOSIS'].astype('str')

#### Identify stroke patients with text in DIAGNOSIS, not complete, changed to ICD-9 code later
admission_data['is_stroke'] = admission_data.apply(lambda row: 1 if row['DIAGNOSIS'].find('STROKE')>=0 else 0, axis=1)


# load ICD9 diagnosis code table into pandas dataframe
file_name = 'DIAGNOSES_ICD.csv'
codes_data = pd.read_csv(file_name)

codes_data['ICD9_CODE'] = codes_data['ICD9_CODE'].astype('str')
codes_data['is_stroke'] = codes_data.apply(lambda row: 1 if row['ICD9_CODE'][:3]=='434' else 0, axis=1)

stroke_codes_data = codes_data[codes_data['is_stroke'] != 0]

# #### removed unused columns joined two tables


# drop unused columns
stroke_codes_data.drop(['is_stroke', 'ICD9_CODE'], axis = 1, inplace = True)
# merge diagnosis code dictionary table to note events/diagnosis code table
stroke_data = pd.merge(admission_data, stroke_codes_data, on = ['SUBJECT_ID', 'HADM_ID'], how = 'inner')

# ### Feature Engineering
# First selected 3 columns for a simple model

X= stroke_data[['ADMISSION_TYPE', 'INSURANCE', 'SEQ_NUM']]

# #### use one-hot encoding on categorical data

X['EMERGENCY'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'EMERGENCY' else 0, axis = 1)
X['ELECTIVE'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'ELECTIVE' else 0, axis = 1)
X['URGENT'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'URGENT' else 0, axis = 1)
X.drop(['ADMISSION_TYPE'],axis=1, inplace=True)



X['Medicare'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Medicare' else 0, axis = 1)
X['Private'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Private' else 0, axis = 1)
X['Medicaid'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Medicaid' else 0, axis = 1)
X['Government'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Government' else 0, axis = 1)
X['SelfPay'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Self Pay' else 0, axis = 1)
X.drop(['INSURANCE'],axis=1, inplace=True)

stroke_data['DISCHARGE_LOCATION'] = stroke_data['DISCHARGE_LOCATION'].astype('str')
stroke_data['DEAD'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('DEAD') >=0 else 0, axis = 1)
stroke_data['SNF'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('SNF') >=0 else 0, axis = 1)
stroke_data['HOME'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('HOME') >=0 else 0, axis = 1)
stroke_data['REHAB'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('REHAB') >=0 else 0, axis = 1)
stroke_data['HOSPITAL'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('HOSPITAL') >=0 else 0, axis = 1)

y = stroke_data[['DEAD', 'SNF', 'HOME', 'REHAB', 'HOSPITAL']]


y = stroke_data['HOME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# fit regularized logistic regression
clf=LogisticRegression(C = 1, class_weight='balanced')
clf.fit(X_train, y_train)

pickle.dump(clf, open("trained_LR_model.sav", 'wb'))