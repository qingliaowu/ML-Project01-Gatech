#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:37:58 2017

@author: cc
"""

import pandas as pd
import pickle


def feature_convert(seq_num, admission_type, insurance):
    X = pd.DataFrame({"ADMISSION_TYPE": [admission_type],
                      "INSURANCE": [insurance],
                      "SEQ_NUM": [seq_num]})
    
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
    
    return X
    

def new_prediction(feature):
    clf = pickle.load(open('trained_LR_model.sav', 'rb'))    
    result = clf.predict_proba(feature)
    
    return result
    


input_seq_num = 1.0
input_admission_type = 'URGENT'
input_insurance = 'Self Pay'

output_feature = feature_convert(input_seq_num, input_admission_type, input_insurance)
output_prediction = new_prediction(output_feature)

NOT_HOME_proba = output_prediction[0,0]
HOME_proba = output_prediction[0,1]