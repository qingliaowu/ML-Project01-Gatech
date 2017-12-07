#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:04:54 2017

@author: cc
"""


import pandas as pd
import numpy as np
import pickle


def feature_convert(gender, admission_location, language, religion, marital_status, ethnicity, is_dead, age, drug_code, admit_time, lab_test_code, lab_test_time, lab_test_result):
    X = pd.DataFrame({"GENDER": [gender],
                      "ADMISSION_LOCATION": [admission_location],
                      "LANGUAGE": [language],
                      "RELIGION": [religion],
                      "MARITAL_STATUS": [marital_status],
                      "ETHNICITY": [ethnicity],
                      "IS_DEAD": [is_dead],
                      "AGE": [age],
                      "DRG_CODE": [drug_code],
                      })
    
    LBL_gender = pickle.load(open('trained_LBL_GENDER.sav', 'rb'))
    X["GENDER"] = LBL_gender.transform([gender])
    LBL_admission_location = pickle.load(open('trained_LBL_ADMISSION_LOCATION.sav', 'rb'))
    X["ADMISSION_LOCATION"] = LBL_admission_location.transform([admission_location])
    LBL_language = pickle.load(open('trained_LBL_LANGUAGE.sav', 'rb'))
    X["LANGUAGE"] = LBL_language.transform([language])
    LBL_religion = pickle.load(open('trained_LBL_RELIGION.sav', 'rb'))
    X["RELIGION"] = LBL_religion.transform([religion])
    LBL_marital_status = pickle.load(open('trained_LBL_MARITAL_STATUS.sav', 'rb'))
    X["MARITAL_STATUS"] = LBL_marital_status.transform([marital_status])
    LBL_ethnicity = pickle.load(open('trained_LBL_ETHNICITY.sav', 'rb'))
    X["ETHNICITY"] = LBL_ethnicity.transform([ethnicity])
    
    X["IS_DEAD"] = X['IS_DEAD'].apply(lambda x: 1 if x == "YES" else 0)
    
    lab_feature = pd.DataFrame({"LOINC_CODE": lab_test_code,
                                "CHARTTIME": lab_test_time,
                                "VALUE": lab_test_result})
    lab_feature['ADMITTIME'] = admit_time
    
    code = pd.read_csv('code.csv')
    
    lab_feature = pd.merge(lab_feature, code, on = 'LOINC_CODE', how = 'inner')
    
    lab_feature["ADMITTIME"] = pd.to_datetime(lab_feature["ADMITTIME"])
    lab_feature["CHARTTIME"] = pd.to_datetime(lab_feature["CHARTTIME"])
    
    # add a new column "TimeDiff"
    lab_feature["TimeDiff"] = abs((lab_feature["ADMITTIME"] - lab_feature["CHARTTIME"]).astype('timedelta64[h]'))
    
    #filter the lab events within 24 hours of admission
    filtered_lab_feature = lab_feature[lab_feature["TimeDiff"] <= 24]
    
    # create a new column "LAB_HOUR"uni
    filtered_lab_feature["LAB_HOUR"] = filtered_lab_feature["ITEMID"].astype("str") + "_" + filtered_lab_feature["TimeDiff"].astype("str")
    
    lab_feature_map = pd.read_csv('lab_feature_map.csv')
    
    filtered_lab_feature = pd.merge(filtered_lab_feature, lab_feature_map, on = 'LAB_HOUR', how = 'inner')
    filtered_lab_feature = filtered_lab_feature.groupby(["feature_idx"])["VALUE"].agg("mean").reset_index()
    
    lab_feature_df = pd.DataFrame(np.zeros((1, 461)))
    
    for i in range(filtered_lab_feature.shape[0]):
        lab_feature_df.iloc[0, int(filtered_lab_feature.iloc[i, 0] - 1)] = filtered_lab_feature.iloc[i, 1]
    
    X = pd.concat([X, lab_feature_df], axis=1)

    return X
    

def new_prediction(feature):
    best_model = pickle.load(open('trained_best_model.sav', 'rb'))    
    result = best_model.predict_proba(feature)
    
    return result
    


input_gender = 'F'
input_admission_location = 'EMERGENCY ROOM ADMIT'
input_language = 'RUSS'
input_religion = 'PROTESTANT QUAKER'
input_marital_status = 'SINGLE'
input_ethnicity = 'HISPANIC OR LATINO'
input_is_dead = 'YES'
input_age = 73
input_drug_code = 8
input_admit_time = '2010-11-06 10:13:00'
input_lab_test_code = ['26454-9', '742-7']
input_lab_test_time = ['2010-11-06 12:47:00', '2010-11-06 14:34:00']
input_lab_test_result = [72.0, 6.7]


output_feature = feature_convert(input_gender, input_admission_location, input_language, input_religion, input_marital_status, input_ethnicity, 
                                 input_is_dead, input_age, input_drug_code, input_admit_time, input_lab_test_code, input_lab_test_time, input_lab_test_result)

output_prediction = new_prediction(output_feature)

DEAD_EXPIRED_proba = output_prediction[0,0] # DEAD/EXPIRED
HOME_proba = output_prediction[0,1] # HOME
HOME_HEALTH_CARE_proba = output_prediction[0,2] # HOME HEALTH CARE
LTCH_proba = output_prediction[0,3] # LONG TERM CARE HOSPITAL
REHAB_DPH_proba = output_prediction[0,4] # REHAB/DISTINCT PART HOSP
SNF_proba = output_prediction[0,5] # SNF

print "DEAD/EXPIRED: " + str(DEAD_EXPIRED_proba)
print "HOME: " + str(HOME_proba)
print "HOME HEALTH CARE: " + str(HOME_HEALTH_CARE_proba)
print "LONG TERM CARE HOSPITAL: " + str(LTCH_proba)
print "REHAB/DISTINCT PART HOSP: " + str(REHAB_DPH_proba)
print "SNF: " + str(SNF_proba)
