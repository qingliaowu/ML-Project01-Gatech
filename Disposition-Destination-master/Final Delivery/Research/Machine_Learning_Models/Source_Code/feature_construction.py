#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:26:20 2017

@author: cc
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file


# load medical charge summaries as DataFrame with pandas
admission_data = pd.read_csv('../data/ADMISSIONS.csv')

# load ICD9 diagnosis code table into pandas dataframe
file_name = '../data/DIAGNOSES_ICD.csv'
codes_data = pd.read_csv(file_name)

codes_data['ICD9_CODE'] = codes_data['ICD9_CODE'].astype('str')
codes_data['is_stroke'] = codes_data.apply(lambda row: 1 if row['ICD9_CODE'][:3]=='434' else 0, axis=1)

stroke_codes_data = codes_data[codes_data['is_stroke'] != 0]

# merge diagnosis code dictionary table to note events/diagnosis code table
stroke_data = pd.merge(admission_data, stroke_codes_data, on = ['SUBJECT_ID', 'HADM_ID'], how = 'inner')

# load PATIENTS.csv file
patient_information = pd.read_csv('../data/PATIENTS.csv')

# join stroke_data table with patient_information table
stroke_data = pd.merge(stroke_data, patient_information, on = ['SUBJECT_ID'], how = 'inner')

## convert ADMITTIME AND DOB columns to datetime type
#stroke_data["ADMITTIME"] = pd.to_datetime(stroke_data["ADMITTIME"])
#stroke_data["DOB"] = pd.to_datetime(stroke_data["DOB"])
#
## add a new column "age" by substracting DOB from ADMITTIME
#stroke_data["Age"] = stroke_data["ADMITTIME"].dt.year - stroke_data["DOB"].dt.year

#load ICUSTAYS table
icu = pd.read_csv("../data/ICUSTAYS.csv")

# join stroke_data with ICUSTAYS
patients = pd.merge(stroke_data, icu, on = ['SUBJECT_ID'], how = 'inner')

patients = patients.rename(columns={'ROW_ID_x':'ROW_ID', 'HADM_ID_x':'HADM_ID'})

columns = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DEATHTIME','ADMISSION_TYPE','ADMISSION_LOCATION','INSURANCE',
'LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY','GENDER', 'DOB', 'ICD9_CODE', 'FIRST_CAREUNIT',
           'LAST_CAREUNIT', 'LOS','DISCHARGE_LOCATION']
           
patient2 = patients[columns]

patient2['AGE'] = (pd.to_datetime(patient2['ADMITTIME'])-
                   pd.to_datetime(patient2['DOB'])).apply(lambda x: round(x.days/365.25))

patient2['ICU_day'] = patient2['ADMITTIME'].apply(lambda x: 'Day' if int(x[11:13])>=7 and int(x[11:13]) < 19 else 'Night' )

#### Let's see how many patients are recurrence
#ICU_count=patient2[['SUBJECT_ID', u'HADM_ID', 'ADMITTIME']].groupby(['SUBJECT_ID', u'HADM_ID']).count().reset_index()
#ICU_count= ICU_count.rename(columns={'ADMITTIME':'RECURRENCE'})
#
#patient2 = pd.merge(patient2, ICU_count, on=['SUBJECT_ID', 'HADM_ID'], how = 'inner')
#
## The first time a patient entering ICU should not considered as a re-ocuurence patient
#patient2 = patient2.sort_values(['SUBJECT_ID','HADM_ID','INTIME'])
#
#row = 0
#while row < patient2.shape[0]:
#    count = patient2.loc[row].RECURRENCE
#    patient2.ix[row,'RECURRENCE'] = 0
#    row += count
#    
#patient2.RECURRENCE = patient2['RECURRENCE'].apply(lambda x: 1 if x>1 else 0)

#check if a patient is dead within 24 hours after admission
patient2["IS_DEAD"] = abs((pd.to_datetime(patient2["DEATHTIME"]) - pd.to_datetime(patient2["ADMITTIME"])).astype('timedelta64[h]'))
patient2['IS_DEAD'] = patient2['IS_DEAD'].apply(lambda x: 1 if x <= 24 else 0)

columns = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 
           'ICU_day','ADMISSION_TYPE', 'ICD9_CODE', 'ADMISSION_LOCATION','FIRST_CAREUNIT', 'LAST_CAREUNIT','IS_DEAD',
           'AGE','GENDER','INSURANCE','LANGUAGE','RELIGION','MARITAL_STATUS','ETHNICITY',
           'LOS', 'DISCHARGE_LOCATION']
patient2 = patient2[columns]

patient2.index = range(2344)
outcome =  ['REHAB/DISTINCT PART HOSP','DEAD/EXPIRED', 'SNF','HOME','HOME HEALTH CARE', 'LONG TERM CARE HOSPITAL']
for row in range(patient2.shape[0]):
    if patient2.loc[row]['DISCHARGE_LOCATION'] not in outcome:
        patient2 = patient2.drop(row)

# load RGCODES table
drug = pd.read_csv("../data/DRGCODES.csv")
patientsID = patient2[['SUBJECT_ID','HADM_ID']]
patientsID.columns =['SUBJECT_ID','HADM_ID']
stroke_drug = pd.merge(patientsID[['SUBJECT_ID','HADM_ID']], drug, on = ['SUBJECT_ID','HADM_ID'], how = 'inner')

num_type=stroke_drug[['SUBJECT_ID','HADM_ID','DRG_TYPE']].groupby(['SUBJECT_ID','HADM_ID']).count().reset_index()
num_drug = stroke_drug[['SUBJECT_ID','HADM_ID','DRG_CODE']].groupby(['SUBJECT_ID','HADM_ID']).count().reset_index()
drugcode_type = pd.merge(num_type, num_drug,on = ['SUBJECT_ID','HADM_ID'], how = 'inner' )
drugcode_type.columns = ['SUBJECT_ID','HADM_ID','DRG_TYPE','DRG_CODE']
patients_2 = pd.merge(patient2, drugcode_type,on = ['SUBJECT_ID','HADM_ID'], how = 'left')



#read LABEVENTS and D_LABITEMS tables
lab_event = pd.read_csv("../data/LABEVENTS.csv")
lab_id = pd.read_csv("../data/D_LABITEMS.csv")

#join lab_event and lab_id
lab_event = pd.merge(lab_event, lab_id, on = "ITEMID", how = "inner").drop_duplicates()

#filter the lab_event table to include the stroke patients' lab events only
stroke_id = patients_2[["SUBJECT_ID", "ADMITTIME"]]
stroke_lab_event = pd.merge(lab_event, stroke_id, on = "SUBJECT_ID", how = "inner").drop_duplicates()

#filter the lab events that happen within 24 hours of admission
# convert ADMITTIME AND CHARTTIME columns to datetime type
stroke_lab_event["ADMITTIME"] = pd.to_datetime(stroke_lab_event["ADMITTIME"])
stroke_lab_event["CHARTTIME"] = pd.to_datetime(stroke_lab_event["CHARTTIME"])

# add a new column "TimeDiff"
stroke_lab_event["TimeDiff"] = abs((stroke_lab_event["ADMITTIME"] - stroke_lab_event["CHARTTIME"]).astype('timedelta64[h]'))
stroke_lab_event.head()

#filter the lab events within 24 hours of admission
filtered_stroke_lab_event = stroke_lab_event[stroke_lab_event["TimeDiff"] <= 24]

# filter out the rows that are not lab tests related to stroke
filtered_stroke_lab_event = filtered_stroke_lab_event.loc[filtered_stroke_lab_event["LOINC_CODE"].isin(['26454-9','26465-5','2160-0','2823-3','3094-0','2345-7','33914-3','5902-2','742-7','711-2','6768-6','8122-4','2039-6','1751-7','4544-3','2085-9','2951-2','2093-3','2500-7','2498-4','2075-0'])]

# filter out the rows with VALUENUM to be NULL
filtered_stroke_lab_event = filtered_stroke_lab_event[["SUBJECT_ID", "ITEMID", "VALUENUM", "TimeDiff"]]
filtered_stroke_lab_event = filtered_stroke_lab_event.dropna()

#Aggregate the filtered_stroke_lab_event table by SUBJECT_ID
# create a new column "LAB_HOUR"uni
filtered_stroke_lab_event["LAB_HOUR"] = filtered_stroke_lab_event["ITEMID"].astype("str") + "_" + filtered_stroke_lab_event["TimeDiff"].astype("str")

# map the created feature_id "LAB_HOUR" to index
lab_feature_map = pd.DataFrame(filtered_stroke_lab_event["LAB_HOUR"].unique())
lab_feature_map["feature_idx"] = lab_feature_map.index + 1
lab_feature_map.columns = ["LAB_HOUR", "feature_idx"]

#join filtered_stroke_lab_event and lab_feature_map
filtered_stroke_lab_event = pd.merge(filtered_stroke_lab_event, lab_feature_map, on = "LAB_HOUR", how = "inner")

#if multiple same lab events happen at the same time, take the mean value
filtered_stroke_lab_event = filtered_stroke_lab_event.groupby(["SUBJECT_ID", "feature_idx"])["VALUENUM"].agg("mean").reset_index()

# create a new column "LABID_VALUE" which is a tuple (feature_idx, VALUE)
filtered_stroke_lab_event["LABID_VALUE"] = filtered_stroke_lab_event[["feature_idx", "VALUENUM"]].apply(tuple, axis=1)

# create a dictionary of patient_lab_features
new_filtered_stroke_lab_event = filtered_stroke_lab_event[["SUBJECT_ID", "LABID_VALUE"]]
patient_lab_features = new_filtered_stroke_lab_event.groupby("SUBJECT_ID")["LABID_VALUE"].apply(lambda x: [x for x in x.values]).to_dict()

#convert patient_lab_features to svmlight format
lab_feature = open("../data/patient_lab_features_svmlight.data", 'w')
data = []

for key, value in sorted(patient_lab_features.items()):
    each_data = []
    each_data.append(str(int(key)))
    
    for (x1, x2) in sorted(value):
        each_data.append("%s:%.4f" % (int(x1), x2))
        
    each_data_str = ' '.join(each_data)
        
    data.append(each_data_str)
            
lab_feature.write("\n".join(data))
lab_feature.close()

#reload patient_lab_features_svmlight.data into sparse matrix
patient_lab_features_sparse = load_svmlight_file("../data/patient_lab_features_svmlight.data")

patient_lab_features_sparse_value = patient_lab_features_sparse[0]
patient_lab_features_sparse_id = patient_lab_features_sparse[1]

patient_lab_features_sparse_dataframe = pd.SparseDataFrame([ pd.SparseSeries(patient_lab_features_sparse_value[i].toarray().ravel()) 
                              for i in np.arange(patient_lab_features_sparse_value.shape[0]) ])


patient_lab_features_sparse_dataframe["SUBJECT_ID"] = patient_lab_features_sparse_id

patients_3 = pd.merge(patients_2, patient_lab_features_sparse_dataframe, on='SUBJECT_ID', how='left')

patients_3.to_csv("../data/patients_3.csv", index=False)


# read MICROBIOLOGYEVENTS.csv
bio_event = pd.read_csv("../data/MICROBIOLOGYEVENTS.csv")

# Filter the bio_event table to include the stroke patients' microbiology events only
stroke_bio_event = pd.merge(bio_event, stroke_id, on = "SUBJECT_ID", how = "inner").drop_duplicates()

#Filter the microbiology events that happen within 24 hours of admission
# convert ADMITTIME AND CHARTTIME columns to datetime type
stroke_bio_event["ADMITTIME"] = pd.to_datetime(stroke_bio_event["ADMITTIME"])
stroke_bio_event["CHARTTIME"] = pd.to_datetime(stroke_bio_event["CHARTTIME"])

# add a new column "TimeDiff"
stroke_bio_event["TimeDiff"] = abs((stroke_bio_event["ADMITTIME"] - stroke_bio_event["CHARTTIME"]).astype('timedelta64[h]'))

#filter the lab events within 24 hours of admission
filtered_stroke_bio_event = stroke_bio_event[stroke_bio_event["TimeDiff"] <= 24]

# filter out the rows that are not microbiology tests related to stroke
filtered_stroke_bio_event = filtered_stroke_bio_event.loc[filtered_stroke_bio_event["SPEC_ITEMID"].isin(['70062', '70012'])]

# create a new column "BIO_HOUR"
filtered_stroke_bio_event["BIO_HOUR"] = filtered_stroke_bio_event["SPEC_ITEMID"].astype("str") + "_" + filtered_stroke_bio_event["TimeDiff"].astype("str")

# map the created feature_id "BIO_HOUR" to index
bio_feature_map = pd.DataFrame(filtered_stroke_bio_event["BIO_HOUR"].unique())
bio_feature_map["feature_idx"] = bio_feature_map.index + 1
bio_feature_map.columns = ["BIO_HOUR", "feature_idx"]

#join filtered_stroke_bio_event and bio_feature_map
filtered_stroke_bio_event = pd.merge(filtered_stroke_bio_event, bio_feature_map, on = "BIO_HOUR", how = "inner")
filtered_stroke_bio_event["VALUE"] = 1

# create a new column "BIOID_VALUE" which is a tuple (feature_idx, VALUE)
filtered_stroke_bio_event["BIOID_VALUE"] = filtered_stroke_bio_event[["feature_idx", "VALUE"]].apply(tuple, axis=1)

# create a dictionary of patient_med_features
new_filtered_stroke_bio_event = filtered_stroke_bio_event[["SUBJECT_ID", "BIOID_VALUE"]].drop_duplicates()
patient_bio_features = new_filtered_stroke_bio_event.groupby("SUBJECT_ID")["BIOID_VALUE"].apply(lambda x: [x for x in x.values]).to_dict()

bio_feature = open("../data/patient_bio_features_svmlight.data", 'w')
data = []

for key, value in sorted(patient_bio_features.items()):
    each_data = []
    each_data.append(str(int(key)))
    
    for (x1, x2) in sorted(value):
        each_data.append("%s:%s" % (int(x1), int(x2)))
        
    each_data_str = ' '.join(each_data)
        
    data.append(each_data_str)
            
bio_feature.write("\n".join(data))
bio_feature.close()

patient_bio_features_sparse = load_svmlight_file("../data/patient_bio_features_svmlight.data")

patient_bio_features_sparse_value = patient_bio_features_sparse[0]
patient_bio_features_sparse_id = patient_bio_features_sparse[1]

patient_bio_features_sparse_dataframe = pd.SparseDataFrame([ pd.SparseSeries(patient_bio_features_sparse_value[i].toarray().ravel()) 
                              for i in np.arange(patient_bio_features_sparse_value.shape[0]) ])


patient_bio_features_sparse_dataframe["SUBJECT_ID"] = patient_bio_features_sparse_id

patients_4 = pd.merge(patients_3, patient_bio_features_sparse_dataframe, on='SUBJECT_ID', how='left')

patients_4.to_csv("../data/patients_4.csv", index=False)


#Read PRESCRIPTIONS.csv
med_event = pd.read_csv("../data/PRESCRIPTIONS.csv")
stroke_med_event = pd.merge(med_event, stroke_id, on = "SUBJECT_ID", how = "inner").drop_duplicates()

# convert ADMITTIME AND CHARTTIME columns to datetime type
stroke_med_event["ADMITTIME"] = pd.to_datetime(stroke_med_event["ADMITTIME"])
stroke_med_event["STARTDATE"] = pd.to_datetime(stroke_med_event["STARTDATE"])

# add a new column "TimeDiff"
stroke_med_event["TimeDiff"] = abs((stroke_med_event["ADMITTIME"] - stroke_med_event["STARTDATE"]).astype('timedelta64[h]'))

#filter the lab events within 24 hours of admission
filtered_stroke_med_event = stroke_med_event[stroke_med_event["TimeDiff"] <= 24]

# filter out the rows that are not drugs related to ischemic stroke
filtered_stroke_med_event = filtered_stroke_med_event.loc[filtered_stroke_med_event["DRUG"].isin(['Alteplase','Aspirin', 'Insulin', 'Heparin', 'Danaparoid', 'Warfarin', 'Dabigatran', 'Rivaroxaban', 'Apixaban', 'Clopidogrel'])]

# create a new column "MED_HOUR"
filtered_stroke_med_event["MED_HOUR"] = filtered_stroke_med_event["DRUG"].astype("str") + "_" + filtered_stroke_med_event["TimeDiff"].astype("str")

# map the created feature_id "MED_HOUR" to index
med_feature_map = pd.DataFrame(filtered_stroke_med_event["MED_HOUR"].unique())
med_feature_map["feature_idx"] = med_feature_map.index + 1
med_feature_map.columns = ["MED_HOUR", "feature_idx"]

#join filtered_stroke_med_event and med_feature_map
filtered_stroke_med_event = pd.merge(filtered_stroke_med_event, med_feature_map, on = "MED_HOUR", how = "inner")
filtered_stroke_med_event["VALUE"] = 1

# create a new column "BIOID_VALUE" which is a tuple (feature_idx, VALUE)
filtered_stroke_med_event["MEDID_VALUE"] = filtered_stroke_med_event[["feature_idx", "VALUE"]].apply(tuple, axis=1)

# create a dictionary of patient_med_features
new_filtered_stroke_med_event = filtered_stroke_med_event[["SUBJECT_ID", "MEDID_VALUE"]].drop_duplicates()
patient_med_features = new_filtered_stroke_med_event.groupby("SUBJECT_ID")["MEDID_VALUE"].apply(lambda x: [x for x in x.values]).to_dict()

med_feature = open("../data/patient_med_features_svmlight.data", 'w')
data = []

for key, value in sorted(patient_med_features.items()):
    each_data = []
    each_data.append(str(int(key)))
    
    for (x1, x2) in sorted(value):
        each_data.append("%s:%s" % (int(x1), int(x2)))
        
    each_data_str = ' '.join(each_data)
        
    data.append(each_data_str)
            
med_feature.write("\n".join(data))
med_feature.close()

patient_med_features_sparse = load_svmlight_file("../data/patient_med_features_svmlight.data")

patient_med_features_sparse_value = patient_med_features_sparse[0]
patient_med_features_sparse_id = patient_med_features_sparse[1]

patient_med_features_sparse_dataframe = pd.SparseDataFrame([ pd.SparseSeries(patient_med_features_sparse_value[i].toarray().ravel()) 
                              for i in np.arange(patient_med_features_sparse_value.shape[0]) ])


patient_med_features_sparse_dataframe["SUBJECT_ID"] = patient_med_features_sparse_id

patients_5 = pd.merge(patients_4, patient_med_features_sparse_dataframe, on='SUBJECT_ID', how='left')

patients_5.to_csv("../data/patients_5.csv", index=False)
