
# coding: utf-8

# ## PREDICTING DISPOSITION DESTINATION IN HOSPITALIZED STROKE PATIENTS
# 
# ### Introduction
# 
# Patients hospitalized with acute stroke frequently require referrals to rehabilitation, some of which require few social work referrals and arrangements, whereas some require many that can significantly increase hospital length of stay. From the hospital’s and stroke neurologist’s perspective, there is therefore a need to identify which patients need which types of rehabilitation early on in their hospital course so as to remove bottlenecks to patient throughput.
# 
# ### Objective
# In the first 24 hours of a stroke patient’s admission (including time spent in the emergency department), accurately predict the likelihood of discharge to specific disposition destinations.

# ### Data cleaning and exploration

# In[181]:

import pandas as pd
# load medical charge summaries as DataFrame with pandas
admission_data = pd.read_csv('ADMISSIONS.csv')
# check the loaded data
print 'There are %d rows and %d columns in the note events table.'%(admission_data.shape[0], admission_data.shape[1])
admission_data.head()


# #### Remove rows with missing hospital addimission ID (HADM_ID) if there is any

# In[3]:

for col in admission_data.columns:
    print 'The number of NULL values in column %s is %d'%(col, sum(pd.isnull(admission_data[col])))


# In[9]:

# change data type of HADM_ID to be integer
admission_data['DIAGNOSIS'] = admission_data['DIAGNOSIS'].astype('str')


# #### Identify stroke patients with text in DIAGNOSIS, not complete, changed to ICD-9 code later

# In[16]:

admission_data['is_stroke'] = admission_data.apply(lambda row: 1 if row['DIAGNOSIS'].find('STROKE')>=0 else 0, axis=1)
admission_data.head()


# In[17]:

admission_data['is_stroke'].value_counts()


# #### Identify stroke patients with ICD-9 code

# In[7]:

# load ICD9 diagnosis code table into pandas dataframe
file_name = 'DIAGNOSES_ICD.csv'
codes_data = pd.read_csv(file_name)
# examine the diagnosis table
print 'There are %d rows %d and columns in the diagnosis code table.'        %(codes_data.shape[0], codes_data.shape[1])
codes_data.head()


#  In mentor feedback, they mentioned "... define “stroke” as “ischemic stroke” only, which is the most common type of storke", starting with "434" and 1480 data points found

# In[24]:


codes_data['ICD9_CODE'] = codes_data['ICD9_CODE'].astype('str')
codes_data['is_stroke'] = codes_data.apply(lambda row: 1 if row['ICD9_CODE'][:3]=='434' else 0, axis=1)


# In[25]:

codes_data['is_stroke'].value_counts()


# #### remove data of non-stroke patients

# In[45]:

stroke_codes_data = codes_data[codes_data['is_stroke'] != 0]
stroke_codes_data.head()


# #### removed unused columns joined two tables

# In[46]:


# drop unused columns
stroke_codes_data.drop(['is_stroke', 'ICD9_CODE'], axis = 1, inplace = True)
# merge diagnosis code dictionary table to note events/diagnosis code table
stroke_data = pd.merge(admission_data, stroke_codes_data, on = ['SUBJECT_ID', 'HADM_ID'], how = 'inner')

# examine the joined table
stroke_data.head()


# ### Feature Engineering
# First selected 3 columns for a simple model

# In[166]:

X= stroke_data[['ADMISSION_TYPE', 'INSURANCE', 'SEQ_NUM']]
X.head()


# In[167]:

stroke_data['SEQ_NUM'].mean()


# In[168]:

stroke_data['ADMISSION_TYPE'].value_counts()


# #### use one-hot encoding on categorical data

# In[169]:

X['EMERGENCY'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'EMERGENCY' else 0, axis = 1)
X['ELECTIVE'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'ELECTIVE' else 0, axis = 1)
X['URGENT'] = X.apply(lambda row: 1 if row['ADMISSION_TYPE'] == 'URGENT' else 0, axis = 1)
X.head()
X.drop(['ADMISSION_TYPE'],axis=1, inplace=True)


# #### the same trick for coding insurance type

# In[170]:

stroke_data['INSURANCE'].value_counts()


# In[171]:

X['Medicare'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Medicare' else 0, axis = 1)
X['Private'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Private' else 0, axis = 1)
X['Medicaid'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Medicaid' else 0, axis = 1)
X['Government'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Government' else 0, axis = 1)
X['SelfPay'] = X.apply(lambda row: 1 if row['INSURANCE'] == 'Self Pay' else 0, axis = 1)
X.head()
X.drop(['INSURANCE'],axis=1, inplace=True)


# #### The output to be predicted has many categories and a long trailing tail. Bin them into five groups

# In[172]:

stroke_data['DISCHARGE_LOCATION'].value_counts()


# In[173]:

stroke_data['DISCHARGE_LOCATION'] = stroke_data['DISCHARGE_LOCATION'].astype('str')
stroke_data['DEAD'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('DEAD') >=0 else 0, axis = 1)
stroke_data['SNF'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('SNF') >=0 else 0, axis = 1)
stroke_data['HOME'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('HOME') >=0 else 0, axis = 1)
stroke_data['REHAB'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('REHAB') >=0 else 0, axis = 1)
stroke_data['HOSPITAL'] = stroke_data.apply(lambda row: 1 if row['DISCHARGE_LOCATION'].find('HOSPITAL') >=0 else 0, axis = 1)

y = stroke_data[['DEAD', 'SNF', 'HOME', 'REHAB', 'HOSPITAL']]


# ### At first simplified to binary classification: HOME or not HOME
# 
# split data into testing and training data and train with logistic regression

# In[174]:

y = stroke_data['HOME']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# In[175]:

# fit regularized logistic regression
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C = 1, class_weight='balanced')
clf.fit(X_train, y_train)

score_train_clf = clf.score(X_train, y_train)
score_test_clf = clf.score(X_test, y_test)
print('Logistic Regression')
print('Training Accuracy: %.3f' % clf.score(X_train, y_train))
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


# In[176]:

from sklearn import metrics
expected = y_test
predicted = clf.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# #### plotting the significance of features

# In[177]:

feature_names = list(X)
print feature_names
print clf.coef_
import numpy as np
c_features = len(feature_names)
plt.barh(range(c_features), clf.coef_[0] )
plt.xlabel("Feature coef")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), feature_names)
plt.show()


# Medicare is a federal program that provides health coverage if you are 65 or older or have a severe disability, no matter your income.
# Medicaid is a state and federal program that provides health coverage if you have a very low income.

# ### Training with random forest

# In[178]:

print("Random Forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100, max_depth = 5, class_weight ='balanced') 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( X_train, y_train)

score_train_forest = forest.score(X_train, y_train)
score_test_forest = forest.score(X_test, y_test)


print('Training Accuracy: %.3f' % forest.score(X_train, y_train))
print('Test Accuracy: %.3f' % forest.score(X_test, y_test))


# In[179]:

feature_names = list(X)
print feature_names
print forest.feature_importances_


# In[180]:


c_features = len(feature_names)
plt.barh(range(c_features), forest.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), feature_names)
plt.show()


# ## NEXT STEP
# 1. Generate additional features
# 2. binary --> multi-class classification

# In[ ]:



