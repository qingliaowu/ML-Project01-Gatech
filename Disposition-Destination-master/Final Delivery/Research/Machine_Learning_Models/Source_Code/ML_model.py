#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:34:03 2017

@author: cc
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, confusion_matrix

from gradient_boost import gradientBoost
from grid_search import search, multiclass_roc_auc_score
from logistic_regression import logistricRegression
from random_forest import randomForest
from svc import svcClf
from metric import performance


logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s",
                        filename="ML_model.log",
                        filemode="w")

# load training and test sets
trn_data_path = "../data/train.npz"
tst_data_path = "../data/test.npz"

train = np.load(trn_data_path)
X_train = train["X_train"]
y_train = train["y_train"]
test = np.load(tst_data_path)
X_test = test["X_test"]
y_test = test["y_test"]


########################################
#Logistic regression
print("Running Logistic Regression Classifier...")

params_lr = {"C": np.logspace(-5, 0, 10), "penalty": ['l1', 'l2'], "tol": [1e-3, 1e-4, 1e-5], "class_weight":[None, "balanced"]}
lr = logistricRegression(params_lr, X_train, y_train)
best_lr = lr.get_best_model()
logging.info("Best parameters for logistic regression: {}\n".format(lr.best_params))
best_lr.fit(X_train, y_train)
y_train_pred_lr = best_lr.predict(X_train)
y_test_pred_lr = best_lr.predict(X_test)

AUC_train_lr = multiclass_roc_auc_score(y_train, y_train_pred_lr)
AUC_test_lr = multiclass_roc_auc_score(y_test, y_test_pred_lr)

print("AUC for training set is: " + str(AUC_train_lr))
print("AUC for test set is: " + str(AUC_test_lr))

logging.info("AUC of {} on training data: {}".format("logistic regression", AUC_train_lr))
logging.info("AUC of {} on test data: {}".format("logistic regression", AUC_test_lr))

#c_features = len(feature_names)
#plt.barh(range(c_features), best_lr.coef_[0])
#plt.xlabel("Feature Coefficient")
#plt.ylabel("Feature Name")
#plt.yticks(np.arange(c_features), feature_names)
#plt.title("Logistic Regression Classifier: Feature Importance")
#plt.show()

y_test_pred_prob_lr = best_lr.predict_proba(X_test)
perform_lr = performance(y_test, y_test_pred_lr, y_test_pred_prob_lr)
scores_lr = perform_lr.get_scores()
logging.info("Accuracy of {} on test data: {}".format("logistic regression", scores_lr.accu_score))
logging.info("Recall of {} on test data: {}".format("logistic regression", scores_lr.recall))
logging.info("Precision of {} on test data: {}".format("logistic regression", scores_lr.precision))
logging.info("f1 score of {} on test data: {}".format("logistic regression", scores_lr.f1_score))

## plot feature importance
#perform.feature_analysis(feature_names, best_lr.coef_[0], "Logistic Regression Classifier: Feature Importance")

# plot ROC curve for test set
perform_lr.roc_auc_curve(title="Logistic Regression Classifier (ROC)")

# plot confusion matrix
cm_lr = confusion_matrix(y_test, y_test_pred_lr)
perform_lr.confusion_matrix(cm_lr, title = "Logistic Regression Classifier (Confusion Matrix)")


#########################################
#Random Forest
print("Running Random Forest Classifier...")

params_rf = {"n_estimators": [10, 30, 50, 70, 90, 100, 120, 160, 200, 240], 
             "min_samples_split": [2, 4, 6, 8], 
             "min_samples_leaf": [1, 2, 3, 4]}
rf = randomForest(params_rf, X_train, y_train)
best_rf = rf.get_best_model()
logging.info("Best parameters for random forest: {}\n".format(rf.best_params))
best_rf.fit(X_train, y_train)
y_train_pred_rf = best_rf.predict(X_train)
y_test_pred_rf = best_rf.predict(X_test)

AUC_train_rf = multiclass_roc_auc_score(y_train, y_train_pred_rf)
AUC_test_rf = multiclass_roc_auc_score(y_test, y_test_pred_rf)

print("AUC for training set is: " + str(AUC_train_rf))
print("AUC for test set is: " + str(AUC_test_rf))

logging.info("AUC of {} on training data: {}".format("random forest", AUC_train_rf))
logging.info("AUC of {} on test data: {}".format("random forest", AUC_test_rf))


#c_features = len(feature_names)
#plt.barh(range(c_features), best_rf.feature_importances_)
#plt.xlabel("Feature Importance")
#plt.ylabel("Feature Name")
#plt.yticks(np.arange(c_features), feature_names)
#plt.title("Random Forest Classifier: Feature Importance")
#plt.show()

y_test_pred_prob_rf = best_rf.predict_proba(X_test)
perform_rf = performance(y_test, y_test_pred_rf, y_test_pred_prob_rf)

scores_rf = perform_rf.get_scores()
logging.info("Accuracy of {} on test data: {}".format("random forest", scores_rf.accu_score))
logging.info("Recall of {} on test data: {}".format("random forest", scores_rf.recall))
logging.info("Precision of {} on test data: {}".format("random forest", scores_rf.precision))
logging.info("f1 score of {} on test data: {}".format("random forest", scores_rf.f1_score))


## plot feature importance
#perform.feature_analysis(feature_names, best_rf.feature_importances_, "Random Forest Classifier: Feature Importance")

# plot ROC curve for test set

perform_rf.roc_auc_curve(title="Random Forest Classifier (ROC)")

# plot confusion matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
perform_rf.confusion_matrix(cm_rf, title = "Random Forest Classifier (Confusion Matrix)")


##########################################
##Support Vector Classifier
#print("Running Support Vector Classifier...")
#
#params_svc = {"C": np.logspace(-5, 0, 10), 
#             "kernel": ["linear", "poly", "rbf", "sigmoid"], 
#             "degree": [2, 3, 4],
#             "coef0": [0.0, 0.1, 0.2]}
#
#svc = svcClf(params_svc, X_train, y_train)
#best_svc = svc.get_best_model()
#logging.info("Best parameters for svc: {}\n".format(svc.best_params))
#best_svc.fit(X_train, y_train)
#y_train_pred_svc = best_svc.predict(X_train)
#y_test_pred_svc = best_svc.predict(X_test)
#
#AUC_train_svc = multiclass_roc_auc_score(y_train, y_train_pred_svc)
#AUC_test_svc = multiclass_roc_auc_score(y_test, y_test_pred_svc)
#
#print("AUC for training set is: " + str(AUC_train_svc))
#print("AUC for test set is: " + str(AUC_test_svc))
#
#logging.info("AUC of {} on training data: {}".format("support vector classifier", AUC_train_svc))
#logging.info("AUC of {} on test data: {}".format("support vector classifier", AUC_test_svc))
#
### plot feature importance
##c_features = len(feature_names)
##plt.barh(range(c_features), best_svc.coef_[0])
##plt.xlabel("Feature Coefficient")
##plt.ylabel("Feature Name")
##plt.yticks(np.arange(c_features), feature_names)
##plt.title("Support Vector Classifier: Feature Importance")
##plt.show()
#
#y_test_pred_prob_svc = best_svc.predict_proba(X_test)
#perform_svc = performance(y_test, y_test_pred_svc, y_test_pred_prob_svc)
#
#scores_svc = perform_svc.get_scores()
#logging.info("Accuracy of {} on test data: {}".format("svc", scores_svc.accu_score))
#logging.info("Recall of {} on test data: {}".format("svc", scores_svc.recall))
#logging.info("Precision of {} on test data: {}".format("svc", scores_svc.precision))
#logging.info("f1 score of {} on test data: {}".format("svc", scores_svc.f1_score))
#
#
### plot feature importance
##perform.feature_analysis(feature_names, best_svc.coef_[0], "Support Vector Classifier: Feature Importance")
#
## plot ROC curve for test set
#
#perform_svc.roc_auc_curve(title="Support Vector Classifier (ROC)")
#
## plot confusion matrix
#cm_svc = confusion_matrix(y_test, y_test_pred_svc)
#perform_svc.confusion_matrix(cm_svc, title = "Support Vector Classifier (Confusion Matrix)")


##########################################
#Gradient Boosting
print("Running Gradient Boosting Classifier...")

params_gbc = {"n_estimators": [100, 120, 160, 200, 240], 
             "learning_rate": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
             "max_depth": [1, 2, 3, 4, 5]}

gbc = gradientBoost(params_gbc, X_train, y_train)
best_gbc = gbc.get_best_model()
logging.info("Best parameters for gradient boosting: {}\n".format(gbc.best_params))
best_gbc.fit(X_train, y_train)
y_train_pred_gbc = best_gbc.predict(X_train)
y_test_pred_gbc = best_gbc.predict(X_test)

AUC_train_gbc = multiclass_roc_auc_score(y_train, y_train_pred_gbc)
AUC_test_gbc = multiclass_roc_auc_score(y_test, y_test_pred_gbc)

print("AUC for training set is: " + str(AUC_train_gbc))
print("AUC for test set is: " + str(AUC_test_gbc))

logging.info("AUC of {} on training data: {}".format("gradient boosting classifier", AUC_train_gbc))
logging.info("AUC of {} on test data: {}".format("gradient boosting classifier", AUC_test_gbc))

## plot feature importance
#c_features = len(feature_names)
#plt.barh(range(c_features), best_ada.feature_importances_)
#plt.xlabel("Feature Importance")
#plt.ylabel("Feature Name")
#plt.yticks(np.arange(c_features), feature_names)
#plt.title("Ada Boost Classifier: Feature Importance")
#plt.show()

y_test_pred_prob_gbc = best_gbc.predict_proba(X_test)
perform_gbc = performance(y_test, y_test_pred_gbc, y_test_pred_prob_gbc)

scores_gbc = perform_gbc.get_scores()
logging.info("Accuracy of {} on test data: {}".format("gradient boosting", scores_gbc.accu_score))
logging.info("Recall of {} on test data: {}".format("gradient boosting", scores_gbc.recall))
logging.info("Precision of {} on test data: {}".format("gradient boosting", scores_gbc.precision))
logging.info("f1 score of {} on test data: {}".format("gradient boosting", scores_gbc.f1_score))


## plot feature importance
#perform.feature_analysis(feature_names, best_ada.feature_importances_, "Ada Boost Classifier: Feature Importance")

# plot ROC curve for test set

perform_gbc.roc_auc_curve(title="Gradient Boosting Classifier (ROC)")

# plot confusion matrix
cm_gbc = confusion_matrix(y_test, y_test_pred_gbc)
perform_gbc.confusion_matrix(cm_gbc, title = "Gradient Boosting Classifier (Confusion Matrix)")

def multiclass_roc_auc_score(truth, pred, average="weighted"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return make_scorer(roc_auc_score(truth, pred, average=average), greater_is_better=True)

