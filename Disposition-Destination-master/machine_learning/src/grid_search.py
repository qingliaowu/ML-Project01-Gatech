from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, make_scorer
import logging

from const import RANDOM_STATE

def search(clf, clf_name, X, Y, params):
    '''
    find optimized parameters for a single model based on accuracy
    :param clf:
    :param clf_name:
    :param X:
    :param Y:
    :param params:
    :return: best parameters
    '''
    # set cross validation rule
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    # set grid search
    model = GridSearchCV(clf, params, cv=cv, n_jobs=1, scoring=score_metric())
    model.fit(X, Y)
    logging.info("Best AUC score of {} model: {}".format(clf_name, model.best_score_))
    return model.best_params_, model.best_score_

def score_metric():
    return make_scorer(multiclass_roc_auc_score, greater_is_better=True)

def multiclass_roc_auc_score(truth, pred, average="weighted"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)
