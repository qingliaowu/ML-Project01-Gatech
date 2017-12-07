from sklearn.linear_model import LogisticRegression
from grid_search import search

from const import RANDOM_STATE

class logistricRegression(object):
    def __init__(self, params, X, y, model_name = "Logistic Regression"):
        self.params = params
        self.X = X
        self.y = y
        self.model_name = model_name

    def get_default_model(self):
        self.model = LogisticRegression(random_state=RANDOM_STATE)

    def optimization(self):
        self.get_default_model()
        self.best_params, self.best_scores = search(self.model, self.get_model_name(), self.X, self.y, self.params)

    def get_model_name(self):
        return self.model_name

    def get_best_model(self):
        self.optimization()
        return LogisticRegression(random_state=RANDOM_STATE, **self.best_params)

    def get_import_features(self):
        best_model = self.get_best_model().fit(self.X, self.y)
        return best_model.coef_