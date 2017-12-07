import itertools
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
seaborn.set(style='ticks')

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from const import N_CLASS, CLASS_NAME
from grid_search import multiclass_roc_auc_score


class performance(object):
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob

    def get_scores(self, average="weighted"):
        auc_score = multiclass_roc_auc_score(self.y_true, self.y_pred)
        accu_score = accuracy_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred, average=average)
        precision = precision_score(self.y_true, self.y_pred, average=average)
        f1 = f1_score(self.y_true, self.y_pred, average=average)

        return scores(auc_score, accu_score, recall, precision, f1)

    def roc_auc_curve(self, title):

        plt.figure()
        y_true = label_binarize(self.y_true, classes=[0, 1, 2, 3, 4, 5])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(N_CLASS):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], self.y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(["aqua", "darkorange", "cornflowerblue", "purple", "pink", "gray"])
        for i, color in zip(range(N_CLASS), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label="ROC curve of class {0} (area = {1:0.2f})"
                           "".format(CLASS_NAME[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig("../results/" + title + ".png", bbox_inches="tight")

    def confusion_matrix(self, cm, title, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)
        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(CLASS_NAME))
        plt.xticks(tick_marks, CLASS_NAME, rotation=90)
        plt.yticks(tick_marks, CLASS_NAME)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig("../results/" + title + ".png", bbox_inches="tight")

    def feature_analysis(self):
        pass


class scores(object):
    def __init__(self, auc_score, accu_score, recall, precision, f1):
        self.auc_score = auc_score
        self.accu_score = accu_score
        self.recall = recall
        self.precision = precision
        self.f1_score = f1
