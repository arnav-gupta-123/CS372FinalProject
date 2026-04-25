import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, accuracy_score

#General LDA Classifier recommended for P300 Studies

class LDAClassifierNCB:
    def __init__(self, solver='svd', shrinkage=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        self.trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def update(self, X_new, y_new):
        X_all = np.vstack((self.X_train, X_new))
        y_all = np.hstack((self.y_train, y_new.flatten()))
        self.fit(X_all, y_all)

    def predict_scores(self, X, y):
        scores = self.model.decision_function(X)
        auc = roc_auc_score(y, scores)
        acc = accuracy_score(y, self.model.predict(X))
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        return scores