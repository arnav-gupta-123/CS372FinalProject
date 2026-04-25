import numpy as np

class BaselineClassifier:
    # always predicts nontarget as a baseline for comparison because of the extreme class imbalance. 

    def predict_scores(self, X, y):
        scores = np.zeros(X.shape[0])
        acc = np.sum(y == 0) / len(y)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: 0.5000")
        return scores