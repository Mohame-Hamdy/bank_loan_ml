import numpy as np
from .config import K_FOR_KNN
import numpy as np
from collections import Counter



# ----------------------------------------
# Linear Regression using Normal Equation
# y = Xw
# w = (XᵀX)⁻¹ Xᵀ y
# ----------------------------------------

class LinearRegression:
    """
    Linear regression using the normal equation with pseudo-inverse for stability.
    """
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        Xb = np.c_[np.ones((X.shape[0], 1)), X]  # shape (n, d+1)
        XtX = Xb.T @ Xb
        self.coef_ = np.linalg.pinv(XtX) @ Xb.T @ y

    def predict(self, X):
        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return Xb @ self.coef_


#  KNN Regressor
class KNNRegressor:
    """
    KNN regressor: for each test point, find k nearest neighbors by Euclidean distance
    and return the mean of their y values.
    """
    def __init__(self, k=K_FOR_KNN):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distances(self, x):
        return np.linalg.norm(self.X_train - x, axis=1)

    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            d = self._distances(x)
            idx = np.argsort(d)[:self.k]
            preds.append(np.mean(self.y_train[idx]))
        return np.array(preds)
