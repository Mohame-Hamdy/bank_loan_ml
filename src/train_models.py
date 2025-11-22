# implementation of linear regression and knn as regressors
import numpy as np
from .config import MODELS

# ----------------------------------------
# Linear Regression using Normal Equation
# y = Xw
# w = (XᵀX)⁻¹ Xᵀ y
# ----------------------------------------

class CustomLinearRegression:

    def fit(self, X, y):
        # Add bias term (column of 1s)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Normal equation: w = (XᵀX)^(-1) Xᵀ y
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return X_bias @ self.weights


# ----------------------------------------
# KNN Regressor
# For each sample:
#   Find k nearest neighbors (euclidean)
#   Average their target values
# ----------------------------------------

class CustomKNNRegressor:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # KNN just stores the dataset
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        preds = []

        for x in X:
            # Compute euclidean distance to all points
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of k nearest neighbors
            k_idx = np.argsort(distances)[:self.k]

            # Average their y values
            pred = np.mean(self.y_train[k_idx])
            preds.append(pred)

        return np.array(preds)


# ----------------------------------------
# Wrapper function to train all models
# ----------------------------------------

def train_all_models(X_train, y_train):

    trained = {}

    if "linear" in MODELS:
        lr = CustomLinearRegression()
        lr.fit(X_train, y_train)
        trained["linear"] = lr

    if "knn" in MODELS:
        knn = CustomKNNRegressor(k=5)
        knn.fit(X_train, y_train)
        trained["knn"] = knn

    return trained
