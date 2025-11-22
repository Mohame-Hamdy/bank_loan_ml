# EVALUATE.PY
# Regression metrics + clean visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model_name, model, X_test, y_test):

    # -------------------
    # Predictions
    # -------------------
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # -------------------
    # Metrics
    # -------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n===== {model_name.upper()} RESULTS =====")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # -------------------
    # Plot 1 — Predicted vs Actual
    # -------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--')    # perfect prediction line
    plt.xlabel("Actual (True Values)")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.grid(True)
    plt.show()

    # -------------------
    # Plot 2 — Residuals vs Predicted
    # -------------------
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Error)")
    plt.title(f"{model_name} - Residuals vs Predicted")
    plt.grid(True)
    plt.show()

    # -------------------
    # Plot 3 — Residual Distribution
    # -------------------
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.title(f"{model_name} - Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
