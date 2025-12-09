import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from .config import THRESHOLD


# Classification evaluation (using regressors + threshold)
def evaluate_classification_from_regressor(model_name, model, X_test, y_test, threshold=THRESHOLD, show_plots=True):
    """
    model: regressor instance with predict(X) -> continuous score (0..1 or any)
    threshold: decision threshold to convert continuous score to class label
    """
    scores = model.predict(X_test)  # continuous outputs
    # ensure in numpy
    scores = np.array(scores)

    # predicted classes by threshold
    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    # AUC: requires scores; handle single-class case
    try:
        auc_score = roc_auc_score(y_test, scores)
    except Exception:
        auc_score = float("nan")

    # Print summary
    print(f"\n===== {model_name} (Classification via regressor + threshold={threshold}) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")
    print(f"AUC      : {auc_score if not np.isnan(auc_score) else 'N/A'}")
    print("Confusion Matrix:")
    print(cm)

    if show_plots:
        # Confusion matrix heatmap
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.colorbar()

        # Add numeric values + quadrant labels
        labels = [["TN", "FP"],
                ["FN", "TP"]]

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                base_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                text = f"{labels[i][j]}\n{cm[i, j]}"
                plt.text(j, i, text, ha="center", va="center", color=base_color, fontsize=10, fontweight="bold")

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0,1], ["0", "1"])
        plt.yticks([0,1], ["0", "1"])
        plt.tight_layout()
        plt.show()


        # ROC curve (if valid)
        try:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_sc = roc_auc_score(y_test, scores)
            plt.figure(figsize=(6,4))
            plt.plot(fpr, tpr, label=f"AUC={auc_sc:.3f}")
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.title(f"{model_name} - ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception:
            # ROC could fail if only one class present in y_test
            pass

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc_score, "confusion_matrix": cm
    }


# Regression evaluation
def evaluate_regression(model_name, model, X_test, y_test, show_plots=True):
    preds = model.predict(X_test)
    preds = np.array(preds)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"\n===== {model_name} (Regression) =====")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    if show_plots:
        # Predicted vs Actual
        plt.figure(figsize=(6,5))
        plt.scatter(y_test, preds, alpha=0.6)
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, color="red", label="Perfect Fit")

        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.grid(True)
        plt.show()


    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
