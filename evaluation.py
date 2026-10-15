import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true, probabilities):
    """Return ROC-AUC, or NaN if it cannot be calculated."""
    try:
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return np.nan


def evaluate_models(models, X_test, y_test):
    rows = []
    probabilities = {}
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # all 7 models support it

        probabilities[name] = y_prob
        predictions[name] = y_pred

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": safe_auc(y_test, y_prob),
            "Average Precision": average_precision_score(y_test, y_prob),
            "Brier Score": brier_score_loss(y_test, y_prob),
        })

    metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    return metrics, probabilities, predictions
