"""
utils/metrics.py
-----------------
Metric computation helpers for anomaly detection evaluation.
"""

import numpy as np


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC).

    Uses scikit-learn if available, falls back to a simple trapezoidal
    approximation otherwise.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except ImportError:
        # Manual ROC-AUC via trapezoidal rule
        order = np.argsort(-y_scores)
        y_sorted = y_true[order]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0
        tp, fp = 0, 0
        for label in y_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
        return float(auc)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
