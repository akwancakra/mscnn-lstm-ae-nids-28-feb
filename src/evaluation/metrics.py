"""Evaluation metrics: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC,
per-attack detection rate, FPR.

Evaluates on both CIC-IDS-2017 and CSE-CIC-IDS-2018.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_roc_auc_safe(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, bool]:
    """Compute ROC-AUC with auto-inversion if AUC < 0.5 (score convention reversed).

    Returns (auc, was_inverted). If was_inverted is True, caller should use -scores
    for all subsequent metrics so that high score = attack consistently.
    """
    try:
        auc_val = roc_auc_score(y_true, scores)
    except ValueError:
        return 0.0, False
    inverted = False
    if auc_val < 0.5:
        logger.warning(
            "ROC-AUC=%.4f < 0.5 — score convention reversed (high=benign?). Inverting scores.",
            auc_val,
        )
        auc_val = roc_auc_score(y_true, -scores)
        inverted = True
        logger.info("After inversion: ROC-AUC=%.4f", auc_val)
    return float(auc_val), inverted


def binary_labels(labels: np.ndarray | pd.Series, benign_label: str = "BENIGN") -> np.ndarray:
    """Convert string labels to binary: 0=benign, 1=attack."""
    if isinstance(labels, pd.Series):
        labels = labels.values
    labels_str = np.array([str(l).strip().upper() for l in labels])
    return (labels_str != benign_label.upper()).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    dataset_name: str = "",
) -> dict:
    """Compute full set of evaluation metrics.

    Args:
        y_true: binary labels (0=benign, 1=attack)
        scores: anomaly scores (higher = more anomalous)
        threshold: decision threshold
        dataset_name: for logging

    Returns:
        dict with all metrics
    """
    y_pred = (scores > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = 0.0
        logger.warning("ROC-AUC computation failed (single class?)")

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = 0.0

    results = {
        "dataset": dataset_name,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_samples": int(len(y_true)),
        "n_attacks": int(y_true.sum()),
        "n_benign": int((y_true == 0).sum()),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(
        "[%s] ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, "
        "Precision=%.4f, Recall=%.4f, FPR=%.4f",
        dataset_name, roc_auc, pr_auc, f1,
        precision, recall, fpr,
    )
    return results


def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compute ROC curve points."""
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)
    return {
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc(fpr_arr, tpr_arr)),
    }


def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Compute Precision-Recall curve points."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, scores)
    return {
        "precision": precision_arr.tolist(),
        "recall": recall_arr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc(recall_arr, precision_arr)),
    }


def per_attack_detection_rate(
    labels_str: np.ndarray | pd.Series,
    scores: np.ndarray,
    threshold: float,
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """Compute detection rate per attack category.

    Returns DataFrame with columns: attack_type, n_samples, n_detected, detection_rate.
    """
    if isinstance(labels_str, pd.Series):
        labels_str = labels_str.values
    labels_str = np.array([str(l).strip() for l in labels_str])
    y_pred = (scores > threshold).astype(int)

    results = []
    unique_labels = np.unique(labels_str)
    for label in unique_labels:
        mask = labels_str == label
        n = mask.sum()
        n_detected = y_pred[mask].sum()
        dr = n_detected / n if n > 0 else 0.0
        results.append({
            "attack_type": label,
            "n_samples": int(n),
            "n_detected": int(n_detected),
            "detection_rate": float(dr),
        })

    df = pd.DataFrame(results).sort_values("detection_rate", ascending=False)
    return df.reset_index(drop=True)


def build_threshold_comparison_table(
    threshold_results: dict,
    y_true: np.ndarray,
    scores: np.ndarray,
    dataset_name: str = "",
) -> pd.DataFrame:
    """Evaluate all threshold strategies on a labeled dataset."""
    rows = []
    all_t = threshold_results.get("all_thresholds", {})
    for name, info in all_t.items():
        t = info["threshold"]
        y_pred = (scores > t).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "strategy": name,
            "threshold": t,
            "fpr_benign_val": info["fpr_on_benign_val"],
            f"precision_{dataset_name}": prec,
            f"recall_{dataset_name}": rec,
            f"f1_{dataset_name}": f1_val,
            f"fpr_{dataset_name}": fpr_val,
        })

    return pd.DataFrame(rows)
