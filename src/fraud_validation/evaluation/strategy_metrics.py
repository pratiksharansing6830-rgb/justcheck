"""Binary classification metrics for fraud evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_fraud_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: pd.Series | None = None,
) -> dict:
    """Calculate fraud detection metrics with class 1 representing blocked fraud."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of rows")
    if scores is not None and len(scores) != len(y_true):
        raise ValueError("scores must have the same number of rows as y_true")

    actual = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="raise")
    predicted = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors="raise")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("y_true and y_pred must not contain missing values")
    if not actual.isin([0, 1]).all() or not predicted.isin([0, 1]).all():
        raise ValueError("y_true and y_pred values must be binary (0 or 1)")

    actual_values = actual.to_numpy(dtype=np.int8)
    predicted_values = predicted.to_numpy(dtype=np.int8)
    true_positives = int(((actual_values == 1) & (predicted_values == 1)).sum())
    true_negatives = int(((actual_values == 0) & (predicted_values == 0)).sum())
    false_positives = int(((actual_values == 0) & (predicted_values == 1)).sum())
    false_negatives = int(((actual_values == 1) & (predicted_values == 0)).sum())

    def safe_divide(numerator: int | float, denominator: int | float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = safe_divide(
        true_positives, true_positives + false_positives
    )
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    metrics = {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": safe_divide(
            false_positives, false_positives + true_negatives
        ),
        "false_negative_rate": safe_divide(
            false_negatives, false_negatives + true_positives
        ),
        "fraud_detection_rate": recall,
        "legitimate_transactions_blocked_rate": safe_divide(
            false_positives, false_positives + true_negatives
        ),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [
                [true_negatives, false_positives],
                [false_negatives, true_positives],
            ],
        },
    }

    if scores is not None:
        numeric_scores = pd.to_numeric(
            pd.Series(scores).reset_index(drop=True), errors="coerce"
        )
        valid_scores = numeric_scores.dropna()
        metrics["score_count"] = int(len(valid_scores))
        metrics["mean_score"] = (
            float(valid_scores.mean()) if len(valid_scores) else None
        )

    return metrics
