import numpy as np
import pandas as pd
import pytest

from fraud_validation.evaluation.evaluate_strategy1 import (
    build_strategy1_features,
    chronological_split,
    evaluate_strategy1,
)
from fraud_validation.evaluation.strategy_metrics import calculate_fraud_metrics


def _transactions(times):
    size = len(times)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, size + 1),
            "TransactionDT": times,
            "TransactionAmt": np.arange(size, dtype=float) + 40,
            "ProductCD": ["C", "W"] * (size // 2) + (["C"] if size % 2 else []),
            "card1": ["same-card"] * size,
            "card4": ["visa"] * size,
            "card6": ["debit"] * size,
            "P_emaildomain": ["example.com"] * size,
            "R_emaildomain": [None] * size,
            "isFraud": [0, 1] * (size // 2) + ([0] if size % 2 else []),
        }
    )


def test_chronological_split_uses_earliest_seventy_percent_and_later_rows():
    transactions = _transactions(range(10)).sample(frac=1, random_state=8)

    training, validation = chronological_split(transactions)

    assert len(training) == 7
    assert len(validation) == 3
    assert training["TransactionDT"].tolist() == list(range(7))
    assert validation["TransactionDT"].tolist() == list(range(7, 10))
    assert training["TransactionDT"].max() < validation["TransactionDT"].min()


def test_validation_metrics_do_not_depend_on_row_order_after_split():
    transactions = _transactions(range(10))
    transactions["prediction"] = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    _, validation = chronological_split(transactions)

    original = calculate_fraud_metrics(
        validation["isFraud"], validation["prediction"]
    )
    reordered = validation.sample(frac=1, random_state=13)
    shuffled = calculate_fraud_metrics(
        reordered["isFraud"], reordered["prediction"]
    )

    assert original == shuffled


def test_chronological_split_keeps_equal_boundary_times_together():
    transactions = _transactions([0, 1, 2, 3, 4, 5, 5, 5, 6, 7])

    training, validation = chronological_split(transactions)

    assert training["TransactionDT"].max() == 5
    assert validation["TransactionDT"].min() == 6


def test_split_rejects_invalid_or_unseparable_input():
    with pytest.raises(ValueError, match="TransactionDT"):
        chronological_split(pd.DataFrame({"isFraud": [0, 1]}))
    with pytest.raises(ValueError, match="no validation rows"):
        chronological_split(_transactions([1, 1, 1, 1]))


def test_feature_pipeline_excludes_label_and_is_label_independent():
    transaction_data = _transactions(range(4))
    labels = transaction_data.pop("isFraud")
    identities = pd.DataFrame(
        {"TransactionID": [1, 3], "DeviceType": ["mobile", "desktop"]}
    )

    features = build_strategy1_features(transaction_data, identities)
    changed_label_copy = transaction_data.copy()
    changed_labels = 1 - labels

    assert "isFraud" not in features.columns
    assert changed_labels.tolist() != labels.tolist()
    pd.testing.assert_frame_equal(
        features, build_strategy1_features(changed_label_copy, identities)
    )
    with pytest.raises(ValueError, match="must not contain"):
        build_strategy1_features(
            transaction_data.assign(isFraud=labels), identities
        )


def test_behavioral_validation_features_use_training_history_only():
    transaction_data = _transactions(range(10))
    identities = pd.DataFrame({"TransactionID": [], "DeviceType": []})
    features = build_strategy1_features(
        transaction_data.drop(columns=["isFraud"]), identities
    )
    training, validation = chronological_split(transaction_data)

    assert features.loc[len(training), "prior_transaction_count"] == len(training)
    # Each later validation row sees prior rows, but no future rows.
    assert features.loc[len(training) + 1, "prior_transaction_count"] == len(training) + 1
    assert validation["TransactionDT"].min() > training["TransactionDT"].max()


def test_evaluation_reports_chronological_periods_and_confusion_matrix():
    transactions = _transactions(range(10))
    transactions["isFraud"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    identities = pd.DataFrame({"TransactionID": [], "DeviceType": []})

    report, confusion = evaluate_strategy1(transactions, identities)

    assert report["train_rows"] == 7
    assert report["validation_rows"] == 3
    assert report["train_time_end"] < report["validation_time_start"]
    assert confusion.to_numpy().sum() == 3
    assert {
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "fraud_detection_rate",
        "legitimate_transactions_blocked_rate",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    }.issubset(report)
