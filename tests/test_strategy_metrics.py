import pandas as pd
import pytest

from fraud_validation.evaluation.strategy_metrics import calculate_fraud_metrics


@pytest.fixture
def known_predictions():
    return pd.Series([1, 1, 0, 0, 0]), pd.Series([1, 0, 1, 0, 0])


def test_precision(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(actual, predicted)["precision"] == 0.5


def test_recall(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(actual, predicted)["recall"] == 0.5


def test_f1(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(actual, predicted)["f1"] == 0.5


def test_false_positive_rate(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(actual, predicted)["false_positive_rate"] == pytest.approx(1 / 3)


def test_false_negative_rate(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(actual, predicted)["false_negative_rate"] == 0.5


def test_fraud_detection_rate_matches_recall(known_predictions):
    actual, predicted = known_predictions
    metrics = calculate_fraud_metrics(actual, predicted)
    assert metrics["fraud_detection_rate"] == 0.5
    assert metrics["fraud_detection_rate"] == metrics["recall"]


def test_legitimate_transactions_blocked_rate(known_predictions):
    actual, predicted = known_predictions
    assert calculate_fraud_metrics(
        actual, predicted
    )["legitimate_transactions_blocked_rate"] == pytest.approx(1 / 3)


def test_zero_denominators_are_safe():
    all_legitimate = calculate_fraud_metrics(
        pd.Series([0, 0]), pd.Series([0, 0])
    )
    all_fraud = calculate_fraud_metrics(
        pd.Series([1, 1]), pd.Series([0, 0])
    )
    assert all_legitimate["precision"] == 0
    assert all_legitimate["recall"] == 0
    assert all_legitimate["f1"] == 0
    assert all_legitimate["false_negative_rate"] == 0
    assert all_fraud["false_positive_rate"] == 0
    assert all_fraud["legitimate_transactions_blocked_rate"] == 0


def test_confusion_matrix_is_json_serializable(known_predictions):
    import json

    actual, predicted = known_predictions
    matrix = calculate_fraud_metrics(actual, predicted)["confusion_matrix"]
    assert matrix == {"labels": [0, 1], "matrix": [[2, 1], [1, 1]]}
    json.dumps(matrix)


def test_metrics_are_independent_of_row_order_after_split():
    actual = pd.Series([0, 1, 0, 1, 1])
    predicted = pd.Series([1, 1, 0, 0, 1])
    permutation = [4, 2, 0, 3, 1]
    original = calculate_fraud_metrics(actual, predicted)
    reordered = calculate_fraud_metrics(
        actual.iloc[permutation], predicted.iloc[permutation]
    )
    assert original == reordered
