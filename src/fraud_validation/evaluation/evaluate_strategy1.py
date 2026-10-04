"""Chronological holdout evaluation for Strategy 1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_validation.features.engineering import (
    build_behavioral_features,
    build_identity_features,
    build_transaction_features,
    merge_identity_data,
)
from fraud_validation.evaluation.strategy_metrics import calculate_fraud_metrics
from fraud_validation.strategies import RuleBasedFraudStrategy

TRAIN_FRACTION = 0.7
TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
]
TRANSACTION_DTYPES = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "isFraud": "int8",
    "TransactionAmt": "float32",
    "ProductCD": "category",
    "card1": "category",
    "card4": "category",
    "card6": "category",
    "P_emaildomain": "category",
    "R_emaildomain": "category",
}
RULE_INPUT_COLUMNS = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
]


def _chronologically_ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        raise ValueError("data must contain 'TransactionDT'")
    transaction_time = pd.to_numeric(df["TransactionDT"], errors="coerce")
    time_values = transaction_time.to_numpy(dtype=float)
    if not np.isfinite(time_values).all():
        raise ValueError("TransactionDT must contain only finite numeric values")
    order = np.argsort(time_values, kind="stable")
    return df.iloc[order].reset_index(drop=True)


def _split_ordered(
    ordered: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(ordered) < 2:
        raise ValueError("at least two transactions are required for a split")

    target_train_rows = int(len(ordered) * train_fraction)
    if target_train_rows == 0:
        raise ValueError("train_fraction leaves no training transactions")
    times = pd.to_numeric(ordered["TransactionDT"], errors="raise").to_numpy()
    boundary_time = times[target_train_rows - 1]
    split_at = int(np.searchsorted(times, boundary_time, side="right"))
    if split_at >= len(ordered):
        raise ValueError(
            "chronological split has no validation rows after keeping tied "
            "TransactionDT values together"
        )
    return ordered.iloc[:split_at], ordered.iloc[split_at:]


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the earliest training rows and later validation rows.

    Transactions with an equal TransactionDT are kept in the same period so
    the validation period always starts strictly after the training period.
    """
    ordered = _chronologically_ordered(df)
    return _split_ordered(ordered, train_fraction)


def build_strategy1_features(
    transactions: pd.DataFrame,
    identity_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build Strategy 1 inputs without exposing the target label to features."""
    if "isFraud" in transactions.columns:
        raise ValueError("feature input must not contain the 'isFraud' label")

    merged = merge_identity_data(transactions, identity_data)
    if len(merged) != len(transactions):
        raise ValueError(
            "identity merge changed the transaction row count; "
            "check for duplicate TransactionID values in identity data"
        )

    transaction_features = build_transaction_features(merged)
    identity_features = build_identity_features(merged)
    behavioral_features = build_behavioral_features(merged)
    return pd.concat(
        [
            merged.loc[:, RULE_INPUT_COLUMNS],
            transaction_features,
            identity_features,
            behavioral_features,
        ],
        axis=1,
    )


def _time_value(value: object) -> int | float:
    numeric_value = float(value)
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def evaluate_strategy1(
    transaction_data: pd.DataFrame,
    identity_data: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Fit Strategy 1 on the earliest 70% and evaluate on the latest 30%."""
    if "isFraud" not in transaction_data.columns:
        raise ValueError("transaction_data must contain 'isFraud'")

    ordered = _chronologically_ordered(transaction_data)
    labels = pd.to_numeric(ordered.pop("isFraud"), errors="raise").reset_index(
        drop=True
    )
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("isFraud labels must contain only 0 or 1")
    training_rows, validation_rows = _split_ordered(ordered, TRAIN_FRACTION)
    train_count = len(training_rows)

    # Compute history once over the entire sorted timeline. The behavioral
    # builder only exposes strictly earlier timestamps, so validation rows
    # cannot alter history for preceding validation rows.
    feature_inputs = build_strategy1_features(ordered, identity_data)
    strategy_training_data = pd.DataFrame(
        {
            "P_emaildomain": feature_inputs.loc[: train_count - 1, "P_emaildomain"],
            "isFraud": labels.iloc[:train_count],
        }
    )
    strategy = RuleBasedFraudStrategy().fit(strategy_training_data)

    validation_features = feature_inputs.iloc[train_count:]
    scored = strategy.score_dataframe(validation_features)
    actual = labels.iloc[train_count:].reset_index(drop=True)
    blocked = scored["risk_level"].eq("HIGH").astype("int8").reset_index(drop=True)
    metrics = calculate_fraud_metrics(actual, blocked, scored["risk_score"])

    train_times = training_rows["TransactionDT"]
    validation_times = validation_rows["TransactionDT"]
    report = {
        "strategy": "Strategy 1: explainable rule-based scoring",
        "split_method": (
            "chronological by TransactionDT; earliest 70% train and latest 30% "
            "validation, keeping equal timestamps together; no random split"
        ),
        "train_rows": int(train_count),
        "validation_rows": int(len(validation_rows)),
        "train_time_start": _time_value(train_times.iloc[0]),
        "train_time_end": _time_value(train_times.iloc[-1]),
        "validation_time_start": _time_value(validation_times.iloc[0]),
        "validation_time_end": _time_value(validation_times.iloc[-1]),
        **metrics,
    }
    confusion_matrix = pd.DataFrame(
        metrics["confusion_matrix"]["matrix"],
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    )
    return report, confusion_matrix


def run() -> dict:
    """Load the standard IEEE-CIS files, evaluate, and save report artifacts."""
    data_dir = Path("data")
    transaction_data = pd.read_csv(
        data_dir / "train_transaction.csv",
        usecols=TRANSACTION_COLUMNS,
        dtype=TRANSACTION_DTYPES,
    )
    identity_data = pd.read_csv(data_dir / "train_identity.csv")
    report, confusion_matrix = evaluate_strategy1(
        transaction_data, identity_data
    )

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "strategy1_validation_metrics.json"
    confusion_path = report_dir / "strategy1_validation_confusion_matrix.csv"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    confusion_matrix.to_csv(confusion_path, index_label="actual")
    return report


if __name__ == "__main__":
    evaluation_report = run()
    print(json.dumps(evaluation_report, indent=2))
