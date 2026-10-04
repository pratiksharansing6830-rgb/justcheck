import numpy as np
import pandas as pd
import pytest

from fraud_validation.features.engineering import (
    build_behavioral_features,
    build_identity_features,
    build_transaction_features,
    merge_identity_data,
)


def test_build_transaction_features():
    transactions = pd.DataFrame(
        {
            "TransactionAmt": [0.0, -2.0, 35.95, 159.95, 10.0, 160.0],
            "TransactionDT": [0, 3600, 86400, 90000, 86399, 90000],
            "ProductCD": ["C", "S", "H", "R", "W", "C"],
            "card6": ["credit", "debit", None, "credit", "debit", "credit"],
            "card4": ["visa", "discover", "visa", None, "mastercard", "visa"],
            "P_emaildomain": ["a.com", None, " ", "b.com", "c.com", None],
            "R_emaildomain": [None, "b.com", " ", "d.com", None, "e.com"],
        }
    )

    features = build_transaction_features(transactions)

    assert features.columns.tolist() == [
        "transaction_amount",
        "transaction_time",
        "transaction_hour",
        "amount_log",
        "is_low_amount",
        "is_high_amount",
        "product_code",
        "card_type",
        "card_network",
        "has_p_email",
        "has_r_email",
    ]
    assert features["transaction_amount"].tolist() == transactions["TransactionAmt"].tolist()
    assert features["transaction_time"].tolist() == transactions["TransactionDT"].tolist()
    assert features["transaction_hour"].tolist() == [0, 1, 0, 1, 23, 1]
    np.testing.assert_allclose(
        features["amount_log"],
        [0, 0, np.log1p(35.95), np.log1p(159.95), np.log1p(10), np.log1p(160)],
    )
    assert features["is_low_amount"].tolist() == [1, 1, 0, 0, 1, 0]
    assert features["is_high_amount"].tolist() == [0, 0, 0, 0, 0, 1]
    assert features["product_code"].tolist() == transactions["ProductCD"].tolist()
    assert features["card_type"].tolist() == transactions["card6"].tolist()
    assert features["card_network"].tolist() == transactions["card4"].tolist()
    assert features["has_p_email"].tolist() == [1, 0, 0, 1, 1, 0]
    assert features["has_r_email"].tolist() == [0, 1, 0, 1, 0, 1]


def test_build_transaction_features_preserves_missing_amounts():
    transactions = pd.DataFrame(
        {
            "TransactionAmt": [np.nan],
            "TransactionDT": [np.nan],
            "ProductCD": [None],
            "card6": [None],
            "card4": [None],
            "P_emaildomain": [None],
            "R_emaildomain": [None],
        }
    )

    features = build_transaction_features(transactions)

    assert pd.isna(features.loc[0, "amount_log"])
    assert pd.isna(features.loc[0, "transaction_hour"])
    assert features.loc[0, "is_low_amount"] == 0
    assert features.loc[0, "is_high_amount"] == 0


def test_merge_identity_data_left_joins_without_mutating_inputs():
    transactions = pd.DataFrame(
        {
            "TransactionID": [3, 1, 2],
            "TransactionAmt": [30.0, 10.0, 20.0],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["mobile", "desktop"],
            "DeviceInfo": ["phone", "computer"],
        }
    )
    original_transactions = transactions.copy(deep=True)
    original_identity = identity.copy(deep=True)

    merged = merge_identity_data(transactions, identity)

    assert merged["TransactionID"].tolist() == [3, 1, 2]
    assert merged["DeviceType"].tolist()[:2] == ["desktop", "mobile"]
    assert merged["DeviceInfo"].tolist()[:2] == ["computer", "phone"]
    assert pd.isna(merged.loc[2, "DeviceType"])
    assert pd.isna(merged.loc[2, "DeviceInfo"])
    assert merged.columns.tolist().count("TransactionID") == 1
    pd.testing.assert_frame_equal(transactions, original_transactions)
    pd.testing.assert_frame_equal(identity, original_identity)


def test_build_identity_features():
    identity = pd.DataFrame(
        {
            "DeviceType": ["mobile", None, None, None],
            "DeviceInfo": [" phone ", None, "   ", "tablet"],
            "id_12": [1, None, None, None],
            "id_13": [None, 2, None, None],
            "id_38": ["a", "b", None, None],
            "isFraud": [1, 0, 0, 1],
        },
        index=[9, 4, 7, 2],
    )
    original = identity.copy(deep=True)

    features = build_identity_features(identity)

    assert features.index.tolist() == identity.index.tolist()
    assert len(features) == len(identity)
    assert features["has_identity"].tolist() == [1, 1, 0, 1]
    assert features.loc[9, "device_type"] == "mobile"
    assert pd.isna(features.loc[4, "device_type"])
    assert pd.isna(features.loc[7, "device_type"])
    assert features["has_device_info"].tolist() == [1, 0, 0, 1]
    assert features["identity_match_count"].tolist() == [2, 2, 0, 0]
    assert "isFraud" not in features.columns
    pd.testing.assert_frame_equal(identity, original)


def test_build_identity_features_handles_missing_identity_columns():
    identity = pd.DataFrame({"TransactionID": [1, 2]}, index=[5, 3])

    features = build_identity_features(identity)

    assert features.index.tolist() == [5, 3]
    assert len(features) == 2
    assert features["has_identity"].tolist() == [0, 0]
    assert features["has_device_info"].tolist() == [0, 0]
    assert features["identity_match_count"].tolist() == [0, 0]
    assert features["device_type"].isna().all()


def test_build_behavioral_features_are_leakage_safe_and_restore_input_order():
    transactions = pd.DataFrame(
        {
            "TransactionDT": [
                7201, 3600, 7200, 0, 3600, 7200, 7200, 86400, 90000,
                *([100] * 10), 86500, 86501,
            ],
            "TransactionAmt": [
                50, 20, 34, 10, 99, 30, 32, 60, 100,
                *([1] * 10), 5, 7,
            ],
            "card1": ["A"] * 9 + ["B"] * 12,
            "isFraud": [0, 1] * 10 + [0],
        },
        index=[50, 11, 12, 13, 14, 15, 16, 17, 18, *range(20, 30), 31, 32],
    )
    original = transactions.copy(deep=True)
    features = build_behavioral_features(transactions)

    assert features.index.tolist() == transactions.index.tolist()
    assert len(features) == len(transactions)
    # The earliest event has no history, despite appearing later in the input.
    first_position = 3
    assert features.iloc[first_position]["prior_transaction_count"] == 0
    assert features.iloc[first_position]["prior_amount_mean"] == 0
    assert features.iloc[first_position]["amount_vs_prior_mean"] == 0
    assert features.iloc[first_position]["previous_transaction_amount"] == 0
    # Equal-time rows see only transactions strictly before their timestamp.
    assert features.iloc[2]["prior_transaction_count"] == 3
    assert features.iloc[5]["prior_transaction_count"] == 3
    assert features.iloc[6]["prior_transaction_count"] == 3
    assert features.iloc[2]["prior_amount_mean"] == 43
    # The later transaction sees all three 7200 rows but not the future event.
    later = features.iloc[0]
    assert later["prior_transaction_count"] == 6
    assert later["prior_amount_mean"] == 37.5
    assert later["amount_vs_prior_mean"] == 50 / 37.5
    assert later["previous_transaction_amount"] == 32
    assert later["prior_transaction_count_1h"] == 3
    assert later["prior_transaction_count_24h"] == 6
    assert later["is_amount_above_prior_mean"] == 1
    assert later["is_high_velocity_1h"] == 1
    assert later["is_high_velocity_24h"] == 0
    # Exactly 24 hours is included; one second beyond the window is excluded.
    assert features.iloc[19]["prior_transaction_count_24h"] == 10
    assert features.iloc[19]["is_high_velocity_24h"] == 1
    assert features.iloc[20]["prior_transaction_count_24h"] == 1
    assert not np.isinf(features.select_dtypes(include="number").to_numpy()).any()
    pd.testing.assert_frame_equal(transactions, original)

    changed_labels = transactions.copy()
    changed_labels["isFraud"] = 1 - changed_labels["isFraud"]
    pd.testing.assert_frame_equal(
        features, build_behavioral_features(changed_labels)
    )


def test_build_behavioral_features_handles_zero_prior_mean_without_infinity():
    transactions = pd.DataFrame(
        {
            "TransactionDT": [0, 1],
            "TransactionAmt": [0.0, 10.0],
            "card1": [1, 1],
        }
    )

    features = build_behavioral_features(transactions)

    assert features["amount_vs_prior_mean"].tolist() == [0.0, 0.0]
    assert np.isfinite(features.select_dtypes(include="number").to_numpy()).all()
    assert features["is_amount_above_prior_mean"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("columns", "missing"),
    [
        (["TransactionAmt", "card1"], "TransactionDT"),
        (["TransactionDT", "card1"], "TransactionAmt"),
        (["TransactionDT", "TransactionAmt"], "card1"),
    ],
)
def test_build_behavioral_features_requires_columns(columns, missing):
    frame = pd.DataFrame({column: [1] for column in columns})

    with pytest.raises(ValueError, match=missing):
        build_behavioral_features(frame)


def test_transaction_identity_behavioral_pipeline():
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [0, 60],
            "TransactionAmt": [20.0, 80.0],
            "ProductCD": ["C", "S"],
            "card4": ["visa", "discover"],
            "card6": ["credit", "debit"],
            "P_emaildomain": ["a.com", None],
            "R_emaildomain": [None, "b.com"],
            "card1": [123, 123],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "DeviceType": ["mobile", None],
            "DeviceInfo": ["phone", None],
            "id_12": [1, None],
        }
    )

    transaction_features = build_transaction_features(transactions)
    merged = merge_identity_data(transactions, identity)
    identity_features = build_identity_features(merged)
    behavioral_features = build_behavioral_features(transactions)
    result = pd.concat(
        [transaction_features, identity_features, behavioral_features], axis=1
    )

    expected_columns = {
        "transaction_amount",
        "transaction_time",
        "transaction_hour",
        "amount_log",
        "product_code",
        "has_identity",
        "device_type",
        "has_device_info",
        "identity_match_count",
        "prior_transaction_count",
        "prior_amount_mean",
        "amount_vs_prior_mean",
        "prior_transaction_count_1h",
        "prior_transaction_count_24h",
        "previous_transaction_amount",
        "is_amount_above_prior_mean",
        "is_high_velocity_1h",
        "is_high_velocity_24h",
    }
    assert expected_columns.issubset(result.columns)
    assert len(result) == len(transactions)
