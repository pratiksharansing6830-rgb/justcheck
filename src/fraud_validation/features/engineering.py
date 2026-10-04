"""Leakage-safe feature engineering for the IEEE-CIS fraud dataset."""

import numpy as np
import pandas as pd

ONE_HOUR_SECONDS = 3600
ONE_DAY_SECONDS = 86400
HIGH_VELOCITY_1H_THRESHOLD = 3
HIGH_VELOCITY_24H_THRESHOLD = 10


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic transaction-level features from transaction data.

    ``transaction_hour`` is a 24-hour bucket derived from the dataset's
    relative ``TransactionDT`` seconds; it is not a real-world clock hour.
    Non-positive amounts map to zero for ``amount_log``. Missing amounts remain
    missing in ``amount_log`` and do not trigger either amount flag.
    """
    amount = pd.to_numeric(df["TransactionAmt"], errors="raise")
    relative_time = pd.to_numeric(df["TransactionDT"], errors="raise")

    features = pd.DataFrame(index=df.index)
    features["transaction_amount"] = df["TransactionAmt"]
    features["transaction_time"] = df["TransactionDT"]
    features["transaction_hour"] = (relative_time // 3600) % 24
    features["amount_log"] = np.log1p(amount.clip(lower=0))
    features["is_low_amount"] = (amount < 35.95).fillna(False).astype("int8")
    features["is_high_amount"] = (amount > 159.95).fillna(False).astype("int8")
    features["product_code"] = df["ProductCD"]
    features["card_type"] = df["card6"]
    features["card_network"] = df["card4"]
    for source, target in (
        ("P_emaildomain", "has_p_email"),
        ("R_emaildomain", "has_r_email"),
    ):
        email = df[source].astype("string").str.strip()
        features[target] = (email.notna() & email.ne("")).astype("int8")

    return features


def merge_identity_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach identity columns to transactions with a left join on TransactionID."""
    return transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        sort=False,
    )


def build_identity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic identity and device features without using labels."""
    features = pd.DataFrame(index=df.index)
    features["device_type"] = (
        df["DeviceType"] if "DeviceType" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )

    if "DeviceInfo" in df.columns:
        device_info = df["DeviceInfo"].astype("string").str.strip()
        features["has_device_info"] = (
            device_info.notna() & device_info.ne("")
        ).astype("int8")
    else:
        features["has_device_info"] = 0

    identity_match_columns = [
        f"id_{number}" for number in range(12, 39)
        if f"id_{number}" in df.columns
    ]
    if identity_match_columns:
        features["identity_match_count"] = (
            df[identity_match_columns].notna().sum(axis=1).astype("int64")
        )
    else:
        features["identity_match_count"] = 0

    identity_columns = [
        column for column in df.columns
        if (column.startswith("id_") and column[3:].isdigit())
        or column in ("DeviceType", "DeviceInfo")
    ]
    if identity_columns:
        values = df[identity_columns]
        present = values.notna()
        for column in identity_columns:
            if pd.api.types.is_object_dtype(values[column].dtype) or pd.api.types.is_string_dtype(values[column].dtype):
                nonempty = values[column].astype("string").str.strip().ne("").fillna(False)
                present[column] = nonempty
        features["has_identity"] = present.any(axis=1).astype("int8")
    else:
        features["has_identity"] = 0

    return features[
        ["has_identity", "device_type", "has_device_info", "identity_match_count"]
    ]


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe card-level history features in original row order.

    History is based only on transactions with a strictly earlier
    ``TransactionDT`` for the same non-missing ``card1``. Transactions sharing
    a timestamp are evaluated as a batch, so none can see another tied row.
    """
    required_columns = ("TransactionDT", "TransactionAmt", "card1")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns for behavioral features: "
            + ", ".join(missing_columns)
        )

    row_count = len(df)
    transaction_time = pd.to_numeric(df["TransactionDT"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(df["TransactionAmt"], errors="coerce").to_numpy(dtype=float)
    card = df["card1"].to_numpy()

    prior_count = np.zeros(row_count, dtype=np.int64)
    prior_mean = np.zeros(row_count, dtype=float)
    prior_count_1h = np.zeros(row_count, dtype=np.int64)
    prior_count_24h = np.zeros(row_count, dtype=np.int64)
    previous_amount = np.zeros(row_count, dtype=float)

    valid_positions = np.flatnonzero(pd.notna(card) & np.isfinite(transaction_time))
    grouped = pd.DataFrame(
        {
            "card1": card[valid_positions],
            "position": valid_positions,
        }
    )

    for group_positions in grouped.groupby("card1", sort=False).indices.values():
        positions = valid_positions[np.asarray(group_positions, dtype=np.int64)]
        stable_order = np.argsort(transaction_time[positions], kind="stable")
        positions = positions[stable_order]
        times = transaction_time[positions]

        starts = np.r_[0, np.flatnonzero(times[1:] != times[:-1]) + 1]
        ends = np.r_[starts[1:], len(positions)]
        unique_times = times[starts]
        group_sizes = ends - starts
        cumulative_counts = np.r_[0, np.cumsum(group_sizes)]

        ordered_amounts = amount[positions]
        valid_amounts = np.isfinite(ordered_amounts)
        amount_sums = np.where(valid_amounts, ordered_amounts, 0.0)
        cumulative_sums = np.r_[0.0, np.cumsum(amount_sums)]
        cumulative_valid_amounts = np.r_[0, np.cumsum(valid_amounts)]

        for group_index, (start, end) in enumerate(zip(starts, ends)):
            batch = positions[start:end]
            count = int(cumulative_counts[group_index])
            amount_count = int(cumulative_valid_amounts[start])
            mean = (
                cumulative_sums[start] / amount_count
                if amount_count
                else 0.0
            )
            prior_count[batch] = count
            prior_mean[batch] = mean
            if start:
                previous_amount[batch] = (
                    ordered_amounts[start - 1]
                    if valid_amounts[start - 1]
                    else 0.0
                )

            time = unique_times[group_index]
            one_hour_start = np.searchsorted(
                unique_times, time - ONE_HOUR_SECONDS, side="left"
            )
            one_day_start = np.searchsorted(
                unique_times, time - ONE_DAY_SECONDS, side="left"
            )
            prior_count_1h[batch] = (
                cumulative_counts[group_index] - cumulative_counts[one_hour_start]
            )
            prior_count_24h[batch] = (
                cumulative_counts[group_index] - cumulative_counts[one_day_start]
            )

    amount_vs_mean = np.divide(
        amount,
        prior_mean,
        out=np.zeros(row_count, dtype=float),
        where=(prior_mean != 0) & np.isfinite(amount),
    )
    amount_vs_mean[~np.isfinite(amount_vs_mean)] = 0.0
    above_prior_mean = (
        np.isfinite(amount)
        & (prior_count > 0)
        & (amount > prior_mean)
    ).astype("int8")

    return pd.DataFrame(
        {
            "prior_transaction_count": prior_count,
            "prior_amount_mean": prior_mean,
            "amount_vs_prior_mean": amount_vs_mean,
            "prior_transaction_count_1h": prior_count_1h,
            "prior_transaction_count_24h": prior_count_24h,
            "previous_transaction_amount": previous_amount,
            "is_amount_above_prior_mean": above_prior_mean,
            "is_high_velocity_1h": (
                prior_count_1h >= HIGH_VELOCITY_1H_THRESHOLD
            ).astype("int8"),
            "is_high_velocity_24h": (
                prior_count_24h >= HIGH_VELOCITY_24H_THRESHOLD
            ).astype("int8"),
        },
        index=df.index,
    )
