"""Incremental behavioral features for a live transaction."""

from __future__ import annotations

from fraud_validation.product.history_store import (
    StoredTransaction,
)
from fraud_validation.product.schemas import (
    BehavioralContext,
    TransactionInput,
)

ONE_HOUR_SECONDS = 3600
ONE_DAY_SECONDS = 86400
HIGH_VELOCITY_1H_THRESHOLD = 3
HIGH_VELOCITY_24H_THRESHOLD = 10
AMOUNT_DEVIATION_THRESHOLD = 3


class BehavioralContextBuilder:
    """Build features from supplied historical transactions only."""

    def build(
        self,
        transaction: TransactionInput,
        previous_transactions: tuple[StoredTransaction, ...],
    ) -> BehavioralContext:
        current_time = transaction.transaction_time
        # Guard strictly here as well as in the store: context remains leakage
        # safe if a different history provider supplies future or tied events.
        prior = [
            event
            for event in previous_transactions
            if event.transaction_time < current_time
        ]

        amounts = [event.transaction_amount for event in prior]
        prior_mean = sum(amounts) / len(amounts) if amounts else None
        previous_amount = amounts[-1] if amounts else None
        amount_ratio = (
            transaction.transaction_amount / prior_mean
            if prior_mean is not None and prior_mean != 0
            else None
        )
        count_1h = sum(
            event.transaction_time >= current_time - ONE_HOUR_SECONDS
            for event in prior
        )
        count_24h = sum(
            event.transaction_time >= current_time - ONE_DAY_SECONDS
            for event in prior
        )
        return BehavioralContext(
            prior_transaction_count=len(prior),
            prior_transaction_count_1h=count_1h,
            prior_transaction_count_24h=count_24h,
            previous_transaction_amount=previous_amount,
            prior_amount_mean=prior_mean,
            amount_vs_prior_mean=amount_ratio,
            amount_above_prior_mean=(
                prior_mean is not None
                and transaction.transaction_amount > prior_mean
            ),
            is_high_velocity_1h=count_1h >= HIGH_VELOCITY_1H_THRESHOLD,
            is_high_velocity_24h=count_24h >= HIGH_VELOCITY_24H_THRESHOLD,
        )
