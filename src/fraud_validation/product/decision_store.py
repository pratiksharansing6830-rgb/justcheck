"""Process-local store for transaction decision records."""

from __future__ import annotations

from threading import RLock

from fraud_validation.product.schemas import (
    MonitoringSummary,
    TransactionDecisionRecord,
)


class DuplicateDecisionError(ValueError):
    """Raised when a transaction already has a stored decision."""


class InMemoryDecisionStore:
    """Keep one decision per transaction ID in chronological insertion order."""

    def __init__(self) -> None:
        self._records: dict[int, TransactionDecisionRecord] = {}
        self._insertion_order: list[int] = []
        self._lock = RLock()

    def add(self, record: TransactionDecisionRecord) -> None:
        with self._lock:
            if record.transaction_id in self._records:
                raise DuplicateDecisionError(
                    f"transaction {record.transaction_id} has already been evaluated"
                )
            self._records[record.transaction_id] = record.model_copy(deep=True)
            self._insertion_order.append(record.transaction_id)

    def get(self, transaction_id: int) -> TransactionDecisionRecord | None:
        with self._lock:
            record = self._records.get(transaction_id)
            return record.model_copy(deep=True) if record is not None else None

    def remove(self, transaction_id: int) -> None:
        """Remove one stored record when later pipeline persistence fails."""
        with self._lock:
            if transaction_id in self._records:
                del self._records[transaction_id]
                self._insertion_order.remove(transaction_id)

    def list(self, limit: int = 20) -> list[TransactionDecisionRecord]:
        """Return the most recent records in chronological insertion order."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            transaction_ids = self._insertion_order[-limit:] if limit else []
            return [
                self._records[transaction_id].model_copy(deep=True)
                for transaction_id in transaction_ids
            ]

    def filter(
        self,
        *,
        risk_level: str | None = None,
        decision: str | None = None,
        limit: int = 20,
    ) -> list[TransactionDecisionRecord]:
        """Return recent matching records, retaining their insertion order."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            matching_ids = [
                transaction_id
                for transaction_id in self._insertion_order
                if (
                    risk_level is None
                    or self._records[transaction_id].risk_level == risk_level
                )
                and (
                    decision is None
                    or self._records[transaction_id].decision == decision
                )
            ]
            selected_ids = matching_ids[-limit:] if limit else []
            return [
                self._records[transaction_id].model_copy(deep=True)
                for transaction_id in selected_ids
            ]

    def summary(self) -> MonitoringSummary:
        with self._lock:
            records = list(self._records.values())
            total = len(records)
            low = sum(record.risk_level == "LOW" for record in records)
            medium = sum(record.risk_level == "MEDIUM" for record in records)
            high = sum(record.risk_level == "HIGH" for record in records)
            allow = sum(record.decision == "ALLOW" for record in records)
            verify = sum(record.decision == "VERIFY" for record in records)
            block = sum(record.decision == "BLOCK" for record in records)
            return MonitoringSummary(
                total_transactions=total,
                low_risk_count=low,
                medium_risk_count=medium,
                high_risk_count=high,
                allow_count=allow,
                verify_count=verify,
                block_count=block,
                high_risk_rate=high / total if total else 0.0,
                block_rate=block / total if total else 0.0,
            )
