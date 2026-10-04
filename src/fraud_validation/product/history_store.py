"""Small process-local store for prior transactions used in live context."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

from fraud_validation.product.schemas import TransactionInput


@dataclass(frozen=True)
class StoredTransaction:
    """Minimal transaction data retained for behavioral calculations."""

    transaction_id: int
    transaction_time: float
    transaction_amount: float
    history_key: str


class InMemoryTransactionHistoryStore:
    """In-memory, process-local history; this is not production persistence.

    ``card1`` is a prototype history key because IEEE-CIS has no universal
    customer ID. If it is absent, the transaction ID isolates that transaction.
    A production system should define a proper customer/account/device identity.
    """

    def __init__(self) -> None:
        self._transactions: dict[str, list[StoredTransaction]] = {}
        self._lock = RLock()

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Serialize a complete request's history read/evaluate/store sequence."""
        with self._lock:
            yield

    @staticmethod
    def history_key_for(transaction: TransactionInput) -> str:
        card_id = transaction.card1
        if card_id is not None and str(card_id).strip():
            return f"card1:{str(card_id).strip()}"
        return f"transaction_id:{transaction.transaction_id}"

    def add(self, transaction: TransactionInput) -> StoredTransaction:
        """Store the current transaction after its evaluation completes."""
        history_key = self.history_key_for(transaction)
        stored = StoredTransaction(
            transaction_id=transaction.transaction_id,
            transaction_time=transaction.transaction_time,
            transaction_amount=transaction.transaction_amount,
            history_key=history_key,
        )
        with self._lock:
            entries = self._transactions.setdefault(history_key, [])
            insert_at = bisect_right(
                entries,
                stored.transaction_time,
                key=lambda entry: entry.transaction_time,
            )
            entries.insert(insert_at, stored)
        return stored

    def get_prior_transactions(
        self,
        transaction: TransactionInput,
    ) -> tuple[StoredTransaction, ...]:
        """Get same-key entries with strictly earlier transaction times.

        Equal-time entries and any later timestamps are deliberately excluded.
        """
        history_key = self.history_key_for(transaction)
        with self._lock:
            entries = self._transactions.get(history_key, [])
            cutoff = bisect_left(
                entries,
                transaction.transaction_time,
                key=lambda entry: entry.transaction_time,
            )
            return tuple(entries[:cutoff])

    def get_by_history_key(
        self,
        history_key: str,
    ) -> tuple[StoredTransaction, ...]:
        """Return stored entries for one exact key in chronological order."""
        with self._lock:
            return tuple(self._transactions.get(history_key, ()))
