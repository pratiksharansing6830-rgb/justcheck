"""Orchestration for one real-time transaction risk evaluation."""

from __future__ import annotations

from fraud_validation.product.behavioral_context import BehavioralContextBuilder
from fraud_validation.product.decision_store import (
    DuplicateDecisionError,
    InMemoryDecisionStore,
)
from fraud_validation.product.history_store import InMemoryTransactionHistoryStore
from fraud_validation.product.risk_engine import RiskEngine
from fraud_validation.product.schemas import (
    RiskResult,
    TransactionDecisionRecord,
    TransactionInput,
)


class RealTimeRiskPipeline:
    """Serialize history-read, scoring, then history-write for each request."""

    def __init__(
        self,
        history_store: InMemoryTransactionHistoryStore,
        decision_store: InMemoryDecisionStore,
        risk_engine: RiskEngine,
        context_builder: BehavioralContextBuilder | None = None,
    ) -> None:
        self.history_store = history_store
        self.decision_store = decision_store
        self.risk_engine = risk_engine
        self.context_builder = context_builder or BehavioralContextBuilder()

    def evaluate(self, transaction: TransactionInput) -> TransactionDecisionRecord:
        """Evaluate, persist the decision, then append behavioral history."""
        with self.history_store.atomic():
            if self.decision_store.get(transaction.transaction_id) is not None:
                raise DuplicateDecisionError(
                    f"transaction {transaction.transaction_id} has already been evaluated"
                )

            previous = self.history_store.get_prior_transactions(transaction)
            context = self.context_builder.build(transaction, previous)
            result: RiskResult = self.risk_engine.evaluate(transaction, context)
            record = TransactionDecisionRecord(
                transaction_id=transaction.transaction_id,
                transaction_amount=transaction.transaction_amount,
                transaction_time=transaction.transaction_time,
                card1=transaction.card1,
                risk_score=result.risk_score,
                risk_level=result.risk_level,
                decision=result.decision,
                reasons=result.reasons,
                fraud_probability=result.fraud_probability,
                strategy_name=result.strategy_name,
                model_version=result.model_version,
                behavioral_context=context,
                metadata=result.metadata,
            )
            self.decision_store.add(record)
            try:
                self.history_store.add(transaction)
            except Exception:
                self.decision_store.remove(transaction.transaction_id)
                raise
            return record
