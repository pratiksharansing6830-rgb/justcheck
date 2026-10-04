"""Strategy contract and deliberately non-production mock implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_validation.product.schemas import (
    BehavioralContext,
    StrategyMetadata,
    TransactionInput,
    _validate_strategy_metadata,
)

AMOUNT_DEVIATION_THRESHOLD = 3


class StrategySignal(BaseModel):
    """Validated strategy output consumed by RiskEngine and decision records."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    fraud_probability: float = Field(
        strict=True, ge=0, le=1, allow_inf_nan=False
    )
    rule_score: float = Field(strict=True, ge=0, le=100, allow_inf_nan=False)
    signals: list[str] = Field(default_factory=list)
    strategy_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    metadata: StrategyMetadata = Field(default_factory=dict)

    @field_validator("strategy_name", "model_version")
    @classmethod
    def metadata_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("metadata must not be blank")
        return value.strip()

    @field_validator("signals")
    @classmethod
    def signals_must_be_nonblank(cls, values: list[str]) -> list[str]:
        normalized = [signal.strip() for signal in values]
        if any(not signal for signal in normalized):
            raise ValueError("signals must not contain blank reasons")
        return normalized

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(
        cls, value: StrategyMetadata
    ) -> StrategyMetadata:
        return _validate_strategy_metadata(value)


class FraudStrategyAdapter(ABC):
    """Strategy boundary: transaction + prior context in, signal out.

    Implementations return probability, explainable signals, truthful strategy
    metadata, and (for the current RiskEngine formula) an explicit 0-100
    compatibility score component. This contract never includes a
    ground-truth fraud label.
    """

    strategy_name: str
    model_version: str

    @abstractmethod
    def evaluate(
        self,
        transaction: TransactionInput,
        behavioral_context: BehavioralContext | None = None,
    ) -> StrategySignal:
        """Evaluate a transaction with only its precomputed historical context."""


class MockStrategyAdapter(FraudStrategyAdapter):
    """Demo-only heuristic adapter; its outputs are not model predictions."""

    strategy_name = "mock_strategy"
    model_version = "demo"

    def evaluate(
        self,
        transaction: TransactionInput,
        behavioral_context: BehavioralContext | None = None,
    ) -> StrategySignal:
        signals: list[str] = []
        rule_score = 0.0

        if transaction.transaction_amount >= 500:
            signals.append("High transaction amount (mock heuristic)")
            rule_score += 60
        if transaction.product_code and transaction.product_code.upper() == "C":
            signals.append("Product code C (mock heuristic)")
            rule_score += 15
        if behavioral_context is not None:
            if behavioral_context.is_high_velocity_1h:
                signals.append("High transaction velocity in the last hour")
                rule_score += 15
            if behavioral_context.is_high_velocity_24h:
                signals.append("High transaction velocity in the last 24 hours")
                rule_score += 15
            if (
                behavioral_context.amount_vs_prior_mean is not None
                and behavioral_context.amount_vs_prior_mean
                >= AMOUNT_DEVIATION_THRESHOLD
            ):
                signals.append(
                    "Transaction amount is significantly above historical average"
                )
                rule_score += 20

        rule_score = min(rule_score, 100.0)
        # This probability is a transparent placeholder derived from the mock
        # rule score; it is not a calibrated fraud probability.
        fraud_probability = rule_score / 100
        return StrategySignal(
            fraud_probability=fraud_probability,
            rule_score=rule_score,
            signals=signals,
            strategy_name="mock_strategy",
            model_version="demo",
        )


class StrategyIntegrationNotConfigured(RuntimeError):
    """Raised when the future validated strategy has not been connected."""


class ValidatedStrategyAdapter(FraudStrategyAdapter):
    """Integration point for a validated strategy supplied by its owners.

    No model or mock fallback is included here. Until an evaluator is supplied,
    calls fail clearly rather than returning results mislabeled as validated.
    """

    def __init__(
        self,
        evaluator: Callable[
            [TransactionInput, BehavioralContext], StrategySignal | Mapping[str, Any]
        ]
        | None = None,
        *,
        strategy_name: str = "validated_strategy",
        model_version: str = "pending",
    ) -> None:
        self._evaluator = evaluator
        self.strategy_name = strategy_name
        self.model_version = model_version

    def evaluate(
        self,
        transaction: TransactionInput,
        behavioral_context: BehavioralContext | None = None,
    ) -> StrategySignal:
        if self._evaluator is None:
            raise StrategyIntegrationNotConfigured(
                "FRAUD_STRATEGY is set to 'validated', but no validated "
                "strategy evaluator has been connected"
            )
        if behavioral_context is None:
            behavioral_context = BehavioralContext(
                prior_transaction_count=0,
                prior_transaction_count_1h=0,
                prior_transaction_count_24h=0,
                previous_transaction_amount=None,
                prior_amount_mean=None,
                amount_vs_prior_mean=None,
                amount_above_prior_mean=False,
                is_high_velocity_1h=False,
                is_high_velocity_24h=False,
            )

        raw_signal = self._evaluator(transaction, behavioral_context)
        if isinstance(raw_signal, StrategySignal):
            raw_values: Any = raw_signal.model_dump()
        elif isinstance(raw_signal, Mapping):
            raw_values = dict(raw_signal)
        else:
            raw_values = raw_signal
        if isinstance(raw_values, dict):
            raw_values.setdefault("strategy_name", self.strategy_name)
            raw_values.setdefault("model_version", self.model_version)
        signal = StrategySignal.model_validate(raw_values)
        return signal


def create_strategy_adapter(strategy_name: str | None = None) -> FraudStrategyAdapter:
    """Create the configured application strategy; defaults to the demo mock."""
    selected = strategy_name
    if selected is None:
        selected = os.getenv("FRAUD_STRATEGY", "mock")
    selected = selected.strip().lower()
    if selected == "mock":
        return MockStrategyAdapter()
    if selected == "validated":
        return ValidatedStrategyAdapter()
    raise ValueError(
        f"Unsupported FRAUD_STRATEGY value {selected!r}; expected 'mock' or "
        "'validated'"
    )
