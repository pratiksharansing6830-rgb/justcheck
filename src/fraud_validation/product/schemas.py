"""Validated request and response models for the product risk layer."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

StrategyMetadata = dict[str, str | int | float | bool | None]


def _validate_strategy_metadata(value: StrategyMetadata) -> StrategyMetadata:
    for key, item in value.items():
        if not key.strip():
            raise ValueError("strategy metadata keys must not be blank")
        if isinstance(item, float) and not isfinite(item):
            raise ValueError("strategy metadata numbers must be finite")
    return value


class TransactionInput(BaseModel):
    """Practical request; transaction_time is relative seconds, not wall time.

    Extra fields are rejected, so ground-truth labels such as ``isFraud`` can
    never enter the real-time request.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(gt=0)
    transaction_amount: float = Field(ge=0, allow_inf_nan=False)
    product_code: str | None = None
    card_type: str | None = None
    card_network: str | None = None
    p_emaildomain: str | None = None
    r_emaildomain: str | None = None
    card1: str | int | None = None
    transaction_time: float = Field(ge=0, allow_inf_nan=False)


class BehavioralContext(BaseModel):
    """Historical transaction features available before the current request."""

    model_config = ConfigDict(extra="forbid")

    prior_transaction_count: int = Field(ge=0)
    prior_transaction_count_1h: int = Field(ge=0)
    prior_transaction_count_24h: int = Field(ge=0)
    previous_transaction_amount: float | None = None
    prior_amount_mean: float | None = None
    amount_vs_prior_mean: float | None = None
    amount_above_prior_mean: bool
    is_high_velocity_1h: bool
    is_high_velocity_24h: bool


class RiskResult(BaseModel):
    """Operational decision for one transaction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int
    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    decision: str
    reasons: list[str]
    fraud_probability: float = Field(ge=0, le=1)
    behavioral_context: BehavioralContext
    strategy_name: str
    model_version: str
    metadata: StrategyMetadata = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(
        cls, value: StrategyMetadata
    ) -> StrategyMetadata:
        return _validate_strategy_metadata(value)


class TransactionDecisionRecord(BaseModel):
    """Runtime decision evidence for investigation and dashboard monitoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: int
    transaction_amount: float
    transaction_time: float
    card1: str | int | None = None
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    risk_score: int = Field(ge=0, le=100)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["ALLOW", "VERIFY", "BLOCK"]
    reasons: list[str]
    fraud_probability: float = Field(ge=0, le=1)
    strategy_name: str
    model_version: str
    behavioral_context: BehavioralContext
    metadata: StrategyMetadata = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(
        cls, value: StrategyMetadata
    ) -> StrategyMetadata:
        return _validate_strategy_metadata(value)


class MonitoringSummary(BaseModel):
    """Aggregates for dashboard cards and simple charts."""

    model_config = ConfigDict(extra="forbid")

    total_transactions: int = Field(ge=0)
    low_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    allow_count: int = Field(ge=0)
    verify_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    high_risk_rate: float = Field(ge=0, le=1)
    block_rate: float = Field(ge=0, le=1)
