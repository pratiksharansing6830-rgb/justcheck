"""Transparent conversion of strategy signals into operational decisions."""

from __future__ import annotations

from pydantic import ValidationError

from fraud_validation.product.schemas import (
    BehavioralContext,
    RiskResult,
    TransactionInput,
)
from fraud_validation.product.strategy_adapter import (
    FraudStrategyAdapter,
    MockStrategyAdapter,
    StrategySignal,
)

LOW_MAX_SCORE = 39
MEDIUM_MAX_SCORE = 69
HIGH_MAX_SCORE = 100
RISK_LEVELS = {"LOW": (0, 39), "MEDIUM": (40, 69), "HIGH": (70, 100)}
DEFAULT_DECISIONS = {"LOW": "ALLOW", "MEDIUM": "VERIFY", "HIGH": "BLOCK"}
PROBABILITY_WEIGHT = 0.5
RULE_SCORE_WEIGHT = 0.5


class RiskEngine:
    """Map strategy signals to score, risk level, decision, and explanations.

    Initial replaceable formula:
    ``round(100 * (0.5 * fraud_probability + 0.5 * rule_score / 100))``.
    This deterministic combination is for product-layer development only and
    is not asserted to be a calibrated or final production formula.
    """

    def __init__(
        self,
        strategy_adapter: FraudStrategyAdapter | None = None,
        *,
        decisions: dict[str, str] | None = None,
        probability_weight: float = PROBABILITY_WEIGHT,
        rule_score_weight: float = RULE_SCORE_WEIGHT,
    ) -> None:
        self.strategy_adapter = strategy_adapter or MockStrategyAdapter()
        selected_decisions = decisions or DEFAULT_DECISIONS
        if set(selected_decisions) != set(RISK_LEVELS):
            raise ValueError("decisions must define LOW, MEDIUM, and HIGH")
        if (
            probability_weight < 0
            or rule_score_weight < 0
            or abs(probability_weight + rule_score_weight - 1.0) > 1e-9
        ):
            raise ValueError(
                "probability_weight and rule_score_weight must be non-negative "
                "and sum to 1"
            )
        self.decisions = dict(selected_decisions)
        self.probability_weight = probability_weight
        self.rule_score_weight = rule_score_weight

    @staticmethod
    def _risk_level(score: int) -> str:
        if score <= LOW_MAX_SCORE:
            return "LOW"
        if score <= MEDIUM_MAX_SCORE:
            return "MEDIUM"
        return "HIGH"

    def _risk_score(self, signal: StrategySignal) -> int:
        raw_score = 100 * (
            self.probability_weight * signal.fraud_probability
            + self.rule_score_weight * signal.rule_score / 100
        )
        return max(0, min(100, round(raw_score)))

    def evaluate(
        self,
        transaction: TransactionInput,
        behavioral_context: BehavioralContext | None = None,
    ) -> RiskResult:
        """Produce an operational result for one validated transaction."""
        context = behavioral_context
        if context is None:
            context = BehavioralContext(
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
        raw_signal = self.strategy_adapter.evaluate(transaction, context)
        try:
            signal = StrategySignal.model_validate(raw_signal)
        except ValidationError as exc:
            raise ValueError(
                "strategy adapter returned an invalid strategy signal"
            ) from exc
        score = self._risk_score(signal)
        level = self._risk_level(score)
        reasons = signal.signals or ["No significant fraud risk signals detected"]
        return RiskResult(
            transaction_id=transaction.transaction_id,
            risk_score=score,
            risk_level=level,
            decision=self.decisions[level],
            reasons=reasons,
            fraud_probability=signal.fraud_probability,
            behavioral_context=context,
            strategy_name=signal.strategy_name,
            model_version=signal.model_version,
            metadata=signal.metadata,
        )
