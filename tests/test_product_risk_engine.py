import pytest

from fraud_validation.product.risk_engine import RiskEngine
from fraud_validation.product.schemas import TransactionInput
from fraud_validation.product.strategy_adapter import FraudStrategyAdapter, StrategySignal


class FixedAdapter(FraudStrategyAdapter):
    def __init__(self, signal):
        self.signal = signal

    def evaluate(self, transaction, behavioral_context=None):
        return self.signal


def _transaction():
    return TransactionInput(
        transaction_id=45,
        transaction_amount=100,
        transaction_time=120,
    )


@pytest.mark.parametrize(
    ("score", "level", "decision"),
    [
        (0, "LOW", "ALLOW"),
        (39, "LOW", "ALLOW"),
        (40, "MEDIUM", "VERIFY"),
        (69, "MEDIUM", "VERIFY"),
        (70, "HIGH", "BLOCK"),
        (100, "HIGH", "BLOCK"),
    ],
)
def test_score_boundaries_and_decisions(score, level, decision):
    signal = StrategySignal(
        fraud_probability=score / 100,
        rule_score=score,
        signals=[],
        strategy_name="test_strategy",
        model_version="test",
    )

    result = RiskEngine(FixedAdapter(signal)).evaluate(_transaction())

    assert result.risk_score == score
    assert result.risk_level == level
    assert result.decision == decision


def test_supplied_strategy_signals_are_the_result_reasons():
    signal = StrategySignal(
        fraud_probability=0.82,
        rule_score=65,
        signals=["High transaction amount", "High transaction velocity"],
        strategy_name="test_strategy",
        model_version="test",
    )

    result = RiskEngine(FixedAdapter(signal)).evaluate(_transaction())

    assert result.reasons == signal.signals
    assert result.fraud_probability == 0.82
    assert result.transaction_id == 45


def test_no_suspicious_signals_have_a_sensible_explanation():
    signal = StrategySignal(
        fraud_probability=0,
        rule_score=0,
        signals=[],
        strategy_name="test_strategy",
        model_version="test",
    )

    result = RiskEngine(FixedAdapter(signal)).evaluate(_transaction())

    assert result.risk_level == "LOW"
    assert result.decision == "ALLOW"
    assert result.reasons == ["No significant fraud risk signals detected"]


def test_decision_mapping_can_be_configured():
    signal = StrategySignal(
        fraud_probability=0.4,
        rule_score=40,
        signals=["demo"],
        strategy_name="test_strategy",
        model_version="test",
    )
    engine = RiskEngine(
        FixedAdapter(signal),
        decisions={"LOW": "PASS", "MEDIUM": "REVIEW", "HIGH": "DECLINE"},
    )

    result = engine.evaluate(_transaction())

    assert result.risk_level == "MEDIUM"
    assert result.decision == "REVIEW"


def test_score_weights_can_be_configured():
    signal = StrategySignal(
        fraud_probability=0.5,
        rule_score=0,
        signals=[],
        strategy_name="test_strategy",
        model_version="test",
    )
    engine = RiskEngine(
        FixedAdapter(signal), probability_weight=0.8, rule_score_weight=0.2
    )

    result = engine.evaluate(_transaction())

    assert result.risk_score == 40
    assert result.risk_level == "MEDIUM"


def test_score_weights_must_be_nonnegative_and_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        RiskEngine(probability_weight=0.8, rule_score_weight=0.3)


@pytest.mark.parametrize(
    "fraud_probability",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_risk_engine_rejects_invalid_strategy_probability(fraud_probability):
    class InvalidAdapter(FraudStrategyAdapter):
        def evaluate(self, transaction, behavioral_context=None):
            return StrategySignal.model_construct(
                fraud_probability=fraud_probability,
                rule_score=10,
                signals=["test"],
                strategy_name="untrusted",
                model_version="test",
            )

    with pytest.raises(ValueError, match="invalid strategy signal"):
        RiskEngine(InvalidAdapter()).evaluate(_transaction())
