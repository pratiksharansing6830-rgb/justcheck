import pytest
from pydantic import ValidationError

from fraud_validation.product.schemas import BehavioralContext
from fraud_validation.product.schemas import TransactionInput
from fraud_validation.product.strategy_adapter import (
    FraudStrategyAdapter,
    MockStrategyAdapter,
    StrategyIntegrationNotConfigured,
    StrategySignal,
    ValidatedStrategyAdapter,
    create_strategy_adapter,
)


def _transaction(**overrides):
    payload = {
        "transaction_id": 123,
        "transaction_amount": 100.0,
        "transaction_time": 3600,
    }
    payload.update(overrides)
    return payload


def test_valid_transaction_with_optional_fields_missing():
    transaction = TransactionInput.model_validate(_transaction())

    assert transaction.transaction_id == 123
    assert transaction.transaction_amount == 100
    assert transaction.product_code is None
    assert transaction.card1 is None


def test_negative_amount_is_rejected():
    with pytest.raises(ValidationError):
        TransactionInput.model_validate(_transaction(transaction_amount=-0.01))


def test_missing_transaction_id_is_rejected():
    payload = _transaction()
    payload.pop("transaction_id")

    with pytest.raises(ValidationError):
        TransactionInput.model_validate(payload)


def test_is_fraud_is_not_accepted_as_request_input():
    with pytest.raises(ValidationError, match="isFraud"):
        TransactionInput.model_validate(_transaction(isFraud=1))


@pytest.mark.parametrize("transaction_time", [-1, float("nan"), float("inf"), "not-time"])
def test_invalid_transaction_time_is_rejected(transaction_time):
    with pytest.raises(ValidationError):
        TransactionInput.model_validate(
            _transaction(transaction_time=transaction_time)
        )


def test_mock_adapter_returns_structured_bounded_signal():
    transaction = TransactionInput.model_validate(
        _transaction(transaction_amount=750, product_code="C")
    )

    signal = MockStrategyAdapter().evaluate(transaction)

    assert isinstance(signal, StrategySignal)
    assert 0 <= signal.fraud_probability <= 1
    assert 0 <= signal.rule_score <= 100
    assert signal.signals == [
        "High transaction amount (mock heuristic)",
        "Product code C (mock heuristic)",
    ]
    assert signal.strategy_name == "mock_strategy"
    assert signal.model_version == "demo"


def test_mock_adapter_is_deterministic_and_says_when_no_mock_rule_applies():
    transaction = TransactionInput.model_validate(_transaction())

    first = MockStrategyAdapter().evaluate(transaction)
    second = MockStrategyAdapter().evaluate(transaction)

    assert first == second
    assert first.fraud_probability == 0
    assert first.rule_score == 0
    assert first.signals == []


def test_validated_adapter_is_an_explicit_unconfigured_integration_point():
    adapter = ValidatedStrategyAdapter()
    transaction = TransactionInput.model_validate(_transaction())

    assert isinstance(adapter, FraudStrategyAdapter)
    assert adapter.strategy_name == "validated_strategy"
    assert adapter.model_version == "pending"
    with pytest.raises(StrategyIntegrationNotConfigured, match="no validated"):
        adapter.evaluate(transaction)


def test_validated_adapter_wraps_provider_signal_with_neutral_metadata():
    received = {}

    def evaluate(transaction, behavioral_context):
        received["transaction"] = transaction
        received["behavioral_context"] = behavioral_context
        return {
            "fraud_probability": 0.42,
            "rule_score": 30,
            "signals": ["provider supplied reason"],
        }

    transaction = TransactionInput.model_validate(_transaction())
    context = BehavioralContext(
        prior_transaction_count=1,
        prior_transaction_count_1h=1,
        prior_transaction_count_24h=1,
        previous_transaction_amount=50,
        prior_amount_mean=50,
        amount_vs_prior_mean=2,
        amount_above_prior_mean=True,
        is_high_velocity_1h=False,
        is_high_velocity_24h=False,
    )
    signal = ValidatedStrategyAdapter(evaluate).evaluate(transaction, context)

    assert signal.strategy_name == "validated_strategy"
    assert signal.model_version == "pending"
    assert signal.fraud_probability == 0.42
    assert signal.signals == ["provider supplied reason"]
    assert received == {
        "transaction": transaction,
        "behavioral_context": context,
    }


def test_validated_adapter_preserves_strategy_metadata_from_evaluator():
    def evaluate(_transaction, _behavioral_context):
        return {
            "fraud_probability": 0.75,
            "rule_score": 60,
            "signals": ["test signal"],
            "strategy_name": "test_validated_strategy",
            "model_version": "test-v1",
            "metadata": {"validation_run": "run-17", "selected": True},
        }

    signal = ValidatedStrategyAdapter(evaluate).evaluate(
        TransactionInput.model_validate(_transaction())
    )

    assert signal.strategy_name == "test_validated_strategy"
    assert signal.model_version == "test-v1"
    assert signal.fraud_probability == 0.75
    assert signal.signals == ["test signal"]
    assert signal.metadata == {"validation_run": "run-17", "selected": True}


@pytest.mark.parametrize("strategy_name", ["mock", "validated"])
def test_strategy_configuration_selects_expected_adapter(monkeypatch, strategy_name):
    monkeypatch.setenv("FRAUD_STRATEGY", strategy_name)

    adapter = create_strategy_adapter()

    expected_type = {
        "mock": MockStrategyAdapter,
        "validated": ValidatedStrategyAdapter,
    }[strategy_name]
    assert isinstance(adapter, expected_type)


def test_strategy_configuration_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("FRAUD_STRATEGY", raising=False)

    assert isinstance(create_strategy_adapter(), MockStrategyAdapter)


def test_invalid_strategy_configuration_fails_clearly():
    with pytest.raises(ValueError, match="expected 'mock' or 'validated'"):
        create_strategy_adapter("other")


def test_strategy_signal_rejects_nonfinite_or_out_of_range_outputs():
    for probability in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValidationError):
            StrategySignal(
                fraud_probability=probability,
                rule_score=0,
                signals=[],
                strategy_name="test",
                model_version="v1",
            )


@pytest.mark.parametrize("probability", ["0.7", True])
def test_strategy_signal_rejects_non_numeric_probability_types(probability):
    with pytest.raises(ValidationError):
        StrategySignal(
            fraud_probability=probability,
            rule_score=0,
            signals=[],
            strategy_name="test",
            model_version="v1",
        )


def test_strategy_signal_rejects_invalid_metadata_and_reasons():
    with pytest.raises(ValidationError):
        StrategySignal(
            fraud_probability=0,
            rule_score=0,
            signals=["  "],
            strategy_name=" ",
            model_version="v1",
        )


def test_strategy_signal_rejects_non_json_safe_metadata():
    with pytest.raises(ValidationError):
        StrategySignal(
            fraud_probability=0,
            rule_score=0,
            signals=[],
            strategy_name="test",
            model_version="v1",
            metadata={"invalid_number": float("nan")},
        )
