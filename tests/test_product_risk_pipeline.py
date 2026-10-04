import pytest

from fraud_validation.product.decision_store import InMemoryDecisionStore
from fraud_validation.product.history_store import InMemoryTransactionHistoryStore
from fraud_validation.product.risk_engine import RiskEngine
from fraud_validation.product.risk_pipeline import RealTimeRiskPipeline
from fraud_validation.product.schemas import BehavioralContext, TransactionInput
from fraud_validation.product.strategy_adapter import (
    MockStrategyAdapter,
    ValidatedStrategyAdapter,
)


def _transaction(transaction_id=1, transaction_time=100, card1="card-a"):
    return TransactionInput(
        transaction_id=transaction_id,
        transaction_time=transaction_time,
        transaction_amount=100,
        card1=card1,
    )


def test_decision_store_failure_does_not_add_transaction_to_history():
    history = InMemoryTransactionHistoryStore()
    decisions = InMemoryDecisionStore()

    def fail_add(record):
        raise RuntimeError("simulated decision-store failure")

    decisions.add = fail_add
    pipeline = RealTimeRiskPipeline(
        history_store=history,
        decision_store=decisions,
        risk_engine=RiskEngine(),
    )
    transaction = _transaction()

    with pytest.raises(RuntimeError, match="simulated decision-store failure"):
        pipeline.evaluate(transaction)

    assert history.get_prior_transactions(
        _transaction(transaction_id=2, transaction_time=101)
    ) == ()
    assert decisions.list() == []


def test_duplicate_decision_is_rejected_before_history_is_updated_again():
    history = InMemoryTransactionHistoryStore()
    decisions = InMemoryDecisionStore()
    pipeline = RealTimeRiskPipeline(
        history_store=history,
        decision_store=decisions,
        risk_engine=RiskEngine(),
    )
    pipeline.evaluate(_transaction())

    with pytest.raises(ValueError, match="already been evaluated"):
        pipeline.evaluate(_transaction(transaction_time=101))

    assert len(history.get_by_history_key("card1:card-a")) == 1
    assert len(decisions.list()) == 1


def test_decision_record_uses_active_strategy_metadata():
    pipeline = RealTimeRiskPipeline(
        history_store=InMemoryTransactionHistoryStore(),
        decision_store=InMemoryDecisionStore(),
        risk_engine=RiskEngine(),
    )

    result = pipeline.evaluate(_transaction())

    assert result.strategy_name == "mock_strategy"
    assert result.model_version == "demo"
    record = pipeline.decision_store.get(result.transaction_id)
    assert record is not None
    assert record.strategy_name == "mock_strategy"
    assert record.model_version == "demo"


def test_mock_and_validated_strategies_share_risk_engine_pipeline():
    observed_contexts: list[BehavioralContext] = []

    def test_evaluator(transaction, context):
        observed_contexts.append(context)
        return {
            "fraud_probability": 0.75,
            "rule_score": 60,
            "signals": ["test evaluator signal"],
            "strategy_name": "test_validated_strategy",
            "model_version": "test-v1",
            "metadata": {"evaluation_id": "test-eval-1"},
        }

    mock_engine = RiskEngine(strategy_adapter=MockStrategyAdapter())
    validated_engine = RiskEngine(
        strategy_adapter=ValidatedStrategyAdapter(test_evaluator)
    )
    assert type(mock_engine) is type(validated_engine)

    mock_pipeline = RealTimeRiskPipeline(
        InMemoryTransactionHistoryStore(),
        InMemoryDecisionStore(),
        mock_engine,
    )
    validated_pipeline = RealTimeRiskPipeline(
        InMemoryTransactionHistoryStore(),
        InMemoryDecisionStore(),
        validated_engine,
    )

    mock_result = mock_pipeline.evaluate(_transaction(transaction_id=10))
    validated_pipeline.evaluate(_transaction(transaction_id=20, transaction_time=100))
    validated_result = validated_pipeline.evaluate(
        _transaction(transaction_id=21, transaction_time=101)
    )
    stored = validated_pipeline.decision_store.get(21)

    assert mock_result.strategy_name == "mock_strategy"
    assert validated_result.behavioral_context.prior_transaction_count == 1
    assert observed_contexts[-1].prior_transaction_count == 1
    assert validated_result.fraud_probability == 0.75
    assert validated_result.strategy_name == "test_validated_strategy"
    assert validated_result.model_version == "test-v1"
    assert validated_result.metadata == {"evaluation_id": "test-eval-1"}
    assert validated_result.reasons == ["test evaluator signal"]
    assert validated_result.risk_score == 68
    assert validated_result.risk_level == "MEDIUM"
    assert validated_result.decision == "VERIFY"
    assert stored is not None
    assert stored.strategy_name == "test_validated_strategy"
    assert stored.model_version == "test-v1"
    assert stored.metadata == {"evaluation_id": "test-eval-1"}
