from datetime import datetime, timezone

import pytest

from fraud_validation.product.decision_store import (
    DuplicateDecisionError,
    InMemoryDecisionStore,
)
from fraud_validation.product.schemas import (
    BehavioralContext,
    TransactionDecisionRecord,
)


def _record(
    transaction_id,
    risk_level="LOW",
    decision="ALLOW",
    *,
    transaction_time=None,
):
    return TransactionDecisionRecord(
        transaction_id=transaction_id,
        transaction_amount=100,
        transaction_time=(
            transaction_id if transaction_time is None else transaction_time
        ),
        card1="card-a",
        evaluated_at=datetime.now(timezone.utc),
        risk_score={"LOW": 0, "MEDIUM": 50, "HIGH": 80}[risk_level],
        risk_level=risk_level,
        decision=decision,
        reasons=["test signal"],
        fraud_probability=0.5,
        strategy_name="mock_strategy",
        model_version="demo",
        behavioral_context=BehavioralContext(
            prior_transaction_count=0,
            prior_transaction_count_1h=0,
            prior_transaction_count_24h=0,
            previous_transaction_amount=None,
            prior_amount_mean=None,
            amount_vs_prior_mean=None,
            amount_above_prior_mean=False,
            is_high_velocity_1h=False,
            is_high_velocity_24h=False,
        ),
    )


def test_add_and_get_record_by_transaction_id():
    store = InMemoryDecisionStore()
    record = _record(1)

    store.add(record)

    assert store.get(1) == record


def test_list_returns_records_in_chronological_insertion_order():
    store = InMemoryDecisionStore()
    store.add(_record(3, transaction_time=300))
    store.add(_record(1, transaction_time=100))
    store.add(_record(2, transaction_time=200))

    assert [record.transaction_id for record in store.list()] == [3, 1, 2]


def test_filter_by_risk_level_and_decision():
    store = InMemoryDecisionStore()
    store.add(_record(1, "LOW", "ALLOW"))
    store.add(_record(2, "MEDIUM", "VERIFY"))
    store.add(_record(3, "HIGH", "BLOCK"))
    store.add(_record(4, "HIGH", "BLOCK"))

    assert [r.transaction_id for r in store.filter(risk_level="HIGH")] == [3, 4]
    assert [r.transaction_id for r in store.filter(decision="VERIFY")] == [2]
    assert [
        r.transaction_id
        for r in store.filter(risk_level="HIGH", decision="BLOCK")
    ] == [3, 4]


def test_limit_returns_most_recent_records_in_insertion_order():
    store = InMemoryDecisionStore()
    for transaction_id in range(1, 5):
        store.add(_record(transaction_id))

    assert [r.transaction_id for r in store.list(limit=2)] == [3, 4]
    assert [r.transaction_id for r in store.filter(limit=1)] == [4]


def test_duplicate_transaction_ids_are_rejected_without_replacing_record():
    store = InMemoryDecisionStore()
    original = _record(1, "LOW", "ALLOW")
    store.add(original)

    with pytest.raises(DuplicateDecisionError):
        store.add(_record(1, "HIGH", "BLOCK"))

    assert store.list() == [original]
    assert store.get(1) == original


def test_get_and_list_return_copies_not_mutable_internal_records():
    store = InMemoryDecisionStore()
    store.add(_record(1))

    returned = store.get(1)
    returned.reasons.append("local mutation")

    assert store.get(1).reasons == ["test signal"]
