from fraud_validation.product.behavioral_context import BehavioralContextBuilder
from fraud_validation.product.history_store import InMemoryTransactionHistoryStore
from fraud_validation.product.schemas import TransactionInput


def _transaction(
    transaction_id=1000,
    transaction_time=100_000,
    amount=100,
    card1="card-a",
):
    return TransactionInput(
        transaction_id=transaction_id,
        transaction_time=transaction_time,
        transaction_amount=amount,
        card1=card1,
    )


def _build(current, earlier):
    store = InMemoryTransactionHistoryStore()
    for transaction in earlier:
        store.add(transaction)
    return BehavioralContextBuilder().build(
        current, store.get_prior_transactions(current)
    )


def test_first_transaction_has_zero_and_null_defaults():
    context = _build(_transaction(), [])

    assert context.prior_transaction_count == 0
    assert context.prior_transaction_count_1h == 0
    assert context.prior_transaction_count_24h == 0
    assert context.previous_transaction_amount is None
    assert context.prior_amount_mean is None
    assert context.amount_vs_prior_mean is None
    assert context.amount_above_prior_mean is False
    assert context.is_high_velocity_1h is False
    assert context.is_high_velocity_24h is False


def test_previous_transaction_count_and_amount_are_reported():
    earlier = [
        _transaction(1, 10, 80),
        _transaction(2, 20, 120),
    ]

    context = _build(_transaction(3, 30, 90), earlier)

    assert context.prior_transaction_count == 2
    assert context.previous_transaction_amount == 120
    assert context.prior_amount_mean == 100


def test_three_prior_transactions_within_one_hour_trigger_velocity():
    current_time = 100_000
    earlier = [
        _transaction(i, current_time - offset, 50)
        for i, offset in enumerate((3000, 2000, 1000), start=1)
    ]

    context = _build(_transaction(4, current_time), earlier)

    assert context.prior_transaction_count_1h == 3
    assert context.is_high_velocity_1h is True


def test_ten_prior_transactions_within_one_day_trigger_velocity():
    current_time = 100_000
    earlier = [
        _transaction(i, current_time - (i * 600), 50)
        for i in range(1, 11)
    ]

    context = _build(_transaction(11, current_time), earlier)

    assert context.prior_transaction_count_24h == 10
    assert context.is_high_velocity_24h is True


def test_amount_anomaly_uses_only_prior_amount_mean():
    current_time = 100_000
    earlier = [
        _transaction(1, current_time - 300, 100),
        _transaction(2, current_time - 200, 100),
        _transaction(3, current_time - 100, 100),
    ]

    context = _build(_transaction(4, current_time, 500), earlier)

    assert context.prior_amount_mean == 100
    assert context.amount_vs_prior_mean == 5
    assert context.amount_above_prior_mean is True


def test_same_timestamp_and_future_transactions_are_excluded():
    current_time = 100_000
    earlier = [
        _transaction(1, current_time - 100, 100),
        _transaction(2, current_time, 200),
        _transaction(3, current_time + 100, 300),
    ]

    context = _build(_transaction(4, current_time, 500), earlier)

    assert context.prior_transaction_count == 1
    assert context.previous_transaction_amount == 100
    assert context.prior_amount_mean == 100
    assert context.amount_vs_prior_mean == 5


def test_zero_prior_mean_does_not_divide_by_zero():
    current_time = 100_000
    earlier = [_transaction(1, current_time - 1, 0)]

    context = _build(_transaction(2, current_time, 500), earlier)

    assert context.prior_amount_mean == 0
    assert context.amount_vs_prior_mean is None
