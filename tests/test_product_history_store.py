from fraud_validation.product.history_store import InMemoryTransactionHistoryStore
from fraud_validation.product.schemas import TransactionInput


def _transaction(
    transaction_id,
    transaction_time,
    *,
    amount=100,
    card1="card-a",
):
    return TransactionInput(
        transaction_id=transaction_id,
        transaction_time=transaction_time,
        transaction_amount=amount,
        card1=card1,
    )


def test_transaction_can_be_stored_and_retrieved_by_history_key():
    store = InMemoryTransactionHistoryStore()
    transaction = _transaction(1, 10)

    stored = store.add(transaction)

    assert stored.transaction_id == 1
    assert stored.transaction_amount == 100
    assert store.get_by_history_key("card1:card-a") == (stored,)


def test_entries_are_returned_in_chronological_order():
    store = InMemoryTransactionHistoryStore()
    store.add(_transaction(3, 30))
    store.add(_transaction(1, 10))
    store.add(_transaction(2, 20))

    history = store.get_by_history_key("card1:card-a")

    assert [entry.transaction_time for entry in history] == [10, 20, 30]


def test_future_transactions_are_not_returned_as_history():
    store = InMemoryTransactionHistoryStore()
    store.add(_transaction(3, 300))
    store.add(_transaction(1, 100))
    current = _transaction(2, 200)

    history = store.get_prior_transactions(current)

    assert [entry.transaction_id for entry in history] == [1]


def test_same_timestamp_transactions_are_not_previous():
    store = InMemoryTransactionHistoryStore()
    store.add(_transaction(1, 100))
    store.add(_transaction(2, 100))

    assert store.get_prior_transactions(_transaction(3, 100)) == ()
    assert len(store.get_prior_transactions(_transaction(4, 101))) == 2


def test_different_history_keys_remain_isolated():
    store = InMemoryTransactionHistoryStore()
    store.add(_transaction(1, 10, card1="card-a"))
    store.add(_transaction(2, 20, card1="card-b"))

    assert store.get_prior_transactions(_transaction(3, 30, card1="card-a"))[0].transaction_id == 1
    assert store.get_prior_transactions(_transaction(4, 30, card1="card-b"))[0].transaction_id == 2


def test_transaction_id_is_used_as_fallback_key_without_card1():
    store = InMemoryTransactionHistoryStore()
    transaction = _transaction(23, 10, card1=None)

    stored = store.add(transaction)

    assert stored.history_key == "transaction_id:23"
    assert store.get_by_history_key("transaction_id:23") == (stored,)
