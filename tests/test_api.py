import pytest
from fastapi.testclient import TestClient

from fraud_validation.api.main import create_app
from fraud_validation.product.strategy_adapter import (
    ValidatedStrategyAdapter,
)


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def _transaction(transaction_id=123, **overrides):
    transaction = {
        "transaction_id": transaction_id,
        "transaction_amount": 100,
        "transaction_time": 3600,
    }
    transaction.update(overrides)
    return transaction


def test_health_endpoint(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "fraud-detection-api",
    }


def test_vite_development_origin_is_allowed_by_cors(client):
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_api_strategy_configuration_uses_environment_and_records_metadata(monkeypatch):
    monkeypatch.setenv("FRAUD_STRATEGY", "validated")
    with TestClient(create_app()) as configured_client:
        pipeline = configured_client.app.state.risk_pipeline
        assert isinstance(
            pipeline.risk_engine.strategy_adapter, ValidatedStrategyAdapter
        )
        response = configured_client.post(
            "/api/v1/fraud/check",
            json=_transaction(transaction_id=765, transaction_time=12),
        )

    assert response.status_code == 503
    assert "no validated strategy evaluator" in response.json()["detail"]


def test_api_invalid_strategy_configuration_fails_at_app_creation(monkeypatch):
    monkeypatch.setenv("FRAUD_STRATEGY", "xgboost")

    with pytest.raises(ValueError, match="expected 'mock' or 'validated'"):
        create_app()


def test_api_mock_strategy_metadata_is_recorded_on_success(client):
    response = client.post(
        "/api/v1/fraud/check",
        json=_transaction(transaction_id=766, transaction_time=12),
    )

    assert response.status_code == 200
    assert response.json()["strategy_name"] == "mock_strategy"
    assert response.json()["model_version"] == "demo"


def test_validated_strategy_metadata_flows_to_api_and_decision_record():
    evaluator_output = {
        "fraud_probability": 0.75,
        "rule_score": 60,
        "signals": ["API contract test signal"],
        "strategy_name": "test_validated_strategy",
        "model_version": "test-v1",
        "metadata": {"validation_run": "api-test-run"},
    }
    adapter = ValidatedStrategyAdapter(
        lambda _transaction, _context: evaluator_output
    )
    with TestClient(create_app(strategy_adapter=adapter)) as configured_client:
        response = configured_client.post(
            "/api/v1/fraud/check",
            json=_transaction(transaction_id=767, transaction_time=12),
        )
        stored_response = configured_client.get("/api/v1/transactions/767")

    assert response.status_code == 200
    result = response.json()
    assert result["fraud_probability"] == 0.75
    assert result["risk_score"] == 68
    assert result["risk_level"] == "MEDIUM"
    assert result["decision"] == "VERIFY"
    assert result["reasons"] == ["API contract test signal"]
    assert result["strategy_name"] == "test_validated_strategy"
    assert result["model_version"] == "test-v1"
    assert result["metadata"] == {"validation_run": "api-test-run"}
    assert stored_response.status_code == 200
    assert stored_response.json()["metadata"] == {"validation_run": "api-test-run"}


def test_fraud_check_returns_risk_result(client):
    response = client.post("/api/v1/fraud/check", json=_transaction())

    assert response.status_code == 200
    result = response.json()
    assert result["transaction_id"] == 123
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["decision"] in {"ALLOW", "VERIFY", "BLOCK"}
    assert isinstance(result["reasons"], list)
    assert 0 <= result["fraud_probability"] <= 1
    assert "isFraud" not in result
    assert result["behavioral_context"]["prior_transaction_count"] == 0
    assert result["strategy_name"] == "mock_strategy"
    assert result["model_version"] == "demo"
    assert result["metadata"] == {}


def test_invalid_request_payloads_return_422(client):
    missing_id = _transaction()
    del missing_id["transaction_id"]
    invalid_payloads = [
        missing_id,
        _transaction(transaction_amount=-1),
        _transaction(transaction_time=-1),
        _transaction(isFraud=1),
    ]

    for payload in invalid_payloads:
        response = client.post("/api/v1/fraud/check", json=payload)
        assert response.status_code == 422


def test_invalid_transaction_time_types_return_422(client):
    response = client.post(
        "/api/v1/fraud/check",
        json=_transaction(transaction_time="not-a-time"),
    )

    assert response.status_code == 422


def test_risk_boundaries_are_exercised_through_the_api(client):
    cases = [
        (_transaction(1, transaction_amount=100), "LOW", "ALLOW"),
        (_transaction(2, transaction_amount=500), "MEDIUM", "VERIFY"),
        (
            _transaction(3, transaction_amount=500, product_code="C"),
            "HIGH",
            "BLOCK",
        ),
    ]

    for payload, expected_level, expected_decision in cases:
        response = client.post("/api/v1/fraud/check", json=payload)

        assert response.status_code == 200
        assert response.json()["risk_level"] == expected_level
        assert response.json()["decision"] == expected_decision


def test_openapi_and_docs_are_exposed(client):
    openapi_response = client.get("/openapi.json")
    docs_response = client.get("/docs")

    assert openapi_response.status_code == 200
    assert docs_response.status_code == 200
    assert {
        "/api/v1/health",
        "/api/v1/fraud/check",
        "/api/v1/transactions",
        "/api/v1/transactions/{transaction_id}",
        "/api/v1/monitoring/summary",
    }.issubset(openapi_response.json()["paths"])


def test_behavioral_context_flows_through_api_for_same_card(client):
    first = client.post(
        "/api/v1/fraud/check",
        json=_transaction(1, transaction_time=1000, card1="card-a"),
    ).json()
    second = client.post(
        "/api/v1/fraud/check",
        json=_transaction(2, transaction_time=1100, card1="card-a"),
    ).json()
    third = client.post(
        "/api/v1/fraud/check",
        json=_transaction(
            3,
            transaction_amount=500,
            transaction_time=1200,
            card1="card-a",
        ),
    ).json()

    assert first["behavioral_context"]["prior_transaction_count"] == 0
    assert second["behavioral_context"]["prior_transaction_count"] == 1
    assert second["behavioral_context"]["previous_transaction_amount"] == 100
    assert third["behavioral_context"]["prior_transaction_count"] == 2
    assert third["behavioral_context"]["prior_amount_mean"] == 100
    assert third["behavioral_context"]["amount_vs_prior_mean"] == 5
    assert (
        "Transaction amount is significantly above historical average"
        in third["reasons"]
    )
    assert third["risk_score"] > second["risk_score"]


def test_different_card_does_not_inherit_history(client):
    client.post(
        "/api/v1/fraud/check",
        json=_transaction(1, transaction_time=1000, card1="card-a"),
    )
    other_card = client.post(
        "/api/v1/fraud/check",
        json=_transaction(2, transaction_time=1100, card1="card-b"),
    ).json()

    assert other_card["behavioral_context"]["prior_transaction_count"] == 0
    assert other_card["behavioral_context"]["prior_amount_mean"] is None


def test_same_timestamp_request_is_not_previous_for_current_request(client):
    client.post(
        "/api/v1/fraud/check",
        json=_transaction(1, transaction_time=1000, card1="card-a"),
    )
    same_time = client.post(
        "/api/v1/fraud/check",
        json=_transaction(2, transaction_time=1000, card1="card-a"),
    ).json()
    later = client.post(
        "/api/v1/fraud/check",
        json=_transaction(3, transaction_time=1001, card1="card-a"),
    ).json()

    assert same_time["behavioral_context"]["prior_transaction_count"] == 0
    assert later["behavioral_context"]["prior_transaction_count"] == 2


def test_duplicate_transaction_id_returns_conflict_without_reprocessing(client):
    first = client.post(
        "/api/v1/fraud/check",
        json=_transaction(1, transaction_time=1000, card1="card-a"),
    )
    duplicate = client.post(
        "/api/v1/fraud/check",
        json=_transaction(
            1,
            transaction_amount=500,
            transaction_time=1100,
            card1="card-a",
        ),
    )

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert "already been evaluated" in duplicate.json()["detail"]
    assert len(client.get("/api/v1/transactions").json()) == 1
    subsequent = client.post(
        "/api/v1/fraud/check",
        json=_transaction(2, transaction_time=1200, card1="card-a"),
    ).json()
    assert subsequent["behavioral_context"]["prior_transaction_count"] == 1


def test_transaction_list_filters_and_limits(client):
    cases = [
        _transaction(1, transaction_time=1000),
        _transaction(2, transaction_amount=500, transaction_time=1100),
        _transaction(
            3, transaction_amount=500, product_code="C", transaction_time=1200
        ),
    ]
    for payload in cases:
        assert client.post("/api/v1/fraud/check", json=payload).status_code == 200

    all_records = client.get("/api/v1/transactions").json()
    high_records = client.get(
        "/api/v1/transactions", params={"risk_level": "HIGH"}
    ).json()
    blocked_records = client.get(
        "/api/v1/transactions", params={"decision": "BLOCK", "limit": 10}
    ).json()
    recent_records = client.get(
        "/api/v1/transactions", params={"limit": 2}
    ).json()

    assert [record["transaction_id"] for record in all_records] == [1, 2, 3]
    assert [record["risk_level"] for record in high_records] == ["HIGH"]
    assert [record["decision"] for record in blocked_records] == ["BLOCK"]
    assert [record["transaction_id"] for record in recent_records] == [2, 3]


@pytest.mark.parametrize(
    ("params",),
    [
        ({"risk_level": "CRITICAL"},),
        ({"decision": "DENY"},),
        ({"limit": 101},),
        ({"limit": 0},),
    ],
)
def test_transaction_list_rejects_invalid_filters(client, params):
    response = client.get("/api/v1/transactions", params=params)

    assert response.status_code == 422


def test_get_transaction_and_missing_transaction(client):
    submitted = client.post(
        "/api/v1/fraud/check",
        json=_transaction(45, transaction_time=1000, card1="card-a"),
    )

    found = client.get("/api/v1/transactions/45")
    missing = client.get("/api/v1/transactions/999")

    assert submitted.status_code == 200
    assert found.status_code == 200
    assert found.json() == submitted.json()
    assert missing.status_code == 404


def test_monitoring_summary_works_for_empty_store(client):
    response = client.get("/api/v1/monitoring/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_transactions": 0,
        "low_risk_count": 0,
        "medium_risk_count": 0,
        "high_risk_count": 0,
        "allow_count": 0,
        "verify_count": 0,
        "block_count": 0,
        "high_risk_rate": 0,
        "block_rate": 0,
    }


def test_monitoring_summary_aggregates_risk_and_decisions(client):
    cases = [
        _transaction(1, transaction_time=1000),
        _transaction(2, transaction_amount=500, transaction_time=1100),
        _transaction(
            3, transaction_amount=500, product_code="C", transaction_time=1200
        ),
    ]
    for payload in cases:
        client.post("/api/v1/fraud/check", json=payload)

    summary = client.get("/api/v1/monitoring/summary").json()

    assert summary == {
        "total_transactions": 3,
        "low_risk_count": 1,
        "medium_risk_count": 1,
        "high_risk_count": 1,
        "allow_count": 1,
        "verify_count": 1,
        "block_count": 1,
        "high_risk_rate": 1 / 3,
        "block_rate": 1 / 3,
    }
