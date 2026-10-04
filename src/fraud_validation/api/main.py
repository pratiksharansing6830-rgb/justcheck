"""FastAPI integration for the fraud-risk product layer."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from fraud_validation.product import (
    FraudStrategyAdapter,
    RiskEngine,
    TransactionInput,
)
from fraud_validation.product.decision_store import (
    DuplicateDecisionError,
    InMemoryDecisionStore,
)
from fraud_validation.product.history_store import InMemoryTransactionHistoryStore
from fraud_validation.product.risk_pipeline import RealTimeRiskPipeline
from fraud_validation.product.schemas import (
    MonitoringSummary,
    TransactionDecisionRecord,
)
from fraud_validation.product.strategy_adapter import (
    StrategyIntegrationNotConfigured,
    create_strategy_adapter,
)


def create_app(
    *,
    strategy_adapter: FraudStrategyAdapter | None = None,
    history_store: InMemoryTransactionHistoryStore | None = None,
    decision_store: InMemoryDecisionStore | None = None,
) -> FastAPI:
    """Build an API app with isolated process-local product components."""
    api = FastAPI(
        title="Adaptive Fraud-Risk Detection API",
        description=(
            "API for evaluating transactions through the fraud-risk product "
            "layer. The configured strategy is a temporary mock adapter, not a "
            "production machine-learning model."
        ),
        version="0.1.0",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    selected_history_store = (
        history_store
        if history_store is not None
        else InMemoryTransactionHistoryStore()
    )
    selected_decision_store = (
        decision_store
        if decision_store is not None
        else InMemoryDecisionStore()
    )
    active_strategy = (
        strategy_adapter
        if strategy_adapter is not None
        else create_strategy_adapter()
    )
    api.state.decision_store = selected_decision_store
    api.state.risk_pipeline = RealTimeRiskPipeline(
        history_store=selected_history_store,
        decision_store=selected_decision_store,
        risk_engine=RiskEngine(strategy_adapter=active_strategy),
    )

    @api.get("/api/v1/health")
    def health() -> dict[str, str]:
        """Return a minimal service health response."""
        return {"status": "ok", "service": "fraud-detection-api"}

    @api.post(
        "/api/v1/fraud/check",
        response_model=TransactionDecisionRecord,
    )
    def check_fraud(
        transaction: TransactionInput,
        pipeline: Annotated[
            RealTimeRiskPipeline,
            Depends(get_risk_pipeline),
        ],
    ) -> TransactionDecisionRecord:
        """Evaluate one validated transaction through the product pipeline."""
        try:
            return pipeline.evaluate(transaction)
        except StrategyIntegrationNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except DuplicateDecisionError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Transaction {transaction.transaction_id} has already been evaluated",
            ) from exc

    @api.get(
        "/api/v1/transactions",
        response_model=list[TransactionDecisionRecord],
    )
    def list_transactions(
        decision_store: Annotated[
            InMemoryDecisionStore,
            Depends(get_decision_store),
        ],
        risk_level: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
        decision: Literal["ALLOW", "VERIFY", "BLOCK"] | None = None,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[TransactionDecisionRecord]:
        return decision_store.filter(
            risk_level=risk_level,
            decision=decision,
            limit=limit,
        )

    @api.get(
        "/api/v1/transactions/{transaction_id}",
        response_model=TransactionDecisionRecord,
    )
    def get_transaction(
        transaction_id: int,
        decision_store: Annotated[
            InMemoryDecisionStore,
            Depends(get_decision_store),
        ],
    ) -> TransactionDecisionRecord:
        record = decision_store.get(transaction_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {transaction_id} was not found",
            )
        return record

    @api.get(
        "/api/v1/monitoring/summary",
        response_model=MonitoringSummary,
    )
    def monitoring_summary(
        decision_store: Annotated[
            InMemoryDecisionStore,
            Depends(get_decision_store),
        ],
    ) -> MonitoringSummary:
        return decision_store.summary()

    return api


def get_risk_pipeline(request: Request) -> RealTimeRiskPipeline:
    """Resolve the app-scoped pipeline for dependency injection."""
    return request.app.state.risk_pipeline


def get_decision_store(request: Request) -> InMemoryDecisionStore:
    """Resolve the app-scoped decision store for dependency injection."""
    return request.app.state.decision_store


app = create_app()
