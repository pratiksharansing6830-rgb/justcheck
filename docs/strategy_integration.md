# Strategy adapter integration

The product integrates a strategy through `FraudStrategyAdapter`. The adapter
isolates the real-time product from the strategy's internal implementation;
the product does not need to know whether the evaluator uses rules, XGBoost,
Hybrid logic, or another validated approach.

`FRAUD_STRATEGY=mock` remains the default demo configuration.
`FRAUD_STRATEGY=validated` selects `ValidatedStrategyAdapter`. Without a
configured evaluator, it raises an explicit unconfigured error and the fraud
check API returns HTTP 503. There is no silent fallback to the mock.

## Handoff contract

### Input to the strategy

The adapter receives:

- a validated `TransactionInput`
- `BehavioralContext` computed from strictly earlier transactions

The transaction input does not include `isFraud`. Ground-truth labels are not
part of the real-time product request or strategy handoff.

### Output from the strategy

The evaluator returns a validated result containing:

- `fraud_probability`: finite numeric probability in `[0, 1]`
- `signals`: explainable, nonblank text that can be shown to an operator
- `strategy_name`: the identity of the strategy that produced the result
- `model_version`: the version identifier for that strategy
- `metadata`: optional JSON-safe scalar metadata for traceability

The current `RiskEngine` also requires a `rule_score` compatibility component
from `0` through `100`. This field is required by the existing product scoring
formula, which combines probability and this component with the current
configured weights. It is retained to avoid changing product scoring in this
integration step. A connected evaluator must provide its genuine, validated
component if it supports one; the adapter does not derive it from probability,
invent it, or substitute a mock value. An evaluator that cannot provide this
component is not yet compatible with the current RiskEngine contract; changing
that scoring contract requires a separate explicit product decision.

`ValidatedStrategyAdapter` validates evaluator output and supplies neutral
`validated_strategy` / `pending` metadata only when the evaluator omits
strategy identity fields. Identity and optional metadata returned by a
configured evaluator are preserved through the product.

## Responsibilities

### Strategy-validation side

- Produce the final validated evaluator/result according to its own validation
  process.
- Supply probability, explainable signals, strategy identity/version, optional
  metadata, and the explicit compatibility component required by the current
  RiskEngine.
- Own all training, validation, backtesting, strategy comparison, threshold
  selection, and model selection.

### Real-time product side

- Supply transaction data and earlier behavioral context to the adapter.
- Validate and consume its output.
- Calculate the operational risk score using the existing RiskEngine formula.
- Assign risk level and action using existing product rules.
- Store the decision and expose it through the API and dashboard.

Only the adapter knows the concrete strategy implementation. The RiskEngine,
decision store, API, and dashboard consume the stable result contract and do
not implement or select Rule, XGBoost, or Hybrid strategies.

## Result flow

`StrategySignal` is consumed by `RiskEngine`, which produces a `RiskResult`.
The real-time pipeline persists the result and its strategy metadata in a
`TransactionDecisionRecord`. The API returns those fields, and the dashboard
displays the strategy name, model version, and provided metadata alongside the
backend decision and explainable reasons.
