import { useRef, useState } from "react";
import {
  ApiError,
  checkFraudTransaction,
  type FraudCheckPayload,
  type TransactionRecord,
} from "../services/api";
import { StatusBadge } from "./StatusBadge";

type ScenarioName = "normal" | "behavioral" | "high-amount";

interface ScenarioStep {
  amount: number;
  cardId: string;
}

interface Scenario {
  label: string;
  steps: ScenarioStep[];
}

interface ScenarioEvent {
  index: number;
  total: number;
  payload: FraudCheckPayload;
  status: "pending" | "success" | "error";
  result?: TransactionRecord;
  error?: string;
}

interface TransactionSimulatorProps {
  onSuccess: (record: TransactionRecord) => void | Promise<void>;
}

let demoIdCounter = 0;
let lastDemoId = 0;

function createDemoTransactionId(): number {
  const candidate = Date.now() * 1000 + demoIdCounter++;
  lastDemoId = Math.max(candidate, lastDemoId + 1);
  return lastDemoId;
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const fields = error.fieldErrors.length
      ? ` ${error.fieldErrors.join(" ")}`
      : "";
    return `${error.message}${fields}`;
  }
  return error instanceof Error
    ? error.message
    : "Unable to process this demo transaction.";
}

function createScenario(name: ScenarioName): Scenario {
  if (name === "behavioral") {
    return {
      label: "Behavioral Change",
      steps: [
        { amount: 100, cardId: "demo-card-01" },
        { amount: 100, cardId: "demo-card-01" },
        { amount: 500, cardId: "demo-card-01" },
      ],
    };
  }
  if (name === "high-amount") {
    return {
      label: "High Amount",
      steps: [{ amount: 5000, cardId: "demo-card-high-amount" }],
    };
  }
  return {
    label: "Normal Transaction",
    steps: [{ amount: 100, cardId: "demo-card-normal" }],
  };
}

export function TransactionSimulator({
  onSuccess,
}: TransactionSimulatorProps) {
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [events, setEvents] = useState<ScenarioEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [scenarioError, setScenarioError] = useState<string | null>(null);
  const nextTransactionTime = useRef(1000);
  const runLock = useRef(false);

  const runScenario = async (name: ScenarioName) => {
    if (runLock.current) return;
    runLock.current = true;
    const selectedScenario = createScenario(name);
    setScenario(selectedScenario);
    setEvents([]);
    setScenarioError(null);
    setRunning(true);

    let submitted = 0;
    let refreshFailed = false;
    try {
      for (const [stepIndex, step] of selectedScenario.steps.entries()) {
        const payload: FraudCheckPayload = {
          transaction_id: createDemoTransactionId(),
          transaction_amount: step.amount,
          transaction_time: nextTransactionTime.current,
          card1: step.cardId,
        };
        nextTransactionTime.current += 100;

        setEvents((current) => [
          ...current,
          {
            index: stepIndex + 1,
            total: selectedScenario.steps.length,
            payload,
            status: "pending",
          },
        ]);

        let result: TransactionRecord;
        try {
          result = await checkFraudTransaction(payload);
        } catch (error) {
          const readableError = describeError(error);
          setEvents((current) =>
            current.map((event) =>
              event.payload.transaction_id === payload.transaction_id
                ? { ...event, status: "error", error: readableError }
                : event,
            ),
          );
          setScenarioError(
            `Transaction ${stepIndex + 1} failed. The sequence was stopped: ${readableError}`,
          );
          break;
        }

        submitted += 1;
        setEvents((current) =>
          current.map((event) =>
            event.payload.transaction_id === payload.transaction_id
              ? { ...event, status: "success", result }
              : event,
          ),
        );
        try {
          await onSuccess(result);
        } catch (error) {
          refreshFailed = true;
          setScenarioError(
            `Transaction ${stepIndex + 1} succeeded, but the dashboard could not refresh: ${describeError(error)}`,
          );
          break;
        }
      }

      if (submitted === selectedScenario.steps.length && !refreshFailed) {
        setScenarioError(null);
      }
    } finally {
      setRunning(false);
      runLock.current = false;
    }
  };

  const successfulEvents = events.filter((event) => event.result);
  const latestResult = successfulEvents.at(-1)?.result;
  const completedCount = successfulEvents.length;

  return (
    <section className="panel simulator-panel" aria-label="Demo scenarios">
      <div className="simulator-heading">
        <div>
          <span className="eyebrow">LIVE API DEMONSTRATION</span>
          <h2>Demo Scenarios</h2>
          <p>Generated inputs go to the live pipeline; decisions below are returned by the backend.</p>
        </div>
        <span className="simulator-pipeline-tag">BACKEND DECISIONS</span>
      </div>

      <div className="simulator-actions">
        <button
          type="button"
          className="simulator-button"
          onClick={() => void runScenario("normal")}
          disabled={running}
        >
          Normal Transaction
        </button>
        <button
          type="button"
          className="simulator-button"
          onClick={() => void runScenario("behavioral")}
          disabled={running}
        >
          Behavioral Change
        </button>
        <button
          type="button"
          className="simulator-button"
          onClick={() => void runScenario("high-amount")}
          disabled={running}
        >
          High Amount
        </button>
        <button
          type="button"
          className="clear-button simulator-clear"
          onClick={() => {
            setScenario(null);
            setEvents([]);
            setScenarioError(null);
          }}
          disabled={running}
        >
          Clear Demo
        </button>
      </div>

      {scenario && (
        <div className="scenario-progress" aria-live="polite">
          <div className="scenario-progress-heading">
            <div>
              <span className="eyebrow">SCENARIO PROGRESS</span>
              <h3>{scenario.label}</h3>
            </div>
            <span className={`scenario-state ${running ? "is-running" : scenarioError ? "has-error" : "is-complete"}`}>
              {running ? "Running" : scenarioError ? "Stopped" : "Complete"}
            </span>
          </div>
          <div className="scenario-progress-meta">
            <span>Progress: {events.length} / {scenario.steps.length} submitted · {completedCount} successful</span>
            {running && <span className="scenario-running-indicator"><span className="button-spinner" /> Evaluating</span>}
          </div>
          <div className="scenario-events">
            {events.map((event) => (
              <article
                className={`scenario-event ${event.error ? "scenario-event-error" : ""}`}
                key={event.payload.transaction_id}
              >
                <div className="scenario-event-top">
                  <div>
                    <strong>Transaction {event.index}</strong>
                    <span className="scenario-event-id">#{event.payload.transaction_id}</span>
                  </div>
                  {event.status === "pending" && (
                    <span className="scenario-pending-label">Evaluating</span>
                  )}
                  {event.error && <span className="scenario-failed-label">Failed</span>}
                </div>

                {event.result ? (
                  <>
                    <div className="scenario-event-metrics">
                      <span>Amount <strong>{event.payload.transaction_amount.toLocaleString()}</strong></span>
                      <span>Risk score <strong>{event.result.risk_score} / 100</strong></span>
                      <span>Previous transactions <strong>{event.result.behavioral_context.prior_transaction_count}</strong></span>
                    </div>
                    <div className="scenario-backend-result">
                      <span>Backend Decision</span>
                      <div className="scenario-event-badges">
                        <StatusBadge value={event.result.risk_level} />
                        <StatusBadge value={event.result.decision} kind="decision" />
                      </div>
                    </div>
                    {event.result.reasons.length > 0 ? (
                      <ul className="scenario-reasons">
                        {event.result.reasons.map((reason, index) => (
                          <li key={`${event.payload.transaction_id}-${index}`}>{reason}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="scenario-no-reasons">No reasons were returned by the backend.</p>
                    )}
                  </>
                ) : (
                  <p className="scenario-event-error-message">{event.error}</p>
                )}
              </article>
            ))}
          </div>

          {scenarioError && (
            <div className="simulator-error" role="alert">{scenarioError}</div>
          )}

          {latestResult && (
            <div className="scenario-latest">
              <span className="eyebrow">LATEST BACKEND DECISION</span>
              <div>
                <strong>#{latestResult.transaction_id} · {latestResult.risk_score} / 100</strong>
                <span className="scenario-event-badges">
                  <StatusBadge value={latestResult.risk_level} />
                  <StatusBadge value={latestResult.decision} kind="decision" />
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
