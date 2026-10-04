import { useState, type FormEvent } from "react";
import {
  ApiError,
  checkFraudTransaction,
  type FraudCheckPayload,
  type TransactionRecord,
} from "../services/api";
import { StatusBadge } from "./StatusBadge";

interface FraudCheckPanelProps {
  latestRecord: TransactionRecord | null;
  onSuccess: (record: TransactionRecord) => void | Promise<void>;
}

interface FormValues {
  transactionId: string;
  transactionAmount: string;
  transactionTime: string;
  cardId: string;
}

const EMPTY_FORM: FormValues = {
  transactionId: "",
  transactionAmount: "",
  transactionTime: "",
  cardId: "",
};

function formatAmount(value: number | null): string {
  return value === null
    ? "No prior history"
    : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatRatio(value: number | null): string {
  return value === null ? "No prior history" : `${value.toFixed(2)}×`;
}

export function FraudCheckPanel({
  latestRecord,
  onSuccess,
}: FraudCheckPanelProps) {
  const [values, setValues] = useState<FormValues>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const updateValue = (field: keyof FormValues, value: string) => {
    setValues((current) => ({ ...current, [field]: value }));
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const payload: FraudCheckPayload = {
      transaction_id: Number(values.transactionId),
      transaction_amount: Number(values.transactionAmount),
      transaction_time: Number(values.transactionTime),
      ...(values.cardId.trim() ? { card1: values.cardId.trim() } : {}),
    };

    setSubmitting(true);
    try {
      const result = await checkFraudTransaction(payload);
      await onSuccess(result);
    } catch (requestError) {
      if (requestError instanceof ApiError) {
        setError(requestError);
      } else {
        setError(
          new ApiError(
            "Unable to process transaction. Please check the backend connection.",
          ),
        );
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className="fraud-demo-grid" aria-label="Live fraud check">
      <div className="panel fraud-form-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">LIVE EVALUATION</span>
            <h2>Run Fraud Check</h2>
            <p>Submit a transaction to the risk pipeline</p>
          </div>
          <span className="demo-tag">DEMO</span>
        </div>

        <form className="fraud-form" onSubmit={submit}>
          <div className="form-grid">
            <label className="form-field">
              <span>Transaction ID <b>*</b></span>
              <input
                type="number"
                inputMode="numeric"
                min="1"
                step="1"
                required
                placeholder="812341"
                value={values.transactionId}
                onChange={(event) => updateValue("transactionId", event.target.value)}
              />
            </label>
            <label className="form-field">
              <span>Transaction Amount <b>*</b></span>
              <input
                type="number"
                min="0"
                step="any"
                required
                placeholder="2500"
                value={values.transactionAmount}
                onChange={(event) => updateValue("transactionAmount", event.target.value)}
              />
            </label>
            <label className="form-field">
              <span>Transaction Time (relative) <b>*</b></span>
              <input
                type="number"
                min="0"
                step="any"
                required
                placeholder="1200"
                value={values.transactionTime}
                onChange={(event) => updateValue("transactionTime", event.target.value)}
              />
              <small>Relative time in seconds; not a clock timestamp.</small>
            </label>
            <label className="form-field">
              <span>Card ID <em>Optional</em></span>
              <input
                type="text"
                placeholder="card-demo-001"
                value={values.cardId}
                onChange={(event) => updateValue("cardId", event.target.value)}
              />
            </label>
          </div>

          {error && (
            <div className="form-error" role="alert">
              <strong>{error.message}</strong>
              {error.fieldErrors.length > 0 && (
                <ul>
                  {error.fieldErrors.map((fieldError) => (
                    <li key={fieldError}>{fieldError}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <div className="form-actions">
            <button
              className="check-button"
              type="submit"
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <span className="button-spinner" aria-hidden="true" />
                  Checking...
                </>
              ) : (
                "Check Transaction"
              )}
            </button>
            <button
              className="clear-button"
              type="button"
              onClick={() => {
                setValues(EMPTY_FORM);
                setError(null);
              }}
              disabled={submitting}
            >
              Clear
            </button>
          </div>
        </form>
      </div>

      <LatestDecision record={latestRecord} />
    </section>
  );
}

function LatestDecision({ record }: { record: TransactionRecord | null }) {
  return (
    <section className="panel latest-decision-panel" aria-live="polite">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MOST RECENT EVALUATION</span>
          <h2>Latest Fraud Decision</h2>
        </div>
        {record && <span className="record-id">#{record.transaction_id}</span>}
      </div>

      {!record ? (
        <div className="latest-empty">
          <span className="latest-empty-icon" aria-hidden="true">↳</span>
          <p>Submit a transaction to see its backend risk assessment.</p>
        </div>
      ) : (
        <>
          <div className="latest-score-row">
            <div>
              <span className="summary-label">Risk score</span>
              <strong className={`latest-score score-${record.risk_level.toLowerCase()}`}>
                {record.risk_score}<span> / 100</span>
              </strong>
            </div>
            <div className="latest-badges">
              <StatusBadge value={record.risk_level} />
              <StatusBadge value={record.decision} kind="decision" />
            </div>
          </div>
          <div className="latest-meta">
            <span>Fraud probability</span>
            <strong>{(record.fraud_probability * 100).toFixed(1)}%</strong>
          </div>
          <div className="latest-reasons">
            <h3>Reasons from strategy</h3>
            {record.reasons.length > 0 ? (
              <ul className="reason-list">
                {record.reasons.map((reason) => <li key={reason}>{reason}</li>)}
              </ul>
            ) : (
              <p className="latest-no-reasons">No reasons returned.</p>
            )}
          </div>
          <div className="latest-context">
            <h3>Behavioral context</h3>
            <div className="context-chip-grid">
              <ContextValue label="Prior transactions" value={record.behavioral_context.prior_transaction_count} />
              <ContextValue label="Previous amount" value={formatAmount(record.behavioral_context.previous_transaction_amount)} />
              <ContextValue label="Prior mean" value={formatAmount(record.behavioral_context.prior_amount_mean)} />
              <ContextValue label="Amount / mean" value={formatRatio(record.behavioral_context.amount_vs_prior_mean)} />
              <ContextValue label="Last 1 hour" value={record.behavioral_context.prior_transaction_count_1h} />
              <ContextValue label="Last 24 hours" value={record.behavioral_context.prior_transaction_count_24h} />
              <ContextValue label="High velocity 1h" value={record.behavioral_context.is_high_velocity_1h ? "Yes" : "No"} />
              <ContextValue label="High velocity 24h" value={record.behavioral_context.is_high_velocity_24h ? "Yes" : "No"} />
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function ContextValue({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="context-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
