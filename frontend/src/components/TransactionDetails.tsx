import type { TransactionRecord } from "../services/api";
import { formatEvaluatedAt, formatRelativeTime } from "./TransactionTable";
import { StatusBadge } from "./StatusBadge";

interface TransactionDetailsProps {
  transaction: TransactionRecord | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}

function displayAmount(value: number | null): string {
  return value === null
    ? "No prior history"
    : value.toLocaleString(undefined, {
        maximumFractionDigits: 2,
        minimumFractionDigits: 2,
      });
}

export function TransactionDetails({
  transaction,
  loading,
  error,
  onClose,
}: TransactionDetailsProps) {
  if (!transaction && !loading && !error) return null;

  const context = transaction?.behavioral_context;
  return (
    <div className="drawer-backdrop" onMouseDown={onClose}>
      <aside
        className="details-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="details-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="drawer-header">
          <div>
            <span className="eyebrow">TRANSACTION INVESTIGATION</span>
            <h2 id="details-title">
              {transaction ? `#${transaction.transaction_id}` : "Details"}
            </h2>
          </div>
          <button
            className="icon-button"
            type="button"
            onClick={onClose}
            aria-label="Close transaction details"
          >
            ×
          </button>
        </div>

        {loading && <p className="drawer-state">Loading transaction details...</p>}
        {error && <p className="drawer-error">{error}</p>}

        {transaction && (
          <div className="drawer-body">
            <section className="detail-score-card">
              <div className="decision-summary-heading">
                <div>
                  <span className="eyebrow">DECISION SUMMARY</span>
                  <div className={`detail-score score-${transaction.risk_level.toLowerCase()}`}>
                    {transaction.risk_score}
                    <span> / 100</span>
                  </div>
                </div>
                <div className="detail-score-meta">
                  <StatusBadge value={transaction.risk_level} />
                  <StatusBadge value={transaction.decision} kind="decision" />
                </div>
              </div>
              <div
                className="risk-scale"
                role="img"
                aria-label={`Risk score ${transaction.risk_score} out of 100`}
              >
                <div className="risk-scale-track">
                  <span className="risk-scale-low" />
                  <span className="risk-scale-medium" />
                  <span className="risk-scale-high" />
                  <span
                    className={`risk-scale-marker score-${transaction.risk_level.toLowerCase()}`}
                    style={{ left: `${transaction.risk_score}%` }}
                  />
                </div>
                <div className="risk-scale-labels">
                  <span>LOW</span><span>MEDIUM</span><span>HIGH</span>
                </div>
              </div>
              <div className="detail-probability">
                <span>Fraud probability</span>
                <strong>{(transaction.fraud_probability * 100).toFixed(1)}%</strong>
              </div>
            </section>

            <section className="detail-section">
              <h3>Transaction Information</h3>
              <dl className="detail-grid">
                <div><dt>Transaction ID</dt><dd>#{transaction.transaction_id}</dd></div>
                <div><dt>Transaction amount</dt><dd>{displayAmount(transaction.transaction_amount)}</dd></div>
                <div><dt>Transaction time</dt><dd>{formatRelativeTime(transaction.transaction_time)}</dd></div>
                <div><dt>Card ID</dt><dd>{transaction.card1 ?? "Not provided"}</dd></div>
                <div><dt>Evaluated at</dt><dd>{formatEvaluatedAt(transaction.evaluated_at)}</dd></div>
              </dl>
            </section>

            {context && (
              <section className="detail-section">
                <h3>Behavioral Evidence</h3>
                <p className="detail-section-caption">Observed behavioral context available to the backend for this transaction.</p>
                <dl className="detail-grid behavioral-grid">
                  <div><dt>Previous transactions</dt><dd>{context.prior_transaction_count}</dd></div>
                  <div><dt>Previous transaction amount</dt><dd>{displayAmount(context.previous_transaction_amount)}</dd></div>
                  <div><dt>Prior transactions in 1 hour</dt><dd>{context.prior_transaction_count_1h}</dd></div>
                  <div><dt>Prior transactions in 24 hours</dt><dd>{context.prior_transaction_count_24h}</dd></div>
                  <div><dt>Prior amount mean</dt><dd>{displayAmount(context.prior_amount_mean)}</dd></div>
                  <div><dt>Amount vs historical average</dt><dd>{context.amount_vs_prior_mean === null ? "No prior history" : `${context.amount_vs_prior_mean.toFixed(2)}×`}</dd></div>
                  <div><dt>Amount above prior mean</dt><dd>{context.amount_above_prior_mean ? "Yes" : "No"}</dd></div>
                  <div><dt>High velocity (1 hour)</dt><dd>{context.is_high_velocity_1h ? "Yes" : "No"}</dd></div>
                  <div><dt>High velocity (24 hours)</dt><dd>{context.is_high_velocity_24h ? "Yes" : "No"}</dd></div>
                </dl>
              </section>
            )}

            <section className="detail-section detail-reasons-section">
              <h3>Why This Decision?</h3>
              {transaction.reasons.length > 0 ? (
                <ul className="reason-list detail-reason-list">
                  {transaction.reasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              ) : (
                <p className="muted-cell">No explanation signals were returned for this transaction.</p>
              )}
            </section>

            <section className="detail-section">
              <h3>Strategy Information</h3>
              <dl className="detail-grid strategy-grid">
                <div><dt>Strategy name</dt><dd>{transaction.strategy_name}</dd></div>
                <div><dt>Model version</dt><dd>{transaction.model_version}</dd></div>
                {Object.entries(transaction.metadata).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key.replaceAll("_", " ")}</dt>
                    <dd>{value === null ? "Not provided" : String(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>
        )}
      </aside>
    </div>
  );
}
