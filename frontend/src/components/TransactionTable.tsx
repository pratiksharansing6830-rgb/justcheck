import type { TransactionRecord } from "../services/api";
import { StatusBadge } from "./StatusBadge";

interface TransactionTableProps {
  transactions: TransactionRecord[];
  loading: boolean;
  error: boolean;
  onSelect: (transactionId: number) => void;
}

function formatAmount(amount: number): string {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(amount);
}

export function formatRelativeTime(seconds: number): string {
  return `T+${new Intl.NumberFormat().format(seconds)} sec`;
}

export function formatEvaluatedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function TransactionTable({
  transactions,
  loading,
  error,
  onSelect,
}: TransactionTableProps) {
  if (loading && transactions.length === 0) {
    return <div className="table-state">Loading transactions...</div>;
  }
  if (transactions.length === 0) {
    return (
      <div className="table-state">
        {error ? "Transactions unavailable." : "No transactions recorded yet."}
      </div>
    );
  }

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Transaction ID</th>
            <th>Amount</th>
            <th>Risk score</th>
            <th>Risk level</th>
            <th>Decision</th>
            <th>Transaction time</th>
            <th>Signals</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction) => (
            <tr
              className="transaction-row"
              key={transaction.transaction_id}
              onClick={() => onSelect(transaction.transaction_id)}
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(transaction.transaction_id);
                }
              }}
              aria-label={`View transaction ${transaction.transaction_id}`}
            >
              <td className="transaction-id">
                #{transaction.transaction_id}
              </td>
              <td>{formatAmount(transaction.transaction_amount)}</td>
              <td>
                <span
                  className={`score-value score-${transaction.risk_level.toLowerCase()}`}
                >
                  {transaction.risk_score}
                </span>
                <span className="score-total"> / 100</span>
              </td>
              <td>
                <StatusBadge value={transaction.risk_level} />
              </td>
              <td>
                <StatusBadge value={transaction.decision} kind="decision" />
              </td>
              <td className="muted-cell">
                {formatRelativeTime(transaction.transaction_time)}
              </td>
              <td>
                <span className="signal-count">
                  {transaction.reasons.length} {transaction.reasons.length === 1 ? "signal" : "signals"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
