import type { MonitoringSummary } from "../services/api";

interface SummaryCardsProps {
  summary: MonitoringSummary | null;
  loading: boolean;
}

const cards = [
  {
    label: "Total transactions",
    field: "total_transactions",
    icon: "↗",
    tone: "blue",
  },
  {
    label: "High risk",
    field: "high_risk_count",
    icon: "!",
    tone: "red",
  },
  {
    label: "Verification required",
    field: "verify_count",
    icon: "◷",
    tone: "amber",
  },
  {
    label: "Blocked",
    field: "block_count",
    icon: "⊘",
    tone: "purple",
  },
] as const;

export function SummaryCards({ summary, loading }: SummaryCardsProps) {
  return (
    <div className="summary-grid">
      {cards.map((card) => (
        <article className="summary-card panel" key={card.field}>
          <div className="summary-card-top">
            <span className="summary-label">{card.label}</span>
            <span className={`summary-icon ${card.tone}`}>{card.icon}</span>
          </div>
          {loading && !summary ? (
            <div className="skeleton skeleton-number" />
          ) : (
            <strong className="summary-value">
              {summary?.[card.field].toLocaleString() ?? "—"}
            </strong>
          )}
          {card.field === "high_risk_count" && summary && (
            <span className="summary-footnote">
              {(summary.high_risk_rate * 100).toFixed(1)}% of evaluated
            </span>
          )}
          {card.field === "block_count" && summary && (
            <span className="summary-footnote">
              {(summary.block_rate * 100).toFixed(1)}% of evaluated
            </span>
          )}
        </article>
      ))}
    </div>
  );
}
