import type { Decision, RiskLevel } from "../services/api";

type StatusValue = RiskLevel | Decision;

interface StatusBadgeProps {
  value: StatusValue;
  kind?: "risk" | "decision";
}

export function StatusBadge({
  value,
  kind = "risk",
}: StatusBadgeProps) {
  return (
    <span className={`status-badge ${kind}-${value.toLowerCase()}`}>
      <span className="status-dot" />
      {value}
    </span>
  );
}
