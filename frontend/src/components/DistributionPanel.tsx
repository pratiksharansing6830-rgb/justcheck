interface DistributionPanelProps {
  title: string;
  subtitle: string;
  values: { label: string; value: number; tone: string }[];
  empty: boolean;
}

export function DistributionPanel({
  title,
  subtitle,
  values,
  empty,
}: DistributionPanelProps) {
  const total = values.reduce((sum, entry) => sum + entry.value, 0);

  return (
    <section className="panel distribution-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span className="panel-total">{total.toLocaleString()}</span>
      </div>
      {empty ? (
        <div className="chart-empty">No transactions yet</div>
      ) : (
        <div className="distribution-content">
          <div className="bar-list">
            {values.map((entry) => {
              const percentage = (entry.value / total) * 100;
              return (
                <div className="bar-row" key={entry.label}>
                  <div className="bar-label">
                    <span className={`legend-dot ${entry.tone}`} />
                    <span>{entry.label}</span>
                    <strong>{entry.value.toLocaleString()}</strong>
                  </div>
                  <div
                    className="bar-track"
                    role="img"
                    aria-label={`${entry.label}: ${entry.value} (${percentage.toFixed(1)}%)`}
                  >
                    <span
                      className={`bar-fill ${entry.tone}`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
