import { useCallback, useEffect, useRef, useState } from "react";
import {
  getHealth,
  getSummary,
  getTransaction,
  getTransactions,
  type Decision,
  type HealthResponse,
  type MonitoringSummary,
  type RiskLevel,
  type TransactionRecord,
} from "../services/api";
import { DistributionPanel } from "../components/DistributionPanel";
import { FraudCheckPanel } from "../components/FraudCheckPanel";
import { SummaryCards } from "../components/SummaryCards";
import { StatusBadge } from "../components/StatusBadge";
import { TransactionDetails } from "../components/TransactionDetails";
import { TransactionSimulator } from "../components/TransactionSimulator";
import { TransactionTable } from "../components/TransactionTable";

const REFRESH_INTERVAL_MS = 8000;

export function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [transactions, setTransactions] = useState<TransactionRecord[]>([]);
  const [riskFilter, setRiskFilter] = useState<RiskLevel | "">("");
  const [decisionFilter, setDecisionFilter] = useState<Decision | "">("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTransaction, setSelectedTransaction] =
    useState<TransactionRecord | null>(null);
  const [latestDecision, setLatestDecision] =
    useState<TransactionRecord | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [detailsError, setDetailsError] = useState<string | null>(null);
  const requestSequence = useRef(0);

  const refreshDashboard = useCallback(async (manual = false) => {
    const sequence = ++requestSequence.current;
    if (manual) setRefreshing(true);
    else setLoading(true);
    setError(null);

    const filters = {
      ...(riskFilter ? { risk_level: riskFilter } : {}),
      ...(decisionFilter ? { decision: decisionFilter } : {}),
    };
    const [healthResult, summaryResult, transactionsResult] =
      await Promise.allSettled([
        getHealth(),
        getSummary(),
        getTransactions(filters),
      ]);
    if (sequence !== requestSequence.current) return;

    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setHealth(null);
    if (summaryResult.status === "fulfilled") setSummary(summaryResult.value);
    if (transactionsResult.status === "fulfilled") {
      setTransactions(transactionsResult.value);
    }
    if (
      healthResult.status === "rejected" ||
      summaryResult.status === "rejected" ||
      transactionsResult.status === "rejected"
    ) {
      const rejection = [
        healthResult,
        summaryResult,
        transactionsResult,
      ].find((result) => result.status === "rejected");
      setError(
        rejection?.status === "rejected"
          ? rejection.reason instanceof Error
            ? rejection.reason.message
            : "Backend unavailable. Make sure the FastAPI server is running."
          : "Backend unavailable. Make sure the FastAPI server is running.",
      );
    }

    setLoading(false);
    setRefreshing(false);
  }, [riskFilter, decisionFilter]);

  useEffect(() => {
    void refreshDashboard();
  }, [refreshDashboard]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshDashboard(true);
    }, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [refreshDashboard]);

  const openTransaction = async (transactionId: number) => {
    setSelectedTransaction(null);
    setDetailsError(null);
    setDetailsLoading(true);
    try {
      const record = await getTransaction(transactionId);
      setSelectedTransaction(record);
    } catch (requestError) {
      setDetailsError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to load transaction details.",
      );
    } finally {
      setDetailsLoading(false);
    }
  };

  const handleSuccessfulDecision = async (record: TransactionRecord) => {
    setLatestDecision(record);
    await refreshDashboard(true);
  };

  const apiConnected = health?.status === "ok";
  const riskDistribution = summary
    ? [
        { label: "LOW", value: summary.low_risk_count, tone: "low" },
        { label: "MEDIUM", value: summary.medium_risk_count, tone: "medium" },
        { label: "HIGH", value: summary.high_risk_count, tone: "high" },
      ]
    : [];
  const decisionDistribution = summary
    ? [
        { label: "ALLOW", value: summary.allow_count, tone: "allow" },
        { label: "VERIFY", value: summary.verify_count, tone: "verify" },
        { label: "BLOCK", value: summary.block_count, tone: "block" },
      ]
    : [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Adaptive Fraud Detection home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M16 3.5 27 8v7.4c0 7-4.5 11.4-11 13.1C9.5 26.8 5 22.4 5 15.4V8l11-4.5Z" />
              <path d="m11.5 16 3 3 6-6" />
            </svg>
          </span>
          <span className="brand-name">Aegis<span>Risk</span></span>
        </a>
        <div className="topbar-right">
          <span className="environment-label">DEMO ENVIRONMENT</span>
          <span className={`connection-status ${apiConnected ? "online" : "offline"}`}>
            <span className="connection-dot" />
            {apiConnected ? "API Connected" : "API Disconnected"}
          </span>
        </div>
      </header>

      <div className="dashboard-content">
        <section className="page-heading">
          <div>
            <div className="eyebrow">FRAUD OPERATIONS</div>
            <h1>Adaptive Fraud Detection</h1>
            <p>Real-Time Risk Monitoring</p>
          </div>
          <div className="heading-actions">
            <span className="refresh-caption">Auto-refresh · 8 sec</span>
            <button
              className="refresh-button"
              type="button"
              onClick={() => void refreshDashboard(true)}
              disabled={refreshing}
            >
              <span aria-hidden="true" className={refreshing ? "spin" : ""}>↻</span>
              {refreshing ? "Refreshing" : "Refresh"}
            </button>
          </div>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-mark">!</span>
            <div>
              <strong>Backend unavailable</strong>
              <p>{error.includes("Make sure") ? error : `${error} Make sure the FastAPI server is running.`}</p>
            </div>
          </div>
        )}

        <section aria-label="Transaction summary">
          <SummaryCards summary={summary} loading={loading} />
        </section>

        <section className="distribution-grid" aria-label="Transaction distributions">
          {summary ? (
            <>
              <DistributionPanel
                title="Risk distribution"
                subtitle="Evaluated transactions by risk level"
                values={riskDistribution}
                empty={summary.total_transactions === 0}
              />
              <DistributionPanel
                title="Decision distribution"
                subtitle="Operational outcomes"
                values={decisionDistribution}
                empty={summary.total_transactions === 0}
              />
            </>
          ) : loading ? (
            <>
              <div className="panel chart-loading"><div className="skeleton skeleton-line" /><div className="skeleton skeleton-chart" /></div>
              <div className="panel chart-loading"><div className="skeleton skeleton-line" /><div className="skeleton skeleton-chart" /></div>
            </>
          ) : (
            <>
              <div className="panel unavailable-panel">Monitoring data unavailable.</div>
              <div className="panel unavailable-panel">Monitoring data unavailable.</div>
            </>
          )}
        </section>

        <TransactionSimulator onSuccess={handleSuccessfulDecision} />

        <section className="panel decision-flow-panel" aria-labelledby="decision-flow-title">
          <div className="decision-flow-heading">
            <div>
              <span className="eyebrow">EXPLAINABLE PIPELINE</span>
              <h2 id="decision-flow-title">How the System Decides</h2>
              <p>Each stage contributes to the backend assessment shown in the dashboard.</p>
            </div>
          </div>
          <ol className="decision-flow-steps">
            <li><span>01</span><strong>Transaction received</strong></li>
            <li><span>02</span><strong>Previous behavior checked</strong></li>
            <li><span>03</span><strong>Fraud strategy evaluates</strong></li>
            <li><span>04</span><strong>Risk score generated</strong></li>
            <li><span>05</span><strong>Risk level assigned</strong></li>
            <li><span>06</span><strong>Action selected</strong></li>
          </ol>
          <div className="decision-flow-outcomes" aria-label="Operational risk and action mapping">
            <div><StatusBadge value="LOW" /><span aria-hidden="true">→</span><StatusBadge value="ALLOW" kind="decision" /></div>
            <div><StatusBadge value="MEDIUM" /><span aria-hidden="true">→</span><StatusBadge value="VERIFY" kind="decision" /></div>
            <div><StatusBadge value="HIGH" /><span aria-hidden="true">→</span><StatusBadge value="BLOCK" kind="decision" /></div>
          </div>
          <p className="decision-flow-note">The values and explanation for each transaction come from the backend response.</p>
        </section>

        <FraudCheckPanel
          latestRecord={latestDecision}
          onSuccess={handleSuccessfulDecision}
        />

        <section className="panel transaction-panel">
          <div className="panel-heading transaction-heading">
            <div>
              <h2>Recent transactions</h2>
              <p>Click a transaction to inspect its risk assessment</p>
            </div>
            <div className="filters">
              <label>
                <span>Risk level</span>
                <select
                  value={riskFilter}
                  onChange={(event) =>
                    setRiskFilter(event.target.value as RiskLevel | "")
                  }
                >
                  <option value="">All levels</option>
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                </select>
              </label>
              <label>
                <span>Decision</span>
                <select
                  value={decisionFilter}
                  onChange={(event) =>
                    setDecisionFilter(event.target.value as Decision | "")
                  }
                >
                  <option value="">All decisions</option>
                  <option value="ALLOW">ALLOW</option>
                  <option value="VERIFY">VERIFY</option>
                  <option value="BLOCK">BLOCK</option>
                </select>
              </label>
            </div>
          </div>
          <TransactionTable
            transactions={transactions}
            loading={loading}
            error={Boolean(error)}
            onSelect={(id) => void openTransaction(id)}
          />
          {transactions.length > 0 && (
            <div className="table-footer">
              Showing {transactions.length} recent transaction{transactions.length === 1 ? "" : "s"}
            </div>
          )}
        </section>

        <footer className="page-footer">
          <span>Adaptive Fraud Detection · Mock strategy demonstration</span>
          <span>Transaction times shown as relative dataset seconds</span>
        </footer>
      </div>

      <TransactionDetails
        transaction={selectedTransaction}
        loading={detailsLoading}
        error={detailsError}
        onClose={() => {
          setSelectedTransaction(null);
          setDetailsError(null);
        }}
      />
    </main>
  );
}
