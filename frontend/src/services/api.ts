export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";
export type Decision = "ALLOW" | "VERIFY" | "BLOCK";
export type StrategyMetadataValue = string | number | boolean | null;

export interface HealthResponse {
  status: string;
  service: string;
}

export interface MonitoringSummary {
  total_transactions: number;
  low_risk_count: number;
  medium_risk_count: number;
  high_risk_count: number;
  allow_count: number;
  verify_count: number;
  block_count: number;
  high_risk_rate: number;
  block_rate: number;
}

export interface BehavioralContext {
  prior_transaction_count: number;
  prior_transaction_count_1h: number;
  prior_transaction_count_24h: number;
  previous_transaction_amount: number | null;
  prior_amount_mean: number | null;
  amount_vs_prior_mean: number | null;
  amount_above_prior_mean: boolean;
  is_high_velocity_1h: boolean;
  is_high_velocity_24h: boolean;
}

export interface TransactionRecord {
  transaction_id: number;
  transaction_amount: number;
  transaction_time: number;
  card1: string | number | null;
  evaluated_at: string;
  risk_score: number;
  risk_level: RiskLevel;
  decision: Decision;
  reasons: string[];
  fraud_probability: number;
  strategy_name: string;
  model_version: string;
  metadata: Record<string, StrategyMetadataValue>;
  behavioral_context: BehavioralContext;
}

export interface TransactionFilters {
  risk_level?: RiskLevel;
  decision?: Decision;
}

export interface FraudCheckPayload {
  transaction_id: number;
  transaction_amount: number;
  transaction_time: number;
  card1?: string;
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
    readonly fieldErrors: string[] = [],
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options);
  } catch {
    throw new ApiError(
      "Backend unavailable. Make sure the FastAPI server is running.",
    );
  }
  if (!response.ok) {
    let body: { detail?: unknown } | null = null;
    try {
      body = (await response.json()) as { detail?: unknown };
    } catch {
      body = null;
    }
    if (response.status === 409) {
      throw new ApiError("Transaction already processed.", 409);
    }
    if (response.status === 422) {
      const details: unknown[] = Array.isArray(body?.detail)
        ? body.detail
        : [];
      const fieldErrors = details
        .map((item) => {
          if (
            typeof item === "object" &&
            item !== null &&
            "loc" in item &&
            "msg" in item &&
            Array.isArray((item as { loc: unknown }).loc)
          ) {
            const validation = item as { loc: unknown[]; msg: unknown };
            const field = validation.loc
              .filter((part): part is string => typeof part === "string")
              .at(-1);
            return field
              ? `${field}: ${String(validation.msg)}`
              : String(validation.msg);
          }
          return "";
        })
        .filter(Boolean);
      throw new ApiError(
        "Please check the transaction details.",
        422,
        fieldErrors,
      );
    }
    if (response.status === 400 || response.status === 404) {
      const detail =
        typeof body?.detail === "string" ? body.detail : undefined;
      throw new ApiError(
        detail ?? `Unable to process transaction (${response.status}).`,
        response.status,
      );
    }
    if (response.status >= 500) {
      throw new ApiError(
        "Unable to process transaction. Please check the backend connection.",
        response.status,
      );
    }
    throw new ApiError(`Backend request failed (${response.status}).`, response.status);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request("/api/v1/health");
}

export function getSummary(): Promise<MonitoringSummary> {
  return request("/api/v1/monitoring/summary");
}

export function getTransactions(
  filters: TransactionFilters = {},
): Promise<TransactionRecord[]> {
  const params = new URLSearchParams();
  if (filters.risk_level) params.set("risk_level", filters.risk_level);
  if (filters.decision) params.set("decision", filters.decision);
  params.set("limit", "100");
  return request(`/api/v1/transactions?${params.toString()}`);
}

export function getTransaction(id: number): Promise<TransactionRecord> {
  return request(`/api/v1/transactions/${encodeURIComponent(id)}`);
}

export function checkFraudTransaction(
  payload: FraudCheckPayload,
): Promise<TransactionRecord> {
  return request("/api/v1/fraud/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
