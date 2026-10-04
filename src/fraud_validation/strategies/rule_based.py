"""Strategy 1: explainable rule-based fraud scoring.

Only the email-domain rule is learned from data (via ``fit``); every other
rule is a fixed, hand-specified threshold. Learned parameters are stored in a
JSON-serialisable config so decisions are reproducible.

Email rule: P_emaildomain high-risk -> +20 (R_emaildomain is NOT used).
Behavioral rules use precomputed, leakage-safe history features when present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

# ---- fixed rule configuration -------------------------------------------------
AMOUNT_LOW = 35.95
AMOUNT_HIGH = 159.95
AMOUNT_LOW_POINTS = 10
AMOUNT_HIGH_POINTS = 15
PRODUCT_POINTS = {"C": 25, "S": 10, "H": 10, "R": 0, "W": 0}
CARD4_POINTS = {"discover": 10}
CARD6_POINTS = {"credit": 10}
EMAIL_POINTS = {"P_emaildomain": 20}
EMAIL_COLUMNS = tuple(EMAIL_POINTS)
VELOCITY_1H_THRESHOLD = 3
VELOCITY_24H_THRESHOLD = 10
AMOUNT_DEVIATION_THRESHOLD = 3
VELOCITY_1H_POINTS = 15
VELOCITY_24H_POINTS = 15
AMOUNT_DEVIATION_POINTS = 20
NEW_PROFILE_POINTS = 5

LOW_MAX = 39
MEDIUM_MAX = 69
MAX_SCORE = 135
DECISIONS = {"LOW": "ALLOW", "MEDIUM": "VERIFY", "HIGH": "BLOCK"}

assert (
    max(PRODUCT_POINTS.values())
    + AMOUNT_HIGH_POINTS
    + sum(EMAIL_POINTS.values())
    + max(CARD4_POINTS.values())
    + max(CARD6_POINTS.values())
    + VELOCITY_1H_POINTS
    + VELOCITY_24H_POINTS
    + AMOUNT_DEVIATION_POINTS
    + NEW_PROFILE_POINTS
    == MAX_SCORE
)

REQUIRED_COLUMNS = ["TransactionAmt", "ProductCD", "card4", "card6", *EMAIL_COLUMNS]


def risk_level_for(score: int | float) -> str:
    if score <= LOW_MAX:
        return "LOW"
    if score <= MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def decision_for(level: str) -> str:
    return DECISIONS[level]


def _norm(df: pd.DataFrame, col: str, upper: bool = False) -> pd.Series:
    """Normalised string series; missing column or value -> <NA>."""
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    s = df[col].astype("string").str.strip()
    s = s.str.upper() if upper else s.str.lower()
    return s.replace("", pd.NA)


def _points_from_map(norm: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    out = np.zeros(len(norm), dtype=np.int64)
    for key, pts in mapping.items():
        mask = (norm == key).fillna(False).to_numpy(dtype=bool)
        out[mask] = pts
    return out


class RuleBasedFraudStrategy:
    """Explainable additive rule scorer with a learned email-domain rule."""

    def __init__(self, min_support: int = 500, lift_threshold: float = 1.5):
        self.min_support = int(min_support)
        self.lift_threshold = float(lift_threshold)
        self.high_risk_domains: dict[str, set[str]] = {c: set() for c in EMAIL_COLUMNS}
        self.email_config: dict[str, Any] | None = None

    # ---- learning ---------------------------------------------------------
    def fit(self, training_data: pd.DataFrame) -> "RuleBasedFraudStrategy":
        """Learn high-risk email domains from ``training_data`` ONLY.

        A domain is high-risk if support >= min_support and
        fraud_rate >= lift_threshold * overall training fraud rate.
        """
        if "isFraud" not in training_data.columns:
            raise ValueError("training_data must contain 'isFraud'")
        y = pd.to_numeric(training_data["isFraud"], errors="coerce")
        valid = y.notna()
        overall = float(y[valid].mean())
        cfg: dict[str, Any] = {
            "min_support": self.min_support,
            "lift_threshold": self.lift_threshold,
            "high_risk_rule": "support >= min_support AND fraud_rate >= lift_threshold * overall_fraud_rate",
            "fit_rows": int(valid.sum()),
            "overall_fraud_rate": overall,
            "risk_fraud_rate_cutoff": self.lift_threshold * overall,
            "points": dict(EMAIL_POINTS),
            "columns": {},
        }
        for col in EMAIL_COLUMNS:
            norm = _norm(training_data, col)
            tmp = pd.DataFrame({"d": norm[valid], "y": y[valid]}).dropna(subset=["d"])
            g = tmp.groupby("d", observed=True)["y"].agg(["count", "sum"])
            kept = g[g["count"] >= self.min_support]
            domains: dict[str, Any] = {}
            high: list[str] = []
            for dom, row in kept.iterrows():
                rate = float(row["sum"] / row["count"])
                is_high = rate >= self.lift_threshold * overall
                domains[str(dom)] = {
                    "n": int(row["count"]),
                    "frauds": int(row["sum"]),
                    "fraud_rate": rate,
                    "lift": rate / overall if overall > 0 else None,
                    "high_risk": bool(is_high),
                }
                if is_high:
                    high.append(str(dom))
            cfg["columns"][col] = {
                "n_distinct_domains": int(len(g)),
                "n_domains_meeting_support": int(len(kept)),
                "n_domains_ignored_low_support": int(len(g) - len(kept)),
                "high_risk_domains": sorted(high),
                "domain_stats": domains,
            }
            self.high_risk_domains[col] = set(high)
        self.email_config = cfg
        return self

    # ---- persistence ------------------------------------------------------
    def save_email_config(self, path: str | Path) -> None:
        if self.email_config is None:
            raise RuntimeError("call fit() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.email_config, indent=2, sort_keys=True))

    @classmethod
    def from_email_config(cls, config: Mapping[str, Any] | str | Path) -> "RuleBasedFraudStrategy":
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        obj = cls(config["min_support"], config["lift_threshold"])
        obj.email_config = dict(config)
        for col in EMAIL_COLUMNS:
            obj.high_risk_domains[col] = set(config["columns"][col]["high_risk_domains"])
        return obj

    # ---- scoring ----------------------------------------------------------
    def _components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce") if "TransactionAmt" in df else pd.Series(np.nan, index=df.index)
        amt = amt.to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):  # NaN comparisons -> False -> 0 points
            amount_pts = np.where(amt < AMOUNT_LOW, AMOUNT_LOW_POINTS, np.where(amt > AMOUNT_HIGH, AMOUNT_HIGH_POINTS, 0))
        comps = {
            "product": _points_from_map(_norm(df, "ProductCD", upper=True), PRODUCT_POINTS),
            "amount": amount_pts.astype(np.int64),
            "card4": _points_from_map(_norm(df, "card4"), CARD4_POINTS),
            "card6": _points_from_map(_norm(df, "card6"), CARD6_POINTS),
        }
        for col in EMAIL_COLUMNS:
            hr = list(self.high_risk_domains[col])
            mask = _norm(df, col).isin(hr).to_numpy(dtype=bool) if hr else np.zeros(len(df), dtype=bool)
            comps[col] = np.where(mask, EMAIL_POINTS[col], 0).astype(np.int64)

        prior_count = self._numeric_column(df, "prior_transaction_count")
        velocity_1h = self._numeric_column(df, "prior_transaction_count_1h")
        velocity_24h = self._numeric_column(df, "prior_transaction_count_24h")
        amount_deviation = self._numeric_column(df, "amount_vs_prior_mean")
        comps["behavior_velocity_1h"] = np.where(
            velocity_1h >= VELOCITY_1H_THRESHOLD, VELOCITY_1H_POINTS, 0
        ).astype(np.int64)
        comps["behavior_velocity_24h"] = np.where(
            velocity_24h >= VELOCITY_24H_THRESHOLD, VELOCITY_24H_POINTS, 0
        ).astype(np.int64)
        comps["behavior_amount_deviation"] = np.where(
            (prior_count > 0)
            & (amount_deviation >= AMOUNT_DEVIATION_THRESHOLD),
            AMOUNT_DEVIATION_POINTS,
            0,
        ).astype(np.int64)
        if "prior_transaction_count" in df.columns:
            new_profile = df["prior_transaction_count"].notna().to_numpy() & (
                prior_count == 0
            )
        else:
            new_profile = np.zeros(len(df), dtype=bool)
        comps["behavior_new_profile"] = np.where(
            new_profile, NEW_PROFILE_POINTS, 0
        ).astype(np.int64)
        return comps

    @staticmethod
    def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
        if column not in df.columns:
            return np.zeros(len(df), dtype=float)
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def score_dataframe(self, df: pd.DataFrame, include_reasons: bool = False) -> pd.DataFrame:
        """Vectorised scoring. Returns risk_score, risk_level, decision (+ reasons)."""
        comps = self._components(df)
        score = sum(comps.values())
        score = np.asarray(score, dtype=np.int64)
        level = np.where(score <= LOW_MAX, "LOW", np.where(score <= MEDIUM_MAX, "MEDIUM", "HIGH"))
        out = pd.DataFrame(
            {"risk_score": score, "risk_level": level, "decision": pd.Series(level, index=df.index).map(DECISIONS).to_numpy()},
            index=df.index,
        )
        if include_reasons:
            out["reasons"] = self._reasons(df, comps)
        return out

    def _reasons(self, df: pd.DataFrame, comps: dict[str, np.ndarray]) -> list[list[str]]:
        prod = _norm(df, "ProductCD", upper=True).to_numpy(dtype=object)
        c4 = _norm(df, "card4").to_numpy(dtype=object)
        c6 = _norm(df, "card6").to_numpy(dtype=object)
        emails = {c: _norm(df, c).to_numpy(dtype=object) for c in EMAIL_COLUMNS}
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").to_numpy(dtype=float) if "TransactionAmt" in df else np.full(len(df), np.nan)
        res: list[list[str]] = []
        for i in range(len(df)):
            r: list[str] = []
            if comps["product"][i]:
                r.append(f"ProductCD={prod[i]} (+{comps['product'][i]})")
            if comps["amount"][i]:
                kind = "High" if amt[i] > AMOUNT_HIGH else "Low"
                r.append(f"{kind} transaction amount (+{comps['amount'][i]})")
            for col in EMAIL_COLUMNS:
                if comps[col][i]:
                    r.append(f"High-risk email domain: {col}={emails[col][i]} (+{comps[col][i]})")
            if comps["card4"][i]:
                r.append(f"card4={c4[i]} (+{comps['card4'][i]})")
            if comps["card6"][i]:
                r.append(f"card6={c6[i]} (+{comps['card6'][i]})")
            if comps["behavior_velocity_1h"][i]:
                r.append("high transaction velocity in the previous hour")
            if comps["behavior_velocity_24h"][i]:
                r.append("high transaction velocity in the previous 24 hours")
            if comps["behavior_amount_deviation"][i]:
                r.append(
                    "transaction amount is significantly above the card's prior average"
                )
            if comps["behavior_new_profile"][i]:
                r.append("no previous transaction history available for this card")
            res.append(r)
        return res

    def score(self, transaction: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        """Score one transaction -> {risk_score, risk_level, decision, reasons}."""
        row = dict(transaction)
        res = self.score_dataframe(pd.DataFrame([row]), include_reasons=True).iloc[0]
        return {
            "risk_score": int(res["risk_score"]),
            "risk_level": str(res["risk_level"]),
            "decision": str(res["decision"]),
            "reasons": list(res["reasons"]),
        }

    def predict(self, transaction: Mapping[str, Any] | pd.Series) -> str:
        """Decision only: ALLOW / VERIFY / BLOCK."""
        return self.score(transaction)["decision"]
