"""Strategy 1 chronological backtest.

Usage:
    python -m fraud_validation.backtest.run_strategy1 --data data/train_transaction.csv --out-dir reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_validation.strategies import rule_based as rb
from fraud_validation.strategies.rule_based import RuleBasedFraudStrategy

USECOLS = ["TransactionDT", "isFraud", "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain"]
DTYPES = {
    "TransactionDT": "int64", "isFraud": "int8", "TransactionAmt": "float32",
    "ProductCD": "category", "card4": "category", "card6": "category",
    "P_emaildomain": "category",
}


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=USECOLS, dtype=DTYPES)


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Sort by TransactionDT (stable); first train_frac rows = history, rest = evaluation."""
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    return df.iloc[:n_train], df.iloc[n_train:]


def compute_metrics(y_true: np.ndarray, flagged: np.ndarray) -> dict:
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(flagged).astype(bool)
    tp = int((y & p).sum()); tn = int((~y & ~p).sum())
    fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    div = lambda a, b: (a / b) if b else None
    precision = div(tp, tp + fp); recall = div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": div(fp, fp + tn),
        "false_negative_rate": div(fn, fn + tp),
        "fraud_detection_rate": recall,  # identical to recall (TP / actual frauds)
    }


def run(data_path: str, out_dir: str, min_support: int = 500, lift_threshold: float = 1.5, train_frac: float = 0.8) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    train, evaluation = chronological_split(df, train_frac)
    del df

    strat = RuleBasedFraudStrategy(min_support, lift_threshold).fit(train)  # history only
    strat.save_email_config(out / "strategy1_email_domain_rules.json")

    scored = strat.score_dataframe(evaluation)  # fixed strategy, no refit
    y = evaluation["isFraud"].to_numpy()
    level = scored["risk_level"].to_numpy()
    flagged = level != "LOW"  # primary: VERIFY or BLOCK counts as "flagged"
    primary = compute_metrics(y, flagged)
    block_only = compute_metrics(y, level == "HIGH")

    cm = pd.DataFrame(
        [[primary["TN"], primary["FP"]], [primary["FN"], primary["TP"]]],
        index=["actual_legit(0)", "actual_fraud(1)"],
        columns=["pred_not_flagged(LOW/ALLOW)", "pred_flagged(MEDIUM+HIGH)"],
    )
    cm.to_csv(out / "strategy1_confusion_matrix.csv")

    crosstab = pd.crosstab(pd.Series(y, name="isFraud"), pd.Series(level, name="risk_level"))
    crosstab = crosstab.reindex(columns=["LOW", "MEDIUM", "HIGH"], fill_value=0)
    t_tr, t_ev = train["TransactionDT"], evaluation["TransactionDT"]
    report = {
        "strategy": "Strategy 1: explainable rule-based scoring (not claimed to be optimal)",
        "split": {
            "method": "chronological by TransactionDT (stable sort), first 80% history / last 20% evaluation; no random split",
            "training_rows": int(len(train)), "training_TransactionDT_range": [int(t_tr.min()), int(t_tr.max())],
            "training_fraud_rate": float(train["isFraud"].mean()),
            "evaluation_rows": int(len(evaluation)), "evaluation_TransactionDT_range": [int(t_ev.min()), int(t_ev.max())],
            "evaluation_fraud_rate": float(evaluation["isFraud"].mean()),
            "boundary_TransactionDT_tie": bool(t_tr.max() == t_ev.min()),
        },
        "rules": {
            "ProductCD": rb.PRODUCT_POINTS,
            "TransactionAmt": {"< %s" % rb.AMOUNT_LOW: rb.AMOUNT_LOW_POINTS, "between (inclusive)": 0, "> %s" % rb.AMOUNT_HIGH: rb.AMOUNT_HIGH_POINTS},
            "card4": rb.CARD4_POINTS, "card6": rb.CARD6_POINTS,
            "email_domain_points": rb.EMAIL_POINTS,
            "max_score": rb.MAX_SCORE,
        },
        "thresholds": {
            "risk_levels": {"LOW": "0-39", "MEDIUM": "40-69", "HIGH": "70-80"},
            "decisions": rb.DECISIONS,
            "amount_cutoffs": [rb.AMOUNT_LOW, rb.AMOUNT_HIGH],
        },
        "learned_email_domain_configuration": {
            "min_support": min_support, "lift_threshold": lift_threshold,
            "overall_training_fraud_rate": strat.email_config["overall_fraud_rate"],
            "fraud_rate_cutoff": strat.email_config["risk_fraud_rate_cutoff"],
            "high_risk_domains": {c: strat.email_config["columns"][c]["high_risk_domains"] for c in rb.EMAIL_COLUMNS},
            "full_stats_file": "strategy1_email_domain_rules.json",
        },
        "evaluation": {
            "n_transactions": int(len(evaluation)),
            "predicted_HIGH_BLOCK": int((level == "HIGH").sum()),
            "predicted_MEDIUM_VERIFY": int((level == "MEDIUM").sum()),
            "predicted_LOW_ALLOW": int((level == "LOW").sum()),
            "positive_definition_primary": "flagged = MEDIUM or HIGH (VERIFY or BLOCK)",
            "metrics_primary_flagged_MEDIUM_plus": primary,
            "metrics_secondary_BLOCK_only": block_only,
            "confusion_matrix_primary": {"rows": list(cm.index), "cols": list(cm.columns), "values": cm.values.tolist()},
            "risk_level_by_actual": {str(k): {c: int(v) for c, v in r.items()} for k, r in crosstab.iterrows()},
            "note": "Accuracy intentionally not reported as a headline metric (3.5% fraud base rate).",
        },
    }
    (out / "strategy1_rule_based_metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_transaction.csv")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--min-support", type=int, default=500)
    ap.add_argument("--lift-threshold", type=float, default=1.5)
    a = ap.parse_args()
    rep = run(a.data, a.out_dir, a.min_support, a.lift_threshold)
    print(json.dumps(rep["evaluation"], indent=2))
