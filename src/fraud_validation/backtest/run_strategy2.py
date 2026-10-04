"""Strategy 2 (XGBoost) chronological backtest.

Usage:
    python -m fraud_validation.backtest.run_strategy2 --data data/train_transaction.csv --out-dir reports
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_validation.backtest.run_strategy1 import chronological_split, compute_metrics
from fraud_validation.data.loader import load_csv
from fraud_validation.data.validation import validate_dataframe
from fraud_validation.strategies.xgboost_strategy import (
    CATEGORICAL_FEATURES, FEATURES, ID_COLUMN, LABEL, NUMERIC_FEATURES, TIME_COLUMN,
    XGBoostConfig, XGBoostFraudStrategy,
)

USECOLS = [ID_COLUMN, TIME_COLUMN, LABEL, *FEATURES]
DTYPES = {ID_COLUMN: "int32", TIME_COLUMN: "int64", LABEL: "int8",
          **{c: "float32" for c in NUMERIC_FEATURES}, **{c: "category" for c in CATEGORICAL_FEATURES}}


def run(data_path: str, out_dir: str, config: XGBoostConfig | None = None, train_frac: float = 0.8) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    cfg = config or XGBoostConfig()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, usecols=USECOLS, dtype=DTYPES)  # only needed columns
    vr = validate_dataframe(df, required_columns=USECOLS)
    if not vr.is_valid:
        raise ValueError(f"dataset failed validation; missing={vr.missing_required_columns} duplicates={vr.duplicate_columns}")

    train, evaluation = chronological_split(df, train_frac)  # same split as Strategy 1
    del df

    t0 = time.time()
    strat = XGBoostFraudStrategy(cfg).fit(train)
    fit_seconds = time.time() - t0

    proba = strat.predict_proba(evaluation)  # label column is never read
    flagged = proba >= cfg.threshold
    y = evaluation[LABEL].to_numpy()
    m = compute_metrics(y, flagged)
    cm = pd.DataFrame([[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
                      index=["actual_legit(0)", "actual_fraud(1)"],
                      columns=["pred_not_fraud", "pred_fraud"])
    cm.to_csv(out / "strategy2_confusion_matrix.csv")

    pd.DataFrame({ID_COLUMN: evaluation[ID_COLUMN].to_numpy(), TIME_COLUMN: evaluation[TIME_COLUMN].to_numpy(),
                  LABEL: y, "fraud_probability": proba, "predicted_fraud": flagged.astype(int)}
                 ).to_csv(out / "strategy2_validation_predictions.csv", index=False)
    strat.feature_importance().to_csv(out / "strategy2_feature_importance.csv", index=False)

    t_tr, t_ev = train[TIME_COLUMN], evaluation[TIME_COLUMN]
    report = {
        "strategy": "Strategy 2: XGBoost classifier (not claimed to be optimal)",
        "split": {
            "method": "chronological by TransactionDT (stable sort), first 80% train / last 20% evaluation; no random split; identical to Strategy 1",
            "training_rows": int(len(train)), "training_TransactionDT_range": [int(t_tr.min()), int(t_tr.max())],
            "training_fraud_rate": float(train[LABEL].mean()),
            "evaluation_rows": int(len(evaluation)), "evaluation_TransactionDT_range": [int(t_ev.min()), int(t_ev.max())],
            "evaluation_fraud_rate": float(evaluation[LABEL].mean()),
            "boundary_TransactionDT_tie": bool(t_tr.max() == t_ev.min()),
        },
        "leakage_controls": [
            "isFraud, TransactionID and raw TransactionDT are not features",
            "category encodings learned from the training portion only; unseen -> NaN",
            "early stopping uses the chronologically latest slice of the TRAINING portion only",
            "final model refit on full training portion; evaluation labels used only for metrics",
        ],
        "model": strat.describe(),
        "fit_seconds": round(fit_seconds, 1),
        "threshold": cfg.threshold,
        "evaluation": {
            "n_transactions": int(len(evaluation)),
            "predicted_fraud_flagged": int(flagged.sum()),
            "predicted_not_fraud": int((~flagged).sum()),
            "metrics": m,
            "supplementary_threshold_independent": {
                "roc_auc": float(roc_auc_score(y, proba)), "pr_auc_average_precision": float(average_precision_score(y, proba)),
            },
            "confusion_matrix": {"rows": list(cm.index), "cols": list(cm.columns), "values": cm.values.tolist()},
            "note": "Accuracy intentionally not reported as a headline metric (3.5% fraud base rate).",
        },
        "files": ["strategy2_xgboost_metrics.json", "strategy2_confusion_matrix.csv",
                  "strategy2_validation_predictions.csv", "strategy2_feature_importance.csv"],
    }
    (out / "strategy2_xgboost_metrics.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_transaction.csv")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    rep = run(a.data, a.out_dir, XGBoostConfig(threshold=a.threshold, n_estimators=a.n_estimators, n_jobs=a.n_jobs))
    print(json.dumps(rep["evaluation"], indent=2))
