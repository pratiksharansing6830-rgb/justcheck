import json

import numpy as np
import pandas as pd
import pytest

from fraud_validation.backtest import run_strategy2
from fraud_validation.strategies.xgboost_strategy import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, CategoryEncoder, XGBoostConfig, XGBoostFraudStrategy,
)

FAST = dict(n_estimators=30, max_depth=3, early_stopping_rounds=5, n_jobs=1)


def make_df(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    d = {"TransactionID": np.arange(n), "TransactionDT": np.sort(rng.integers(86400, 10_000_000, n))}
    for c in NUMERIC_FEATURES:
        d[c] = rng.normal(size=n).astype("float32")
    for c in CATEGORICAL_FEATURES:
        d[c] = rng.choice(["a", "b", "c"], n)
    risky = d["C1"] > 0.8
    d["isFraud"] = ((risky & (rng.random(n) < 0.7)) | (rng.random(n) < 0.02)).astype(int)
    df = pd.DataFrame(d)
    df.loc[rng.random(n) < 0.2, "D1"] = np.nan
    df.loc[rng.random(n) < 0.2, "P_emaildomain"] = None
    return df


def test_features_exclude_label_id_and_raw_time():
    assert "isFraud" not in FEATURES and "TransactionID" not in FEATURES and "TransactionDT" not in FEATURES
    assert len(FEATURES) == len(set(FEATURES))


def test_encoder_learns_from_training_only_and_unseen_is_nan():
    train = pd.DataFrame({"ProductCD": ["W", "C", "W", None]})
    ev = pd.DataFrame({"ProductCD": ["C", "ZZZ", None, "W"]})
    enc = CategoryEncoder(("ProductCD",)).fit(train)
    before = dict(enc.mappings["ProductCD"])
    out = enc.transform(ev)["ProductCD"]
    assert before == {"c": 0, "w": 1} and enc.mappings["ProductCD"] == before  # eval does not change mapping
    assert out.iloc[0] == 0 and out.iloc[3] == 1
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])  # unseen + missing -> NaN


def test_encoder_missing_column_is_nan_not_crash():
    enc = CategoryEncoder(("card4",)).fit(pd.DataFrame({"card4": ["visa"]}))
    assert enc.transform(pd.DataFrame({"x": [1, 2]}))["card4"].isna().all()


def test_fit_predict_proba_valid_and_label_not_needed():
    df = make_df()
    tr, ev = df.iloc[:1200], df.iloc[1200:]
    s = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(tr)
    p = s.predict_proba(ev.drop(columns=["isFraud"]))
    assert len(p) == len(ev) and np.all((p >= 0) & (p <= 1)) and not np.isnan(p).any()


def test_missing_values_do_not_crash():
    df = make_df()
    s = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(df.iloc[:1200])
    sparse = pd.DataFrame({"TransactionAmt": [np.nan, 50.0]})  # almost everything missing
    assert len(s.predict_proba(sparse)) == 2


def test_threshold_configurable_and_validated():
    df = make_df()
    s = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(df.iloc[:1200])
    ev = df.iloc[1200:]
    p = s.predict_proba(ev)
    assert s.predict(ev, threshold=0.0).all()
    assert np.array_equal(s.predict(ev, threshold=0.3), p >= 0.3)
    assert s.predict(ev, threshold=1.0).sum() <= (p >= 1.0).sum()
    with pytest.raises(ValueError):
        s.predict(ev, threshold=1.5)
    with pytest.raises(ValueError):
        XGBoostFraudStrategy(XGBoostConfig(threshold=-0.1))


def test_predictions_do_not_depend_on_evaluation_labels():
    df = make_df()
    tr, ev = df.iloc[:1200], df.iloc[1200:].copy()
    s = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(tr)
    p1 = s.predict_proba(ev)
    ev["isFraud"] = 1 - ev["isFraud"]
    assert np.array_equal(p1, s.predict_proba(ev))


def test_inner_early_stopping_split_is_chronological_within_training():
    tr = make_df().sample(frac=1, random_state=1)
    a, b = XGBoostFraudStrategy._inner_split(tr, 0.1)
    assert a["TransactionDT"].max() <= b["TransactionDT"].min() and len(a) == int(len(tr) * 0.9) and len(a) + len(b) == len(tr)


def test_deterministic_with_fixed_seed():
    df = make_df()
    a = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(df.iloc[:1200]).predict_proba(df.iloc[1200:])
    b = XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(df.iloc[:1200]).predict_proba(df.iloc[1200:])
    assert np.allclose(a, b)


def test_requires_label_and_both_classes():
    df = make_df()
    with pytest.raises(ValueError):
        XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(df.drop(columns=["isFraud"]))
    one = df.iloc[:200].copy(); one["isFraud"] = 0
    with pytest.raises(ValueError):
        XGBoostFraudStrategy(XGBoostConfig(**FAST)).fit(one)


def test_backtest_end_to_end_chronological_and_reports(tmp_path):
    df = make_df(3000).sample(frac=1, random_state=3)  # shuffled input; runner must sort by time
    csv = tmp_path / "train_transaction.csv"
    df.to_csv(csv, index=False)
    rep = run_strategy2.run(str(csv), str(tmp_path / "reports"), XGBoostConfig(**FAST))
    sp = rep["split"]
    assert sp["training_rows"] == 2400 and sp["evaluation_rows"] == 600
    assert sp["training_TransactionDT_range"][1] <= sp["evaluation_TransactionDT_range"][0]
    m = rep["evaluation"]["metrics"]
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == 600
    for k in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "fraud_detection_rate"):
        assert k in m
    files = sorted(p.name for p in (tmp_path / "reports").iterdir())
    assert files and all(f.startswith("strategy2_") for f in files)
    preds = pd.read_csv(tmp_path / "reports" / "strategy2_validation_predictions.csv")
    assert len(preds) == 600 and preds["fraud_probability"].between(0, 1).all()
    assert json.loads((tmp_path / "reports" / "strategy2_xgboost_metrics.json").read_text())["threshold"] == 0.5
