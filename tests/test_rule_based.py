import numpy as np
import pandas as pd
import pytest

from fraud_validation.backtest.run_strategy1 import chronological_split, compute_metrics
from fraud_validation.strategies import RuleBasedFraudStrategy, decision_for, risk_level_for

S = RuleBasedFraudStrategy()  # unfitted: email rule contributes 0


def pts(**kw):
    return S.score(kw)["risk_score"]


def test_productcd_c():
    assert pts(ProductCD="C") == 25


def test_productcd_h():
    assert pts(ProductCD="H") == 10


def test_productcd_w():
    assert pts(ProductCD="W") == 0


def test_amount_low():
    assert pts(TransactionAmt=35.94) == 10


def test_amount_high():
    assert pts(TransactionAmt=159.96) == 15


def test_amount_middle_and_inclusive_bounds():
    assert pts(TransactionAmt=100) == 0
    assert pts(TransactionAmt=35.95) == 0
    assert pts(TransactionAmt=159.95) == 0


def test_discover():
    assert pts(card4="discover") == 10
    assert pts(card4="visa") == 0


def test_credit():
    assert pts(card6="credit") == 10
    assert pts(card6="debit") == 0


def test_missing_values_do_not_crash():
    assert S.score({})["risk_score"] == 0
    r = S.score({"ProductCD": None, "TransactionAmt": np.nan, "card4": np.nan, "card6": None,
                 "P_emaildomain": np.nan})
    assert r["risk_score"] == 0 and r["decision"] == "ALLOW"
    df = pd.DataFrame({"TransactionAmt": [np.nan, 10.0], "ProductCD": [None, "C"]})
    assert S.score_dataframe(df)["risk_score"].tolist() == [0, 35]


@pytest.mark.parametrize("score,level", [(0, "LOW"), (39, "LOW"), (40, "MEDIUM"), (69, "MEDIUM"), (70, "HIGH"), (80, "HIGH")])
def test_risk_boundaries(score, level):
    assert risk_level_for(score) == level


def test_decisions():
    assert decision_for("LOW") == "ALLOW"
    assert decision_for("MEDIUM") == "VERIFY"
    assert decision_for("HIGH") == "BLOCK"


def test_explanations_and_max_score():
    r = S.score({"ProductCD": "C", "TransactionAmt": 500, "card4": "discover", "card6": "credit"})
    assert r["risk_score"] == 60 and r["risk_level"] == "MEDIUM" and r["decision"] == "VERIFY"
    assert "ProductCD=C (+25)" in r["reasons"] and "High transaction amount (+15)" in r["reasons"]
    assert "card6=credit (+10)" in r["reasons"]


def _email_frame():
    # 800 historical rows then 200 later rows; 'late.com' is fraud-heavy ONLY in the later rows.
    hist = pd.DataFrame({
        "TransactionDT": np.arange(800),
        "P_emaildomain": ["risky.com"] * 400 + ["safe.com"] * 400,
        "isFraud": [1] * 160 + [0] * 240 + [1] * 4 + [0] * 396,
    })
    late = pd.DataFrame({"TransactionDT": np.arange(800, 1000), "P_emaildomain": "late.com", "isFraud": 1})
    return pd.concat([hist, late], ignore_index=True)


def test_email_rules_learned_only_from_training():
    df = _email_frame().sample(frac=1, random_state=0)  # shuffled input; split must sort by time
    train, ev = chronological_split(df, 0.8)
    assert train["TransactionDT"].max() < ev["TransactionDT"].min()
    strat = RuleBasedFraudStrategy(min_support=50, lift_threshold=1.5).fit(train)
    hr = strat.email_config["columns"]["P_emaildomain"]["high_risk_domains"]
    assert hr == ["risky.com"]
    assert "late.com" not in strat.email_config["columns"]["P_emaildomain"]["domain_stats"]
    # scoring the evaluation period must not change the learned config
    before = repr(strat.email_config)
    strat.score_dataframe(ev)
    assert repr(strat.email_config) == before
    # sanity: fitting on everything WOULD have leaked late.com
    leaky = RuleBasedFraudStrategy(min_support=50).fit(df)
    assert "late.com" in leaky.email_config["columns"]["P_emaildomain"]["high_risk_domains"]
    assert strat.score({"P_emaildomain": "late.com"})["risk_score"] == 0
    assert strat.score({"P_emaildomain": "risky.com"})["risk_score"] == 20


def test_min_support_ignores_tiny_domains():
    df = pd.DataFrame({"P_emaildomain": ["tiny.com"] * 10 + ["big.com"] * 600,
                       "isFraud": [1] * 10 + [0] * 600})
    s = RuleBasedFraudStrategy(min_support=500).fit(df)
    cfg = s.email_config["columns"]["P_emaildomain"]
    assert cfg["high_risk_domains"] == [] and "tiny.com" not in cfg["domain_stats"]


def test_config_roundtrip(tmp_path):
    s = RuleBasedFraudStrategy(min_support=50).fit(_email_frame().iloc[:800])
    p = tmp_path / "cfg.json"
    s.save_email_config(p)
    s2 = RuleBasedFraudStrategy.from_email_config(p)
    assert s2.score({"P_emaildomain": "risky.com"}) == s.score({"P_emaildomain": "risky.com"})


def test_metrics():
    m = compute_metrics([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (m["TP"], m["TN"], m["FP"], m["FN"]) == (1, 2, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_positive_rate"] == pytest.approx(1 / 3)


def test_r_emaildomain_not_used_and_max_score_80():
    from fraud_validation.strategies import rule_based as rb
    assert rb.EMAIL_COLUMNS == ("P_emaildomain",) and rb.MAX_SCORE == 80
    df = _email_frame().iloc[:800]
    s = RuleBasedFraudStrategy(min_support=50).fit(df)
    assert list(s.email_config["columns"]) == ["P_emaildomain"]
    r = s.score({"ProductCD": "C", "TransactionAmt": 500, "card4": "discover", "card6": "credit",
                 "P_emaildomain": "risky.com", "R_emaildomain": "risky.com"})
    assert r["risk_score"] == 80 and r["risk_level"] == "HIGH" and r["decision"] == "BLOCK"
    assert not any("R_emaildomain" in x for x in r["reasons"])
    assert s.score({"R_emaildomain": "risky.com"})["risk_score"] == 0
