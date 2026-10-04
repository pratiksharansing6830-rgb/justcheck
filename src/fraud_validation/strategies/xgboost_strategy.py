"""Strategy 2: XGBoost fraud classifier on a practical IEEE-CIS transaction-column subset.

Leakage controls:
  * ``isFraud`` is never a feature (asserted at import time).
  * ``TransactionID`` and raw ``TransactionDT`` are not features.
  * Categorical encodings are learned from the training portion only; unseen
    or missing categories become NaN (XGBoost handles NaN natively).
  * Early stopping uses a chronologically LATER slice of the *training* portion,
    never the evaluation period. The final model is refit on the full training
    portion with the selected number of trees.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

LABEL = "isFraud"
TIME_COLUMN = "TransactionDT"
ID_COLUMN = "TransactionID"

NUMERIC_FEATURES: tuple[str, ...] = (
    "TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1",
    *(f"C{i}" for i in range(1, 15)),
    "D1", "D2", "D3", "D4", "D5", "D10", "D15",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

assert LABEL not in FEATURES and TIME_COLUMN not in FEATURES and ID_COLUMN not in FEATURES
assert len(set(FEATURES)) == len(FEATURES)


class CategoryEncoder:
    """Ordinal encoder fit on training data only. Unseen/missing -> NaN."""

    def __init__(self, columns: tuple[str, ...] = CATEGORICAL_FEATURES):
        self.columns = tuple(columns)
        self.mappings: dict[str, dict[str, int]] | None = None

    @staticmethod
    def _norm(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.lower()

    def fit(self, df: pd.DataFrame) -> "CategoryEncoder":
        self.mappings = {}
        for col in self.columns:
            if col not in df.columns:
                self.mappings[col] = {}
                continue
            cats = sorted(self._norm(df[col]).dropna().unique().tolist())
            self.mappings[col] = {c: i for i, c in enumerate(cats)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mappings is None:
            raise RuntimeError("encoder is not fitted")
        out = {}
        for col in self.columns:
            if col not in df.columns:
                out[col] = np.full(len(df), np.nan, dtype=np.float32)
                continue
            mapped = self._norm(df[col]).astype("object").map(self.mappings[col])
            out[col] = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=df.index)


@dataclass
class XGBoostConfig:
    n_estimators: int = 500
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    early_stopping_rounds: int = 30
    early_stopping_fraction: float = 0.1  # latest slice of the TRAINING portion; 0 disables
    threshold: float = 0.5
    random_state: int = 42
    n_jobs: int = 1


def _check_threshold(t: float) -> float:
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {t}")
    return float(t)


class XGBoostFraudStrategy:
    def __init__(self, config: XGBoostConfig | None = None):
        self.config = config or XGBoostConfig()
        _check_threshold(self.config.threshold)
        self.encoder = CategoryEncoder()
        self.model = None
        self.best_iteration: int | None = None
        self.n_trees: int | None = None
        self.fit_info: dict[str, Any] = {}

    # ---- features ---------------------------------------------------------
    def _design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = {}
        for col in NUMERIC_FEATURES:
            numeric[col] = (
                pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
                if col in df.columns else np.full(len(df), np.nan, dtype=np.float32)
            )
        X = pd.concat([pd.DataFrame(numeric, index=df.index), self.encoder.transform(df)], axis=1)
        return X[list(FEATURES)]

    @staticmethod
    def _inner_split(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split INSIDE the training portion (earlier -> fit, later -> early-stop)."""
        if TIME_COLUMN in df.columns:
            df = df.sort_values(TIME_COLUMN, kind="stable")
        cut = int(len(df) * (1 - fraction))
        return df.iloc[:cut], df.iloc[cut:]

    # ---- learning ---------------------------------------------------------
    def _new_model(self, n_estimators: int, early_stopping: bool):
        from xgboost import XGBClassifier  # imported lazily so import errors surface clearly

        c = self.config
        return XGBClassifier(
            n_estimators=n_estimators, learning_rate=c.learning_rate, max_depth=c.max_depth,
            subsample=c.subsample, colsample_bytree=c.colsample_bytree,
            min_child_weight=c.min_child_weight, tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=c.early_stopping_rounds if early_stopping else None,
            random_state=c.random_state, n_jobs=c.n_jobs,
        )

    def fit(self, training_data: pd.DataFrame) -> "XGBoostFraudStrategy":
        if LABEL not in training_data.columns:
            raise ValueError(f"training_data must contain '{LABEL}'")
        train = training_data[training_data[LABEL].notna()]
        if train[LABEL].nunique() < 2:
            raise ValueError("training data must contain both classes")
        self.encoder.fit(train)
        c = self.config
        if c.early_stopping_fraction > 0:
            fit_part, stop_part = self._inner_split(train, c.early_stopping_fraction)
            if stop_part[LABEL].nunique() < 2 or fit_part[LABEL].nunique() < 2:
                raise ValueError("inner early-stopping split lacks both classes; set early_stopping_fraction=0")
            probe = self._new_model(c.n_estimators, True)
            probe.fit(
                self._design_matrix(fit_part), fit_part[LABEL].astype(int),
                eval_set=[(self._design_matrix(stop_part), stop_part[LABEL].astype(int))], verbose=False,
            )
            self.best_iteration = int(probe.best_iteration)
            self.n_trees = self.best_iteration + 1
            self.fit_info = {"early_stopping_fit_rows": int(len(fit_part)), "early_stopping_val_rows": int(len(stop_part)),
                             "best_iteration": self.best_iteration}
        else:
            self.n_trees = c.n_estimators
            self.fit_info = {"early_stopping": "disabled"}
        # final model: full training portion, fixed number of trees chosen without the evaluation period
        self.model = self._new_model(self.n_trees, False)
        self.model.fit(self._design_matrix(train), train[LABEL].astype(int), verbose=False)
        self.fit_info.update({"final_fit_rows": int(len(train)), "n_trees": int(self.n_trees)})
        return self

    # ---- inference --------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Fraud probability per row. Never reads the label column."""
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict_proba(self._design_matrix(df))[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        t = _check_threshold(self.config.threshold if threshold is None else threshold)
        return self.predict_proba(df) >= t

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("call fit() first")
        imp = pd.DataFrame({"feature": list(FEATURES), "importance_gain": self.model.feature_importances_})
        return imp.sort_values("importance_gain", ascending=False).reset_index(drop=True)

    def describe(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config), "fit_info": self.fit_info,
            "numeric_features": list(NUMERIC_FEATURES), "categorical_features": list(CATEGORICAL_FEATURES),
            "categorical_cardinality_learned_from_training": {k: len(v) for k, v in (self.encoder.mappings or {}).items()},
        }
