"""
Inference + explanation API for the v3 ML models.

This module hides all SHAP and sklearn plumbing behind a small surface that
the Streamlit dashboard can call:

  - ``load_model(target)``       calibrated pipeline (cached)
  - ``predict_proba(target, X)`` probability for the positive class
  - ``explain_batch(target, X)`` per-row top drivers with SHAP values, ready
                                 to render as plain-English explanations
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import all_feature_columns, build_feature_frame, humanise_feature

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@lru_cache(maxsize=4)
def load_model(target: str):
    """Return the saved (calibrated) classifier for the given target."""
    return joblib.load(MODELS_DIR / f"{target}_model.joblib")


@lru_cache(maxsize=1)
def load_feature_config() -> dict:
    with (MODELS_DIR / "feature_config.json").open() as f:
        return json.load(f)


def _unwrap_pipeline(cal_model) -> Pipeline:
    """Pull the underlying sklearn Pipeline out of a CalibratedClassifierCV.

    Structure: CalibratedClassifierCV.calibrated_classifiers_[0].estimator is
    a FrozenEstimator wrapping the fitted Pipeline.
    """
    inner = cal_model.calibrated_classifiers_[0].estimator
    if isinstance(inner, FrozenEstimator):
        return inner.estimator
    return inner


def predict_proba(target: str, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated P(positive) for each row in ``X``."""
    model = load_model(target)
    return model.predict_proba(X[all_feature_columns()])[:, 1]


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _explainer_bundle(target: str):
    """Cache the SHAP explainer + preprocessor + transformed feature names."""
    pipeline = _unwrap_pipeline(load_model(target))
    preproc = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    feature_names = list(preproc.get_feature_names_out())

    if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(clf)
    elif isinstance(clf, LogisticRegression):
        df = build_feature_frame()
        bg = df[df["split"] == "train"][all_feature_columns()].sample(80, random_state=42)
        bg_pre = preproc.transform(bg)
        explainer = shap.LinearExplainer(clf, bg_pre)
    else:
        df = build_feature_frame()
        bg = df[df["split"] == "train"][all_feature_columns()].sample(50, random_state=42)
        bg_pre = preproc.transform(bg)
        explainer = shap.Explainer(clf.predict_proba, bg_pre)

    return explainer, preproc, feature_names


def _shap_for_positive_class(raw_shap, n_rows: int, n_features: int) -> np.ndarray:
    """Coerce SHAP output into a (n_rows, n_features) array for the positive class."""
    if isinstance(raw_shap, list):
        sv = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    else:
        sv = np.asarray(raw_shap)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    return np.asarray(sv).reshape(n_rows, n_features)


def shap_values_matrix(target: str, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (shap_values [n_rows, n_features], feature_names)."""
    explainer, preproc, feature_names = _explainer_bundle(target)
    X_pre = preproc.transform(X[all_feature_columns()])
    raw = explainer.shap_values(X_pre)
    sv = _shap_for_positive_class(raw, X_pre.shape[0], X_pre.shape[1])
    return sv, feature_names


def explain_batch(target: str, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
    """For each row, return a list of top-k drivers (largest |shap|).

    Each driver is ``{"feature": <human-readable>, "shap": float, "direction": "+"/"-"}``
    where ``+`` means the feature pushed the prediction toward the positive
    class (more risk) and ``-`` means it pushed away.
    """
    sv, feature_names = shap_values_matrix(target, X)
    human_names = [humanise_feature(n) for n in feature_names]

    out: list[list[dict]] = []
    for i in range(sv.shape[0]):
        row = sv[i]
        order = np.argsort(-np.abs(row))[:top_k]
        out.append(
            [
                {
                    "feature": human_names[j],
                    "shap": float(row[j]),
                    "direction": "+" if row[j] > 0 else "-",
                }
                for j in order
            ]
        )
    return out
