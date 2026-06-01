"""
Train, calibrate, and evaluate the ML risk models.

For each outcome (``withdrew``, ``failed_next_assessment``):
  1. Train Logistic Regression, Random Forest and XGBoost pipelines on the
     train split using class-balanced weighting.
  2. Pick the winner by validation ROC-AUC.
  3. Isotonically calibrate the winner on the validation split so the
     predicted probabilities reflect real frequencies.
  4. Evaluate honestly on the held-out test split.

Outputs (saved to ../models/):
  - ``withdrew_model.joblib`` / ``failed_next_assessment_model.joblib``
  - ``feature_config.json``
  - ``metrics.json``
  - ``metrics_report.md``
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import build_feature_frame, feature_columns

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Minimum recall floors for threshold selection on the validation set.
# We tune the decision threshold to maximise F1 on the validation set
# subject to recall >= this floor (so an early-warning system catches enough
# real positives even if it costs some precision).
MIN_RECALL: dict[str, float] = {
    "withdrew": 0.60,
    "failed_next_assessment": 0.70,
}


def _build_preprocessor() -> ColumnTransformer:
    cols = feature_columns()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols["numeric"]),
            ("bin", "passthrough", cols["binary"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols["categorical"]),
        ],
        remainder="drop",
    )


def _pick_threshold(y_true: np.ndarray, p: np.ndarray, min_recall: float) -> tuple[float, dict]:
    """Pick the threshold that maximises F1 on val, subject to recall >= floor.

    Falls back to the recall-floor threshold if no F1-optimal point satisfies it.
    Returns (threshold, val_metrics_at_that_threshold).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    # precision_recall_curve returns arrays of length n+1; thresholds is length n.
    # We align: precision[:-1] / recall[:-1] correspond to thresholds.
    pr = precision[:-1]
    rc = recall[:-1]
    thr = thresholds

    eligible = rc >= min_recall
    if eligible.any():
        f1 = np.where((pr + rc) > 0, 2 * pr * rc / (pr + rc + 1e-12), 0.0)
        f1_masked = np.where(eligible, f1, -1.0)
        idx = int(np.argmax(f1_masked))
    else:
        # Recall floor unreachable -> pick the lowest threshold that maximises recall.
        idx = int(np.argmax(rc))

    chosen_thr = float(thr[idx])
    pred = (p >= chosen_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    metrics = {
        "precision": float(pr[idx]),
        "recall": float(rc[idx]),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    return chosen_thr, metrics


def _candidate_pipelines() -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline(
            [
                ("pre", _build_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "rf": Pipeline(
            [
                ("pre", _build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgb": Pipeline(
            [
                ("pre", _build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="auc",
                        n_jobs=-1,
                        random_state=42,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
    }


def train_one_target(df: pd.DataFrame, target: str) -> dict:
    cols = feature_columns()
    feature_cols = cols["numeric"] + cols["binary"] + cols["categorical"]

    X = df[feature_cols]
    y = df[target].astype(int)
    split = df["split"]

    X_tr, y_tr = X[split == "train"], y[split == "train"]
    X_val, y_val = X[split == "val"], y[split == "val"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    candidates = _candidate_pipelines()
    leaderboard: list[dict] = []
    for name, pipe in candidates.items():
        if name == "xgb":
            pos = int(y_tr.sum())
            neg = int(len(y_tr) - pos)
            scale_pos_weight = max(1.0, neg / max(1, pos))
            pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
        pipe.fit(X_tr, y_tr)
        p_val = pipe.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, p_val))
        val_pr = float(average_precision_score(y_val, p_val))
        leaderboard.append({"model": name, "val_roc_auc": val_auc, "val_pr_auc": val_pr})
        print(f"  [{target}] {name:<7}  val ROC-AUC={val_auc:.3f}  val PR-AUC={val_pr:.3f}")

    leaderboard_sorted = sorted(leaderboard, key=lambda r: r["val_roc_auc"], reverse=True)
    winner_name = leaderboard_sorted[0]["model"]
    winner = candidates[winner_name]
    print(f"  [{target}] -> winner: {winner_name}")

    # Platt (sigmoid) calibration on the held-out validation set.
    # Sigmoid gives smooth probabilities (fits a logistic on the model output)
    # rather than the stair-step output of isotonic, which is overkill for a
    # 90-row validation set and looks coarse in the dashboard.
    # FrozenEstimator tells CalibratedClassifierCV not to refit the underlying model.
    calibrated = CalibratedClassifierCV(FrozenEstimator(winner), method="sigmoid")
    calibrated.fit(X_val, y_val)

    # ---- Threshold selection on validation ----
    # Pick the threshold that maximises F1 on val, subject to recall floor.
    p_val_cal = calibrated.predict_proba(X_val)[:, 1]
    threshold, val_at_thr = _pick_threshold(y_val.to_numpy(), p_val_cal, min_recall=MIN_RECALL[target])

    p_test = calibrated.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= threshold).astype(int)

    metrics = {
        "winner_model": winner_name,
        "leaderboard": leaderboard_sorted,
        "threshold": float(threshold),
        "min_recall_floor": float(MIN_RECALL[target]),
        "val_at_threshold": val_at_thr,
        "test": {
            "n": int(len(y_test)),
            "positives": int(y_test.sum()),
            "roc_auc": float(roc_auc_score(y_test, p_test)),
            "pr_auc": float(average_precision_score(y_test, p_test)),
            "brier": float(brier_score_loss(y_test, p_test)),
            "log_loss": float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6))),
            "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
            "classification_report": classification_report(
                y_test, pred_test, output_dict=True, zero_division=0
            ),
        },
    }

    joblib.dump(calibrated, MODELS_DIR / f"{target}_model.joblib")
    return metrics


def _write_metrics_report(results: dict[str, dict]) -> None:
    lines: list[str] = [
        "# ML Model Performance Report",
        "",
        "All metrics are computed on the **held-out test split** (90 students) that was never "
        "seen during training or model selection. Validation is used only to choose the winning "
        "model family, fit the sigmoid (Platt) calibrator, and pick the decision threshold.",
        "",
    ]
    for target, m in results.items():
        t = m["test"]
        cm = t["confusion_matrix"]
        cr = t["classification_report"]
        v = m["val_at_threshold"]
        lines += [
            f"## Target: `{target}`",
            "",
            (
                f"**Winning model:** `{m['winner_model']}`  ·  "
                f"**Decision threshold:** {m['threshold']:.3f}  "
                f"(F1-optimal on val, recall floor {m['min_recall_floor']:.2f})"
            ),
            "",
            (
                f"At the chosen threshold the **validation** confusion is "
                f"TP={v['tp']}, FP={v['fp']}, FN={v['fn']}, TN={v['tn']}  "
                f"(precision={v['precision']:.2f}, recall={v['recall']:.2f}, F1={v['f1']:.2f})."
            ),
            "",
            "### Validation leaderboard",
            "",
            "| Model | Val ROC-AUC | Val PR-AUC |",
            "|---|---:|---:|",
        ]
        for row in m["leaderboard"]:
            lines.append(f"| `{row['model']}` | {row['val_roc_auc']:.3f} | {row['val_pr_auc']:.3f} |")
        lines += [
            "",
            "### Test-set performance",
            "",
            f"- N = **{t['n']}** students, positives = **{t['positives']}**",
            f"- ROC-AUC: **{t['roc_auc']:.3f}**",
            f"- PR-AUC:  **{t['pr_auc']:.3f}**",
            f"- Brier score (calibration error, lower is better): {t['brier']:.3f}",
            f"- Log loss: {t['log_loss']:.3f}",
            "",
            f"**Confusion matrix** @ threshold {m['threshold']:.3f}:",
            "",
            "|              | Pred 0 | Pred 1 |",
            "|---           |---:|---:|",
            f"| **Actual 0** | {cm[0][0]} | {cm[0][1]} |",
            f"| **Actual 1** | {cm[1][0]} | {cm[1][1]} |",
            "",
            (
                f"Class 1 (positive) — precision: **{cr['1']['precision']:.3f}**, "
                f"recall: **{cr['1']['recall']:.3f}**, "
                f"F1: **{cr['1']['f1-score']:.3f}**"
            ),
            "",
        ]
    (MODELS_DIR / "metrics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = build_feature_frame()
    print(
        f"Feature frame: {df.shape}  "
        f"(withdrew={int(df['withdrew'].sum())}, "
        f"failed_next_assessment={int(df['failed_next_assessment'].sum())})"
    )

    results: dict[str, dict] = {}
    for target in feature_columns()["targets"]:
        print(f"\n=== Training: {target} ===")
        results[target] = train_one_target(df, target)

    (MODELS_DIR / "feature_config.json").write_text(
        json.dumps(feature_columns(), indent=2), encoding="utf-8"
    )
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _write_metrics_report(results)

    print("\nSaved artifacts to:", MODELS_DIR)
    for f in sorted(MODELS_DIR.iterdir()):
        print(" -", f.name)


if __name__ == "__main__":
    main()
