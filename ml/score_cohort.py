"""
Batch-score every student with both ML models and persist the results.

Run after ``train.py``. Produces two CSVs in ../data/ that the dashboard
loads directly, so the dashboard never has to import sklearn/SHAP at runtime
(faster cold start and a clean separation of concerns).

Outputs
-------
data/predictions.csv     One row per student:
    student_id, withdraw_prob, withdraw_pred,
    fail_prob, fail_pred,
    overall_risk, risk_band

data/shap_drivers.csv    Long format, top 5 drivers per (student, target):
    student_id, target, rank, feature, shap_value, direction
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import all_feature_columns, build_feature_frame
from inference import explain_batch, load_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

TARGETS = ["withdrew", "failed_next_assessment"]
PRED_COL = {"withdrew": "withdraw", "failed_next_assessment": "fail"}


def _load_thresholds() -> dict[str, float]:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return {t: metrics[t]["threshold"] for t in TARGETS}


def _risk_band(p_withdraw: float, p_fail: float) -> tuple[float, str]:
    """Blend the two model outputs into an overall risk score and band.

    Weighting reflects the relative cost of each outcome: withdrawal is
    far more serious than a single failed assessment, so we weight it more.
    """
    overall = 0.65 * p_withdraw + 0.35 * p_fail
    if overall >= 0.40:
        band = "High"
    elif overall >= 0.20:
        band = "Medium"
    else:
        band = "Low"
    return float(overall), band


def main() -> None:
    df = build_feature_frame()
    X = df[all_feature_columns()]
    print(f"Scoring {len(df)} students with {len(TARGETS)} models...")

    thresholds = _load_thresholds()
    print(f"Thresholds: {thresholds}")

    predictions = pd.DataFrame({"student_id": df["student_id"].values})

    shap_rows: list[dict] = []
    for target in TARGETS:
        model = load_model(target)
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= thresholds[target]).astype(int)

        prefix = PRED_COL[target]
        predictions[f"{prefix}_prob"] = np.round(prob, 4)
        predictions[f"{prefix}_pred"] = pred

        print(f"  {target:<22}  mean prob={prob.mean():.3f}  flagged={int(pred.sum())} ({pred.mean()*100:.1f}%)")

        drivers = explain_batch(target, X, top_k=5)
        for sid, row_drivers in zip(df["student_id"].values, drivers):
            for rank, d in enumerate(row_drivers, start=1):
                shap_rows.append(
                    {
                        "student_id": sid,
                        "target": target,
                        "rank": rank,
                        "feature": d["feature"],
                        "shap_value": round(d["shap"], 4),
                        "direction": d["direction"],
                    }
                )

    overall_pairs = [
        _risk_band(w, f)
        for w, f in zip(predictions["withdraw_prob"], predictions["fail_prob"])
    ]
    predictions["overall_risk"] = [round(p[0], 4) for p in overall_pairs]
    predictions["risk_band"] = [p[1] for p in overall_pairs]

    predictions.to_csv(DATA_DIR / "predictions.csv", index=False)
    pd.DataFrame(shap_rows).to_csv(DATA_DIR / "shap_drivers.csv", index=False)

    print()
    print("Risk band distribution:")
    print(predictions["risk_band"].value_counts().to_string())
    print()
    print(f"Wrote {DATA_DIR / 'predictions.csv'}  ({len(predictions)} rows)")
    print(f"Wrote {DATA_DIR / 'shap_drivers.csv'} ({len(shap_rows)} rows)")


if __name__ == "__main__":
    main()
