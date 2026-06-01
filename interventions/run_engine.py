"""
End-to-end script: build the enriched feature frame, run the rule engine,
and persist outputs the dashboard reads at runtime.

Outputs (in ../data/):
  - automated_actions.csv   one row per fired (deduped) action
  - rules_catalogue.json    static catalogue for the dashboard's Rules page

Run after ml/score_cohort.py.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Allow ``python interventions/run_engine.py`` from anywhere.
sys.path.insert(0, str(ROOT / "ml"))

from features import build_feature_frame  # noqa: E402
from interventions.engine import run as run_engine  # noqa: E402
from interventions.rules import rules_catalogue  # noqa: E402

AS_OF_DATE = date(2025, 3, 22)


def _top_driver_for(target: str, shap_long: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    sub = shap_long[(shap_long["target"] == target) & (shap_long["rank"] == 1)]
    return sub[["student_id", "feature", "direction"]].rename(
        columns={
            "feature": f"top_driver_{col_prefix}",
            "direction": f"top_driver_{col_prefix}_dir",
        }
    )


def build_enriched_frame() -> pd.DataFrame:
    feats = build_feature_frame()
    preds = pd.read_csv(DATA_DIR / "predictions.csv")
    shap_long = pd.read_csv(DATA_DIR / "shap_drivers.csv")

    drv_w = _top_driver_for("withdrew", shap_long, "withdraw")
    drv_f = _top_driver_for("failed_next_assessment", shap_long, "fail")

    df = (
        feats.merge(preds, on="student_id", how="left")
        .merge(drv_w, on="student_id", how="left")
        .merge(drv_f, on="student_id", how="left")
    )

    # ---- v3.1: merge in the communications summary so the NON_RESPONSIVE rule
    # has the data it needs. If the comms log hasn't been generated yet (back-
    # compat), fall back to zero-valued columns so the engine still works.
    comms_path = DATA_DIR / "student_comms_summary.csv"
    comms_cols = [
        "comms_30d_count",
        "comms_responded_30d",
        "last_contact_date",
        "last_contact_channel",
        "last_contact_status",
        "is_non_responsive",
    ]
    if comms_path.exists():
        comms_summary = pd.read_csv(comms_path)
        df = df.merge(comms_summary, on="student_id", how="left")
    else:
        for c in comms_cols:
            df[c] = "" if c.startswith("last_") else 0

    df["comms_30d_count"] = df["comms_30d_count"].fillna(0).astype(int)
    df["comms_responded_30d"] = df["comms_responded_30d"].fillna(0).astype(int)
    # is_non_responsive may be read from CSV as object dtype; coerce explicitly
    # to avoid the pandas downcast deprecation warning.
    df["is_non_responsive"] = df["is_non_responsive"].map(
        lambda v: bool(v) if pd.notna(v) else False
    ).astype(bool)
    df["last_contact_date"] = df["last_contact_date"].fillna("").astype(str)
    df["last_contact_channel"] = df["last_contact_channel"].fillna("").astype(str)
    df["last_contact_status"] = df["last_contact_status"].fillna("").astype(str)

    # ---- Make sure the new v3.1 numeric features have sensible defaults so
    # rule lambdas never receive NaN. (This is defensive: the v3.1 enrichment
    # should always populate these.)
    for col in (
        "consecutive_missed_max",
        "unexcused_absences",
        "excused_absences",
        "late_arrivals_total",
        "missing_assignments_count",
        "wellbeing_flags",
        "wellbeing_score",
    ):
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df["top_driver_withdraw"] = df["top_driver_withdraw"].fillna("(no driver available)")
    df["top_driver_fail"] = df["top_driver_fail"].fillna("(no driver available)")
    return df


def main() -> None:
    df = build_enriched_frame()
    print(f"Enriched frame: {df.shape}")

    actions = run_engine(df, as_of=AS_OF_DATE)
    actions.to_csv(DATA_DIR / "automated_actions.csv", index=False)

    with (DATA_DIR / "rules_catalogue.json").open("w", encoding="utf-8") as f:
        json.dump(rules_catalogue(), f, indent=2)

    # ---- summary ----
    print(f"\nActions written: {len(actions)}")
    if actions.empty:
        return
    print("\nSeverity mix:")
    print(actions["severity"].value_counts().to_string())
    print("\nActions per support team:")
    print(actions["route_to"].value_counts().to_string())
    print("\nMost-common rules fired:")
    print(actions["rule_id"].value_counts().to_string())

    per_student = actions.groupby("student_id").size()
    print(f"\nActions per student: mean={per_student.mean():.2f}, "
          f"max={per_student.max()}, "
          f"students with at least one action={per_student.size}, "
          f"students with no action={len(df) - per_student.size}")

    print("\nSample (highest-severity first):")
    cols = ["action_id", "student_id", "severity", "rule_name", "route_to", "rationale"]
    print(actions[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
