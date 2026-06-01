"""
Data loaders for the v3.1 dashboard.

The dashboard consumes only the precomputed CSVs / JSON written by the ML
and intervention pipelines. It does NOT import sklearn, shap, or any model
artefact at runtime - keeping the deployed app lightweight.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from .paths import DATA_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
STUDENTS_CSV = DATA_DIR / "students_v3.csv"
WEEKLY_CSV = DATA_DIR / "weekly_engagement.csv"
ASSESSMENTS_CSV = DATA_DIR / "assessments_v3.csv"
INTERVENTIONS_HISTORY_CSV = DATA_DIR / "interventions_v3.csv"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
SHAP_DRIVERS_CSV = DATA_DIR / "shap_drivers.csv"
ACTIONS_CSV = DATA_DIR / "automated_actions.csv"
RULES_JSON = DATA_DIR / "rules_catalogue.json"
METRICS_JSON = MODELS_DIR / "metrics.json"
METRICS_BASELINE_JSON = MODELS_DIR / "metrics_v3_baseline.json"

# v3.1 layers
MODULES_CSV = DATA_DIR / "modules.csv"
ENROLMENTS_CSV = DATA_DIR / "enrolments.csv"
MODULE_ENGAGEMENT_CSV = DATA_DIR / "module_engagement.csv"
DAILY_ATTENDANCE_CSV = DATA_DIR / "daily_attendance.csv"
COMMS_LOG_CSV = DATA_DIR / "communications_log.csv"
COMMS_SUMMARY_CSV = DATA_DIR / "student_comms_summary.csv"


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "escalation": 3}


# ---------------------------------------------------------------------------
# Cached loaders - base layers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_students() -> pd.DataFrame:
    return pd.read_csv(STUDENTS_CSV)


@st.cache_data(show_spinner=False)
def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(WEEKLY_CSV)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    return df


@st.cache_data(show_spinner=False)
def load_assessments() -> pd.DataFrame:
    df = pd.read_csv(ASSESSMENTS_CSV)
    df["due_date"] = pd.to_datetime(df["due_date"])
    return df


@st.cache_data(show_spinner=False)
def load_interventions_history() -> pd.DataFrame:
    df = pd.read_csv(INTERVENTIONS_HISTORY_CSV)
    df["intervention_date"] = pd.to_datetime(df["intervention_date"])
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_CSV)


@st.cache_data(show_spinner=False)
def load_shap_drivers() -> pd.DataFrame:
    return pd.read_csv(SHAP_DRIVERS_CSV)


@st.cache_data(show_spinner=False)
def load_actions() -> pd.DataFrame:
    df = pd.read_csv(ACTIONS_CSV)
    df["triggered_on"] = pd.to_datetime(df["triggered_on"])
    return df


@st.cache_data(show_spinner=False)
def load_rules_catalogue() -> list[dict]:
    with RULES_JSON.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with METRICS_JSON.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics_baseline() -> dict | None:
    if not METRICS_BASELINE_JSON.exists():
        return None
    with METRICS_BASELINE_JSON.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Cached loaders - v3.1 layers (modules, daily attendance, comms)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_modules() -> pd.DataFrame:
    if not MODULES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(MODULES_CSV)


@st.cache_data(show_spinner=False)
def load_enrolments() -> pd.DataFrame:
    if not ENROLMENTS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(ENROLMENTS_CSV)


@st.cache_data(show_spinner=False)
def load_module_engagement() -> pd.DataFrame:
    if not MODULE_ENGAGEMENT_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(MODULE_ENGAGEMENT_CSV)


@st.cache_data(show_spinner=False)
def load_daily_attendance() -> pd.DataFrame:
    if not DAILY_ATTENDANCE_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DAILY_ATTENDANCE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_communications() -> pd.DataFrame:
    if not COMMS_LOG_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(COMMS_LOG_CSV)
    df["sent_date"] = pd.to_datetime(df["sent_date"])
    if "response_date" in df.columns:
        df["response_date"] = pd.to_datetime(df["response_date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_comms_summary() -> pd.DataFrame:
    if not COMMS_SUMMARY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(COMMS_SUMMARY_CSV)
    if "last_contact_date" in df.columns:
        df["last_contact_date"] = pd.to_datetime(df["last_contact_date"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Joined / derived views
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def students_with_predictions() -> pd.DataFrame:
    """One row per student, with predictions joined on."""
    s = load_students()
    p = load_predictions()
    return s.merge(p, on="student_id", how="left")


@st.cache_data(show_spinner=False)
def consolidated_cases() -> pd.DataFrame:
    """One row per student with at least one open action.

    The highest-severity action is the *primary*; remaining actions become
    *supporting concerns* attached to the same case. This mirrors how a real
    case-management system would present a student's situation: one case file,
    multiple concerns inside it.

    Returns a DataFrame with columns:
        student_id
        primary_action_id, primary_rule_id, primary_rule_name,
        primary_severity, primary_action, primary_team, primary_rationale,
        primary_policy_basis, primary_withdraw_prob, primary_fail_prob,
        supporting   - list[dict] of supporting concern records
        concerns_total
        teams_involved   - list[str]
        highest_sev_rank
    """
    actions = load_actions()
    if actions.empty:
        return pd.DataFrame()

    a = actions.copy()
    a["_sev"] = a["severity"].map(SEVERITY_RANK)
    a = a.sort_values(
        ["student_id", "_sev", "withdraw_prob_at_trigger", "fail_prob_at_trigger"],
        ascending=[True, False, False, False],
    )

    rows: list[dict] = []
    for sid, group in a.groupby("student_id", sort=False):
        primary = group.iloc[0]
        supporting = group.iloc[1:]
        sup_records = []
        for _, sup in supporting.iterrows():
            sup_records.append(
                {
                    "action_id": sup["action_id"],
                    "rule_id": sup["rule_id"],
                    "rule_name": sup["rule_name"],
                    "severity": sup["severity"],
                    "action": sup["action"],
                    "route_to": sup["route_to"],
                    "rationale": sup["rationale"],
                    "policy_basis": sup["policy_basis"],
                }
            )
        rows.append(
            {
                "student_id": sid,
                "primary_action_id": primary["action_id"],
                "primary_rule_id": primary["rule_id"],
                "primary_rule_name": primary["rule_name"],
                "primary_severity": primary["severity"],
                "primary_action": primary["action"],
                "primary_team": primary["route_to"],
                "primary_rationale": primary["rationale"],
                "primary_policy_basis": primary["policy_basis"],
                "primary_withdraw_prob": float(primary["withdraw_prob_at_trigger"]),
                "primary_fail_prob": float(primary["fail_prob_at_trigger"]),
                "supporting": sup_records,
                "concerns_total": int(len(group)),
                "teams_involved": sorted(group["route_to"].unique().tolist()),
                "highest_sev_rank": int(SEVERITY_RANK.get(primary["severity"], 0)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-student lookups
# ---------------------------------------------------------------------------
def student_record(student_id: str) -> dict[str, Any]:
    """Return one student's full row plus joined predictions, as a dict."""
    df = students_with_predictions()
    row = df.loc[df["student_id"] == student_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def student_weekly(student_id: str) -> pd.DataFrame:
    df = load_weekly()
    return df[df["student_id"] == student_id].sort_values("week")


def student_actions(student_id: str) -> pd.DataFrame:
    df = load_actions()
    return df[df["student_id"] == student_id].sort_values(
        "severity", key=lambda s: s.map(SEVERITY_RANK), ascending=False
    )


def student_history(student_id: str) -> pd.DataFrame:
    df = load_interventions_history()
    return df[df["student_id"] == student_id].sort_values("intervention_date", ascending=False)


def student_drivers(student_id: str, target: str) -> pd.DataFrame:
    """Top SHAP drivers for one student and one target."""
    df = load_shap_drivers()
    sub = df[(df["student_id"] == student_id) & (df["target"] == target)]
    return sub.sort_values("rank")


def module_roster(module_code: str) -> pd.DataFrame:
    """Return all students enrolled in one module with their module-level
    metrics joined to overall risk + identity columns from the predictions table.
    """
    me = load_module_engagement()
    if me.empty:
        return pd.DataFrame()
    sub = me[me["module_code"] == module_code].copy()
    if sub.empty:
        return sub
    swp = students_with_predictions()
    keep = [
        "student_id",
        "student_name",
        "program",
        "year_of_study",
        "risk_band",
        "overall_risk",
        "withdraw_prob",
        "fail_prob",
    ]
    sub = sub.merge(swp[keep], on="student_id", how="left")
    return sub


def student_modules(student_id: str) -> pd.DataFrame:
    """Return one row per module the student is enrolled in, with engagement."""
    me = load_module_engagement()
    if me.empty:
        return pd.DataFrame()
    sub = me[me["student_id"] == student_id].copy()
    mods = load_modules()
    if not mods.empty:
        mod_meta = mods[["module_code", "module_name", "programme", "level"]]
        sub = sub.merge(mod_meta, on="module_code", how="left")
    return sub.sort_values("module_code")


def student_daily_attendance(student_id: str) -> pd.DataFrame:
    df = load_daily_attendance()
    if df.empty:
        return df
    return df[df["student_id"] == student_id].sort_values("date")


def student_comms(student_id: str) -> pd.DataFrame:
    df = load_communications()
    if df.empty:
        return df
    return df[df["student_id"] == student_id].sort_values("sent_date", ascending=False)


def student_comms_summary_row(student_id: str) -> dict[str, Any]:
    df = load_comms_summary()
    if df.empty:
        return {}
    row = df[df["student_id"] == student_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def student_case(student_id: str) -> dict[str, Any]:
    """Return the consolidated case for one student (primary + supporting)."""
    cases = consolidated_cases()
    if cases.empty:
        return {}
    row = cases[cases["student_id"] == student_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in ("true", "1", "yes", "t")
