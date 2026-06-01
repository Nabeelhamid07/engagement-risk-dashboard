"""
Feature engineering for the v3 ML pipeline.

Joins ``students_v3.csv`` with per-student aggregates derived from
``weekly_engagement.csv`` to produce a single modelling table.

The cross-sectional engagement features are already on the student row
(those were derived from the weekly snapshots during data generation).
Here we add a handful of *temporal* features that the cross-sectional
view cannot capture:

  - attendance_volatility_12w      week-to-week std of attendance
  - vle_logins_volatility_12w      week-to-week std of VLE logins
  - weeks_with_zero_logins         count of fully-disengaged weeks
  - weeks_since_last_login         recency gap (0 = active this week)
  - max_weekly_attendance_drop     largest week-over-week attendance fall

These five features carry distinct predictive signal beyond the
cross-sectional slope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NUMERIC_FEATURES: list[str] = [
    # cross-sectional engagement (already on student row, derived from weekly)
    "attendance_rate",
    "attendance_slope_4w",
    "vle_logins_last_week",
    "vle_time_hours_last_week",
    "vle_logins_slope_4w",
    "vle_hours_slope_4w",
    "materials_accessed_last_4w",
    # weekly-derived temporal features (added below)
    "attendance_volatility_12w",
    "vle_logins_volatility_12w",
    "weeks_with_zero_logins",
    "weeks_since_last_login",
    "max_weekly_attendance_drop",
    # submissions
    "assignments_submitted_on_time",
    "assignments_submitted_late",
    # grades
    "last_assessment_grade",
    "previous_semester_gpa",
    # upcoming assessment
    "days_to_nearest_assessment",
    # intervention history
    "prior_intervention_count",
    "prior_interventions_improved",
    # context
    "year_of_study",
    # --- v3.1 enrichment ---
    # daily-attendance derived
    "consecutive_missed_max",
    "late_arrivals_total",
    "excused_absences",
    "unexcused_absences",
    # behavioural counts
    "missing_assignments_count",
    "quiz_attempts_count",
    "resource_downloads_count",
    "forum_posts_count",
    # module-aggregated academic
    "mean_module_grade",
    "module_grade_variance",
    # simple wellbeing layer
    "wellbeing_score",
    "wellbeing_flags",
]

BINARY_FEATURES: list[str] = [
    "is_commuter",
    "works_part_time",
    "has_declared_disability",
    "is_international",
    "accessed_upcoming_assessment_brief",
]

CATEGORICAL_FEATURES: list[str] = [
    "program",
    "financial_support",
    "engagement_trend",
]

TARGETS: list[str] = ["withdrew", "failed_next_assessment"]


def _weeks_since_last_login(series: pd.Series) -> int:
    vals = series.to_numpy()
    nz = np.where(vals > 0)[0]
    if len(nz) == 0:
        return int(len(vals))
    return int(len(vals) - 1 - nz[-1])


def _per_student_weekly_aggregates(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["student_id", "week"]).copy()
    weekly["att_diff"] = weekly.groupby("student_id")["attendance_pct"].diff()

    grp = weekly.groupby("student_id", sort=False)

    out = pd.DataFrame({"student_id": list(grp.groups.keys())})
    out["attendance_volatility_12w"] = grp["attendance_pct"].std().round(2).to_numpy()
    out["vle_logins_volatility_12w"] = grp["vle_logins"].std().round(2).to_numpy()
    out["weeks_with_zero_logins"] = grp["vle_logins"].apply(lambda s: int((s == 0).sum())).to_numpy()
    out["weeks_since_last_login"] = grp["vle_logins"].apply(_weeks_since_last_login).to_numpy()
    out["max_weekly_attendance_drop"] = (
        (-grp["att_diff"].min()).fillna(0).round(2).to_numpy()
    )
    return out


def build_feature_frame() -> pd.DataFrame:
    """Return one row per student with features + targets + split."""
    students = pd.read_csv(DATA_DIR / "students_v3.csv")
    weekly = pd.read_csv(DATA_DIR / "weekly_engagement.csv")
    weekly_agg = _per_student_weekly_aggregates(weekly)
    return students.merge(weekly_agg, on="student_id", how="left")


def feature_columns() -> dict[str, list[str]]:
    return {
        "numeric": list(NUMERIC_FEATURES),
        "binary": list(BINARY_FEATURES),
        "categorical": list(CATEGORICAL_FEATURES),
        "targets": list(TARGETS),
    }


def all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


HUMAN_LABELS: dict[str, str] = {
    "attendance_rate": "Recent attendance (4-week avg)",
    "attendance_slope_4w": "Attendance trend (last 4w vs prior 8w)",
    "vle_logins_last_week": "VLE logins last week",
    "vle_time_hours_last_week": "VLE hours last week",
    "vle_logins_slope_4w": "VLE login trend",
    "vle_hours_slope_4w": "VLE hours trend",
    "materials_accessed_last_4w": "Lecture materials accessed (last 4w)",
    "attendance_volatility_12w": "Attendance volatility (12-week std)",
    "vle_logins_volatility_12w": "VLE login volatility",
    "weeks_with_zero_logins": "Weeks with no VLE activity",
    "weeks_since_last_login": "Weeks since last VLE login",
    "max_weekly_attendance_drop": "Largest weekly attendance drop",
    "assignments_submitted_on_time": "On-time submissions",
    "assignments_submitted_late": "Late submissions",
    "last_assessment_grade": "Most recent assessment grade",
    "previous_semester_gpa": "Previous semester GPA",
    "days_to_nearest_assessment": "Days until next assessment",
    "prior_intervention_count": "Past interventions received",
    "prior_interventions_improved": "Past interventions that improved engagement",
    "year_of_study": "Year of study",
    "is_commuter": "Commuter student",
    "works_part_time": "Works part-time",
    "has_declared_disability": "Has declared disability",
    "is_international": "International student",
    "accessed_upcoming_assessment_brief": "Accessed upcoming assessment brief",
    "program": "Programme",
    "financial_support": "Financial support",
    "engagement_trend": "Engagement trend",
    # --- v3.1 enrichment labels ---
    "consecutive_missed_max": "Longest streak of unexcused absences (days)",
    "late_arrivals_total": "Late arrivals (12-week count)",
    "excused_absences": "Excused absences (12-week count)",
    "unexcused_absences": "Unexcused absences (12-week count)",
    "missing_assignments_count": "Missing assignments",
    "quiz_attempts_count": "Quiz attempts",
    "resource_downloads_count": "Resource downloads",
    "forum_posts_count": "Forum / discussion posts",
    "mean_module_grade": "Mean grade across modules",
    "module_grade_variance": "Grade variance across modules",
    "wellbeing_score": "Wellbeing check-in score (1-10)",
    "wellbeing_flags": "Wellbeing concerns raised",
}


def humanise_feature(transformed_name: str) -> str:
    """Translate a ColumnTransformer output name into a readable label."""
    for prefix in ("num__", "bin__", "cat__"):
        if transformed_name.startswith(prefix):
            base = transformed_name[len(prefix):]
            if prefix == "cat__":
                for orig in CATEGORICAL_FEATURES:
                    if base.startswith(orig + "_"):
                        value = base[len(orig) + 1:]
                        orig_label = HUMAN_LABELS.get(orig, orig)
                        return f"{orig_label} = {value}"
                return base
            return HUMAN_LABELS.get(base, base)
    return HUMAN_LABELS.get(transformed_name, transformed_name)
