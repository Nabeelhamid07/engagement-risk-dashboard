"""
Build the 'Why this case was created' checklist for a given rule + student.

For every rule in the catalogue we define a small list of the actual conditions
that have to be true for it to fire. The helper here re-evaluates each condition
against the student's current record and returns formatted rows that the case
card can render side-by-side: condition · current value · threshold.

This is intentionally a plain-language reframing of the raw trigger expression
(e.g. ``withdraw_prob >= 0.40``) so the dashboard reader does not need to read
code to understand why a case was opened.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Value formatters
# ---------------------------------------------------------------------------
def _fmt_pct(v: Any) -> str:
    return f"{float(v) * 100:.1f}%"


def _fmt_raw_pct(v: Any) -> str:
    return f"{float(v):.1f}%"


def _fmt_int(v: Any) -> str:
    return f"{int(v)}"


def _fmt_float(v: Any) -> str:
    return f"{float(v):.2f}"


def _fmt_str(v: Any) -> str:
    return str(v) if v is not None else "-"


def _fmt_bool(v: Any) -> str:
    if isinstance(v, str):
        return "Yes" if v.lower() in ("true", "1", "yes", "t") else "No"
    return "Yes" if bool(v) else "No"


# ---------------------------------------------------------------------------
# Per-rule checklist definitions
# ---------------------------------------------------------------------------
# Each row: (label, student_record_key, value_formatter, threshold_text)
CHECKLIST_DEFS: dict[str, list[tuple[str, str, Callable[[Any], str], str]]] = {
    "WD_CRIT": [
        ("Predicted withdrawal probability exceeded critical threshold",
         "withdraw_prob", _fmt_pct, "at or above 40.0%"),
    ],
    "WD_HIGH": [
        ("Predicted withdrawal probability in targeted-outreach band",
         "withdraw_prob", _fmt_pct, "between 25.0% and 40.0%"),
    ],
    "WD_MOD": [
        ("Predicted withdrawal probability in elevated soft-nudge band",
         "withdraw_prob", _fmt_pct, "between 15.0% and 25.0%"),
    ],
    "ATT_CRIT": [
        ("Four-week attendance below institutional minimum",
         "attendance_rate", _fmt_raw_pct, "below 50.0%"),
    ],
    "ATT_WARN": [
        ("Four-week attendance in warning band",
         "attendance_rate", _fmt_raw_pct, "between 50.0% and 65.0%"),
    ],
    "VLE_DISENG": [
        ("Weeks since last VLE login",
         "weeks_since_last_login", _fmt_int, "3 or more"),
    ],
    "ATT_CONSECUTIVE": [
        ("Consecutive unexcused absences",
         "consecutive_missed_max", _fmt_int, "5 or more"),
    ],
    "FAIL_CRIT": [
        ("Predicted next-assessment failure probability",
         "fail_prob", _fmt_pct, "at or above 70.0%"),
        ("Days until next assessment",
         "days_to_nearest_assessment", _fmt_int, "7 or fewer"),
    ],
    "FAIL_HIGH": [
        ("Predicted next-assessment failure probability",
         "fail_prob", _fmt_pct, "at or above 50.0%"),
        ("Days until next assessment",
         "days_to_nearest_assessment", _fmt_int, "14 or fewer"),
    ],
    "BRIEF_UNREAD": [
        ("Assessment brief accessed",
         "accessed_upcoming_assessment_brief", _fmt_bool, "No"),
        ("Days until next assessment",
         "days_to_nearest_assessment", _fmt_int, "7 or fewer"),
    ],
    "GRADE_DROP": [
        ("Most recent assessment grade",
         "last_assessment_grade", _fmt_raw_pct, "below 50.0%"),
        ("Previous semester GPA",
         "previous_semester_gpa", _fmt_float, "2.50 or higher"),
    ],
    "LATE_PATTERN": [
        ("Late submissions on record",
         "assignments_submitted_late", _fmt_int, "3 or more"),
    ],
    "MISSING_ASSIGNMENTS": [
        ("Missing assignments across enrolled modules",
         "missing_assignments_count", _fmt_int, "3 or more"),
    ],
    "WELLBEING": [
        ("Engagement trend",
         "engagement_trend", _fmt_str, "Declining"),
        ("Four-week attendance",
         "attendance_rate", _fmt_raw_pct, "below 60.0%"),
        ("Balancing part-time work",
         "works_part_time", _fmt_bool, "Yes"),
    ],
    "WELLBEING_FLAGS": [
        ("Open wellbeing concerns logged",
         "wellbeing_flags", _fmt_int, "2 or more"),
    ],
    "FIN_SUPPORT": [
        ("Financial support category",
         "financial_support", _fmt_str, "Self-funded"),
        ("Balancing part-time work",
         "works_part_time", _fmt_bool, "Yes"),
        ("Predicted next-assessment failure probability",
         "fail_prob", _fmt_pct, "at or above 40.0%"),
    ],
    "DIS_SUPPORT": [
        ("Declared disability on record",
         "has_declared_disability", _fmt_bool, "Yes"),
        ("Predicted failure or withdrawal risk elevated",
         "fail_prob", _fmt_pct, "fail at or above 40.0%, or withdraw at or above 20.0%"),
    ],
    "INT_SUPPORT": [
        ("International student",
         "is_international", _fmt_bool, "Yes"),
        ("Four-week attendance",
         "attendance_rate", _fmt_raw_pct, "below 65.0%"),
        ("Weeks since last VLE login",
         "weeks_since_last_login", _fmt_int, "2 or more"),
    ],
    "NON_RESPONSIVE": [
        ("Outreach attempts in the last 30 days",
         "comms_30d_count", _fmt_int, "3 or more"),
        ("Responses received in the last 30 days",
         "comms_responded_30d", _fmt_int, "0"),
    ],
}


def case_reason_rows(rule_id: str, student_data: dict[str, Any]) -> list[dict[str, str]]:
    """Return formatted rows for the 'Why this case was created' block.

    Each row is a dict with keys: ``label``, ``current``, ``threshold``.
    If a rule is unknown, falls back to a single placeholder row.
    """
    defs = CHECKLIST_DEFS.get(rule_id)
    if not defs:
        return [{"label": "Rule trigger fired", "current": "-", "threshold": "-"}]

    rows: list[dict[str, str]] = []
    for label, key, formatter, threshold in defs:
        v = student_data.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            current = "-"
        else:
            try:
                current = formatter(v)
            except (ValueError, TypeError):
                current = str(v)
        rows.append({"label": label, "current": current, "threshold": threshold})
    return rows
