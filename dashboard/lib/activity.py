"""
Build a per-case 'Case activity' timeline.

Each rule firing triggers a deterministic chain of system steps: notifying the
support team, queuing student outreach, watching for a response, and scheduling
the eventual policy action. We derive the state of each step from:

  - the action's trigger timestamp (always present),
  - any matching entry in the communications log (if outreach has actually been
    dispatched), and
  - per-severity SLA windows.

The result is a list of dicts the dashboard renders as a vertical step list.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd


# Rules where the system is expected to dispatch student-facing outreach
# directly to the student (in addition to notifying the team). Other rules
# (e.g. financial referral, disability referral) only involve internal
# notifications - no automated student-facing email/SMS.
RULES_WITH_STUDENT_OUTREACH = {
    "WD_CRIT",
    "WD_HIGH",
    "WD_MOD",
    "ATT_CRIT",
    "ATT_WARN",
    "VLE_DISENG",
    "ATT_CONSECUTIVE",
    "FAIL_CRIT",
    "FAIL_HIGH",
    "BRIEF_UNREAD",
    "MISSING_ASSIGNMENTS",
    "WELLBEING",
    "WELLBEING_FLAGS",
    "INT_SUPPORT",
    "NON_RESPONSIVE",
}


# Escalation window in days, by severity, after which a non-response is
# pushed up to the next layer of support.
ESCALATION_DAYS_BY_SEVERITY = {
    "escalation": 2,
    "high": 3,
    "medium": 5,
    "low": 7,
}


def _fmt_ts(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %d, %H:%M")


def _fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %d")


def case_activity_steps(
    action: dict[str, Any],
    student_comms: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Return the list of activity steps for one case.

    Each step is a dict with keys: ``label``, ``detail`` (optional context),
    ``timestamp`` (display string or empty), ``status`` (``done`` /
    ``pending`` / ``scheduled``).
    """
    rule_id = str(action.get("rule_id", ""))
    rule_name = str(action.get("rule_name", "Rule trigger"))
    triggered_on = pd.Timestamp(action["triggered_on"])
    team = str(action.get("route_to", "Support team"))
    severity = str(action.get("severity", "medium")).lower()
    action_text = str(action.get("action", "")).strip()

    # Has outreach gone out since the rule fired?
    outreach: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    if student_comms is not None and not student_comms.empty:
        after = student_comms[student_comms["sent_date"] >= triggered_on]
        if not after.empty:
            outreach = after.sort_values("sent_date").iloc[0].to_dict()
            if bool(outreach.get("responded", False)):
                response = outreach

    steps: list[dict[str, Any]] = []

    # Step 1 - Rule fired (always done)
    steps.append({
        "label": f"Rule fired: {rule_name}",
        "detail": f"Routed to {team}",
        "timestamp": _fmt_ts(triggered_on),
        "status": "done",
    })

    # Step 2 - Team notified
    steps.append({
        "label": f"{team} notified by system",
        "detail": "Internal alert email queued and delivered",
        "timestamp": _fmt_ts(triggered_on + timedelta(minutes=1)),
        "status": "done",
    })

    # Step 3 - Student outreach (only for rules that involve student contact)
    if rule_id in RULES_WITH_STUDENT_OUTREACH:
        if outreach is not None:
            channel = str(outreach.get("channel", "email")).lower()
            sent = pd.Timestamp(outreach["sent_date"])
            subj = str(outreach.get("subject", "")).strip()
            steps.append({
                "label": f"Auto-{channel} sent to student",
                "detail": f"Subject: \"{subj}\"" if subj else "Outreach dispatched",
                "timestamp": _fmt_ts(sent),
                "status": "done",
            })
        else:
            steps.append({
                "label": "Student outreach queued",
                "detail": "Auto-email will be sent shortly",
                "timestamp": "",
                "status": "pending",
            })

    # Step 4 - Response watcher
    if response is not None:
        resp_date_raw = response.get("response_date")
        try:
            resp_date = pd.Timestamp(resp_date_raw)
            resp_ts = _fmt_ts(resp_date)
        except Exception:
            resp_ts = ""
        steps.append({
            "label": "Student responded",
            "detail": "Response recorded by system",
            "timestamp": resp_ts,
            "status": "done",
        })
    elif outreach is not None:
        sla_days = ESCALATION_DAYS_BY_SEVERITY.get(severity, 5)
        escalate_on = pd.Timestamp(outreach["sent_date"]) + timedelta(days=sla_days)
        steps.append({
            "label": "Awaiting student response",
            "detail": f"Escalates if no reply by {_fmt_date(escalate_on)}",
            "timestamp": "",
            "status": "pending",
        })

    # Step 5 - Policy action (final scheduled step)
    if action_text:
        if response is not None:
            steps.append({
                "label": "Recommended next step",
                "detail": action_text,
                "timestamp": "",
                "status": "scheduled",
            })
        else:
            steps.append({
                "label": "Scheduled policy action",
                "detail": action_text,
                "timestamp": "",
                "status": "scheduled",
            })

    return steps
