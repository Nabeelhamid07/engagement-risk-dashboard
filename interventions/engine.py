"""
Rule engine: evaluates the rule catalogue against a feature frame enriched
with ML predictions and SHAP drivers, and returns an action log.

Behaviour designed to look like a real institutional policy layer:

  - Every rule that fires is recorded with a full audit trail (rule id,
    severity, action, routing team, policy basis, rationale with concrete
    numbers, predicted probabilities at the time of trigger, top SHAP driver).
  - Per-student de-duplication: within each support team, only the
    highest-severity action survives. This prevents a single student from
    being spammed with five overlapping nudges, which would feel like a game.
  - Output schema is stable so the dashboard can read it without coupling
    to the rule implementations.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .rules import RULES, SEVERITY_ORDER


def evaluate(rows: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Run every rule against every row, return raw action log (pre-dedup)."""
    records: list[dict] = []
    for _, row in rows.iterrows():
        for rule in RULES:
            try:
                fired = bool(rule.when(row))
            except Exception:
                fired = False
            if not fired:
                continue
            records.append(
                {
                    "student_id": row["student_id"],
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "action": rule.action,
                    "route_to": rule.route_to,
                    "policy_basis": rule.policy_basis,
                    "rationale": rule.rationale(row),
                    "withdraw_prob_at_trigger": float(row.get("withdraw_prob", 0.0)),
                    "fail_prob_at_trigger": float(row.get("fail_prob", 0.0)),
                    "top_driver_withdraw": row.get("top_driver_withdraw", ""),
                    "top_driver_fail": row.get("top_driver_fail", ""),
                }
            )
    if not records:
        return pd.DataFrame(columns=[
            "student_id", "rule_id", "rule_name", "severity", "action",
            "route_to", "policy_basis", "rationale",
            "withdraw_prob_at_trigger", "fail_prob_at_trigger",
            "top_driver_withdraw", "top_driver_fail",
        ])

    df = pd.DataFrame.from_records(records)
    df["_sev_rank"] = df["severity"].map(SEVERITY_ORDER)
    return df


def deduplicate(actions: pd.DataFrame) -> pd.DataFrame:
    """Keep at most one action per (student, support team) — the highest severity."""
    if actions.empty:
        return actions

    deduped = (
        actions.sort_values(
            ["student_id", "route_to", "_sev_rank"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["student_id", "route_to"], keep="first")
        .drop(columns=["_sev_rank"])
        .reset_index(drop=True)
    )
    return deduped


def assign_action_ids(actions: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Add action_id, triggered_on, status columns."""
    if actions.empty:
        actions["action_id"] = []
        actions["triggered_on"] = []
        actions["status"] = []
        return actions

    actions = actions.copy()
    # Sort by severity (most urgent first) so triage queue is meaningful.
    actions["_sev_rank"] = actions["severity"].map(SEVERITY_ORDER)
    actions = actions.sort_values(
        ["_sev_rank", "withdraw_prob_at_trigger", "fail_prob_at_trigger"],
        ascending=[False, False, False],
    ).drop(columns=["_sev_rank"]).reset_index(drop=True)

    actions["action_id"] = [f"ACT{i + 1:05d}" for i in range(len(actions))]
    actions["triggered_on"] = as_of.isoformat()
    actions["status"] = "Pending"
    return actions


COLUMN_ORDER = [
    "action_id",
    "triggered_on",
    "student_id",
    "severity",
    "rule_id",
    "rule_name",
    "action",
    "route_to",
    "policy_basis",
    "rationale",
    "withdraw_prob_at_trigger",
    "fail_prob_at_trigger",
    "top_driver_withdraw",
    "top_driver_fail",
    "status",
]


def run(rows: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """End-to-end: evaluate, dedupe, assign IDs, return clean DataFrame."""
    raw = evaluate(rows, as_of)
    deduped = deduplicate(raw)
    final = assign_action_ids(deduped, as_of)
    if final.empty:
        return pd.DataFrame(columns=COLUMN_ORDER)
    return final[COLUMN_ORDER]
