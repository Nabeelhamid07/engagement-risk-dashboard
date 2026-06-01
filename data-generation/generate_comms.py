"""
Synthetic communications log for the v3.1 dashboard.

Simulates ~30 days of outreach activity to students. Higher-risk students
receive more contact attempts; high-risk students are also less likely to
respond (which is precisely the operational problem the dashboard exists to
surface).

Outputs
-------
  data/communications_log.csv      one row per contact attempt
  data/student_comms_summary.csv   per-student aggregate (comms count,
                                   responses, last contact date, channel,
                                   non-responsive flag)

Depends on
----------
  data/students_v3.csv     (must include any v3.1 enrichment columns)
  data/predictions.csv     (for risk_band)

Run AFTER ``ml/score_cohort.py`` so risk bands are current.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Independent seed so comms RNG draws don't collide with earlier scripts.
RANDOM_SEED = 44
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AS_OF_DATE = datetime(2025, 3, 22).date()
WINDOW_DAYS = 30

# Channel mix: (channel, weight, [statuses], [status_probs])
CHANNELS = [
    ("email", 0.50, ["Sent", "Opened", "Replied", "No response"], [0.30, 0.20, 0.10, 0.40]),
    ("sms", 0.25, ["Sent", "Replied", "No response"], [0.55, 0.10, 0.35]),
    ("phone", 0.10, ["Answered", "Voicemail", "No answer"], [0.45, 0.25, 0.30]),
    ("meeting", 0.10, ["Attended", "No-show", "Rescheduled"], [0.55, 0.25, 0.20]),
    ("in_person", 0.05, ["Conversation held"], [1.00]),
]
CHANNEL_NAMES = [c[0] for c in CHANNELS]
CHANNEL_WEIGHTS = np.array([c[1] for c in CHANNELS])
CHANNEL_WEIGHTS = CHANNEL_WEIGHTS / CHANNEL_WEIGHTS.sum()

SUBJECTS_BY_CHANNEL: dict[str, list[str]] = {
    "email": [
        "Engagement check-in",
        "Attendance reminder",
        "Assessment support is available",
        "Wellbeing check-in",
        "We've noticed you missed class - are you OK?",
        "Resources to help you catch up",
    ],
    "sms": [
        "Quick check-in - drop by office hours?",
        "We noticed you missed class today",
        "Office hours running today - come by",
    ],
    "phone": [
        "Welfare call from your personal tutor",
        "Tutor check-in call",
        "Catch-up call",
    ],
    "meeting": [
        "1:1 retention meeting",
        "Tutor catch-up meeting",
        "Wellbeing meeting",
    ],
    "in_person": [
        "Corridor catch-up",
        "Class chat",
    ],
}

RESPONSE_STATUSES = {"Replied", "Answered", "Attended", "Conversation held"}


def _is_response(status: str) -> bool:
    return status in RESPONSE_STATUSES


def _channel_index() -> int:
    return int(rng.choice(len(CHANNELS), p=CHANNEL_WEIGHTS))


def _make_one_comm(student_id: str, risk_band: str) -> dict:
    chan_idx = _channel_index()
    channel, _, statuses, status_probs = CHANNELS[chan_idx]
    status = str(rng.choice(statuses, p=status_probs))

    # High-risk students are statistically less likely to respond. We model this
    # by converting a fraction of "would-have-responded" outcomes into non-
    # responses. This is realistic: disengaged students don't reply.
    if risk_band == "High" and _is_response(status) and rng.random() < 0.35:
        non_resp = [s for s in statuses if not _is_response(s)]
        if non_resp:
            status = str(rng.choice(non_resp))

    sent_days_ago = int(rng.integers(0, WINDOW_DAYS + 1))
    sent_date = AS_OF_DATE - timedelta(days=sent_days_ago)

    responded = _is_response(status)
    if responded:
        delay = int(rng.integers(0, 4))
        candidate = sent_date + timedelta(days=delay)
        response_date = min(candidate, AS_OF_DATE).isoformat()
    else:
        response_date = ""

    subject = str(rng.choice(SUBJECTS_BY_CHANNEL[channel]))

    return {
        "student_id": student_id,
        "channel": channel,
        "subject": subject,
        "sent_date": sent_date.isoformat(),
        "status": status,
        "responded": responded,
        "response_date": response_date,
    }


def _comms_for_student(student_id: str, risk_band: str) -> list[dict]:
    if risk_band == "High":
        lam = 3.5
    elif risk_band == "Medium":
        lam = 1.5
    else:
        lam = 0.3
    n = int(rng.poisson(lam))
    return [_make_one_comm(student_id, risk_band) for _ in range(n)]


def main() -> None:
    students = pd.read_csv(DATA_DIR / "students_v3.csv")
    preds = pd.read_csv(DATA_DIR / "predictions.csv")
    base = students.merge(preds[["student_id", "risk_band"]], on="student_id", how="left")
    base["risk_band"] = base["risk_band"].fillna("Low")

    all_rows: list[dict] = []
    for _, s in base.iterrows():
        all_rows.extend(_comms_for_student(s["student_id"], s["risk_band"]))

    comms = pd.DataFrame(all_rows)
    comms = comms.sort_values(["student_id", "sent_date"]).reset_index(drop=True)
    comms["comm_id"] = [f"COM{i + 1:05d}" for i in range(len(comms))]
    comms = comms[
        ["comm_id", "student_id", "channel", "subject", "sent_date", "status",
         "responded", "response_date"]
    ]

    # ---- Per-student aggregates ----
    if comms.empty:
        summary = pd.DataFrame(
            columns=[
                "student_id", "comms_30d_count", "comms_responded_30d",
                "last_contact_date", "last_contact_channel", "last_contact_status",
                "is_non_responsive",
            ]
        )
    else:
        agg = comms.groupby("student_id").agg(
            comms_30d_count=("comm_id", "count"),
            comms_responded_30d=("responded", "sum"),
            last_contact_date=("sent_date", "max"),
        ).reset_index()

        last_per = (
            comms.sort_values("sent_date")
            .groupby("student_id")
            .tail(1)[["student_id", "channel", "status"]]
            .rename(
                columns={
                    "channel": "last_contact_channel",
                    "status": "last_contact_status",
                }
            )
        )
        summary = agg.merge(last_per, on="student_id", how="left")
        summary["is_non_responsive"] = (
            (summary["comms_30d_count"] >= 2) & (summary["comms_responded_30d"] == 0)
        )

    comms.to_csv(DATA_DIR / "communications_log.csv", index=False)
    summary.to_csv(DATA_DIR / "student_comms_summary.csv", index=False)

    # ---- Console summary ----
    total = len(comms)
    n_students = len(base)
    n_active = len(summary)
    print("Communications log generated.")
    print(f"  Total contact attempts (last {WINDOW_DAYS}d): {total}")
    print(f"  Students with at least one contact:         {n_active} / {n_students}")
    if n_active:
        avg_per_active = summary["comms_30d_count"].mean()
        response_rate = (
            summary["comms_responded_30d"].sum() / summary["comms_30d_count"].sum() * 100
        )
        print(f"  Mean contacts per active student:            {avg_per_active:.2f}")
        print(f"  Cohort response rate (overall):              {response_rate:.1f}%")
        print(f"  Non-responsive students (2+ comms, 0 reply): {int(summary['is_non_responsive'].sum())}")
    print()
    if total:
        print("Channel mix")
        print(comms["channel"].value_counts().to_string())
        print()
        print("Status mix")
        print(comms["status"].value_counts().to_string())
    print()
    print("Files written")
    print("  data/communications_log.csv")
    print("  data/student_comms_summary.csv")


if __name__ == "__main__":
    main()
