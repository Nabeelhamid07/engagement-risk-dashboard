"""
v3 synthetic dataset for ML-based student dropout / failure prediction.

Why v3 exists
-------------
The original generator computed a deterministic composite risk score from
hand-picked weights and wrote that score back as the "label". A machine
learning model trained on that data would just be rediscovering the formula.
That's a tautology, not prediction.

v3 fixes this by inverting the generative process:

  1. Each student has a hidden ``latent_engagement`` value (a continuous
     score, roughly Normal(0, 1)). This represents their *true* underlying
     engagement and is never written to any CSV.
  2. All observable features (attendance, VLE logins, hours, materials,
     submissions, grades, GPA, brief access, trend) are sampled as
     **noisy observations** of the latent state plus context.
  3. Each student has a ``latent_slope`` describing whether they are on a
     trajectory of improvement or decline. This drives the weekly snapshots
     and a slow shift in features over the 12-week window.
  4. The outcome labels (``withdrew``, ``failed_next_assessment``) are
     sampled from sigmoid(latent + context + INDEPENDENT GAUSSIAN NOISE).
     They are therefore correlated with the features but not deterministic
     functions of them - an ML model has to recover the relationship through
     the noisy proxies, which is the whole point.

Outputs (in ../data/)
---------------------
  - students_v3.csv          one row per student, includes outcomes + split
  - weekly_engagement.csv    12 weekly snapshots per student
  - interventions_v3.csv     historical interventions, biased toward risky
  - assessments_v3.csv       cohort-level assessment table

The older v1/v2 CSVs (``students_data.csv`` etc.) are intentionally not
overwritten so the existing dashboard keeps working until Stage 4 swaps it.
"""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)
fake = Faker("en_GB")
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Cohort knobs
# ---------------------------------------------------------------------------
N_STUDENTS = 600
N_WEEKS = 12                       # weeks of history per student
AS_OF_DATE = datetime(2025, 3, 22).date()  # "today" in the simulation

PROGRAMS = [
    "Computer Science",
    "Data Science",
    "Business Analytics",
    "Engineering",
    "Psychology",
]
FINANCIAL_SUPPORT = ["Full", "Partial", "Self-funded"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z]+", "", s)
    return s or "student"


# ---------------------------------------------------------------------------
# Per-student generation
# ---------------------------------------------------------------------------
def _generate_student(idx: int) -> tuple[dict, list[dict]]:
    """Return (student_row, list_of_weekly_rows) for a single student."""

    student_id = f"STU{idx + 1:04d}"

    # ---- Hidden latent state (never exported) ----
    latent = float(rng.normal(0.0, 1.0))
    latent_slope = float(rng.normal(0.0, 0.35))  # trajectory direction

    # ---- Independent demographics ----
    first = fake.first_name()
    last = fake.last_name()
    name = f"{first} {last}"
    email = f"{_slug(first)}.{_slug(last)}@student.chester.ac.uk"
    program = str(rng.choice(PROGRAMS))
    year_of_study = int(rng.integers(1, 4))
    enrol_month = int(rng.choice([9, 10]))
    enrol_day = int(rng.integers(1, 29))
    enrol_date = datetime(2024, enrol_month, enrol_day).date()

    is_commuter = bool(rng.random() < 0.30)
    works_part_time = bool(rng.random() < 0.40)
    has_declared_disability = bool(rng.random() < 0.15)
    is_international = bool(rng.random() < 0.25)
    financial_support = str(rng.choice(FINANCIAL_SUPPORT, p=[0.35, 0.35, 0.30]))

    # Context penalties on engagement features (NOT on latent).
    commuter_att_pen = 3.5 if is_commuter else 0.0
    pt_hours_pen = 1.5 if works_part_time else 0.0

    # ---- Weekly snapshots: noisy observations of a drifting latent ----
    wk_records: list[dict] = []
    for w in range(1, N_WEEKS + 1):
        # Latent at this week: anchored at "now" (week N_WEEKS), drifts back
        # in time via the slope, plus weekly noise.
        weeks_back = N_WEEKS - w
        week_latent = (
            latent
            - latent_slope * (weeks_back / 4.0)  # earlier weeks reflect past state
            + float(rng.normal(0.0, 0.15))
        )

        # Weekly attendance: linear in latent + commuter penalty + noise.
        wk_att = 75.0 + 12.0 * week_latent - commuter_att_pen + float(rng.normal(0.0, 7.0))
        wk_att = float(np.clip(wk_att, 0.0, 100.0))

        # Weekly VLE logins: Poisson around a latent-dependent rate.
        login_rate = max(0.1, 5.0 + 3.5 * week_latent)
        wk_logins = int(min(rng.poisson(login_rate), 25))

        # Weekly VLE hours: correlated with logins, minus part-time pressure.
        wk_hours = 0.6 * wk_logins + float(rng.normal(0.0, 1.4)) - pt_hours_pen
        wk_hours = float(np.clip(wk_hours, 0.0, 25.0))

        # Materials accessed (lecture notes, recordings, etc).
        mat_rate = max(0.2, 1.5 + 1.2 * week_latent)
        wk_materials = int(min(rng.poisson(mat_rate), 12))

        week_ending = AS_OF_DATE - timedelta(days=(N_WEEKS - w) * 7)

        wk_records.append(
            {
                "student_id": student_id,
                "week": w,
                "week_ending": week_ending.isoformat(),
                "attendance_pct": round(wk_att, 2),
                "vle_logins": wk_logins,
                "vle_hours": round(wk_hours, 2),
                "materials_accessed": wk_materials,
            }
        )

    wk_df = pd.DataFrame(wk_records)
    latest = wk_df.iloc[-1]
    last_4 = wk_df.tail(4)
    first_8 = wk_df.head(8)

    # ---- Cross-sectional features (derived from weekly history) ----
    attendance_rate = float(last_4["attendance_pct"].mean())  # rolling 4-week mean
    vle_logins_last_week = int(latest["vle_logins"])
    vle_time_hours_last_week = float(latest["vle_hours"])
    materials_accessed_last_4w = int(last_4["materials_accessed"].sum())

    att_slope_4w = float(last_4["attendance_pct"].mean() - first_8["attendance_pct"].mean())
    vle_logins_slope_4w = float(last_4["vle_logins"].mean() - first_8["vle_logins"].mean())
    vle_hours_slope_4w = float(last_4["vle_hours"].mean() - first_8["vle_hours"].mean())

    if att_slope_4w > 3.0:
        engagement_trend = "Improving"
    elif att_slope_4w < -3.0:
        engagement_trend = "Declining"
    else:
        engagement_trend = "Stable"

    # ---- Submission behaviour (correlates with latent + observed attendance) ----
    late_prob = float(np.clip(0.10 - 0.05 * latent + 0.002 * max(0.0, 70.0 - attendance_rate), 0.01, 0.50))
    assignments_submitted_late = int(rng.binomial(3, min(late_prob * 1.2, 0.95)))
    on_time_prob = float(np.clip(0.50 + 0.16 * latent + 0.003 * (attendance_rate - 70.0), 0.05, 0.98))
    assignments_submitted_on_time = int(
        rng.binomial(max(0, 8 - assignments_submitted_late), on_time_prob)
    )

    # ---- Grade history ----
    grade_mu = 60.0 + 12.0 * latent + 0.15 * (attendance_rate - 75.0) + 0.5 * vle_logins_last_week
    last_assessment_grade = int(np.clip(rng.normal(grade_mu, 9.0), 0, 100))

    gpa_mu = 1.5 + 0.025 * last_assessment_grade + 0.10 * latent
    previous_semester_gpa = float(np.clip(rng.normal(gpa_mu, 0.35), 0.0, 4.0))

    # ---- Upcoming-assessment behaviour ----
    brief_prob = float(np.clip(0.20 + 0.45 * latent + 0.003 * (attendance_rate - 70.0), 0.05, 0.97))
    accessed_upcoming_assessment_brief = bool(rng.random() < brief_prob)
    days_to_nearest_assessment = int(rng.integers(3, 22))

    # ---- OUTCOMES (the ML targets) ----
    # Withdrawal: rare event driven by chronic disengagement + life stress + noise.
    withdraw_logit = (
        -3.10
        + -2.10 * latent
        + -1.60 * latent_slope
        + 0.40 * (1 if works_part_time else 0)
        + 0.50 * (1 if financial_support == "Self-funded" else 0)
        + 0.30 * (1 if is_commuter else 0)
        + float(rng.normal(0.0, 0.55))  # INDEPENDENT noise -> not learnable from features alone
    )
    withdrew = bool(rng.random() < _sigmoid(withdraw_logit))

    # Next-assessment failure: more common, driven by short-term engagement.
    fail_logit = (
        -1.40
        + -1.70 * latent
        + -0.85 * latent_slope
        + -0.018 * (attendance_rate - 75.0)
        + 0.40 * (1 if assignments_submitted_late >= 2 else 0)
        + -0.25 * (1 if accessed_upcoming_assessment_brief else 0)
        + float(rng.normal(0.0, 0.60))
    )
    failed_next_assessment = bool(rng.random() < _sigmoid(fail_logit))

    student_row = {
        # identity
        "student_id": student_id,
        "student_name": name,
        "email": email,
        "program": program,
        "year_of_study": year_of_study,
        "enrollment_date": enrol_date.isoformat(),
        # context
        "is_commuter": is_commuter,
        "works_part_time": works_part_time,
        "has_declared_disability": has_declared_disability,
        "is_international": is_international,
        "financial_support": financial_support,
        # engagement (cross-sectional, derived from weekly snapshots)
        "attendance_rate": round(attendance_rate, 2),
        "attendance_slope_4w": round(att_slope_4w, 2),
        "vle_logins_last_week": vle_logins_last_week,
        "vle_time_hours_last_week": round(vle_time_hours_last_week, 2),
        "vle_logins_slope_4w": round(vle_logins_slope_4w, 2),
        "vle_hours_slope_4w": round(vle_hours_slope_4w, 2),
        "materials_accessed_last_4w": materials_accessed_last_4w,
        "engagement_trend": engagement_trend,
        # submissions
        "assignments_submitted_on_time": assignments_submitted_on_time,
        "assignments_submitted_late": assignments_submitted_late,
        # grades
        "last_assessment_grade": last_assessment_grade,
        "previous_semester_gpa": round(previous_semester_gpa, 2),
        # upcoming assessment
        "accessed_upcoming_assessment_brief": accessed_upcoming_assessment_brief,
        "days_to_nearest_assessment": days_to_nearest_assessment,
        # outcomes (ML targets)
        "withdrew": withdrew,
        "failed_next_assessment": failed_next_assessment,
    }

    return student_row, wk_records


# ---------------------------------------------------------------------------
# Interventions + assessments (auxiliary tables)
# ---------------------------------------------------------------------------
def _build_interventions(students: pd.DataFrame) -> pd.DataFrame:
    """Historical interventions, biased toward students who later struggle.

    Generated AFTER outcomes so interventions can be used as features without
    creating circular dependence: they're a proxy for past concern, not a
    cause of the outcome.
    """
    n_total = int(rng.integers(280, 361))
    student_ids = students["student_id"].tolist()
    risky_ids = students.loc[
        students["withdrew"] | students["failed_next_assessment"], "student_id"
    ].tolist()

    types = ["Email", "SMS", "Phone Call", "In-Person Meeting"]
    type_probs = [0.40, 0.30, 0.18, 0.12]
    reasons = [
        "Low Attendance",
        "Missed Deadline",
        "Declining Grades",
        "No VLE Activity",
        "Mental Health Concern",
    ]

    rows: list[dict] = []
    for k in range(n_total):
        if risky_ids and rng.random() < 0.65:
            sid = str(rng.choice(risky_ids))
        else:
            sid = str(rng.choice(student_ids))
        date = AS_OF_DATE - timedelta(days=int(rng.integers(0, 91)))
        itype = str(rng.choice(types, p=type_probs))
        reason = str(rng.choice(reasons))
        responded = bool(rng.random() < 0.55)
        if responded:
            response_time_hours = int(rng.integers(0, 169))
            improved = bool(rng.random() < 0.65)
        else:
            response_time_hours = ""
            improved = bool(rng.random() < 0.18)
        rows.append(
            {
                "intervention_id": f"INT{k + 1:04d}",
                "student_id": sid,
                "intervention_date": date.isoformat(),
                "intervention_type": itype,
                "reason": reason,
                "student_responded": responded,
                "response_time_hours": response_time_hours,
                "engagement_improved": improved,
            }
        )
    return pd.DataFrame(rows)


def _build_assessments() -> pd.DataFrame:
    n_total = int(rng.integers(32, 41))
    prefixes = ["CS", "DA", "BA", "EN", "PS"]
    titles_pool = [
        "Research Methods Essay",
        "Data Analysis Project",
        "Statistics Midterm Exam",
        "Group Presentation",
        "Literature Review",
        "Programming Coursework",
        "Case Study Report",
        "Lab Practical Assessment",
        "Final Examination",
        "Reflective Portfolio",
        "Research Proposal",
        "Team Project Deliverable",
    ]
    types = ["Essay", "Project", "Exam", "Presentation"]
    weights = [10, 15, 20, 30, 40, 50]

    rows: list[dict] = []
    for k in range(n_total):
        aid = f"ASS{k + 1:04d}"
        code = f"{rng.choice(prefixes)}{rng.integers(101, 401)}"
        title = str(rng.choice(titles_pool))
        due = AS_OF_DATE + timedelta(days=int(rng.integers(1, 57)))
        atype = str(rng.choice(types))
        weight = int(rng.choice(weights))
        enrolled = int(rng.integers(40, 181))
        accessed_p = float(rng.uniform(0.35, 0.92))
        accessed = int(rng.binomial(enrolled, accessed_p))
        rows.append(
            {
                "assessment_id": aid,
                "module_code": code,
                "assessment_title": title,
                "due_date": due.isoformat(),
                "assessment_type": atype,
                "weight_percentage": weight,
                "students_enrolled": enrolled,
                "students_accessed_brief": accessed,
                "students_not_accessed": enrolled - accessed,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def _stratified_split(df: pd.DataFrame, target: str = "withdrew",
                      test_frac: float = 0.15, val_frac: float = 0.15) -> pd.Series:
    """Return a Series of 'train' / 'val' / 'test' stratified on target."""
    split_col = pd.Series("train", index=df.index, dtype=object)
    for value in df[target].unique():
        idx = df.index[df[target] == value].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        split_col.loc[idx[:n_test]] = "test"
        split_col.loc[idx[n_test:n_test + n_val]] = "val"
    return split_col


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    student_rows: list[dict] = []
    weekly_rows: list[dict] = []

    for i in range(N_STUDENTS):
        s_row, w_rows = _generate_student(i)
        student_rows.append(s_row)
        weekly_rows.extend(w_rows)

    students = pd.DataFrame(student_rows)
    weekly = pd.DataFrame(weekly_rows)

    # Interventions depend on outcomes -> build after.
    interventions = _build_interventions(students)
    assessments = _build_assessments()

    # Attach intervention history aggregates back to students.
    iv_agg = (
        interventions.groupby("student_id")
        .agg(
            prior_intervention_count=("intervention_id", "count"),
            prior_interventions_improved=("engagement_improved", "sum"),
        )
        .reset_index()
    )
    students = students.merge(iv_agg, on="student_id", how="left")
    students["prior_intervention_count"] = students["prior_intervention_count"].fillna(0).astype(int)
    students["prior_interventions_improved"] = students["prior_interventions_improved"].fillna(0).astype(int)

    # Train / val / test split.
    students["split"] = _stratified_split(students, target="withdrew")

    # Write CSVs.
    students.to_csv(DATA_DIR / "students_v3.csv", index=False)
    weekly.to_csv(DATA_DIR / "weekly_engagement.csv", index=False)
    interventions.to_csv(DATA_DIR / "interventions_v3.csv", index=False)
    assessments.to_csv(DATA_DIR / "assessments_v3.csv", index=False)

    # ---- Summary ----
    n = len(students)
    w_rate = students["withdrew"].mean() * 100
    f_rate = students["failed_next_assessment"].mean() * 100
    print("v3 dataset generated.")
    print(f"  Students:           {n}")
    print(f"  Withdrew:           {students['withdrew'].sum()}  ({w_rate:.1f}%)")
    print(f"  Failed next assmt:  {students['failed_next_assessment'].sum()}  ({f_rate:.1f}%)")
    print(
        "  Split sizes:        "
        f"train={(students['split'] == 'train').sum()}, "
        f"val={(students['split'] == 'val').sum()}, "
        f"test={(students['split'] == 'test').sum()}"
    )

    # Per-split withdraw rates (should be similar -> stratification working).
    print("  Withdrew per split:")
    for s in ("train", "val", "test"):
        sub = students[students["split"] == s]
        print(f"    {s:<5}  n={len(sub):>3}  withdraw_rate={sub['withdrew'].mean() * 100:.1f}%")

    print("  Engagement trend distribution:")
    for k, v in students["engagement_trend"].value_counts().items():
        print(f"    {k:<10} {v}")

    print("  Weekly snapshot rows:", len(weekly))
    print("  Intervention rows:   ", len(interventions))
    print("  Assessment rows:     ", len(assessments))
    print()
    print("Files written:")
    for f in ("students_v3.csv", "weekly_engagement.csv", "interventions_v3.csv", "assessments_v3.csv"):
        print(f"  data/{f}")


if __name__ == "__main__":
    generate()
