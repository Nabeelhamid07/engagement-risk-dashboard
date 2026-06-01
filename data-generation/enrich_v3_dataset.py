"""
v3.1 enrichment - layers richer behavioural and module-level data on top of the
base v3 cohort. Run AFTER ``generate_v3_dataset.py``.

What gets added
---------------
1. Module catalogue + per-student enrolments + per-(student, module) engagement.
2. Daily attendance log (5 days x 12 weeks) with Late / Excused / Absent
   granularity. Reconstructs to the existing weekly_engagement aggregates.
3. Derived behavioural counters: consecutive_missed_max, late_arrivals_total,
   excused_absences, unexcused_absences, missing_assignments_count,
   mean_module_grade, module_grade_variance.
4. New raw engagement counts: quiz_attempts_count, resource_downloads_count,
   forum_posts_count.
5. Simple wellbeing layer (synthetic, clearly labelled): wellbeing_score (1-10),
   wellbeing_flags (count), wellbeing_last_checkin (date).

Inputs (read-only)
------------------
  ../data/students_v3.csv          (existing - base outputs preserved)
  ../data/weekly_engagement.csv    (existing)

Outputs
-------
  ../data/modules.csv              (new)
  ../data/enrolments.csv           (new)
  ../data/module_engagement.csv    (new)
  ../data/daily_attendance.csv     (new)
  ../data/students_v3.csv          (overwritten with extra columns appended)

The outcome labels (`withdrew`, `failed_next_assessment`) are NOT regenerated.
They came from the latent state in the base generator and must stay stable so
the ML retrain in 5b is comparable to the original v3 results.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility (different seed from the base generator so RNG draws do not
# overlap with the latent state generation).
# ---------------------------------------------------------------------------
RANDOM_SEED = 43
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

AS_OF_DATE = datetime(2025, 3, 22).date()
N_WEEKS = 12

# ---------------------------------------------------------------------------
# Module catalogue
# ---------------------------------------------------------------------------
# (code, name, programme, level)
MODULE_CATALOGUE = [
    ("CS101", "Programming Fundamentals", "Computer Science", 1),
    ("CS201", "Data Structures and Algorithms", "Computer Science", 2),
    ("CS301", "Software Engineering", "Computer Science", 3),
    ("DS101", "Statistics and Probability", "Data Science", 1),
    ("DS201", "Machine Learning Foundations", "Data Science", 2),
    ("DS301", "Big Data and Cloud Computing", "Data Science", 3),
    ("BA101", "Foundations of Business", "Business Analytics", 1),
    ("BA201", "Marketing Analytics", "Business Analytics", 2),
    ("BA301", "Strategic Decision Analytics", "Business Analytics", 3),
    ("EN101", "Engineering Mathematics", "Engineering", 1),
    ("EN201", "Mechanical Systems", "Engineering", 2),
    ("EN301", "Engineering Project Management", "Engineering", 3),
    ("PS101", "Introduction to Psychology", "Psychology", 1),
    ("PS201", "Cognitive Psychology", "Psychology", 2),
    ("PS301", "Research Methods in Psychology", "Psychology", 3),
    ("GEN001", "Academic Skills and Communication", "Cross-programme", 1),
    ("GEN002", "Research Ethics and Integrity", "Cross-programme", 2),
]

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _consecutive_missed(statuses: list[str]) -> int:
    """Longest streak of 'Absent' entries in a chronologically-ordered list.

    'Excused' and 'Late' don't break the streak in either direction — they're
    not counted. Only an unexplained absence counts as a 'missed session'.
    """
    longest = current = 0
    for status in statuses:
        if status == "Absent":
            current += 1
            longest = max(longest, current)
        elif status in ("Present", "Late"):
            current = 0
        # Excused: doesn't reset, doesn't extend
    return longest


def _build_modules_table() -> pd.DataFrame:
    return pd.DataFrame(
        MODULE_CATALOGUE,
        columns=["module_code", "module_name", "programme", "level"],
    )


def _build_enrolments(students: pd.DataFrame) -> pd.DataFrame:
    """3-4 modules per student, weighted toward same programme + general modules."""
    rows = []
    for _, s in students.iterrows():
        prog = s["program"]
        year = int(s["year_of_study"])

        core = next(
            (m for m in MODULE_CATALOGUE if m[2] == prog and m[3] == year),
            None,
        )
        chosen: list[str] = []
        if core is not None:
            chosen.append(core[0])

        n_extra = int(rng.choice([2, 3], p=[0.55, 0.45]))
        pool = [m for m in MODULE_CATALOGUE if not chosen or m[0] != chosen[0]]
        weights = np.array(
            [3.0 if m[2] == prog else 2.5 if m[2] == "Cross-programme" else 1.0 for m in pool]
        )
        weights = weights / weights.sum()

        n_extra = min(n_extra, len(pool))
        extra_idx = rng.choice(len(pool), size=n_extra, replace=False, p=weights)
        for i in extra_idx:
            chosen.append(pool[int(i)][0])

        for code in chosen:
            rows.append({"student_id": s["student_id"], "module_code": code})
    return pd.DataFrame(rows)


def _build_module_engagement(students: pd.DataFrame, enrolments: pd.DataFrame) -> pd.DataFrame:
    """Per (student, module) attendance + grade + engagement composite.

    Per-module values are noisy variants of the student-level cross-sectional
    figures so that aggregating module-level back to student-level recovers
    approximately the same numbers.
    """
    s_lookup = students.set_index("student_id")
    rows = []
    for _, e in enrolments.iterrows():
        sid = e["student_id"]
        mcode = e["module_code"]
        srow = s_lookup.loc[sid]

        mod_att = float(np.clip(srow["attendance_rate"] + rng.normal(0, 8.0), 0, 100))
        mod_grade = float(np.clip(srow["last_assessment_grade"] + rng.normal(0, 9.0), 0, 100))

        assessments_total = int(rng.choice([2, 3], p=[0.4, 0.6]))
        complete_prob = float(np.clip(0.55 + 0.005 * mod_att, 0.25, 0.97))
        completed = int(rng.binomial(assessments_total, complete_prob))
        missed = assessments_total - completed

        mod_engagement = float(
            np.clip(0.6 * mod_att + 0.4 * mod_grade + rng.normal(0, 5.0), 0, 100)
        )

        rows.append(
            {
                "student_id": sid,
                "module_code": mcode,
                "module_attendance_pct": round(mod_att, 2),
                "module_grade": round(mod_grade, 1),
                "assessments_total": assessments_total,
                "assessments_completed": completed,
                "assessments_missed": missed,
                "module_engagement_score": round(mod_engagement, 1),
            }
        )
    return pd.DataFrame(rows)


def _build_daily_attendance(weekly: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct daily M-F attendance from the weekly aggregates.

    For each student-week, the number of attended days is round(weekly_pct/100 * 5).
    Attended days are randomly distributed across M-F, with a small fraction of
    attended days flagged 'Late' and a small fraction of absent days 'Excused'.
    """
    rows = []
    for (sid, _wk_idx), wk_group in weekly.groupby(["student_id", "week"]):
        wk = wk_group.iloc[0]
        wk_att = float(wk["attendance_pct"])
        week_num = int(wk["week"])
        week_ending = pd.to_datetime(wk["week_ending"]).date()
        week_start = week_ending - timedelta(days=4)

        attended_count = int(round(wk_att / 100.0 * 5.0))
        attended_count = max(0, min(5, attended_count))

        if attended_count > 0:
            present_days = sorted(
                int(i) for i in rng.choice(5, size=attended_count, replace=False)
            )
        else:
            present_days = []

        for i, day_name in enumerate(DAYS):
            day_date = week_start + timedelta(days=i)
            if i in present_days:
                status = "Late" if rng.random() < 0.12 else "Present"
            else:
                status = "Excused" if rng.random() < 0.18 else "Absent"
            rows.append(
                {
                    "student_id": sid,
                    "week": week_num,
                    "day": day_name,
                    "date": day_date.isoformat(),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def _derive_attendance_counters(daily: pd.DataFrame) -> pd.DataFrame:
    """consecutive_missed_max, late_arrivals_total, excused_absences, unexcused_absences."""
    daily = daily.sort_values(["student_id", "week", "day"])
    grouped = daily.groupby("student_id")

    rows = []
    for sid, g in grouped:
        statuses = g["status"].tolist()
        rows.append(
            {
                "student_id": sid,
                "consecutive_missed_max": _consecutive_missed(statuses),
                "late_arrivals_total": int((g["status"] == "Late").sum()),
                "excused_absences": int((g["status"] == "Excused").sum()),
                "unexcused_absences": int((g["status"] == "Absent").sum()),
            }
        )
    return pd.DataFrame(rows)


def _new_engagement_counts(students: pd.DataFrame) -> pd.DataFrame:
    """Quiz attempts, resource downloads, forum posts.

    All correlated positively with attendance (a proxy for the latent engagement
    we no longer have direct access to) plus Poisson noise.
    """
    att = students["attendance_rate"].values
    quiz = rng.poisson(np.clip(2.0 + 0.05 * np.maximum(0, att - 50), 0.5, 25))
    downloads = rng.poisson(np.clip(5.0 + 0.10 * np.maximum(0, att - 50), 1.0, 50))
    forum = rng.poisson(np.clip(0.5 + 0.03 * np.maximum(0, att - 50), 0.2, 18))
    return pd.DataFrame(
        {
            "student_id": students["student_id"].values,
            "quiz_attempts_count": np.clip(quiz, 0, 30).astype(int),
            "resource_downloads_count": np.clip(downloads, 0, 60).astype(int),
            "forum_posts_count": np.clip(forum, 0, 20).astype(int),
        }
    )


def _wellbeing_layer(students: pd.DataFrame) -> pd.DataFrame:
    """Simple synthetic wellbeing layer. Clearly NOT clinical data.

    - wellbeing_score (1-10): mild correlation with attendance, trend, and
      declared support needs. Noisy.
    - wellbeing_flags (count): concerns raised by tutors / peers. Higher for
      low-attendance students and those with declared support needs.
    - wellbeing_last_checkin (date in the last 30 days).
    """
    n = len(students)
    score = np.clip(
        np.round(
            7.0
            + 0.04 * (students["attendance_rate"].values - 75.0)
            + 0.3 * (students["engagement_trend"].values == "Improving").astype(int)
            - 0.5 * students["has_declared_disability"].astype(int).values
            + rng.normal(0, 1.5, size=n)
        ),
        1,
        10,
    ).astype(int)

    flags = rng.poisson(
        np.clip(
            0.5
            + 0.02 * np.maximum(0, 60.0 - students["attendance_rate"].values)
            + 0.5 * students["has_declared_disability"].astype(int).values,
            0,
            8,
        )
    ).astype(int)

    last = [
        (AS_OF_DATE - timedelta(days=int(rng.integers(1, 31)))).isoformat()
        for _ in range(n)
    ]

    return pd.DataFrame(
        {
            "student_id": students["student_id"].values,
            "wellbeing_score": score,
            "wellbeing_flags": flags,
            "wellbeing_last_checkin": last,
        }
    )


def _module_aggregates_per_student(mod_engagement: pd.DataFrame) -> pd.DataFrame:
    agg = mod_engagement.groupby("student_id").agg(
        mean_module_grade=("module_grade", "mean"),
        module_grade_variance=("module_grade", "var"),
    ).reset_index()
    agg["mean_module_grade"] = agg["mean_module_grade"].round(1)
    agg["module_grade_variance"] = agg["module_grade_variance"].fillna(0.0).round(1)
    return agg


def _missing_assignments(students: pd.DataFrame) -> np.ndarray:
    """Approximate missing-assignment count. Driven by low on-time submissions."""
    base = np.maximum(0, 4 - students["assignments_submitted_on_time"].values)
    missing = rng.poisson(0.4 + 0.5 * base)
    return np.clip(missing, 0, 10).astype(int)


# ---------------------------------------------------------------------------
# Module-level KPIs (for the catalogue table)
# ---------------------------------------------------------------------------
def _attach_module_aggs(modules: pd.DataFrame, mod_engagement: pd.DataFrame) -> pd.DataFrame:
    aggs = (
        mod_engagement.groupby("module_code")
        .agg(
            n_students=("student_id", "count"),
            mean_attendance=("module_attendance_pct", "mean"),
            mean_grade=("module_grade", "mean"),
            mean_engagement=("module_engagement_score", "mean"),
            mean_missed=("assessments_missed", "mean"),
        )
        .reset_index()
    )
    aggs["mean_attendance"] = aggs["mean_attendance"].round(2)
    aggs["mean_grade"] = aggs["mean_grade"].round(2)
    aggs["mean_engagement"] = aggs["mean_engagement"].round(2)
    aggs["mean_missed"] = aggs["mean_missed"].round(2)
    return modules.merge(aggs, on="module_code", how="left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def enrich() -> None:
    students_path = DATA_DIR / "students_v3.csv"
    weekly_path = DATA_DIR / "weekly_engagement.csv"
    if not students_path.exists() or not weekly_path.exists():
        raise FileNotFoundError(
            "Base v3 dataset not found. Run generate_v3_dataset.py first."
        )

    students = pd.read_csv(students_path)
    weekly = pd.read_csv(weekly_path)

    # ---- Strip any previous enrichment columns so re-runs are clean ----
    enrichment_cols = [
        "consecutive_missed_max",
        "late_arrivals_total",
        "excused_absences",
        "unexcused_absences",
        "missing_assignments_count",
        "mean_module_grade",
        "module_grade_variance",
        "quiz_attempts_count",
        "resource_downloads_count",
        "forum_posts_count",
        "wellbeing_score",
        "wellbeing_flags",
        "wellbeing_last_checkin",
    ]
    students = students.drop(columns=[c for c in enrichment_cols if c in students.columns])

    # ---- Build everything ----
    print("Enriching v3 dataset ...")

    modules_df = _build_modules_table()
    enrolments_df = _build_enrolments(students)
    mod_engagement_df = _build_module_engagement(students, enrolments_df)
    daily_df = _build_daily_attendance(weekly)

    att_counters = _derive_attendance_counters(daily_df)
    eng_counts = _new_engagement_counts(students)
    well = _wellbeing_layer(students)
    mod_per_student = _module_aggregates_per_student(mod_engagement_df)
    modules_df = _attach_module_aggs(modules_df, mod_engagement_df)

    # ---- Merge derived columns onto students ----
    students = students.merge(att_counters, on="student_id", how="left")
    students = students.merge(eng_counts, on="student_id", how="left")
    students = students.merge(mod_per_student, on="student_id", how="left")
    students = students.merge(well, on="student_id", how="left")
    students["missing_assignments_count"] = _missing_assignments(students)

    # ---- Persist ----
    students.to_csv(students_path, index=False)
    modules_df.to_csv(DATA_DIR / "modules.csv", index=False)
    enrolments_df.to_csv(DATA_DIR / "enrolments.csv", index=False)
    mod_engagement_df.to_csv(DATA_DIR / "module_engagement.csv", index=False)
    daily_df.to_csv(DATA_DIR / "daily_attendance.csv", index=False)

    # ---- Summary ----
    print()
    print("Enrichment summary")
    print(f"  Modules in catalogue:      {len(modules_df)}")
    print(f"  Enrolments:                {len(enrolments_df)}  "
          f"(avg {len(enrolments_df) / len(students):.2f} per student)")
    print(f"  Module engagement rows:    {len(mod_engagement_df)}")
    print(f"  Daily attendance rows:     {len(daily_df)}")
    print(f"  Students table columns:    {len(students.columns)}")
    print()
    print("Cohort-level snapshots of the new fields")
    print(f"  consecutive_missed_max:    mean={students['consecutive_missed_max'].mean():.2f}  "
          f"max={students['consecutive_missed_max'].max()}")
    print(f"  late_arrivals_total:       mean={students['late_arrivals_total'].mean():.2f}")
    print(f"  excused_absences:          mean={students['excused_absences'].mean():.2f}")
    print(f"  unexcused_absences:        mean={students['unexcused_absences'].mean():.2f}")
    print(f"  missing_assignments_count: mean={students['missing_assignments_count'].mean():.2f}")
    print(f"  quiz_attempts_count:       mean={students['quiz_attempts_count'].mean():.2f}")
    print(f"  resource_downloads_count:  mean={students['resource_downloads_count'].mean():.2f}")
    print(f"  forum_posts_count:         mean={students['forum_posts_count'].mean():.2f}")
    print(f"  wellbeing_score:           mean={students['wellbeing_score'].mean():.2f}  "
          f"(1-10 scale)")
    print(f"  wellbeing_flags:           mean={students['wellbeing_flags'].mean():.2f}")
    print(f"  mean_module_grade:         mean={students['mean_module_grade'].mean():.2f}")
    print()
    print("Top 5 modules by lowest mean attendance (early signal for 'most disengaged classes'):")
    bottom = modules_df.sort_values("mean_attendance").head(5)
    for _, m in bottom.iterrows():
        print(f"  {m['module_code']:<8} {m['module_name'][:36]:<36}  "
              f"att={m['mean_attendance']:.1f}%  grade={m['mean_grade']:.1f}  "
              f"n={int(m['n_students'])}")
    print()
    print("Files written")
    for f in (
        "students_v3.csv (overwritten with extra columns)",
        "modules.csv",
        "enrolments.csv",
        "module_engagement.csv",
        "daily_attendance.csv",
    ):
        print(f"  data/{f}")


if __name__ == "__main__":
    enrich()
