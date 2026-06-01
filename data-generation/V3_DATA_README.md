# v3 Synthetic Dataset — Data Foundation

This document explains the redesigned synthetic dataset that powers the ML risk model. It exists alongside the original v1/v2 CSVs (`students_data.csv`, etc.) which remain untouched until the dashboard is switched over in Stage 4.

## Why we rebuilt the data

The previous dataset had a fundamental problem for an ML-based project: the **risk score was generated using a deterministic formula and then written back into the data as a "label"**. Any model trained on that data is just rediscovering the formula — that is a tautology, not prediction.

The v3 generator inverts the process. We generate the *truth* first, then generate noisy *observations* of it. The ML model has to recover the truth through the noise. That is a real prediction task.

## How v3 works (the mechanics)

For every student we draw two hidden numbers that are never written to any CSV:

- **`latent_engagement`** — the student's true underlying engagement level, roughly Normal(0, 1). High = engaged, low = disengaged.
- **`latent_slope`** — the trajectory direction. Positive = improving over time, negative = declining.

Every observable feature is then a **noisy function of these hidden numbers plus context**:

| Feature | Generated from |
|---|---|
| Attendance per week | latent + commuter penalty + gaussian noise |
| VLE logins per week | Poisson with rate ≈ 5 + 3.5 × latent |
| VLE hours per week | 0.6 × logins − part-time penalty + noise |
| Materials accessed | Poisson(1.5 + 1.2 × latent) |
| Last assessment grade | 60 + 12 × latent + attendance/VLE adjustments + noise (σ=9) |
| Previous semester GPA | 1.5 + 0.025 × grade + 0.10 × latent + noise |
| Late submissions | Binomial(3, p) where p depends on latent and attendance |
| Brief access flag | Bernoulli(p) where p depends on latent and attendance |

Twelve weekly snapshots are generated per student, with the latent drifting back in time according to the slope plus weekly noise. The cross-sectional fields on the student row are then derived from those weekly snapshots:

- `attendance_rate` = mean of the **last 4 weeks**
- `vle_logins_last_week`, `vle_time_hours_last_week` = the **week-12** value
- `attendance_slope_4w` = last-4-weeks mean − first-8-weeks mean
- `engagement_trend` = `Improving` / `Declining` / `Stable` derived from that slope

This guarantees the cross-sectional and weekly tables are internally consistent — the current snapshot is literally the end of the weekly history.

## The outcome labels (what the ML predicts)

Two binary targets are sampled from logistic models on the hidden latent state, with **independent Gaussian noise** baked in:

**`withdrew`** — did the student withdraw within the simulation period?
```
withdraw_logit = -3.10
               + -2.10 * latent
               + -1.60 * latent_slope
               + 0.40 * works_part_time
               + 0.50 * (financial_support == "None")
               + 0.30 * is_commuter
               + Normal(0, 0.55)              # <-- the irreducible noise
withdrew = Bernoulli(sigmoid(withdraw_logit))
```

**`failed_next_assessment`** — will they fail their next assessment (<50%)?
```
fail_logit = -1.40
           + -1.70 * latent
           + -0.85 * latent_slope
           + -0.018 * (attendance_rate - 75)
           + 0.40 * (assignments_submitted_late >= 2)
           + -0.25 * accessed_upcoming_assessment_brief
           + Normal(0, 0.60)                   # <-- noise here too
failed_next_assessment = Bernoulli(sigmoid(fail_logit))
```

The noise term is the whole point. Without it, the outcomes would be a deterministic function of the features and ML AUC would hit 1.0 — fake. With it, the model has a ceiling that's strictly below perfect, and *learning the right pattern* is what closes the gap.

## Realised cohort statistics

Running `python generate_v3_dataset.py` produces:

- **600 students** total
- **Withdrew: ~15%** (91 students) — realistic for a semester window
- **Failed next assessment: ~29%** (172 students) — realistic for early-career assessments
- **Engagement trend**: ~36% Improving, ~33% Declining, ~31% Stable
- **Stratified split**: 420 train / 90 val / 90 test, with withdraw rate ≈ 15% in each split (within 1 pp)

Sanity-check correlations between features and outcomes (all signs correct, magnitudes reasonable):

```
                                withdrew    failed_next_assessment
attendance_rate                  -0.46           -0.57
last_assessment_grade            -0.42           -0.49
previous_semester_gpa            -0.37           -0.46
materials_accessed_last_4w       -0.36           -0.46
vle_logins_last_week             -0.36           -0.50
assignments_submitted_on_time    -0.35           -0.40
assignments_submitted_late       +0.23           +0.21
prior_intervention_count         +0.34           +0.54
```

A baseline logistic-regression on the validation set gives:

- `withdrew`: **ROC-AUC ≈ 0.80**
- `failed_next_assessment`: **ROC-AUC ≈ 0.91**

Both are well below 1.0, both are well above 0.5. XGBoost/LightGBM (Stage 2) will pick up further non-linear gain — but no model will hit 1.0, because the noise term ensures an irreducible error floor.

## Files written to `../data/`

| File | Rows | Description |
|---|---:|---|
| `students_v3.csv` | 600 | One row per student. Features + outcomes + split. **Extended in v3.1 — see below.** |
| `weekly_engagement.csv` | 7,200 | 12 weekly snapshots per student. Drives temporal features. |
| `interventions_v3.csv` | ~290 | Historical intervention log. Generated *after* outcomes; biased toward at-risk students so it acts as a "system already trying to help" proxy. |
| `assessments_v3.csv` | ~35 | Cohort-level upcoming assessments. |
| `modules.csv` | 17 | **v3.1** — module catalogue with module-level aggregates (mean attendance, mean grade, mean engagement). |
| `enrolments.csv` | ~2,080 | **v3.1** — student-to-module enrolments. Each student takes 3–4 modules, biased toward their own programme. |
| `module_engagement.csv` | ~2,080 | **v3.1** — per-(student, module) attendance, grade, assessment record, and engagement composite. |
| `daily_attendance.csv` | 36,000 | **v3.1** — daily M–F attendance log over 12 weeks. Each entry is Present / Late / Excused / Absent. Reconstructs to the weekly attendance aggregates. |

Columns on `students_v3.csv` (after v3.1 enrichment):

```
# --- base v3 ---
student_id, student_name, email, program, year_of_study, enrollment_date,
is_commuter, works_part_time, has_declared_disability, is_international, financial_support,
attendance_rate, attendance_slope_4w,
vle_logins_last_week, vle_time_hours_last_week, vle_logins_slope_4w, vle_hours_slope_4w,
materials_accessed_last_4w,
engagement_trend,
assignments_submitted_on_time, assignments_submitted_late,
last_assessment_grade, previous_semester_gpa,
accessed_upcoming_assessment_brief, days_to_nearest_assessment,
withdrew, failed_next_assessment,        # <-- the ML targets
prior_intervention_count, prior_interventions_improved,
split                                    # train / val / test

# --- v3.1 enrichment ---
# Derived from daily_attendance.csv (12 weeks x 5 days)
consecutive_missed_max,                  # longest streak of unexcused absences
late_arrivals_total,                     # count over 12 weeks
excused_absences, unexcused_absences,
# Derived from module_engagement.csv
mean_module_grade, module_grade_variance,
# New engagement counts
quiz_attempts_count, resource_downloads_count, forum_posts_count,
# Aggregate academic
missing_assignments_count,
# Simple wellbeing layer (clearly synthetic)
wellbeing_score, wellbeing_flags, wellbeing_last_checkin
```

## v3.1 enrichment script

`enrich_v3_dataset.py` is run **after** `generate_v3_dataset.py` to layer the additional structure on top. It:

- Adds the **modules layer** — a 17-module catalogue spanning all programmes plus two cross-programme modules. Each student is enrolled in 3–4 modules weighted toward their own programme. Per-module attendance and grade are noisy variants of the student's cross-sectional figures, so module-level aggregates and student-level aggregates remain consistent with each other.
- Adds **daily attendance** — for each student-week, draws a 5-day Mon–Fri pattern that integrates back to the existing weekly `attendance_pct`. About 12 % of attended days are flagged `Late`, about 18 % of absent days are flagged `Excused`, and the rest are `Absent`. Derived counters (`consecutive_missed_max`, `late_arrivals_total`, `excused_absences`, `unexcused_absences`) are written back to the student row.
- Adds **new engagement counts** — quiz attempts, resource downloads, forum posts — all Poisson-distributed and positively correlated with attendance.
- Adds an **aggregate academic counter** (`missing_assignments_count`) driven by the existing on-time-submissions count.
- Adds a **simple wellbeing layer** — `wellbeing_score` (1–10), `wellbeing_flags` (count of concerns raised), `wellbeing_last_checkin` (date). Synthetic; clearly labelled as such. Mildly correlated with attendance, engagement trend, and declared support needs.

The enrichment uses an independent random seed (43) so its draws do not overlap with the base latent-state RNG.

**Important:** the outcome labels (`withdrew`, `failed_next_assessment`) are **not** regenerated by the enrichment. They were sampled from the latent state in the base generator and stay fixed so the v3.1 ML retrain on the richer feature set is directly comparable to the original v3 results.

## What's intentionally NOT in v3

- **No `risk_score` column.** The old composite is removed from the data. The dashboard will compute it from the trained ML model in Stage 4.
- **No `latent_engagement` or `latent_slope` columns.** These are the hidden truth; if the ML could see them it would be cheating.

## Reproducibility

Seed = 42 across `random`, `numpy`, `numpy.random.default_rng`, and `Faker`. Running the generator twice produces identical files. To change the cohort, edit `N_STUDENTS` or `N_WEEKS` at the top of `generate_v3_dataset.py`.

## Next stages

- **Stage 2** — train ML models (logistic regression, random forest, XGBoost) on these outcomes, with proper validation metrics (ROC-AUC, PR-AUC, calibration, confusion matrix on the held-out test split). SHAP for explanations.
- **Stage 3** — automated intervention engine triggered by model output.
- **Stage 4** — rebuild the dashboard around model predictions and drivers (no more deterministic composite in the main UI).
