# Project state - master briefing (read this first)

> **Purpose of this file:** This is the single most up-to-date description of the project as it currently stands. It complements the other READMEs by filling in everything they do not cover yet: the communications layer, the dashboard structure, the live deployment, the latest cohort numbers, and the limitations to keep in mind when writing the client impact report.
>
> All data, names, and outcomes in the project are **simulated**. There are no real students.

---

## 1. What the project is, in one paragraph

A working prototype of a **student engagement risk monitoring system** that combines a synthetic-but-realistic dataset, a calibrated machine-learning layer (predicting withdrawal and assessment failure), an explainability layer (SHAP), a rules-based automated intervention engine, and a six-page interactive dashboard. The goal is to show how a university could move from **lagging, reactive support** (acting after a student has already withdrawn or failed) to **leading, predictive support** (acting on early signals with a documented audit trail).

---

## 2. Current cohort snapshot (used by the dashboard)

| Metric | Value |
|---|---|
| Total students | **600** |
| True withdraw rate (label) | 15.2% |
| True "failed-next-assessment" rate (label) | 28.7% |
| Mean predicted P(withdraw) | 15.0% |
| Mean predicted P(fail next) | 28.6% |
| Students in **High** risk band | 115 |
| Students in **Medium** risk band | 87 |
| Students in **Low** risk band | 398 |
| Students with critical withdraw risk (>50%) | 37 |
| Modules in catalogue | 17 |
| Daily attendance rows | 36,000 |
| Module engagement rows | 2,080 |
| Communications log rows | 653 |
| Students contacted in last 30 days | 297 |
| Responses received in last 30 days | 107 |
| Students flagged "non-responsive" | 102 |
| Automated actions currently open | **688 across 326 students** |
| Distinct rules that have fired | 16 (of 19 defined) |

These are the live numbers an analyst would see when they open the dashboard today.

---

## 3. ML pipeline - current performance (test split)

Validation methodology and full train/val/test detail are in `ML_PIPELINE_README.md` and `metrics_report.md`.

| Target | Winner model | ROC-AUC | Brier | Recall (positive class) | Precision |
|---|---|---|---|---|---|
| Withdrew | Random Forest (Platt-calibrated) | **0.850** | 0.108 | 0.86 | 0.38 |
| Failed next assessment | XGBoost (Platt-calibrated) | **0.960** | 0.069 | 0.88 | 0.78 |

The withdraw model is deliberately tuned for **high recall** at the cost of precision - in retention work, the cost of missing an at-risk student is much higher than the cost of a false positive. The fail-next model is more balanced because the signal is denser.

A formal v3 vs v3.1 comparison (showing how the enriched features moved performance) is in `metrics_comparison.md`.

---

## 4. Intervention engine - current state

| Aspect | Value |
|---|---|
| Total declarative rules | **19** |
| Rules currently firing on this cohort | 16 |
| Total open actions | 688 |
| Students with at least one action | 326 |

**Action breakdown by routed team:**

| Team | Open actions |
|---|---|
| Personal Tutor | 193 |
| Academic Skills Team | 133 |
| Module Lead | 95 |
| Wellbeing Team | 83 |
| Student Services | 62 |
| Senior Tutor | 47 |
| Disability Services | 29 |
| Financial Aid Office | 29 |
| International Office | 17 |

**Action breakdown by severity:**

| Severity | Count |
|---|---|
| High | 268 |
| Medium | 214 |
| Low | 143 |
| Escalation | 63 |

**Top firing rules:**

| Rule | Actions |
|---|---|
| BRIEF_UNREAD - assessment brief unread close to deadline | 95 |
| WELLBEING_FLAGS - multiple wellbeing concerns | 79 |
| WD_CRIT - critical withdrawal risk | 63 |
| WD_HIGH - high withdrawal risk | 62 |
| FAIL_HIGH - elevated failure risk | 57 |
| ATT_WARN - attendance warning | 48 |
| WD_MOD - elevated withdrawal risk | 47 |
| NON_RESPONSIVE - no response to repeated outreach | 47 |
| MISSING_ASSIGNMENTS - multiple missing assignments | 40 |
| FAIL_CRIT - critical failure risk | 36 |
| GRADE_DROP - sudden grade drop | 34 |
| DIS_SUPPORT - declared disability + concern | 29 |
| FIN_SUPPORT - possible financial pressure | 29 |
| INT_SUPPORT - international student isolation | 17 |
| WELLBEING - possible wellbeing concern | 4 |
| VLE_DISENG - VLE disengagement | 1 |

The full rule definitions (thresholds, rationale templates, suggested actions, policy basis) live in `INTERVENTION_ENGINE_README.md`. The numbers in that file are from the v3 baseline run; the numbers above are the current v3.1 run.

---

## 5. Communications layer (NOT documented elsewhere - read this for the report)

A communications layer was added in v3.1 to make the "automated intervention" story end-to-end and to support the **NON_RESPONSIVE** rule. It is not described in any of the other READMEs.

### What it generates
A `communications_log.csv` of 653 simulated outreach events plus a `student_comms_summary.csv` rollup. For every student the rollup computes:

- `comms_30d_count` - total contacts received in the last 30 days
- `comms_responded_30d` - of those, how many were replied to
- `last_contacted_at`, `last_responded_at`, `days_since_last_contact`, `days_since_last_response`
- `email_30d`, `sms_30d`, `phone_call_30d`, `meeting_30d` - per-channel counts
- `is_non_responsive` - true if the student received 2+ contacts in the last 30 days and replied to none of them within the response window

### How it surfaces in the engine
The new rule `NON_RESPONSIVE` fires when `is_non_responsive == True`. It is routed to the **Senior Tutor** at **high** severity, on the policy basis of *Student Engagement Policy s3.1 (Escalation after silent outreach)*. It is currently the 8th-most-common rule (47 actions).

### How it surfaces in the dashboard
- **Student profile -> Communications tab**: shows every contact for that student, the channel, whether it was responded to, and the response timing.
- **Interventions page**: every case card has a "Comms strip" header summarising recent contact attempts and responses, so a tutor can see at a glance whether the student is silent.
- **Interventions page filter**: an "Outreach status" filter lets you isolate non-responsive students.
- **Activity log per case**: each case has a hidden-by-default activity timeline that includes outreach steps (e.g. *"Auto-email queued"*, *"Auto-email will be sent shortly"*, *"Escalation if no response within X days"*).

### Why it matters for the client impact story
Without this layer the system could only say "we identified risk". With it, the system can say "we identified risk, attempted outreach, tracked responses, and escalated when outreach failed" - which is the full closed loop a university needs to evidence to a regulator.

---

## 6. Dashboard structure (NOT documented elsewhere - read this for the report)

The dashboard is a Streamlit app, rebuilt entirely on top of the v3.1 outputs. It is structured as **six pages** with a consistent visual language (custom CSS, Plotly charts styled through a shared theme helper).

### Page 1 - Overview (executive summary)
- KPI cards: cohort size, % at high risk, % at medium risk, predicted withdrawals, predicted fails next assessment
- Risk-band donut chart
- Cohort attendance trend chart (weekly)
- **"Top 10 students needing attention"** table sorted by P(withdraw)
- **"Most disengaged modules"** widget - module-level lens, ranked by average risk of enrolled students
- Filters: programme, year of study (every KPI on the page updates when filtered)

### Page 2 - Students (roster)
- Searchable, filterable directory of all 600 students
- Columns: name, ID, programme, year, attendance, predicted P(withdraw), predicted P(fail), risk band, open concerns
- Multiselect filters (programme, year, risk band) defaulted to empty so the full roster is visible by default

### Page 3 - Student profile (360 deg view) - 7 tabs
1. **Overview** - header card, profile KPIs, top SHAP drivers, summary chart
2. **Attendance** - weekly attendance bar chart, daily attendance heatmap, KPIs (consecutive missed, late arrivals, excused vs unexcused)
3. **Academics** - per-module attendance, per-module grade vs cohort mean, module breakdown table, KPIs (missing assignments, grade variance)
4. **LMS engagement** - VLE logins, hours trend, lecture-material access, quiz attempts, resource downloads, forum activity
5. **Wellbeing** - wellbeing score, flags raised, last check-in
6. **Case file** - the consolidated case for that student: one **primary concern** plus a list of **supporting concerns** (no longer one card per rule), a **"Why this case was created"** checklist, and a **Case activity log** (hidden behind a click, no jargon like "automation timeline" - just a clean chronological log)
7. **Communications** - per-student comms strip, KPIs for contacts vs responses, full comms log table

### Page 4 - Modules
- Module browser with KPIs (n_modules, lowest attendance, lowest grade, most high-risk students)
- Three ranked bar charts: lowest attendance, lowest average grade, most high-risk
- Module drill-down: header card, module-level KPIs, quick-pick lists of lowest attenders / graders, full sortable roster restricted to that module, attendance and grade distributions

### Page 5 - At-risk students (early-warning table)
- Sorted by predicted withdraw probability
- Each row links straight to that student's profile via "View profile ->"
- A "Concerns on case" column shows how many active concerns each student has (replacing the older "Open actions" label, to align with the consolidated-case framing)

### Page 6 - Interventions (operational case management)
- **One card per student**, not one per rule - the system consolidates concerns
- Each card shows: primary concern, supporting concerns, severity tag, comms strip, "View profile ->" button, and an expander with the case activity log
- **"Why this case was created"** checklist in human language e.g. *"Withdrawal risk exceeded university threshold (61.6% vs 40%)"* rather than raw boolean expressions
- **Five filters**: programme, concern type, outreach status (responsive / non-responsive / not yet contacted), risk band, free-text search by name or ID
- Single-select filter widgets for a cleaner UI; sort order configurable

### What was deliberately removed
- The earlier "Performance" page (model metrics) was removed for the client-facing build - those numbers live in `metrics_report.md` and `metrics_comparison.md` for the academic write-up.
- "AUTOMATED" black tags on card headers were removed after user testing - the automation story is told through the activity log instead.

---

## 7. Live deployment

| Item | Value |
|---|---|
| Public URL | **https://engagement-risk-dashboard-1.streamlit.app/** |
| Hosting | Streamlit Community Cloud |
| Python version | 3.11 |
| Source code | Public GitHub repository (linked from the dashboard footer) |
| Status | Live and shareable; no login required |

Streamlit Community Cloud sleeps apps after about a week of inactivity. Cold start takes ~30 seconds the first time someone opens the link after a long pause; subsequent loads are instant.

---

## 8. Worked example - one real case from the current data

This is the kind of case the report can use as a concrete walk-through.

> **Student STU0168 (Ashleigh Davis)**
>
> - Predicted P(withdraw): **61.6%** (threshold: 40%) - in the "Critical" band
> - Predicted P(fail next assessment): **83.8%**
> - Top SHAP driver for withdraw: *Lecture materials accessed (last 4 weeks)*
> - Top SHAP driver for fail: *Past interventions received*
>
> The engine consolidated this into one case routed to the **Personal Tutor** at **escalation** severity, on the policy basis of *Student Retention Policy s4.2 (Critical Early Warning)*.
>
> The "Why this case was created" checklist reads:
> - Withdrawal risk exceeded university threshold (61.6% vs 40%)
> - Assessment failure risk exceeded university threshold (83.8% vs 60%)
> - (Plus any supporting concerns the engine attached)
>
> The case activity log shows the queued outreach step ("Auto-email will be sent shortly") and the escalation step that will fire if no response is received within the SLA window.

---

## 9. Limitations and caveats (important - the report must be honest about these)

1. **All data is synthetic.** No real student records were used. Names, modules, attendance, grades, and outcomes were generated by a controlled simulator with a fixed random seed for reproducibility.
2. **Labels are simulated outcomes.** The ML models are trained against `withdrew` and `failed_next_assessment` labels that the data generator itself produced. Real-world performance will differ.
3. **Wellbeing data is simplified.** The wellbeing layer is a score + flags + last-checkin proxy. A production system would integrate with a real student-support case management system.
4. **Communications layer is simulated.** Email, SMS, phone, and meeting events are generated, not produced by a real CRM. Response rates are realistic but not measured.
5. **Demographic features are used as inputs.** Disability, financial support, and international status are inputs to both the model and the rules. Any production deployment would need a fairness review and an equality-impact assessment before going live.
6. **Calibration is Platt scaling (sigmoid).** Suitable for the scale of this dataset; isotonic or temperature scaling could be evaluated at larger scale.
7. **This is a prototype, not a production system.** No authentication, no role-based access control, no integration with an SIS, and no monitoring/alerting in place. It is designed to evidence the concept, not to be deployed.

---

## 10. File guide for the report

| File | What it covers | When to cite it |
|---|---|---|
| **PROJECT_STATE.md** (this file) | Current state, dashboard, comms, live URL, limitations | Top of report, executive summary, screenshots/walk-through, limitations section |
| V3_DATA_README.md | Synthetic data methodology, schema, columns, generation process | Methodology / data foundation section |
| ML_PIPELINE_README.md | Feature engineering, model training, calibration, SHAP | Methodology / ML section |
| metrics_report.md | Full model performance breakdown (train/val/test, per model) | Results / accuracy section |
| metrics_comparison.md | v3 baseline vs v3.1 enriched comparison | Impact-of-enrichment / results section |
| INTERVENTION_ENGINE_README.md | All 19 rules - definitions, thresholds, routing, rationale | Intervention layer / operational story section |
| Client Impact Report Template.docx | Required report format | Structure to follow |

Read in that order, the seven files are sufficient to write the full client impact report without needing to inspect any source code.
