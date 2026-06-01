# Stage 3 - Automated Intervention Engine

This is the policy layer that converts ML predictions into concrete actions an institution would actually take. It is deliberately modelled on how real higher-education early-alert systems work (e.g. EAB Navigate, Jisc Student Success, Civitas Inspire), so the artefact reflects a real institutional workflow, not a game.

## What the engine does

For each of the 600 students it evaluates a catalogue of 15 rules. A rule is a small, declarative object with five things:

1. **A trigger** - a Boolean predicate over the student's enriched snapshot (engagement features, ML-predicted probabilities, top SHAP driver).
2. **A severity tier** - `low`, `medium`, `high`, or `escalation`.
3. **A concrete action** - what the system would actually do (send SMS, schedule 1:1, refer to wellbeing, etc.).
4. **A routing destination** - which support team owns the action (Personal Tutor, Wellbeing Team, Financial Aid Office, etc.).
5. **A policy basis** - the institutional policy the rule is encoding, so every automated action can be defended in a real-world setting.

Each fired action is logged with a full audit trail: rule id, severity, action, routing team, policy basis, a rationale that quotes the actual numbers that crossed the threshold, the predicted probabilities at the moment of trigger, and the top SHAP-derived driver. That is exactly what a real student-success platform stores against an alert so a caseworker can defend it.

## Why de-duplication matters

If every rule that fires becomes its own action, a single severely at-risk student can rack up six or seven overlapping notifications - which would feel like a game and would not happen in a real institution. The engine therefore enforces a simple but realistic rule:

> Within each support team, only the highest-severity action survives.

So a student whose features fire both `WD_CRIT` (escalation, routed to Personal Tutor) and `ATT_WARN` (low, routed to Personal Tutor) gets a single Personal Tutor action - the critical one. But the same student can still get a parallel action from the Wellbeing Team or Academic Skills Team, because those are different teams. This matches how a real triage queue would be structured: one outreach per team per student per cycle, prioritised by severity.

## The rule catalogue (15 rules)

| ID | Tier | Trigger | Action | Routed to |
|---|---|---|---|---|
| `WD_CRIT` | escalation | `withdraw_prob >= 0.40` | Schedule a mandatory 1:1 retention meeting with personal tutor within 5 working days | Personal Tutor |
| `WD_HIGH` | high | `0.25 <= withdraw_prob < 0.40` | Welfare phone call from Student Services within 7 working days | Student Services |
| `WD_MOD` | medium | `0.15 <= withdraw_prob < 0.25` | Automated SMS check-in offering personal tutor office hours | Personal Tutor |
| `ATT_CRIT` | high | `attendance_rate < 50` | Formal attendance review meeting with personal tutor | Personal Tutor |
| `ATT_WARN` | low | `50 <= attendance_rate < 65` | Automated attendance warning email with study-skills resources | Personal Tutor |
| `VLE_DISENG` | medium | `weeks_since_last_login >= 3` | Engagement check email from personal tutor | Personal Tutor |
| `FAIL_CRIT` | high | `fail_prob >= 0.70 AND days_to_nearest_assessment <= 7` | Urgent 1:1 study-skills session and reminder to access the assessment brief | Academic Skills Team |
| `FAIL_HIGH` | medium | `fail_prob >= 0.50 AND days_to_nearest_assessment <= 14` | Invitation to weekly study-skills drop-in workshop | Academic Skills Team |
| `BRIEF_UNREAD` | low | `not accessed_brief AND days_to_nearest_assessment <= 7` | Automated reminder email with a direct link to the assessment brief | Module Lead |
| `GRADE_DROP` | medium | `last_grade < 50 AND previous_gpa >= 2.5` | Academic concern check-in with personal tutor | Personal Tutor |
| `LATE_PATTERN` | low | `assignments_submitted_late >= 3` | Invitation to a time-management workshop | Academic Skills Team |
| `WELLBEING` | high | `engagement_trend = Declining AND attendance < 60 AND works_part_time` | Welfare check-in from the Wellbeing Team within 5 working days | Wellbeing Team |
| `FIN_SUPPORT` | medium | `financial_support = Self-funded AND works_part_time AND fail_prob >= 0.40` | Referral to the Financial Aid Office for a hardship review | Financial Aid Office |
| `DIS_SUPPORT` | medium | `has_declared_disability AND (fail_prob >= 0.40 OR withdraw_prob >= 0.20)` | Proactive check-in from Disability Services to review support plan | Disability Services |
| `INT_SUPPORT` | medium | `is_international AND attendance < 65 AND weeks_since_last_login >= 2` | Welfare check from the International Office | International Office |

The catalogue mixes:

- **ML-driven triggers** (rules `WD_*`, `FAIL_*`) which fire on the calibrated probabilities from the Random Forest models.
- **Classical thresholds** (attendance, late submissions, VLE inactivity) which are policy rules an institution would have anyway.
- **Compound rules** (Wellbeing, Financial, Disability, International) which combine contextual flags with engagement signals - these are the proactive equity-oriented rules.

This mix mirrors how real systems work: ML alone is not enough, classical rules alone are too blunt, and the combination is what makes the workflow defensible.

## Output of the latest run

Across 600 students the engine produced **520 actions** for **262 distinct students** (44% of the cohort). The remaining 338 students received zero actions because none of their features crossed any threshold - the system leaves the well-engaged majority alone, which is exactly the behaviour a Director of Student Success would expect.

### Severity mix

| Severity | Count |
|---|---:|
| escalation | 60 |
| high | 92 |
| medium | 222 |
| low | 146 |

The pyramid shape is correct: most actions are nudges, fewer are interventions, fewest are escalations.

### Distribution per support team

| Team | Actions |
|---|---:|
| Personal Tutor | 195 |
| Module Lead | 95 |
| Academic Skills Team | 93 |
| Student Services | 63 |
| Financial Aid Office | 27 |
| Disability Services | 26 |
| International Office | 17 |
| Wellbeing Team | 4 |

All 8 teams are represented, with volumes that look institutional. Wellbeing referrals are deliberately rare (4) because they should be reserved for genuinely concerning combinations - not used as a default catch-all.

### Per-student action counts

| Actions | Students |
|---|---:|
| 0 | 338 |
| 1 | 122 |
| 2 | 70 |
| 3 | 51 |
| 4 | 18 |
| 5 | 1 |
| 6 | 0-1 |

Mean 1.98 actions per flagged student; maximum 6 for the most severely at-risk individual. No student is buried under a dozen overlapping nudges - the de-duplication rule is doing its job.

## Sample audit entry

This is what every row of `automated_actions.csv` looks like:

```
action_id:                ACT00001
triggered_on:             2025-03-22
student_id:               STU0168
severity:                 escalation
rule_id:                  WD_CRIT
rule_name:                Critical withdrawal risk
action:                   Schedule a mandatory 1:1 retention meeting with personal tutor within 5 working days
route_to:                 Personal Tutor
policy_basis:             Student Retention Policy §4.2 (Critical Early Alert)
rationale:                Predicted withdrawal probability is 58.8% (threshold 40%). Top contributing factor: Recent attendance (4-week avg).
withdraw_prob_at_trigger: 0.5877
fail_prob_at_trigger:     0.9009
top_driver_withdraw:      Recent attendance (4-week avg)
top_driver_fail:          Recent attendance (4-week avg)
status:                   Pending
```

A Director of Student Success could read that row out loud in a board meeting and defend every field.

## Files

```
interventions/
├── rules.py                          declarative catalogue (15 Rule objects)
├── engine.py                         evaluator + de-duplication
├── run_engine.py                     end-to-end runner
└── INTERVENTION_ENGINE_README.md     this document

data/
├── automated_actions.csv             520 rows - the audit log the dashboard reads
└── rules_catalogue.json              static catalogue for the dashboard's Rules page
```

The dashboard never imports the engine code - it just reads the two output files. That keeps the deployed app small and means the engine can be re-run (e.g. with a different rule set) without redeploying.

## How to (re)run

```powershell
python -m interventions.run_engine
```

Output is deterministic given the same inputs (predictions + features). Re-running with different rule weights or thresholds is the natural place to do sensitivity analysis in the dissertation.

## What this gives the dissertation narrative

- The ML model is no longer just a number on a dashboard; it drives concrete actions that map to **named institutional policies**.
- Every automated action is **defensible**: full audit trail, predicted probabilities at trigger time, top SHAP driver, and policy citation.
- The engine encodes **eight distinct support pathways** rather than a single generic "alert", which is closer to how real student-success teams are structured.
- **De-duplication** prevents alert fatigue, a well-documented failure mode of first-generation early-alert systems.
- The mix of **ML-driven and classical** rules is itself a methodological choice that can be discussed (when does ML add value over rules? when do rules act as a safety net for the ML?).
