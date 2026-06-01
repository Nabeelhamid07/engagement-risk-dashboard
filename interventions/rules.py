"""
Declarative rule catalogue for the automated intervention engine.

Each rule combines:

  - a **trigger**: a Boolean predicate over a student's enriched feature row
    (their latest engagement snapshot + the ML model's predicted probabilities
    + their top SHAP driver).
  - an **action**: the concrete step the system takes (e.g. send SMS, book
    a 1:1 with a tutor, refer to wellbeing).
  - a **routing destination**: which support team owns the action.
  - a **severity tier**: ``low`` / ``medium`` / ``high`` / ``escalation``.
  - a **policy basis**: the institutional policy the rule encodes, so every
    automated action can be defended in a real-world setting.
  - a **rationale builder**: a short, plain-English sentence that quotes the
    actual numbers that crossed the threshold, for the audit log.

The rule set is intentionally modelled on the kinds of policy triggers used
in real higher-education early-alert systems (e.g. EAB Navigate, Jisc
Student Success, Civitas Inspire). It mixes ML-driven triggers ("predicted
withdrawal probability >= 40%") with classical attendance / submission /
VLE rules, because real systems do both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "escalation": 3}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    description: str
    trigger_summary: str       # one-line condition shown in the rules catalogue
    severity: str              # low / medium / high / escalation
    action: str                # concrete next step
    route_to: str              # which team owns it
    policy_basis: str          # plausible institutional policy reference
    when: Callable[[pd.Series], bool]
    rationale: Callable[[pd.Series], str]


# ---------------------------------------------------------------------------
# ML-driven withdrawal rules
# ---------------------------------------------------------------------------
RULE_WITHDRAW_CRITICAL = Rule(
    id="WD_CRIT",
    name="Critical withdrawal risk",
    description="Triggered when the predictive model estimates a 40%+ probability that the student will withdraw within the semester.",
    trigger_summary="withdraw_prob >= 0.40",
    severity="escalation",
    action="Schedule a mandatory 1:1 retention meeting with personal tutor within 5 working days",
    route_to="Personal Tutor",
    policy_basis="Student Retention Policy §4.2 (Critical Early Alert)",
    when=lambda r: r["withdraw_prob"] >= 0.40,
    rationale=lambda r: (
        f"Predicted withdrawal probability is {r['withdraw_prob']*100:.1f}% "
        f"(threshold 40%). Top contributing factor: {r['top_driver_withdraw']}."
    ),
)

RULE_WITHDRAW_HIGH = Rule(
    id="WD_HIGH",
    name="High withdrawal risk",
    description="Triggered when withdrawal probability sits in the 25–40% band.",
    trigger_summary="0.25 <= withdraw_prob < 0.40",
    severity="high",
    action="Welfare phone call from Student Services within 7 working days",
    route_to="Student Services",
    policy_basis="Student Retention Policy §4.3 (Targeted Outreach)",
    when=lambda r: 0.25 <= r["withdraw_prob"] < 0.40,
    rationale=lambda r: (
        f"Predicted withdrawal probability is {r['withdraw_prob']*100:.1f}% "
        f"(within the 25–40% targeted-outreach band). Top contributing factor: "
        f"{r['top_driver_withdraw']}."
    ),
)

RULE_WITHDRAW_MODERATE = Rule(
    id="WD_MOD",
    name="Elevated withdrawal risk",
    description="Triggered when withdrawal probability sits in the 15–25% band.",
    trigger_summary="0.15 <= withdraw_prob < 0.25",
    severity="medium",
    action="Automated SMS check-in offering personal tutor office hours",
    route_to="Personal Tutor",
    policy_basis="Student Retention Policy §4.4 (Soft Nudge)",
    when=lambda r: 0.15 <= r["withdraw_prob"] < 0.25,
    rationale=lambda r: (
        f"Predicted withdrawal probability is {r['withdraw_prob']*100:.1f}% "
        f"(elevated band, soft-nudge policy)."
    ),
)


# ---------------------------------------------------------------------------
# Attendance rules
# ---------------------------------------------------------------------------
RULE_ATTENDANCE_CRITICAL = Rule(
    id="ATT_CRIT",
    name="Critical attendance",
    description="Four-week rolling attendance has fallen below 50%, the institutional minimum for satisfactory engagement.",
    trigger_summary="attendance_rate < 50",
    severity="high",
    action="Formal attendance review meeting with personal tutor",
    route_to="Personal Tutor",
    policy_basis="Academic Engagement Policy §2.1 (Minimum Attendance)",
    when=lambda r: r["attendance_rate"] < 50,
    rationale=lambda r: (
        f"Four-week attendance has fallen to {r['attendance_rate']:.1f}%, below the "
        f"50% minimum required by the Academic Engagement Policy."
    ),
)

RULE_ATTENDANCE_WARNING = Rule(
    id="ATT_WARN",
    name="Attendance warning",
    description="Four-week rolling attendance is between 50% and 65% - early warning band.",
    trigger_summary="50 <= attendance_rate < 65",
    severity="low",
    action="Automated attendance warning email with study-skills resources",
    route_to="Personal Tutor",
    policy_basis="Academic Engagement Policy §2.2 (Attendance Warnings)",
    when=lambda r: 50 <= r["attendance_rate"] < 65,
    rationale=lambda r: (
        f"Four-week attendance is {r['attendance_rate']:.1f}% - in the 50-65% "
        f"warning band."
    ),
)


# ---------------------------------------------------------------------------
# VLE engagement rules
# ---------------------------------------------------------------------------
RULE_VLE_DISENGAGED = Rule(
    id="VLE_DISENG",
    name="VLE disengagement",
    description="No virtual learning environment activity for three or more consecutive weeks.",
    trigger_summary="weeks_since_last_login >= 3",
    severity="medium",
    action="Engagement check email from personal tutor (template: 'Are you OK?')",
    route_to="Personal Tutor",
    policy_basis="Digital Engagement Policy §3.1 (Inactivity Alerts)",
    when=lambda r: r["weeks_since_last_login"] >= 3,
    rationale=lambda r: (
        f"No VLE activity for {int(r['weeks_since_last_login'])} consecutive weeks "
        f"(threshold: 3 weeks)."
    ),
)


# ---------------------------------------------------------------------------
# ML-driven failure rules (next assessment)
# ---------------------------------------------------------------------------
RULE_FAIL_UPCOMING_CRITICAL = Rule(
    id="FAIL_CRIT",
    name="Critical failure risk on upcoming assessment",
    description="Predicted probability of failing the next assessment is 70%+ and the assessment is within 7 days.",
    trigger_summary="fail_prob >= 0.70 AND days_to_nearest_assessment <= 7",
    severity="high",
    action="Urgent 1:1 study-skills session and reminder to access the assessment brief",
    route_to="Academic Skills Team",
    policy_basis="Assessment Support Policy §5.1 (Pre-assessment Intervention)",
    when=lambda r: (r["fail_prob"] >= 0.70) and (r["days_to_nearest_assessment"] <= 7),
    rationale=lambda r: (
        f"Predicted failure probability is {r['fail_prob']*100:.1f}% with only "
        f"{int(r['days_to_nearest_assessment'])} days until the next assessment. "
        f"Top contributing factor: {r['top_driver_fail']}."
    ),
)

RULE_FAIL_UPCOMING = Rule(
    id="FAIL_HIGH",
    name="Elevated failure risk on upcoming assessment",
    description="Predicted failure probability is 50%+ with the assessment within 14 days.",
    trigger_summary="fail_prob >= 0.50 AND days_to_nearest_assessment <= 14",
    severity="medium",
    action="Invitation to weekly study-skills drop-in workshop",
    route_to="Academic Skills Team",
    policy_basis="Assessment Support Policy §5.2 (Targeted Skills Workshops)",
    when=lambda r: (r["fail_prob"] >= 0.50) and (r["days_to_nearest_assessment"] <= 14),
    rationale=lambda r: (
        f"Predicted failure probability is {r['fail_prob']*100:.1f}% with "
        f"{int(r['days_to_nearest_assessment'])} days until the next assessment."
    ),
)

RULE_BRIEF_UNREAD = Rule(
    id="BRIEF_UNREAD",
    name="Assessment brief unread close to deadline",
    description="Student has not accessed the upcoming assessment brief and the deadline is within 7 days.",
    trigger_summary="not accessed_brief AND days_to_nearest_assessment <= 7",
    severity="low",
    action="Automated reminder email with a direct link to the assessment brief",
    route_to="Module Lead",
    policy_basis="Assessment Support Policy §5.3 (Pre-deadline Nudge)",
    when=lambda r: (not bool(r["accessed_upcoming_assessment_brief"])) and (r["days_to_nearest_assessment"] <= 7),
    rationale=lambda r: (
        f"The assessment brief has not been opened, with only "
        f"{int(r['days_to_nearest_assessment'])} days remaining."
    ),
)


# ---------------------------------------------------------------------------
# Academic performance rules
# ---------------------------------------------------------------------------
RULE_GRADE_DROP = Rule(
    id="GRADE_DROP",
    name="Sudden grade drop from a strong baseline",
    description="Most recent assessment was below 50% despite a previous semester GPA of 2.5 or higher.",
    trigger_summary="last_assessment_grade < 50 AND previous_semester_gpa >= 2.5",
    severity="medium",
    action="Academic concern check-in with personal tutor",
    route_to="Personal Tutor",
    policy_basis="Academic Standards Policy §6.1 (Performance Anomaly)",
    when=lambda r: (r["last_assessment_grade"] < 50) and (r["previous_semester_gpa"] >= 2.5),
    rationale=lambda r: (
        f"Most recent grade {int(r['last_assessment_grade'])}% is sharply below "
        f"the student's prior GPA of {r['previous_semester_gpa']:.2f}."
    ),
)

RULE_LATE_PATTERN = Rule(
    id="LATE_PATTERN",
    name="Pattern of late submissions",
    description="Three or more late submissions in the rolling submission window.",
    trigger_summary="assignments_submitted_late >= 3",
    severity="low",
    action="Invitation to a time-management workshop",
    route_to="Academic Skills Team",
    policy_basis="Academic Skills Policy §7.2 (Submission Patterns)",
    when=lambda r: r["assignments_submitted_late"] >= 3,
    rationale=lambda r: (
        f"{int(r['assignments_submitted_late'])} late submissions on record "
        f"(threshold: 3)."
    ),
)


# ---------------------------------------------------------------------------
# Wellbeing / contextual rules
# ---------------------------------------------------------------------------
RULE_WELLBEING = Rule(
    id="WELLBEING",
    name="Possible wellbeing concern",
    description="Combined signal of declining engagement, low attendance and part-time work commitments - pattern associated with wellbeing pressure.",
    trigger_summary="engagement_trend == 'Declining' AND attendance_rate < 60 AND works_part_time",
    severity="high",
    action="Welfare check-in from the Wellbeing Team within 5 working days",
    route_to="Wellbeing Team",
    policy_basis="Student Wellbeing Policy §8.1 (Proactive Outreach)",
    when=lambda r: (
        r["engagement_trend"] == "Declining"
        and r["attendance_rate"] < 60
        and bool(r["works_part_time"])
    ),
    rationale=lambda r: (
        f"Engagement trend is declining, attendance is {r['attendance_rate']:.1f}%, "
        f"and the student is balancing part-time work - pattern flagged by the "
        f"Wellbeing Policy."
    ),
)

RULE_FINANCIAL = Rule(
    id="FIN_SUPPORT",
    name="Possible financial pressure",
    description="Self-funded student combining part-time work with elevated failure risk - pattern associated with financial hardship.",
    trigger_summary="financial_support == 'Self-funded' AND works_part_time AND fail_prob >= 0.40",
    severity="medium",
    action="Referral to the Financial Aid Office for a hardship review",
    route_to="Financial Aid Office",
    policy_basis="Student Hardship Policy §9.1 (Proactive Referral)",
    when=lambda r: (
        (r["financial_support"] == "Self-funded")
        and bool(r["works_part_time"])
        and (r["fail_prob"] >= 0.40)
    ),
    rationale=lambda r: (
        f"Self-funded student with part-time work commitments and a "
        f"{r['fail_prob']*100:.1f}% predicted failure probability."
    ),
)


# ---------------------------------------------------------------------------
# Inclusion / equity rules
# ---------------------------------------------------------------------------
RULE_DISABILITY = Rule(
    id="DIS_SUPPORT",
    name="Declared disability + academic concern",
    description="Student has a declared disability and is showing elevated failure or withdrawal risk - proactive referral to Disability Services.",
    trigger_summary="has_declared_disability AND (fail_prob >= 0.40 OR withdraw_prob >= 0.20)",
    severity="medium",
    action="Proactive check-in from Disability Services to review support plan",
    route_to="Disability Services",
    policy_basis="Inclusive Education Policy §10.1 (Proactive Disability Support)",
    when=lambda r: bool(r["has_declared_disability"]) and (
        r["fail_prob"] >= 0.40 or r["withdraw_prob"] >= 0.20
    ),
    rationale=lambda r: (
        f"Declared disability on record; predicted failure probability "
        f"{r['fail_prob']*100:.1f}% / withdrawal probability "
        f"{r['withdraw_prob']*100:.1f}%."
    ),
)

RULE_INTERNATIONAL = Rule(
    id="INT_SUPPORT",
    name="International student showing isolation signals",
    description="International student with low attendance and a gap in VLE activity - pattern associated with isolation or visa concerns.",
    trigger_summary="is_international AND attendance_rate < 65 AND weeks_since_last_login >= 2",
    severity="medium",
    action="Welfare check from the International Office",
    route_to="International Office",
    policy_basis="International Student Support Policy §11.1 (Welfare Outreach)",
    when=lambda r: (
        bool(r["is_international"])
        and r["attendance_rate"] < 65
        and r["weeks_since_last_login"] >= 2
    ),
    rationale=lambda r: (
        f"International student with attendance {r['attendance_rate']:.1f}% and "
        f"{int(r['weeks_since_last_login'])} weeks since last VLE login."
    ),
)


# ---------------------------------------------------------------------------
# v3.1 enrichment-driven rules (use the richer behavioural and comms signals)
# ---------------------------------------------------------------------------
RULE_CONSECUTIVE_MISSED = Rule(
    id="ATT_CONSECUTIVE",
    name="Consecutive missed sessions",
    description="Five or more consecutive scheduled sessions missed without explanation - "
                "the institutional automatic-review trigger.",
    trigger_summary="consecutive_missed_max >= 5",
    severity="high",
    action="Personal tutor must initiate contact within 48 hours and log the outcome",
    route_to="Personal Tutor",
    policy_basis="Academic Engagement Policy §2.3 (Consecutive Absence Trigger)",
    when=lambda r: int(r.get("consecutive_missed_max", 0) or 0) >= 5,
    rationale=lambda r: (
        f"{int(r['consecutive_missed_max'])} consecutive unexcused absences on record "
        f"(threshold: 5). Cumulative unexcused absences this term: "
        f"{int(r['unexcused_absences'])}."
    ),
)

RULE_MISSING_ASSIGNMENTS = Rule(
    id="MISSING_ASSIGNMENTS",
    name="Multiple missing assignments",
    description="Three or more missing assignments across the student's enrolled modules.",
    trigger_summary="missing_assignments_count >= 3",
    severity="high",
    action="Academic Skills Team to schedule a recovery-plan meeting",
    route_to="Academic Skills Team",
    policy_basis="Assessment Recovery Policy §5.4 (Missing Assessment Trigger)",
    when=lambda r: int(r.get("missing_assignments_count", 0) or 0) >= 3,
    rationale=lambda r: (
        f"{int(r['missing_assignments_count'])} missing assignments on record "
        f"across enrolled modules (threshold: 3)."
    ),
)

RULE_WELLBEING_FLAGS = Rule(
    id="WELLBEING_FLAGS",
    name="Multiple wellbeing concerns raised",
    description="Two or more wellbeing concerns have been raised about this student by tutors or peers.",
    trigger_summary="wellbeing_flags >= 2",
    severity="high",
    action="Wellbeing Team to make direct contact within 3 working days",
    route_to="Wellbeing Team",
    policy_basis="Student Wellbeing Policy §8.2 (Multiple Flag Threshold)",
    when=lambda r: int(r.get("wellbeing_flags", 0) or 0) >= 2,
    rationale=lambda r: (
        f"{int(r['wellbeing_flags'])} wellbeing concerns raised "
        f"(threshold: 2). Last wellbeing check-in score: "
        f"{int(r['wellbeing_score'])}/10."
    ),
)

RULE_NON_RESPONSIVE = Rule(
    id="NON_RESPONSIVE",
    name="No response to repeated outreach",
    description="Three or more contact attempts in the last 30 days with no response from the student.",
    trigger_summary="comms_30d_count >= 3 AND comms_responded_30d == 0",
    severity="high",
    action="Senior Tutor to attempt in-person engagement and escalate to Head of Programme if no response",
    route_to="Senior Tutor",
    policy_basis="Engagement Escalation Policy §12.1 (Non-Response Escalation)",
    when=lambda r: (
        int(r.get("comms_30d_count", 0) or 0) >= 3
        and int(r.get("comms_responded_30d", 0) or 0) == 0
    ),
    rationale=lambda r: (
        f"{int(r['comms_30d_count'])} outreach attempts in the last 30 days, "
        f"all unanswered. Most recent contact: {r.get('last_contact_date', 'unknown')} "
        f"via {r.get('last_contact_channel', 'unknown')}."
    ),
)


# ---------------------------------------------------------------------------
# Catalogue (evaluation order: more specific / higher severity first)
# ---------------------------------------------------------------------------
RULES: list[Rule] = [
    # ML withdrawal rules (most severe first)
    RULE_WITHDRAW_CRITICAL,
    RULE_WITHDRAW_HIGH,
    RULE_WITHDRAW_MODERATE,
    # Classical attendance + engagement rules
    RULE_ATTENDANCE_CRITICAL,
    RULE_ATTENDANCE_WARNING,
    RULE_VLE_DISENGAGED,
    RULE_CONSECUTIVE_MISSED,
    # ML failure + brief rules
    RULE_FAIL_UPCOMING_CRITICAL,
    RULE_FAIL_UPCOMING,
    RULE_BRIEF_UNREAD,
    # Academic performance
    RULE_GRADE_DROP,
    RULE_LATE_PATTERN,
    RULE_MISSING_ASSIGNMENTS,
    # Wellbeing + context
    RULE_WELLBEING,
    RULE_WELLBEING_FLAGS,
    RULE_FINANCIAL,
    RULE_DISABILITY,
    RULE_INTERNATIONAL,
    # Communication escalation
    RULE_NON_RESPONSIVE,
]


def rules_catalogue() -> list[dict]:
    """Return the static part of every rule as a list of dicts (for the dashboard)."""
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "trigger_summary": r.trigger_summary,
            "severity": r.severity,
            "action": r.action,
            "route_to": r.route_to,
            "policy_basis": r.policy_basis,
        }
        for r in RULES
    ]
