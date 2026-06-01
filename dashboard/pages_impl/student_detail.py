"""Student Profile - 360-degree view of one student, organised into tabs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.activity import case_activity_steps
from lib.data import (
    load_actions,
    load_rules_catalogue,
    student_case,
    student_comms,
    student_comms_summary_row,
    student_daily_attendance,
    student_drivers,
    student_history,
    student_modules,
    student_record,
    student_weekly,
    students_with_predictions,
)
from lib.reasons import case_reason_rows
from lib.theme import (
    CH_CHARCOAL,
    CH_GREEN,
    CH_GREY,
    CH_INK,
    CH_LABEL,
    CH_MUTED,
    CH_ORANGE,
    CH_PAGE,
    CH_PANEL,
    CH_RED,
    activity_log_html,
    case_card,
    comms_strip,
    hero,
    kpi_card,
    kpi_row,
    risk_band_html,
    section_title,
    severity_badge,
    style_fig,
    supporting_concerns_html,
    why_block_html,
)


STATUS_COLOUR = {
    "Present": "#1B7F3B",
    "Late": "#F4C023",
    "Excused": "#9A9A9A",
    "Absent": "#E2231A",
}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _gauge(prob: float, title: str) -> go.Figure:
    if prob < 0.15:
        colour = CH_GREEN
    elif prob < 0.40:
        colour = CH_ORANGE
    else:
        colour = CH_RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 30, "color": CH_INK}},
            title={"text": title, "font": {"size": 13, "color": CH_MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": CH_MUTED}},
                "bar": {"color": colour, "thickness": 0.75},
                "bgcolor": CH_PAGE,
                "borderwidth": 1,
                "bordercolor": CH_GREY,
                "steps": [
                    {"range": [0, 15], "color": "#E8F4EA"},
                    {"range": [15, 40], "color": "#FCEFD4"},
                    {"range": [40, 100], "color": "#FBE3E1"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor=CH_PAGE, height=230, margin=dict(l=15, r=15, t=35, b=10))
    return fig


def _drivers_bar(drivers_df: pd.DataFrame, title: str) -> go.Figure:
    if drivers_df.empty:
        return style_fig(go.Figure(), height=230, title=f"{title} (no drivers)")
    d = drivers_df.copy().sort_values("rank", ascending=False)
    colours = [CH_RED if v > 0 else CH_GREEN for v in d["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=d["shap_value"].tolist(),
            y=d["feature"].tolist(),
            orientation="h",
            marker_color=colours,
            text=[f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in d["shap_value"]],
            textposition="outside",
            textfont=dict(color=CH_INK, size=11),
        )
    )
    fig.update_layout(
        xaxis=dict(title="", gridcolor=CH_GREY, zeroline=True, zerolinecolor=CH_GREY),
        yaxis=dict(title=""),
        showlegend=False,
    )
    return style_fig(fig, height=max(230, 42 * len(d) + 50), title=title)


def _weekly_chart(weekly_df: pd.DataFrame) -> go.Figure:
    if weekly_df.empty:
        return style_fig(go.Figure(), height=320, title="12-week engagement trajectory")
    fig = go.Figure()
    fig.add_scatter(
        x=weekly_df["week"],
        y=weekly_df["attendance_pct"],
        mode="lines+markers",
        name="Attendance (%)",
        line=dict(color=CH_RED, width=2.5),
        marker=dict(size=8),
        yaxis="y",
    )
    fig.add_scatter(
        x=weekly_df["week"],
        y=weekly_df["vle_logins"],
        mode="lines+markers",
        name="VLE logins",
        line=dict(color=CH_ORANGE, width=2, dash="dot"),
        marker=dict(size=7),
        yaxis="y2",
    )
    fig.update_layout(
        xaxis=dict(title="Week", dtick=1, gridcolor=CH_GREY),
        yaxis=dict(title="Attendance (%)", range=[0, 105], gridcolor=CH_GREY),
        yaxis2=dict(title="VLE logins", overlaying="y", side="right", showgrid=False, range=[0, 28]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return style_fig(fig, height=320)


def _attendance_heatmap(daily: pd.DataFrame) -> go.Figure:
    """Weeks x days heatmap of daily attendance status, colour-coded."""
    if daily.empty:
        return style_fig(go.Figure(), height=260, title="Daily attendance")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    code = {"Present": 3, "Late": 2, "Excused": 1, "Absent": 0}
    grid = daily.pivot_table(index="day", columns="week", values="status", aggfunc="first")
    grid = grid.reindex(days)

    z = grid.map(lambda s: code.get(s, None) if pd.notna(s) else None).values.tolist()
    text = grid.fillna("").values.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"W{w}" for w in grid.columns],
            y=days,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10, color=CH_INK),
            colorscale=[
                [0.00, STATUS_COLOUR["Absent"]],
                [0.33, STATUS_COLOUR["Excused"]],
                [0.66, STATUS_COLOUR["Late"]],
                [1.00, STATUS_COLOUR["Present"]],
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate="%{y}, %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="", side="top"),
        yaxis=dict(title="", autorange="reversed"),
    )
    return style_fig(fig, height=260)


def _module_attendance_chart(mods: pd.DataFrame) -> go.Figure:
    if mods.empty:
        return style_fig(go.Figure(), height=300, title="Attendance by module")
    d = mods.sort_values("module_attendance_pct", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=d["module_attendance_pct"].tolist(),
            y=d["module_code"].tolist(),
            orientation="h",
            marker_color=[
                CH_RED if v < 60 else (CH_ORANGE if v < 80 else CH_GREEN)
                for v in d["module_attendance_pct"]
            ],
            text=[f"{v:.0f}%" for v in d["module_attendance_pct"]],
            textposition="outside",
            textfont=dict(color=CH_INK, size=11),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 110], gridcolor=CH_GREY, title=""),
        yaxis=dict(title=""),
        showlegend=False,
    )
    return style_fig(fig, height=max(220, 38 * len(d) + 60))


def _module_grade_chart(mods: pd.DataFrame, modules_meta: pd.DataFrame) -> go.Figure:
    if mods.empty:
        return style_fig(go.Figure(), height=300, title="Grade by module vs cohort mean")
    d = mods.merge(modules_meta[["module_code", "mean_grade"]], on="module_code", how="left")
    d = d.sort_values("module_code")
    fig = go.Figure()
    fig.add_bar(
        y=d["module_code"],
        x=d["module_grade"],
        orientation="h",
        marker_color=CH_INK,
        name="This student",
        text=[f"{v:.0f}" for v in d["module_grade"]],
        textposition="outside",
        textfont=dict(color=CH_INK, size=11),
    )
    fig.add_bar(
        y=d["module_code"],
        x=d["mean_grade"],
        orientation="h",
        marker_color=CH_GREY,
        name="Cohort mean",
        opacity=0.7,
    )
    fig.update_layout(
        barmode="group",
        xaxis=dict(range=[0, 110], gridcolor=CH_GREY, title=""),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_fig(fig, height=max(260, 50 * len(d) + 60))


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------
def _context_strip(rec: dict) -> str:
    flags: list[str] = []
    if rec.get("is_commuter"):
        flags.append("Commuter")
    if rec.get("works_part_time"):
        flags.append("Works part-time")
    if rec.get("has_declared_disability"):
        flags.append("Declared disability")
    if rec.get("is_international"):
        flags.append("International")
    flags.append(f"Financial support: {rec.get('financial_support', '')}")
    flags.append(f"Year {int(rec.get('year_of_study', 0))}")
    return " &middot; ".join(flags)


def _profile_header(rec: dict) -> None:
    st.markdown(
        f"""
        <div class="profile-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                <div>
                    <div class="profile-name">{rec["student_name"]} <span style="color:{CH_MUTED}; font-weight:500; font-size:1rem;">({rec["student_id"]})</span></div>
                    <div class="profile-meta">{rec["program"]} &middot; {_context_strip(rec)}</div>
                </div>
                <div style="text-align:right; min-width:120px;">
                    <div class="kpi-label">OVERALL RISK</div>
                    <div style="font-size:1.6rem; margin-top:0.3rem;">{risk_band_html(rec["risk_band"])}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Individual tabs
# ---------------------------------------------------------------------------
def _tab_overview(rec: dict, student_id: str) -> None:
    kpi_row(
        [
            kpi_card(
                "P(withdraw)",
                f"{rec['withdraw_prob'] * 100:.1f}%",
                caption="probability of withdrawal",
                colour="red" if rec["withdraw_prob"] > 0.40 else ("orange" if rec["withdraw_prob"] > 0.15 else "green"),
            ),
            kpi_card(
                "P(fail next)",
                f"{rec['fail_prob'] * 100:.1f}%",
                caption="probability of failing next assessment",
                colour="red" if rec["fail_prob"] > 0.40 else ("orange" if rec["fail_prob"] > 0.15 else "green"),
            ),
            kpi_card(
                "Attendance (4w avg)",
                f"{rec['attendance_rate']:.1f}%",
                caption=f"trend: {rec['engagement_trend']}",
            ),
        ]
    )

    st.markdown(
        f"<div style='color:{CH_MUTED}; font-size:0.85rem; margin: 0.8rem 0 0.2rem 0;'>"
        "Predicted probabilities come from calibrated ML models trained on past cohorts. "
        "These scores are computed for <strong>every</strong> student, not just at-risk ones."
        "</div>",
        unsafe_allow_html=True,
    )

    section_title("Predicted risk and the features driving it")
    cg, cd = st.columns([0.85, 1.4], gap="medium")
    with cg:
        st.plotly_chart(_gauge(rec["withdraw_prob"], "Withdrawal probability"), width="stretch")
    with cd:
        st.plotly_chart(
            _drivers_bar(student_drivers(student_id, "withdrew"), "Top drivers - withdrawal"),
            width="stretch",
        )

    cg2, cd2 = st.columns([0.85, 1.4], gap="medium")
    with cg2:
        st.plotly_chart(_gauge(rec["fail_prob"], "Failure probability"), width="stretch")
    with cd2:
        st.plotly_chart(
            _drivers_bar(
                student_drivers(student_id, "failed_next_assessment"),
                "Top drivers - next-assessment failure",
            ),
            width="stretch",
        )

    st.caption(
        "**Red bars** push the predicted risk up; **green bars** push it down. "
        "Magnitudes reflect each feature's contribution to the model's output (SHAP-derived)."
    )

    section_title("12-week trajectory")
    st.plotly_chart(_weekly_chart(student_weekly(student_id)), width="stretch")


def _tab_attendance(rec: dict, student_id: str) -> None:
    kpi_row(
        [
            kpi_card(
                "Attendance (4w avg)",
                f"{rec['attendance_rate']:.1f}%",
                caption="rolling 4-week average",
                colour="red" if rec["attendance_rate"] < 60 else ("orange" if rec["attendance_rate"] < 80 else "green"),
            ),
            kpi_card(
                "Consecutive missed",
                f"{int(rec.get('consecutive_missed_max', 0) or 0)}",
                caption="longest absence streak (days)",
                colour="red" if int(rec.get("consecutive_missed_max", 0) or 0) >= 3 else None,
            ),
            kpi_card(
                "Late arrivals",
                f"{int(rec.get('late_arrivals_total', 0) or 0)}",
                caption="total this semester",
            ),
        ]
    )

    kpi_row(
        [
            kpi_card(
                "Unexcused absences",
                f"{int(rec.get('unexcused_absences', 0) or 0)}",
                caption="not authorised",
                colour="red" if int(rec.get("unexcused_absences", 0) or 0) >= 5 else None,
            ),
            kpi_card(
                "Excused absences",
                f"{int(rec.get('excused_absences', 0) or 0)}",
                caption="authorised / certified",
            ),
            kpi_card(
                "Attendance trend",
                rec.get("engagement_trend", "stable").title(),
                caption="direction over last 4 weeks",
            ),
        ]
    )

    section_title("Weekly attendance trajectory")
    weekly = student_weekly(student_id)
    if weekly.empty:
        st.info("No weekly engagement records available.")
    else:
        fig = go.Figure(
            go.Bar(
                x=weekly["week"],
                y=weekly["attendance_pct"],
                marker_color=[
                    CH_RED if v < 60 else (CH_ORANGE if v < 80 else CH_GREEN)
                    for v in weekly["attendance_pct"]
                ],
                text=[f"{v:.0f}%" for v in weekly["attendance_pct"]],
                textposition="outside",
                textfont=dict(color=CH_INK, size=10),
                hovertemplate="Week %{x}: %{y:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis=dict(title="Week", dtick=1, gridcolor=CH_GREY),
            yaxis=dict(title="Attendance (%)", range=[0, 110], gridcolor=CH_GREY),
            showlegend=False,
        )
        st.plotly_chart(style_fig(fig, height=300), width="stretch")

    section_title("Daily attendance log (12 weeks)")
    daily = student_daily_attendance(student_id)
    if daily.empty:
        st.info("Daily attendance log not available for this student.")
    else:
        st.plotly_chart(_attendance_heatmap(daily), width="stretch")
        legend_html = " &nbsp;&middot;&nbsp; ".join(
            [
                f"<span style='display:inline-block;width:10px;height:10px;background:{c};border-radius:3px;'></span>"
                f" {s}"
                for s, c in STATUS_COLOUR.items()
            ]
        )
        st.markdown(
            f"<div style='color:{CH_MUTED}; font-size:0.78rem; margin-top:0.4rem;'>"
            f"Legend: {legend_html}"
            "</div>",
            unsafe_allow_html=True,
        )


def _tab_academics(rec: dict, student_id: str) -> None:
    kpi_row(
        [
            kpi_card(
                "Last assessment grade",
                f"{int(rec['last_assessment_grade'])}%",
                caption="most recent mark",
                colour="red" if rec["last_assessment_grade"] < 40 else ("orange" if rec["last_assessment_grade"] < 55 else "green"),
            ),
            kpi_card(
                "Previous GPA",
                f"{rec['previous_semester_gpa']:.2f}",
                caption="cumulative",
            ),
            kpi_card(
                "Missing assignments",
                f"{int(rec.get('missing_assignments_count', 0) or 0)}",
                caption="overdue submissions",
                colour="red" if int(rec.get("missing_assignments_count", 0) or 0) >= 2 else None,
            ),
        ]
    )

    kpi_row(
        [
            kpi_card(
                "Submitted on time",
                f"{int(rec['assignments_submitted_on_time'])}",
                caption="assignments delivered punctually",
            ),
            kpi_card(
                "Submitted late",
                f"{int(rec['assignments_submitted_late'])}",
                caption="past deadline",
            ),
            kpi_card(
                "Mean module grade",
                f"{rec.get('mean_module_grade', 0) or 0:.1f}",
                caption=f"variance {rec.get('module_grade_variance', 0) or 0:.1f}",
            ),
        ]
    )

    mods = student_modules(student_id)
    if mods.empty:
        st.info("Per-module breakdown not available for this student.")
        return

    from lib.data import load_modules
    modules_meta = load_modules()

    section_title("Per-module attendance")
    st.plotly_chart(_module_attendance_chart(mods), width="stretch")

    section_title("Per-module grade vs cohort mean")
    st.plotly_chart(_module_grade_chart(mods, modules_meta), width="stretch")

    section_title("Module breakdown")
    show = mods.copy()
    keep = [
        "module_code",
        "module_name",
        "module_attendance_pct",
        "module_grade",
        "assessments_completed",
        "assessments_total",
        "assessments_missed",
        "module_engagement_score",
    ]
    show = show[[c for c in keep if c in show.columns]].rename(
        columns={
            "module_code": "Module",
            "module_name": "Title",
            "module_attendance_pct": "Attendance %",
            "module_grade": "Grade",
            "assessments_completed": "Completed",
            "assessments_total": "Total",
            "assessments_missed": "Missed",
            "module_engagement_score": "Engagement",
        }
    )
    st.dataframe(show, hide_index=True, width="stretch")


def _tab_engagement(rec: dict) -> None:
    kpi_row(
        [
            kpi_card(
                "VLE logins (last wk)",
                f"{int(rec['vle_logins_last_week'])}",
                caption="login count",
            ),
            kpi_card(
                "VLE hours (last wk)",
                f"{rec['vle_time_hours_last_week']:.1f}",
                caption="time on platform",
            ),
            kpi_card(
                "Materials accessed (4w)",
                f"{int(rec['materials_accessed_last_4w'])}",
                caption="resources opened",
            ),
        ]
    )

    kpi_row(
        [
            kpi_card(
                "Quiz attempts",
                f"{int(rec.get('quiz_attempts_count', 0) or 0)}",
                caption="formative quizzes taken",
            ),
            kpi_card(
                "Resource downloads",
                f"{int(rec.get('resource_downloads_count', 0) or 0)}",
                caption="materials downloaded",
            ),
            kpi_card(
                "Forum posts",
                f"{int(rec.get('forum_posts_count', 0) or 0)}",
                caption="discussion contributions",
            ),
        ]
    )

    section_title("Assessment context")
    brief_accessed = "Yes" if rec.get("accessed_upcoming_assessment_brief") in (True, 1, "True", "true") else "No"
    days_to_next = int(rec.get("days_to_nearest_assessment", 0) or 0)
    st.markdown(
        f"""
        <div style='display:flex; gap:1.5rem; flex-wrap:wrap;'>
          <div><span class='kpi-label'>NEXT ASSESSMENT IN</span><br>
            <span style='font-size:1.4rem; color:{CH_INK}; font-weight:700;'>{days_to_next} days</span></div>
          <div><span class='kpi-label'>BRIEF ACCESSED?</span><br>
            <span style='font-size:1.4rem; color:{CH_RED if brief_accessed == "No" else CH_GREEN};
                       font-weight:700;'>{brief_accessed}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tab_wellbeing(rec: dict) -> None:
    wb_score = float(rec.get("wellbeing_score", 0) or 0)
    wb_flags = int(rec.get("wellbeing_flags", 0) or 0)
    wb_last = rec.get("wellbeing_last_checkin") or ""

    kpi_row(
        [
            kpi_card(
                "Wellbeing score",
                f"{wb_score:.1f}",
                caption="0 = severe concern, 10 = thriving",
                colour="red" if wb_score < 4 else ("orange" if wb_score < 6 else "green"),
            ),
            kpi_card(
                "Open wellbeing flags",
                f"{wb_flags}",
                caption="active concerns logged",
                colour="red" if wb_flags >= 2 else ("orange" if wb_flags == 1 else None),
            ),
            kpi_card(
                "Last check-in",
                str(wb_last) if wb_last else "Never",
                caption="counselling / pastoral",
            ),
        ]
    )

    section_title("Personal & contextual factors")
    chips: list[tuple[str, bool]] = [
        ("Commuter", bool(rec.get("is_commuter"))),
        ("Works part-time", bool(rec.get("works_part_time"))),
        ("Declared disability", bool(rec.get("has_declared_disability"))),
        ("International student", bool(rec.get("is_international"))),
        (
            f"Financial support: {rec.get('financial_support', '-')}",
            (str(rec.get("financial_support", "")).lower() in {"self-funded", "self funded"}),
        ),
    ]
    chip_html_parts: list[str] = []
    for label, on in chips:
        bg = CH_RED if on and "Financial" in label else (CH_PANEL if on else "#FFFFFF")
        colour = CH_RED if on and "Financial" in label else CH_CHARCOAL
        border = CH_GREY
        weight = 600 if on else 400
        if "Financial" in label and on:
            chip_html_parts.append(
                f"<span style='background:{CH_RED}; color:white; padding:6px 12px; "
                f"border-radius:999px; font-size:0.82rem; font-weight:600; margin-right:0.45rem;'>{label}</span>"
            )
        else:
            chip_html_parts.append(
                f"<span style='background:{bg}; color:{colour}; padding:6px 12px; "
                f"border:1px solid {border}; border-radius:999px; font-size:0.82rem; "
                f"font-weight:{weight}; margin-right:0.45rem;'>{label}</span>"
            )
    st.markdown(
        f"<div style='line-height:2.2;'>{''.join(chip_html_parts)}</div>",
        unsafe_allow_html=True,
    )

    if wb_flags == 0 and wb_score >= 6:
        st.markdown(
            f"<div style='color:{CH_GREEN}; margin-top:1rem; font-size:0.88rem;'>"
            "No wellbeing concerns currently logged for this student."
            "</div>",
            unsafe_allow_html=True,
        )


def _tab_case_file(rec: dict, student_id: str) -> None:
    case = student_case(student_id)

    if not case:
        st.success(
            "No open case for this student. The engine has not flagged any policy violations. "
            "ML probabilities are still computed and shown on the Overview tab."
        )
        return

    sev = str(case["primary_severity"]).lower()
    title_html = f"{case['primary_rule_name']}{severity_badge(sev)}"

    st.markdown(
        case_card(
            title_html=title_html,
            reason="",
            action=case["primary_action"],
            severity=sev,
            assigned_to=case["primary_team"],
            policy_basis=case["primary_policy_basis"],
        ),
        unsafe_allow_html=True,
    )

    # 'Why this case was created' - readable conditions checklist.
    why_rows = case_reason_rows(case["primary_rule_id"], rec)
    st.markdown(why_block_html(why_rows), unsafe_allow_html=True)

    n_extra = int(case["concerns_total"]) - 1
    if n_extra > 0:
        section_title(f"{n_extra} other concern{'s' if n_extra != 1 else ''} on this case")
        st.markdown(
            supporting_concerns_html(case["supporting"] or []),
            unsafe_allow_html=True,
        )

    with st.expander("Show case activity"):
        actions = load_actions()
        match = actions[actions["action_id"] == case["primary_action_id"]]
        triggered_on = match["triggered_on"].iloc[0] if not match.empty else pd.Timestamp.now()
        primary_action_record = {
            "rule_id": case["primary_rule_id"],
            "rule_name": case["primary_rule_name"],
            "triggered_on": triggered_on,
            "route_to": case["primary_team"],
            "severity": case["primary_severity"],
            "action": case["primary_action"],
            "student_id": student_id,
        }
        this_student_comms = student_comms(student_id)
        steps = case_activity_steps(primary_action_record, this_student_comms)
        st.markdown(activity_log_html(steps), unsafe_allow_html=True)

    section_title("Historical interventions on record")
    hist = student_history(student_id)
    if hist.empty:
        st.caption("No prior interventions recorded.")
    else:
        hist_view = hist[
            ["intervention_date", "intervention_type", "reason", "student_responded", "engagement_improved"]
        ].copy()
        hist_view["intervention_date"] = hist_view["intervention_date"].dt.strftime("%Y-%m-%d")
        hist_view = hist_view.rename(
            columns={
                "intervention_date": "Date",
                "intervention_type": "Type",
                "reason": "Reason",
                "student_responded": "Responded",
                "engagement_improved": "Engagement improved",
            }
        )
        st.dataframe(hist_view, hide_index=True, width="stretch")


def _tab_communications(student_id: str) -> None:
    summary = student_comms_summary_row(student_id)
    comms = student_comms(student_id)

    if not summary and (comms is None or comms.empty):
        st.info("No communications have been sent to this student yet.")
        return

    last_date = ""
    if summary.get("last_contact_date") and not pd.isna(summary["last_contact_date"]):
        last_date = pd.Timestamp(summary["last_contact_date"]).strftime("%Y-%m-%d")

    st.markdown(
        comms_strip(
            last_date=last_date,
            last_channel=str(summary.get("last_contact_channel", "") or ""),
            last_status=str(summary.get("last_contact_status", "") or ""),
            count_30d=int(summary.get("comms_30d_count", 0) or 0),
            responded_30d=int(summary.get("comms_responded_30d", 0) or 0),
        ),
        unsafe_allow_html=True,
    )

    total = len(comms) if comms is not None else 0
    responded = int(comms["responded"].sum()) if comms is not None and not comms.empty else 0
    resp_rate = (responded / total * 100) if total else 0.0

    kpi_row(
        [
            kpi_card(
                "Total contacts (all time)",
                f"{total}",
                caption="across all channels",
            ),
            kpi_card(
                "Replies received",
                f"{responded}",
                caption=f"{resp_rate:.0f}% response rate",
                colour="green" if resp_rate >= 50 else ("orange" if resp_rate >= 20 else "red"),
            ),
            kpi_card(
                "Non-responsive flag",
                "YES" if summary.get("is_non_responsive") else "No",
                caption="2+ contacts, 0 replies in 30d",
                colour="red" if summary.get("is_non_responsive") else "green",
            ),
        ]
    )

    section_title("Communications log")
    if comms is None or comms.empty:
        st.caption("No communications recorded.")
        return
    show = comms.copy()
    show["sent_date"] = show["sent_date"].dt.strftime("%Y-%m-%d")
    show["response_date"] = show["response_date"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-"
    )
    show = show[
        ["sent_date", "channel", "subject", "status", "responded", "response_date"]
    ].rename(
        columns={
            "sent_date": "Sent",
            "channel": "Channel",
            "subject": "Subject",
            "status": "Status",
            "responded": "Responded",
            "response_date": "Response date",
        }
    )
    st.dataframe(show, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------
def render() -> None:
    hero(
        "Student Profile",
        "Drill into a single student. The tabs below organise attendance, academics, "
        "LMS engagement, wellbeing, the consolidated case file, and the communications "
        "log all under one consistent header.",
    )

    df = students_with_predictions()

    label_to_id = {
        f"{row['student_id']} - {row['student_name']} - "
        f"{row['risk_band']} ({row['overall_risk'] * 100:.0f}%)": row["student_id"]
        for _, row in df.sort_values("overall_risk", ascending=False).iterrows()
    }
    labels = list(label_to_id.keys())

    target_id = st.session_state.pop("target_student_id", None)
    if target_id:
        for lbl in labels:
            if label_to_id[lbl] == target_id:
                st.session_state["student_picker"] = lbl
                break

    choice = st.selectbox("Choose a student", labels, key="student_picker")
    student_id = label_to_id[choice]
    rec = student_record(student_id)
    if not rec:
        st.error("Student not found.")
        return

    _profile_header(rec)

    tab_overview, tab_att, tab_acad, tab_eng, tab_wb, tab_case, tab_comms = st.tabs(
        ["Overview", "Attendance", "Academics", "LMS engagement", "Wellbeing", "Case file", "Communications"]
    )
    with tab_overview:
        _tab_overview(rec, student_id)
    with tab_att:
        _tab_attendance(rec, student_id)
    with tab_acad:
        _tab_academics(rec, student_id)
    with tab_eng:
        _tab_engagement(rec)
    with tab_wb:
        _tab_wellbeing(rec)
    with tab_case:
        _tab_case_file(rec, student_id)
    with tab_comms:
        _tab_communications(student_id)
