"""Modules - module-level lens on the cohort.

The other pages are student-lens. This page flips the question and asks:
which classes are the problem? Pick a module, see the students in that module
sorted by their *module-level* attendance and grade (not their overall
attendance, which would mislead).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import (
    load_modules,
    load_module_engagement,
    module_roster,
    students_with_predictions,
)
from lib.theme import (
    CH_GREEN,
    CH_GREY,
    CH_INK,
    CH_LABEL,
    CH_MUTED,
    CH_ORANGE,
    CH_PAGE,
    CH_RED,
    hero,
    kpi_card,
    kpi_row,
    risk_band_html,
    section_title,
    style_fig,
)


# ---------------------------------------------------------------------------
# Module browser - compact horizontal bar with short code-only labels so 3 bars
# fit comfortably in one row.
# ---------------------------------------------------------------------------
def _ranked_bar(
    d: pd.DataFrame,
    value_col: str,
    fmt: str,
    colours: list[str],
    hover_unit: str = "",
) -> go.Figure:
    """Build a horizontal bar chart from already-sorted-and-ranked module data."""
    labels = d["module_code"].tolist()
    customdata = d["module_name"].tolist()
    fig = go.Figure(
        go.Bar(
            x=d[value_col].tolist(),
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[fmt.format(v) for v in d[value_col]],
            textposition="outside",
            textfont=dict(color=CH_INK, size=11),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata}</b> (%{y})<br>"
                f"%{{x}}{hover_unit}"
                "<extra></extra>"
            ),
        )
    )
    x_max = max(d[value_col].max() * 1.18, 1)
    fig.update_layout(
        xaxis=dict(range=[0, x_max], gridcolor=CH_GREY),
        yaxis=dict(title="", tickfont=dict(size=11)),
        showlegend=False,
    )
    return style_fig(fig, height=max(260, 32 * len(d) + 70))


# ---------------------------------------------------------------------------
# Charts - drill-down for one module
# ---------------------------------------------------------------------------
def _hist(values: list[float], colour: str, title: str, x_label: str, x_max: int = 100) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=20,
            marker_color=colour,
            opacity=0.85,
            hovertemplate=x_label + ": %{x}<br>Students: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title=x_label, range=[0, x_max], gridcolor=CH_GREY),
        yaxis=dict(title="Students", gridcolor=CH_GREY),
        bargap=0.06,
        showlegend=False,
    )
    return style_fig(fig, height=270, title=title)


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------
def _make_jump_callback(student_id: str):
    def _cb() -> None:
        st.session_state["target_student_id"] = student_id
        st.session_state["nav_radio"] = "Student Profile"

    return _cb


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------
def render() -> None:
    hero(
        "Modules",
        "Module-level lens on the cohort. Use this page to ask 'which classes are the "
        "problem?' Pick a module and drill into the students enrolled in it, sorted by "
        "their module-specific attendance and grade.",
    )

    mods = load_modules()
    me = load_module_engagement()

    if mods.empty or me.empty:
        st.info("Module-level data is not available.")
        return

    # Add per-module high-risk-student count by joining module enrolments
    # against the predicted risk band on each student.
    swp = students_with_predictions()[["student_id", "risk_band"]]
    me_with_band = me.merge(swp, on="student_id", how="left")
    high_per_module = (
        me_with_band[me_with_band["risk_band"] == "High"]
        .groupby("module_code")
        .size()
        .to_dict()
    )
    mods = mods.copy()
    mods["n_high_risk"] = mods["module_code"].map(high_per_module).fillna(0).astype(int)

    # ---- KPI strip ----
    n_modules = len(mods)
    lowest_att = mods.nsmallest(1, "mean_attendance").iloc[0]
    lowest_grade = mods.nsmallest(1, "mean_grade").iloc[0]
    most_risk = mods.nlargest(1, "n_high_risk").iloc[0]

    kpi_row(
        [
            kpi_card("Modules in view", f"{n_modules}", caption="this term"),
            kpi_card(
                "Lowest-attended module",
                f"{lowest_att['mean_attendance']:.1f}%",
                caption=f"{lowest_att['module_code']} - {lowest_att['module_name']}",
                colour="red",
            ),
            kpi_card(
                "Module with most high-risk students",
                f"{int(most_risk['n_high_risk'])}",
                caption=f"{most_risk['module_code']} - {most_risk['module_name']}",
                colour="red",
            ),
        ]
    )

    # ---- Module browser: three ranked bars in one row ----
    section_title("Module browser")
    st.caption(
        "Three quick rankings across the cohort. Hover any bar to see the full module name and value."
    )

    n_top = 10

    # Lowest attendance: red where attendance < 60, orange < 75, grey otherwise.
    d_att = mods.nsmallest(n_top, "mean_attendance").sort_values(
        "mean_attendance", ascending=False
    )
    colours_att = [
        CH_RED if v < 60 else (CH_ORANGE if v < 75 else "#A0A0A0")
        for v in d_att["mean_attendance"]
    ]

    # Lowest grade: red where grade < 50, orange < 60, grey otherwise.
    d_grade = mods.nsmallest(n_top, "mean_grade").sort_values(
        "mean_grade", ascending=False
    )
    colours_grade = [
        CH_RED if v < 50 else (CH_ORANGE if v < 60 else "#A0A0A0")
        for v in d_grade["mean_grade"]
    ]

    # Most high-risk students: red bands relative to the highest count in view.
    d_risk = mods.nlargest(n_top, "n_high_risk").sort_values(
        "n_high_risk", ascending=True
    )
    max_risk = max(int(d_risk["n_high_risk"].max() or 1), 1)
    colours_risk = [
        CH_RED if v >= max_risk * 0.7 else (CH_ORANGE if v >= max_risk * 0.4 else "#A0A0A0")
        for v in d_risk["n_high_risk"]
    ]

    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.5rem;'>LOWEST ATTENDANCE</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _ranked_bar(d_att, "mean_attendance", "{:.1f}%", colours_att, hover_unit="%"),
            width="stretch",
        )
    with c2:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.5rem;'>LOWEST AVERAGE GRADE</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _ranked_bar(d_grade, "mean_grade", "{:.1f}", colours_grade),
            width="stretch",
        )
    with c3:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.5rem;'>MOST HIGH-RISK STUDENTS</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _ranked_bar(d_risk, "n_high_risk", "{:.0f}", colours_risk, hover_unit=" students"),
            width="stretch",
        )

    # ---- Module drill-down ----
    section_title("Drill into a module")

    mod_options = [
        f"{row.module_code} - {row.module_name}" for row in mods.sort_values("module_code").itertuples()
    ]
    mod_index = {f"{row.module_code} - {row.module_name}": row.module_code for row in mods.itertuples()}

    chosen = st.selectbox("Choose a module", mod_options, key="mod_picker")
    module_code = mod_index[chosen]
    mod_row = mods[mods["module_code"] == module_code].iloc[0]

    # Module header card
    st.markdown(
        f"""
        <div class="profile-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                <div>
                    <div class="profile-name">{mod_row['module_name']} <span style="color:{CH_MUTED}; font-weight:500; font-size:1rem;">({mod_row['module_code']})</span></div>
                    <div class="profile-meta">{mod_row['programme']} &middot; Level {int(mod_row['level'])} &middot; {int(mod_row['n_students'])} students enrolled</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    roster = module_roster(module_code)
    if roster.empty:
        st.info("No enrolled students found for this module.")
        return

    # Module-level KPI strip
    mean_att = float(roster["module_attendance_pct"].mean())
    mean_grade = float(roster["module_grade"].mean())
    mean_eng = float(roster["module_engagement_score"].mean())
    n_at_risk = int(roster["risk_band"].eq("High").sum())

    kpi_row(
        [
            kpi_card(
                "Mean attendance",
                f"{mean_att:.1f}%",
                caption="across enrolled students",
                colour="red" if mean_att < 60 else ("orange" if mean_att < 75 else None),
            ),
            kpi_card(
                "Mean grade",
                f"{mean_grade:.1f}",
                caption="this module only",
                colour="red" if mean_grade < 50 else ("orange" if mean_grade < 60 else None),
            ),
            kpi_card(
                "High-risk students in this module",
                f"{n_at_risk}",
                caption=f"{n_at_risk / len(roster) * 100:.0f}% of the class",
                colour="red" if n_at_risk > 5 else None,
            ),
        ]
    )

    # Quick-pick: top 5 lowest attenders and lowest graders
    section_title("Who needs the most attention in this module")
    cq1, cq2 = st.columns(2, gap="medium")
    with cq1:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.4rem;'>LOWEST ATTENDERS</div>",
            unsafe_allow_html=True,
        )
        low_att = roster.sort_values("module_attendance_pct", ascending=True).head(5)
        for _, r in low_att.iterrows():
            ca, cb = st.columns([4, 1.4], gap="small")
            with ca:
                st.markdown(
                    f"<div style='padding:0.55rem 0; border-bottom: 1px solid {CH_GREY};'>"
                    f"<div style='font-weight:600; color:{CH_INK}; font-size:0.92rem;'>{r['student_name']} <span style='color:{CH_MUTED}; font-weight:500; font-size:0.85rem;'>({r['student_id']})</span></div>"
                    f"<div style='color:{CH_MUTED}; font-size:0.8rem; margin-top:0.15rem;'>"
                    f"Attendance <span style='color:{CH_RED}; font-weight:700;'>{r['module_attendance_pct']:.1f}%</span>"
                    f" &middot; Grade {r['module_grade']:.0f}"
                    f" &middot; {risk_band_html(r['risk_band'])} overall</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with cb:
                st.button(
                    "Profile →",
                    key=f"mod_low_att_{module_code}_{r['student_id']}",
                    width="stretch",
                    on_click=_make_jump_callback(r["student_id"]),
                    type="secondary",
                )

    with cq2:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.4rem;'>LOWEST GRADERS</div>",
            unsafe_allow_html=True,
        )
        low_grade = roster.sort_values("module_grade", ascending=True).head(5)
        for _, r in low_grade.iterrows():
            ca, cb = st.columns([4, 1.4], gap="small")
            with ca:
                st.markdown(
                    f"<div style='padding:0.55rem 0; border-bottom: 1px solid {CH_GREY};'>"
                    f"<div style='font-weight:600; color:{CH_INK}; font-size:0.92rem;'>{r['student_name']} <span style='color:{CH_MUTED}; font-weight:500; font-size:0.85rem;'>({r['student_id']})</span></div>"
                    f"<div style='color:{CH_MUTED}; font-size:0.8rem; margin-top:0.15rem;'>"
                    f"Grade <span style='color:{CH_RED}; font-weight:700;'>{r['module_grade']:.0f}</span>"
                    f" &middot; Attendance {r['module_attendance_pct']:.1f}%"
                    f" &middot; {risk_band_html(r['risk_band'])} overall</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            with cb:
                st.button(
                    "Profile →",
                    key=f"mod_low_grade_{module_code}_{r['student_id']}",
                    width="stretch",
                    on_click=_make_jump_callback(r["student_id"]),
                    type="secondary",
                )

    # Full roster table
    section_title("Full roster for this module")
    show = roster[
        [
            "student_id",
            "student_name",
            "program",
            "year_of_study",
            "module_attendance_pct",
            "module_grade",
            "assessments_completed",
            "assessments_total",
            "assessments_missed",
            "module_engagement_score",
            "risk_band",
        ]
    ].rename(
        columns={
            "student_id": "ID",
            "student_name": "Student",
            "program": "Programme",
            "year_of_study": "Year",
            "module_attendance_pct": "Module attendance %",
            "module_grade": "Module grade",
            "assessments_completed": "Completed",
            "assessments_total": "Total",
            "assessments_missed": "Missed",
            "module_engagement_score": "Engagement",
            "risk_band": "Overall risk band",
        }
    )
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Module attendance %": st.column_config.ProgressColumn(
                "Module attendance %", format="%.1f%%", min_value=0, max_value=100
            ),
            "Module grade": st.column_config.NumberColumn(
                "Module grade", format="%.0f"
            ),
            "Engagement": st.column_config.NumberColumn(
                "Engagement", format="%.2f"
            ),
        },
    )

    # Distributions
    section_title("How this class is distributed")
    ch1, ch2 = st.columns(2, gap="medium")
    with ch1:
        st.plotly_chart(
            _hist(
                roster["module_attendance_pct"].tolist(),
                CH_RED,
                "Module attendance - student distribution",
                "Attendance (%)",
                x_max=100,
            ),
            width="stretch",
        )
    with ch2:
        st.plotly_chart(
            _hist(
                roster["module_grade"].tolist(),
                CH_INK,
                "Module grade - student distribution",
                "Grade",
                x_max=100,
            ),
            width="stretch",
        )
