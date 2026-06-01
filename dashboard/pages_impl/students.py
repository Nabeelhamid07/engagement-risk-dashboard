"""Students - searchable, filterable roster of the entire cohort."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.data import students_with_predictions
from lib.theme import (
    CH_GREEN,
    CH_LABEL,
    CH_ORANGE,
    CH_RED,
    hero,
    kpi_card,
    kpi_row,
)

RISK_BAND_ORDER = ["High", "Medium", "Low"]


def render() -> None:
    hero(
        "Students",
        "The full cohort, searchable and filterable. Use this page to find a specific student, "
        "or to browse a cohort segment. Click a row to open the student's profile.",
    )

    df = students_with_predictions().copy()

    # ---- KPI strip ----
    n_total = len(df)
    n_high = int((df["risk_band"] == "High").sum())
    n_medium = int((df["risk_band"] == "Medium").sum())

    kpi_row(
        [
            kpi_card("Total students", f"{n_total:,}", caption="in the cohort"),
            kpi_card("High risk", f"{n_high}", caption=f"{n_high / n_total * 100:.1f}% of cohort", colour="red"),
            kpi_card("Medium risk", f"{n_medium}", caption=f"{n_medium / n_total * 100:.1f}% of cohort", colour="orange"),
        ]
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # ---- Filters ----
    programs = sorted(df["program"].dropna().unique().tolist())
    years = sorted(df["year_of_study"].unique().tolist())

    f1, f2, f3, f4 = st.columns([1.6, 1.2, 1, 1])
    with f1:
        q = st.text_input("Search by name or ID", placeholder="e.g. STU0168 or Ruth")
    with f2:
        program_sel = st.multiselect(
            "Programme", programs, default=[], placeholder="All programmes"
        )
    with f3:
        year_sel = st.multiselect("Year", years, default=[], placeholder="All years")
    with f4:
        band_sel = st.multiselect(
            "Risk band", RISK_BAND_ORDER, default=[], placeholder="All risk bands"
        )

    view = df.copy()
    if program_sel:
        view = view[view["program"].isin(program_sel)]
    if year_sel:
        view = view[view["year_of_study"].isin(year_sel)]
    if band_sel:
        view = view[view["risk_band"].isin(band_sel)]
    if q.strip():
        qq = q.strip().lower()
        view = view[
            view["student_id"].str.lower().str.contains(qq, na=False)
            | view["student_name"].str.lower().str.contains(qq, na=False)
        ]

    # ---- Sort ----
    sort_col = st.selectbox(
        "Sort by",
        ["Risk (highest first)", "Predicted P(withdraw)", "Attendance (lowest first)", "Name (A-Z)"],
    )
    if sort_col == "Risk (highest first)":
        view = view.sort_values("overall_risk", ascending=False)
    elif sort_col == "Predicted P(withdraw)":
        view = view.sort_values("withdraw_prob", ascending=False)
    elif sort_col == "Attendance (lowest first)":
        view = view.sort_values("attendance_rate")
    else:
        view = view.sort_values("student_name")

    st.markdown(
        f"<div style='color:{CH_LABEL}; font-size:0.85rem; margin: 0.5rem 0 1rem 0;'>"
        f"Showing <b>{len(view)}</b> of <b>{len(df)}</b> students."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Table ----
    show = view.copy()
    show["Attendance %"] = show["attendance_rate"].round(1)
    show["Last grade"] = show["last_assessment_grade"]
    show["GPA"] = show["previous_semester_gpa"].round(2)
    show["P(withdraw)"] = (show["withdraw_prob"] * 100).round(1)
    show["P(fail next)"] = (show["fail_prob"] * 100).round(1)

    display = show[
        [
            "student_id",
            "student_name",
            "program",
            "year_of_study",
            "Attendance %",
            "Last grade",
            "GPA",
            "P(withdraw)",
            "P(fail next)",
            "risk_band",
        ]
    ].rename(
        columns={
            "student_id": "ID",
            "student_name": "Student",
            "program": "Programme",
            "year_of_study": "Year",
            "risk_band": "Risk band",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=620,
        column_config={
            "Attendance %": st.column_config.NumberColumn("Attendance %", format="%.1f"),
            "Last grade": st.column_config.NumberColumn("Last grade", format="%d"),
            "GPA": st.column_config.NumberColumn("GPA", format="%.2f"),
            "P(withdraw)": st.column_config.ProgressColumn(
                "P(withdraw)", format="%.1f%%", min_value=0, max_value=100
            ),
            "P(fail next)": st.column_config.ProgressColumn(
                "P(fail next)", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

    st.caption(
        "To inspect a student, copy their ID and open the **Student Profile** page from the sidebar."
    )

    st.download_button(
        "Export filtered roster (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="students_filtered.csv",
        mime="text/csv",
    )
