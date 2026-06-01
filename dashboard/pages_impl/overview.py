"""Overview - the home page. Institutional KPIs, cohort filters,
disengaged-modules widget, Top 10 watchlist."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.data import (
    load_actions,
    load_module_engagement,
    load_modules,
    load_students,
    load_weekly,
    students_with_predictions,
)
from lib.theme import (
    CH_GREY,
    CH_INK,
    CH_LABEL,
    CH_MUTED,
    CH_ORANGE,
    CH_PAGE,
    CH_RED,
    RISK_BAND_COLOURS,
    hero,
    kpi_card,
    kpi_row,
    section_title,
    style_fig,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _risk_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_band"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    colours = [RISK_BAND_COLOURS[k] for k in counts.index]
    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.62,
            marker=dict(colors=colours, line=dict(color=CH_PAGE, width=3)),
            textinfo="label+value",
            textposition="outside",
            textfont=dict(size=13, color=CH_INK),
        )
    )
    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{int(counts.sum())}</b><br><span style='color:{CH_MUTED}; font-size:11px;'>students</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color=CH_INK),
            )
        ],
    )
    return style_fig(fig, height=340)


def _cohort_attendance_trend(weekly_subset: pd.DataFrame) -> go.Figure:
    weekly_mean = weekly_subset.groupby("week")["attendance_pct"].mean().round(2)
    fig = go.Figure()
    fig.add_scatter(
        x=weekly_mean.index.tolist(),
        y=weekly_mean.values.tolist(),
        mode="lines+markers",
        line=dict(color=CH_RED, width=3),
        marker=dict(size=8, color=CH_RED),
        fill="tozeroy",
        fillcolor="rgba(226,35,26,0.07)",
        name="Cohort mean attendance",
    )
    fig.update_layout(
        xaxis=dict(title="Week", dtick=1, gridcolor=CH_GREY),
        yaxis=dict(title="Mean attendance (%)", range=[max(0, weekly_mean.min() - 5), 100], gridcolor=CH_GREY),
        hovermode="x unified",
        showlegend=False,
    )
    return style_fig(fig, height=340)


def _disengaged_modules_chart(top_disengaged: pd.DataFrame) -> go.Figure:
    """Horizontal bar of modules with the lowest mean attendance."""
    if top_disengaged.empty:
        return style_fig(go.Figure(), height=300, title="Most disengaged modules (no data)")
    d = top_disengaged.sort_values("mean_attendance", ascending=True).copy()
    fig = go.Figure(
        go.Bar(
            x=d["mean_attendance"].tolist(),
            y=[f"{r.module_code} - {r.module_name}" for r in d.itertuples()],
            orientation="h",
            marker_color=[
                CH_RED if v < 60 else (CH_ORANGE if v < 75 else "#A0A0A0")
                for v in d["mean_attendance"]
            ],
            text=[f"{v:.1f}%" for v in d["mean_attendance"]],
            textposition="outside",
            textfont=dict(color=CH_INK, size=11),
            customdata=d[["mean_grade", "mean_engagement", "n_students"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Mean attendance: %{x:.1f}%<br>"
                "Mean grade: %{customdata[0]:.1f}<br>"
                "Engagement score: %{customdata[1]:.2f}<br>"
                "Students enrolled: %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 110], gridcolor=CH_GREY, title="Mean attendance (%)"),
        yaxis=dict(title=""),
        showlegend=False,
    )
    return style_fig(fig, height=max(280, 42 * len(d) + 80))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render() -> None:
    hero(
        "Overview",
        "Institution-wide snapshot of engagement and predicted risk for the current cohort. "
        "Filter by programme or year of study to focus on a specific population.",
    )

    df_all = students_with_predictions()
    actions = load_actions()
    weekly = load_weekly()
    modules = load_modules()
    module_engagement = load_module_engagement()

    # ---- Filters: programme + year ----
    programmes = sorted(df_all["program"].dropna().unique().tolist())
    years = sorted(df_all["year_of_study"].dropna().unique().tolist())

    fc1, fc2, fc3 = st.columns([1.3, 1.0, 0.6], gap="small")
    with fc1:
        prog_sel = st.multiselect(
            "Programme",
            programmes,
            placeholder="All programmes",
        )
    with fc2:
        year_sel = st.multiselect(
            "Year of study",
            years,
            placeholder="All years",
            format_func=lambda y: f"Year {int(y)}",
        )
    with fc3:
        if st.button("Reset", width="stretch"):
            st.rerun()

    df = df_all.copy()
    if prog_sel:
        df = df[df["program"].isin(prog_sel)]
    if year_sel:
        df = df[df["year_of_study"].isin(year_sel)]
    if df.empty:
        st.warning("No students match the current filters.")
        return

    weekly_subset = weekly[weekly["student_id"].isin(df["student_id"])]
    actions_subset = actions[actions["student_id"].isin(df["student_id"])] if not actions.empty else actions
    enrolled_modules = (
        module_engagement[module_engagement["student_id"].isin(df["student_id"])]["module_code"].unique().tolist()
        if not module_engagement.empty else []
    )

    # ---- KPI row 1: institutional headlines ----
    n_total = len(df)
    mean_attendance = float(df["attendance_rate"].mean())
    n_high_risk = int((df["risk_band"] == "High").sum())

    kpi_row(
        [
            kpi_card(
                "Students in view",
                f"{n_total:,}",
                caption="after filters",
            ),
            kpi_card(
                "Mean attendance",
                f"{mean_attendance:.1f}%",
                caption="4-week rolling average",
            ),
            kpi_card(
                "Students flagged as high risk",
                f"{n_high_risk}",
                caption=f"{n_high_risk / n_total * 100:.1f}% of this view",
                colour="red",
            ),
        ]
    )

    # ---- KPI row 2: predictive headlines ----
    mean_withdraw = float(df["withdraw_prob"].mean())
    n_critical = int((df["withdraw_prob"] > 0.50).sum())
    n_open_cases = (
        int(actions_subset["student_id"].nunique()) if not actions_subset.empty else 0
    )

    kpi_row(
        [
            kpi_card(
                "Predicted withdrawal rate",
                f"{mean_withdraw * 100:.1f}%",
                caption="cohort-average probability",
            ),
            kpi_card(
                "Students at critical risk",
                f"{n_critical}",
                caption="withdrawal probability above 50%",
                colour="red",
            ),
            kpi_card(
                "Students with open case",
                f"{n_open_cases}",
                caption="at least one active intervention",
            ),
        ]
    )

    # ---- Risk donut + cohort attendance trend ----
    section_title("Cohort health")
    col_a, col_b = st.columns([1, 1.25], gap="large")
    with col_a:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.6rem;'>RISK BAND DISTRIBUTION</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_risk_donut(df), width="stretch")
    with col_b:
        st.markdown(
            f"<div class='kpi-label' style='margin-bottom:0.6rem;'>COHORT ATTENDANCE - LAST 12 WEEKS</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_cohort_attendance_trend(weekly_subset), width="stretch")

    # ---- Most disengaged modules ----
    section_title("Most disengaged modules")
    st.caption(
        "Modules ranked by mean attendance across all enrolled students in the current view. "
        "These are candidate modules where teaching design, scheduling, or assessment load may "
        "be driving disengagement at the class level."
    )

    if modules.empty:
        st.info("Module-level data not available.")
    else:
        mods = modules.copy()
        if enrolled_modules:
            mods = mods[mods["module_code"].isin(enrolled_modules)]
        if prog_sel:
            mods = mods[mods["programme"].isin(prog_sel)]
        top_disengaged = mods.sort_values("mean_attendance", ascending=True).head(8)
        st.plotly_chart(_disengaged_modules_chart(top_disengaged), width="stretch")

    # ---- Top 10 students needing attention ----
    section_title("Top 10 students needing attention")

    top = df.sort_values("overall_risk", ascending=False).head(10).copy()
    top["P(withdraw)"] = (top["withdraw_prob"] * 100).round(1)
    top["P(fail next)"] = (top["fail_prob"] * 100).round(1)
    top["Risk band"] = top["risk_band"]

    view = top[
        [
            "student_id",
            "student_name",
            "program",
            "year_of_study",
            "attendance_rate",
            "P(withdraw)",
            "P(fail next)",
            "Risk band",
        ]
    ].rename(
        columns={
            "student_id": "ID",
            "student_name": "Student",
            "program": "Programme",
            "year_of_study": "Year",
            "attendance_rate": "Attendance %",
        }
    )

    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "P(withdraw)": st.column_config.ProgressColumn(
                "P(withdraw)", format="%.1f%%", min_value=0, max_value=100
            ),
            "P(fail next)": st.column_config.ProgressColumn(
                "P(fail next)", format="%.1f%%", min_value=0, max_value=100
            ),
            "Attendance %": st.column_config.NumberColumn("Attendance %", format="%.1f"),
        },
    )

    st.caption(
        "Open the **Students** or **At-Risk Students** pages to filter the full cohort, "
        "or click into **Student Profile** to drill into any individual."
    )
