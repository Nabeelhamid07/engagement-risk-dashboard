"""At-Risk Students - early warning page. Predictions + primary driver per student."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.data import (
    load_actions,
    load_shap_drivers,
    students_with_predictions,
)
from lib.theme import (
    CH_LABEL,
    hero,
    kpi_card,
    kpi_row,
)


def _top_driver_table(target: str) -> pd.DataFrame:
    """Top-1 driver per student for a given target."""
    shap = load_shap_drivers()
    sub = shap[(shap["target"] == target) & (shap["rank"] == 1)][
        ["student_id", "feature", "direction"]
    ].copy()
    sub["primary_driver"] = sub.apply(
        lambda r: f"↑ {r['feature']}" if r["direction"] == "+" else f"↓ {r['feature']}",
        axis=1,
    )
    return sub[["student_id", "primary_driver"]]


def render() -> None:
    hero(
        "At-Risk Students",
        "Students the predictive model has flagged as elevated risk, sorted with the highest concern at the top. "
        "Each row shows the model's predicted probabilities and the single biggest factor pushing the prediction.",
    )

    df = students_with_predictions().copy()
    actions = load_actions()

    # ---- KPI strip ----
    n_high = int((df["risk_band"] == "High").sum())
    n_medium = int((df["risk_band"] == "Medium").sum())
    n_critical = int((df["withdraw_prob"] > 0.50).sum())

    kpi_row(
        [
            kpi_card("High-risk students", f"{n_high}", caption=f"{n_high / len(df) * 100:.1f}% of cohort", colour="red"),
            kpi_card("Medium-risk students", f"{n_medium}", caption=f"{n_medium / len(df) * 100:.1f}% of cohort", colour="orange"),
            kpi_card("Critical risk", f"{n_critical}", caption="withdrawal probability above 50%", colour="red"),
        ]
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    # ---- Filters ----
    f1, f2, f3 = st.columns([1.2, 1.4, 1])
    with f1:
        bands = st.multiselect(
            "Risk band",
            ["High", "Medium", "Low"],
            default=["High", "Medium"],
            placeholder="All risk bands",
        )
    with f2:
        progs = sorted(df["program"].dropna().unique().tolist())
        program_sel = st.multiselect(
            "Programme", progs, default=[], placeholder="All programmes"
        )
    with f3:
        min_withdraw = st.slider("Minimum P(withdraw) %", 0, 100, 0, step=5)

    view = df.copy()
    if bands:
        view = view[view["risk_band"].isin(bands)]
    if program_sel:
        view = view[view["program"].isin(program_sel)]
    view = view[view["withdraw_prob"] >= min_withdraw / 100.0]

    # Pull primary withdraw and fail drivers
    drv_w = _top_driver_table("withdrew").rename(columns={"primary_driver": "Primary driver (withdraw)"})
    drv_f = _top_driver_table("failed_next_assessment").rename(columns={"primary_driver": "Primary driver (fail)"})
    view = view.merge(drv_w, on="student_id", how="left").merge(drv_f, on="student_id", how="left")

    if not actions.empty:
        action_counts = actions.groupby("student_id").size().rename("Concerns on case").reset_index()
        view = view.merge(action_counts, on="student_id", how="left")
        view["Concerns on case"] = view["Concerns on case"].fillna(0).astype(int)
    else:
        view["Concerns on case"] = 0

    view = view.sort_values("overall_risk", ascending=False)

    st.markdown(
        f"<div style='color:{CH_LABEL}; font-size:0.85rem; margin: 0.5rem 0 1rem 0;'>"
        f"Showing <b>{len(view)}</b> students out of <b>{len(df)}</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Table ----
    show = view.copy()
    show["P(withdraw)"] = (show["withdraw_prob"] * 100).round(1)
    show["P(fail next)"] = (show["fail_prob"] * 100).round(1)
    show["Attendance %"] = show["attendance_rate"].round(1)
    show["Days to next assessment"] = show["days_to_nearest_assessment"]

    display = show[
        [
            "student_id",
            "student_name",
            "program",
            "P(withdraw)",
            "P(fail next)",
            "Attendance %",
            "Days to next assessment",
            "Primary driver (withdraw)",
            "Primary driver (fail)",
            "risk_band",
            "Concerns on case",
        ]
    ].rename(
        columns={
            "student_id": "ID",
            "student_name": "Student",
            "program": "Programme",
            "risk_band": "Risk band",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=620,
        column_config={
            "P(withdraw)": st.column_config.ProgressColumn(
                "P(withdraw)", format="%.1f%%", min_value=0, max_value=100
            ),
            "P(fail next)": st.column_config.ProgressColumn(
                "P(fail next)", format="%.1f%%", min_value=0, max_value=100
            ),
            "Attendance %": st.column_config.NumberColumn("Attendance %", format="%.1f"),
            "Days to next assessment": st.column_config.NumberColumn(
                "Days to next assessment", format="%d"
            ),
            "Concerns on case": st.column_config.NumberColumn("Concerns on case", format="%d"),
        },
    )

    st.caption(
        "**↑** means the feature is pushing predicted risk up; **↓** means it is pushing risk down. "
        "Drivers come from a SHAP analysis of the underlying Random Forest models."
    )

    st.download_button(
        "Export filtered list (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="at_risk_students.csv",
        mime="text/csv",
    )
