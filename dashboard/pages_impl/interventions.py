"""Interventions - case management view. One card per STUDENT.

Each card consolidates everything the engine has fired for that student: a
single primary concern as the headline, plus any supporting concerns nested
inside an expander. Every card carries a clear AUTOMATED badge and a readable
'Why this case was created' checklist showing the exact conditions that fired
the rule, alongside an on-demand activity log of every step the system has
already taken on the case.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.activity import case_activity_steps
from lib.data import (
    consolidated_cases,
    load_actions,
    load_comms_summary,
    load_communications,
    load_rules_catalogue,
    load_students,
    students_with_predictions,
)
from lib.reasons import case_reason_rows
from lib.theme import (
    CH_GREY,
    CH_INK,
    CH_LABEL,
    CH_MUTED,
    CH_RED,
    SEVERITY_ORDER,
    activity_log_html,
    case_card,
    comms_strip,
    hero,
    kpi_card,
    kpi_row,
    section_title,
    severity_badge,
    style_fig,
    supporting_concerns_html,
    why_block_html,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _team_bar(actions: pd.DataFrame) -> go.Figure:
    counts = actions["route_to"].value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.values.tolist(),
            y=counts.index.tolist(),
            orientation="h",
            marker_color=CH_RED,
            text=counts.values.tolist(),
            textposition="outside",
            textfont=dict(color=CH_INK, size=12),
            hovertemplate="%{y}: %{x} cases<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="", gridcolor=CH_GREY, range=[0, counts.max() * 1.18]),
        yaxis=dict(title=""),
        showlegend=False,
        bargap=0.45,
    )
    return style_fig(fig, height=340)


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------
def _make_jump_callback(student_id: str):
    def _cb() -> None:
        st.session_state["target_student_id"] = student_id
        st.session_state["nav_radio"] = "Student Profile"

    return _cb


# ---------------------------------------------------------------------------
# Outreach status classification
# ---------------------------------------------------------------------------
def _outreach_status(comms_row: pd.Series | None) -> str:
    if comms_row is None or pd.isna(comms_row.get("comms_30d_count", 0)):
        return "Silent (never contacted)"
    n = int(comms_row.get("comms_30d_count", 0) or 0)
    r = int(comms_row.get("comms_responded_30d", 0) or 0)
    if n == 0:
        return "Silent (never contacted)"
    if r == 0 and n >= 2:
        return "Non-responsive (2+ contacts, no reply)"
    if r > 0:
        return "Recently responded"
    return "Awaiting reply"


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    hero(
        "Interventions",
        "Consolidated case file per student. The engine fires multiple policy rules in parallel; "
        "this view groups them into one case per student with a primary concern surfaced and the "
        "rest treated as supporting evidence.",
    )

    actions = load_actions()
    students_full = load_students()
    students_lookup = students_full.set_index("student_id")["student_name"].to_dict()
    preds = students_with_predictions()
    cases = consolidated_cases()
    comms = load_comms_summary()
    comms_log = load_communications()
    rules = {r["id"]: r for r in load_rules_catalogue()}

    if cases.empty:
        st.success("No active cases. The cohort is healthy.")
        return

    pred_cols = preds[["student_id", "student_name", "program", "year_of_study", "risk_band"]]
    cases = cases.merge(pred_cols, on="student_id", how="left")

    if not comms.empty:
        comms_cols = comms[
            [
                "student_id",
                "comms_30d_count",
                "comms_responded_30d",
                "last_contact_date",
                "last_contact_channel",
                "last_contact_status",
            ]
        ].copy()
        comms_cols["last_contact_date"] = comms_cols["last_contact_date"].dt.strftime("%Y-%m-%d")
        cases = cases.merge(comms_cols, on="student_id", how="left")
    cases["comms_30d_count"] = cases.get("comms_30d_count", pd.Series(dtype=int)).fillna(0).astype(int)
    cases["comms_responded_30d"] = cases.get("comms_responded_30d", pd.Series(dtype=int)).fillna(0).astype(int)
    cases["last_contact_date"] = cases.get("last_contact_date", pd.Series(dtype=str)).fillna("")
    cases["last_contact_channel"] = cases.get("last_contact_channel", pd.Series(dtype=str)).fillna("")
    cases["last_contact_status"] = cases.get("last_contact_status", pd.Series(dtype=str)).fillna("")
    cases["outreach_status"] = cases.apply(_outreach_status, axis=1)

    # ---- Three top KPIs ----
    n_cases_attention = int(
        cases[cases["primary_severity"].isin(["escalation", "high"])]["student_id"].nunique()
    )
    n_open_cases = int(cases["student_id"].nunique())
    n_critical = int((preds["withdraw_prob"] > 0.50).sum())

    kpi_row(
        [
            kpi_card(
                "Cases needing attention",
                f"{n_cases_attention}",
                caption="this week (escalation + high severity)",
                colour="red",
            ),
            kpi_card(
                "Students with open case",
                f"{n_open_cases}",
                caption="across the cohort",
            ),
            kpi_card(
                "Students at critical risk",
                f"{n_critical}",
                caption="withdrawal probability above 50%",
                colour="red",
            ),
        ]
    )

    # ---- Outreach effectiveness banner ----
    if not comms.empty:
        total_comms = int(comms["comms_30d_count"].sum())
        total_resp = int(comms["comms_responded_30d"].sum())
        resp_rate = (total_resp / total_comms * 100) if total_comms else 0.0
        n_non_resp = int(comms["is_non_responsive"].sum()) if "is_non_responsive" in comms.columns else 0
        kpi_row(
            [
                kpi_card(
                    "Outreach attempts (30d)",
                    f"{total_comms:,}",
                    caption=f"across {len(comms)} students",
                ),
                kpi_card(
                    "Cohort response rate",
                    f"{resp_rate:.1f}%",
                    caption=f"{total_resp:,} replies out of {total_comms:,} contacts",
                ),
                kpi_card(
                    "Non-responsive students",
                    f"{n_non_resp}",
                    caption="2 or more contacts, no replies",
                    colour="red" if n_non_resp >= 30 else "orange",
                ),
            ]
        )

    # ---- Bar chart by team ----
    section_title("Cases by support team")
    st.plotly_chart(_team_bar(actions), width="stretch")

    # ---- Filters: 5 filters + 1 sort, all on one row ----
    section_title("Open cases")

    programmes = sorted(cases["program"].dropna().unique().tolist())
    concern_options = sorted(cases["primary_rule_name"].dropna().unique().tolist())
    outreach_options = [
        "Non-responsive (2+ contacts, no reply)",
        "Awaiting reply",
        "Recently responded",
        "Silent (never contacted)",
    ]
    band_options = ["High", "Medium", "Low"]

    f1, f2, f3, f4, f5, f6 = st.columns(
        [1.2, 1.4, 1.4, 1.0, 1.4, 1.3], gap="small"
    )
    with f1:
        prog_sel = st.multiselect(
            "Programme", programmes, placeholder="All programmes", key="iv_prog"
        )
    with f2:
        concern_sel = st.multiselect(
            "Concern type", concern_options, placeholder="All concerns", key="iv_concern"
        )
    with f3:
        out_sel = st.multiselect(
            "Outreach status", outreach_options, placeholder="Any outreach status", key="iv_out"
        )
    with f4:
        band_sel = st.multiselect(
            "Risk band", band_options, placeholder="Any risk band", key="iv_band"
        )
    with f5:
        search = st.text_input(
            "Search by name or ID", placeholder="e.g. Davis or STU0168", key="iv_search"
        )
    with f6:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Severity (highest first)",
                "Predicted withdrawal probability",
                "Number of concerns (most first)",
            ],
            key="iv_sort",
        )

    view = cases.copy()
    if prog_sel:
        view = view[view["program"].isin(prog_sel)]
    if concern_sel:
        view = view[view["primary_rule_name"].isin(concern_sel)]
    if out_sel:
        view = view[view["outreach_status"].isin(out_sel)]
    if band_sel:
        view = view[view["risk_band"].isin(band_sel)]
    if search:
        s = search.strip().lower()
        view = view[
            view["student_name"].str.lower().str.contains(s, na=False)
            | view["student_id"].str.lower().str.contains(s, na=False)
        ]

    if sort_by == "Severity (highest first)":
        view = view.sort_values(
            ["highest_sev_rank", "primary_withdraw_prob", "primary_fail_prob"],
            ascending=[False, False, False],
        )
    elif sort_by == "Predicted withdrawal probability":
        view = view.sort_values("primary_withdraw_prob", ascending=False)
    else:
        view = view.sort_values(["concerns_total", "highest_sev_rank"], ascending=[False, False])

    page_size = 20
    if "interv_show_n" not in st.session_state:
        st.session_state["interv_show_n"] = page_size
    show_n = st.session_state["interv_show_n"]
    visible = view.head(show_n)

    st.markdown(
        f"<div style='color:{CH_LABEL}; font-size:0.85rem; margin: 0.5rem 0 1.2rem 0;'>"
        f"Showing <b>{len(visible)}</b> of <b>{len(view)}</b> student case files."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Render case cards ----
    student_records = preds.set_index("student_id").to_dict(orient="index")

    for _, row in visible.iterrows():
        sid = row["student_id"]
        name = row.get("student_name") or students_lookup.get(sid, sid)
        sev = str(row["primary_severity"]).lower()

        title_html = (
            f"{name}"
            f"<span class='case-id-soft'>({sid})</span>"
            f"{severity_badge(sev)}"
        )

        col_card, col_side = st.columns([5.5, 1.55], gap="small")

        with col_card:
            # The compact case card (severity strip, name, action, policy basis).
            st.markdown(
                case_card(
                    title_html=title_html,
                    reason="",
                    action=row["primary_action"],
                    severity=sev,
                    assigned_to="",
                    policy_basis=row["primary_policy_basis"],
                ),
                unsafe_allow_html=True,
            )

            # 'Why this case was created' - readable conditions checklist.
            student_data = student_records.get(sid, {})
            why_rows = case_reason_rows(row["primary_rule_id"], student_data)
            st.markdown(why_block_html(why_rows), unsafe_allow_html=True)

            # Communications strip footer.
            st.markdown(
                comms_strip(
                    last_date=str(row.get("last_contact_date", "") or ""),
                    last_channel=str(row.get("last_contact_channel", "") or ""),
                    last_status=str(row.get("last_contact_status", "") or ""),
                    count_30d=int(row.get("comms_30d_count", 0) or 0),
                    responded_30d=int(row.get("comms_responded_30d", 0) or 0),
                ),
                unsafe_allow_html=True,
            )

            # Optional 'other concerns' expander.
            n_extra = int(row["concerns_total"]) - 1
            if n_extra > 0:
                with st.expander(
                    f"Show {n_extra} other concern" + ("s" if n_extra != 1 else "")
                    + " on this case"
                ):
                    st.markdown(
                        supporting_concerns_html(row["supporting"] or []),
                        unsafe_allow_html=True,
                    )

            # Activity log (steps the system has already executed) - hidden by default.
            with st.expander("Show case activity"):
                primary_action_record = {
                    "rule_id": row["primary_rule_id"],
                    "rule_name": row["primary_rule_name"],
                    "triggered_on": actions.loc[
                        actions["action_id"] == row["primary_action_id"], "triggered_on"
                    ].iloc[0]
                    if not actions[actions["action_id"] == row["primary_action_id"]].empty
                    else pd.Timestamp.now(),
                    "route_to": row["primary_team"],
                    "severity": row["primary_severity"],
                    "action": row["primary_action"],
                    "student_id": sid,
                }
                this_student_comms = (
                    comms_log[comms_log["student_id"] == sid] if not comms_log.empty else pd.DataFrame()
                )
                steps = case_activity_steps(primary_action_record, this_student_comms)
                st.markdown(activity_log_html(steps), unsafe_allow_html=True)

        with col_side:
            st.markdown(
                f"""
                <div style='padding-top: 0.95rem;'>
                    <div style='color:{CH_LABEL}; font-size:0.65rem; letter-spacing:0.1em;
                                text-transform:uppercase; font-weight:600;'>PRIMARY OWNER</div>
                    <div style='color:{CH_INK}; font-weight:600; font-size:0.88rem;
                                margin-top:0.2rem; line-height:1.25;'>
                        {row['primary_team']}
                    </div>
                    <div style='color:{CH_MUTED}; font-size:0.72rem; margin-top:0.5rem;'>
                        {row['concerns_total']} concern(s) &middot;
                        {len(row.get('teams_involved') or [])} team(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button(
                "View profile →",
                key=f"vp_case_{sid}",
                width="stretch",
                on_click=_make_jump_callback(sid),
                type="secondary",
            )

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    if len(visible) < len(view):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button(
                f"Show {min(page_size, len(view) - len(visible))} more cases",
                width="stretch",
            ):
                st.session_state["interv_show_n"] = show_n + page_size
                st.rerun()

    # ---- Rules catalogue tucked into an expander ----
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    with st.expander(f"View all {len(rules)} policy rules"):
        fire_counts = actions["rule_id"].value_counts().to_dict()
        sev_rank_map = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        rules_sorted = sorted(
            rules.values(), key=lambda r: sev_rank_map.get(r["severity"], 99)
        )

        st.markdown(
            f"<div style='color:{CH_MUTED}; font-size:0.85rem; margin-bottom: 0.6rem;'>"
            "The complete set of triggers the engine evaluates. Each rule maps a condition to "
            "an action, a support team, and an institutional policy."
            "</div>",
            unsafe_allow_html=True,
        )

        rules_df = pd.DataFrame(rules_sorted)
        rules_df["fired"] = rules_df["id"].map(fire_counts).fillna(0).astype(int)
        rules_df = rules_df[
            ["severity", "id", "name", "trigger_summary", "action", "route_to", "policy_basis", "fired"]
        ].rename(
            columns={
                "severity": "Severity",
                "id": "ID",
                "name": "Rule",
                "trigger_summary": "Trigger",
                "action": "Action",
                "route_to": "Routed to",
                "policy_basis": "Policy basis",
                "fired": "Fired this cycle",
            }
        )
        st.dataframe(rules_df, hide_index=True, width="stretch", height=560)
