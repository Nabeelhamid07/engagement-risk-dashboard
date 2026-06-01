"""University of Chester theme: palette, CSS, KPI card helper, Plotly defaults.

Visual brief (v2):
  - calm, spacious, modern layout. Linear/Notion/Stripe-style breathing room.
  - 3 KPI cards per row, not 4 or 5
  - large numbers, small uppercase labels
  - ~1180px page cap so wide monitors don't sprawl
  - ~40-50px between major sections
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
CH_RED = "#E2231A"
CH_RED_SOFT = "#FBE3E1"
CH_CHARCOAL = "#2E2E27"
CH_INK = "#1B1B17"
CH_GREY = "#EAEAEA"
CH_GREY_DARK = "#9A9A9A"
CH_MUTED = "#8A8A85"
CH_LABEL = "#888888"
CH_YELLOW = "#F4C023"
CH_GREEN = "#1B7F3B"
CH_ORANGE = "#C77800"
CH_PAGE = "#FFFFFF"
CH_PANEL = "#F7F7F6"
CH_PANEL_DARK = "#EDECE9"

RISK_BAND_COLOURS = {"Low": CH_GREEN, "Medium": CH_ORANGE, "High": CH_RED}

SEVERITY_COLOURS = {
    "low": "#1F8FB2",
    "medium": "#E0A030",
    "high": "#D14A2C",
    "escalation": CH_RED,
}
SEVERITY_ORDER = ["escalation", "high", "medium", "low"]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
            /* Page container - capped width, generous vertical padding */
            .block-container {{
                padding-top: 2rem;
                padding-bottom: 4rem;
                max-width: 1180px;
            }}
            header[data-testid="stHeader"] {{
                background: linear-gradient(90deg, {CH_RED} 0%, {CH_RED} 4px, {CH_PAGE} 4px, {CH_PAGE} 100%);
            }}

            /* Sidebar - clean panel feel */
            div[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {CH_PANEL} 0%, {CH_PAGE} 40%);
                border-right: 1px solid {CH_GREY};
            }}
            div[data-testid="stSidebar"] > div:first-child {{
                padding-top: 0.5rem;
            }}
            div[data-testid="stSidebar"] .stRadio > div {{
                gap: 0.15rem;
            }}
            div[data-testid="stSidebar"] label[data-baseweb="radio"] {{
                padding: 0.5rem 0.75rem;
                border-radius: 7px;
                color: {CH_CHARCOAL};
            }}
            div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
                background: {CH_PANEL_DARK};
            }}

            /* Hero: page title block with red left border */
            .ch-hero {{
                font-family: system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: {CH_CHARCOAL};
                border-left: 4px solid {CH_RED};
                padding: 0.5rem 0 0.5rem 1rem;
                margin-bottom: 2.5rem;
            }}
            .ch-hero h1 {{
                font-size: 1.9rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0 0 0.4rem 0;
                color: {CH_INK};
            }}
            .ch-hero p {{
                margin: 0;
                color: {CH_MUTED};
                font-size: 0.95rem;
                line-height: 1.5;
                max-width: 760px;
            }}

            /* KPI card - large numbers, small uppercase labels, generous padding */
            .kpi-card {{
                background: {CH_PAGE};
                border: 1px solid {CH_GREY};
                border-radius: 12px;
                padding: 1.5rem 1.6rem 1.4rem 1.6rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.025);
                min-height: 140px;
            }}
            .kpi-label {{
                font-size: 0.7rem;
                color: {CH_LABEL};
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin: 0;
            }}
            .kpi-value {{
                font-size: 2.5rem;
                font-weight: 700;
                color: {CH_INK};
                margin: 0.55rem 0 0.35rem 0;
                line-height: 1.05;
            }}
            .kpi-value-red {{ color: {CH_RED}; }}
            .kpi-value-orange {{ color: {CH_ORANGE}; }}
            .kpi-value-green {{ color: {CH_GREEN}; }}
            .kpi-caption {{
                font-size: 0.78rem;
                color: {CH_MUTED};
                margin: 0;
            }}

            /* Section spacing helpers */
            .section-gap {{ margin-top: 2.5rem; margin-bottom: 1rem; }}
            .section-title {{
                font-size: 1.05rem;
                font-weight: 650;
                color: {CH_INK};
                margin: 0 0 1.1rem 0;
            }}

            /* Case / list cards (used on Interventions + Student Profile) */
            .case-card {{
                background: {CH_PAGE};
                border: 1px solid {CH_GREY};
                border-left: 5px solid {CH_RED};
                border-radius: 8px;
                padding: 1.15rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.025);
            }}
            .case-card-low {{ border-left-color: {SEVERITY_COLOURS["low"]}; }}
            .case-card-medium {{ border-left-color: {SEVERITY_COLOURS["medium"]}; }}
            .case-card-high {{ border-left-color: {SEVERITY_COLOURS["high"]}; }}
            .case-card-escalation {{ border-left-color: {SEVERITY_COLOURS["escalation"]}; }}

            .case-row {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
            }}
            .case-main {{
                flex: 1;
                min-width: 0;
            }}
            .case-aside {{
                text-align: right;
                font-size: 0.78rem;
                color: {CH_MUTED};
                flex-shrink: 0;
                padding-top: 0.2rem;
            }}
            .case-title {{
                font-size: 1rem;
                font-weight: 650;
                color: {CH_INK};
                margin: 0 0 0.35rem 0;
            }}
            .case-id-soft {{
                color: {CH_MUTED};
                font-weight: 500;
                font-size: 0.88rem;
                margin-left: 0.35rem;
            }}
            .case-reason {{
                font-size: 0.88rem;
                color: {CH_CHARCOAL};
                line-height: 1.45;
                margin: 0 0 0.35rem 0;
            }}
            .case-action {{
                font-size: 0.9rem;
                font-weight: 600;
                color: {CH_INK};
                margin: 0;
            }}
            .case-policy {{
                font-size: 0.72rem;
                color: {CH_MUTED};
                font-style: italic;
                margin: 0.5rem 0 0 0;
            }}

            /* Severity pills */
            .badge {{
                display: inline-block;
                padding: 2px 9px;
                border-radius: 999px;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                color: white;
                text-transform: uppercase;
                vertical-align: middle;
                margin-left: 0.4rem;
            }}
            .badge-low {{ background: {SEVERITY_COLOURS["low"]}; }}
            .badge-medium {{ background: {SEVERITY_COLOURS["medium"]}; }}
            .badge-high {{ background: {SEVERITY_COLOURS["high"]}; }}
            .badge-escalation {{ background: {SEVERITY_COLOURS["escalation"]}; }}

            .band-Low {{ color: {CH_GREEN}; font-weight: 700; }}
            .band-Medium {{ color: {CH_ORANGE}; font-weight: 700; }}
            .band-High {{ color: {CH_RED}; font-weight: 700; }}

            /* Profile header card */
            .profile-card {{
                background: {CH_PAGE};
                border: 1px solid {CH_GREY};
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 2rem;
            }}
            .profile-name {{
                font-size: 1.4rem;
                font-weight: 700;
                color: {CH_INK};
                margin: 0;
            }}
            .profile-meta {{
                color: {CH_MUTED};
                font-size: 0.9rem;
                margin-top: 0.3rem;
            }}

            /* Metric chrome */
            span[data-testid="stMetricValue"] {{
                color: {CH_INK} !important;
                font-weight: 700;
            }}
            span[data-testid="stMetricLabel"] {{
                color: {CH_LABEL} !important;
                font-weight: 600;
                letter-spacing: 0.04em;
            }}

            /* Table polish */
            div[data-testid="stDataFrame"] {{
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid {CH_GREY};
            }}

            /* Automated banner: a dark pill that makes the automation obvious */
            .auto-banner {{
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                background: {CH_INK};
                color: white;
                padding: 4px 11px;
                border-radius: 999px;
                font-size: 0.65rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}
            .auto-banner-dot {{
                width: 6px; height: 6px; border-radius: 50%;
                background: #2ecc71;
                box-shadow: 0 0 0 3px rgba(46,204,113,0.20);
            }}

            /* Trigger summary box - shows the exact condition that fired */
            .trigger-box {{
                background: #FAFAF8;
                border: 1px solid {CH_GREY};
                border-radius: 6px;
                padding: 0.6rem 0.85rem;
                margin-top: 0.65rem;
                font-family: ui-monospace, "SF Mono", Consolas, monospace;
                font-size: 0.78rem;
                color: {CH_CHARCOAL};
                line-height: 1.45;
            }}
            .trigger-label {{
                font-family: system-ui, sans-serif;
                font-size: 0.65rem;
                color: {CH_LABEL};
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 0.25rem;
            }}

            /* Communications strip: small footer on a case card */
            .comms-strip {{
                margin-top: 0.7rem;
                padding-top: 0.6rem;
                border-top: 1px dashed {CH_GREY};
                font-size: 0.78rem;
                color: {CH_MUTED};
            }}
            .comms-strip strong {{
                color: {CH_CHARCOAL};
                font-weight: 600;
            }}
            .comms-strip-warn {{ color: {CH_RED}; font-weight: 600; }}

            /* 'Why this case was created' checklist */
            .why-block {{
                background: #FAFAF8;
                border: 1px solid {CH_GREY};
                border-radius: 6px;
                padding: 0.7rem 0.9rem;
                margin-top: 0.65rem;
            }}
            .why-label {{
                font-size: 0.65rem;
                color: {CH_LABEL};
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin: 0 0 0.4rem 0;
            }}
            .why-row {{
                display: grid;
                grid-template-columns: 22px 1fr auto auto;
                column-gap: 0.6rem;
                align-items: baseline;
                padding: 0.3rem 0;
                border-top: 1px dashed {CH_GREY};
                font-size: 0.85rem;
            }}
            .why-row:first-of-type {{ border-top: 0; padding-top: 0.1rem; }}
            .why-check {{
                color: {CH_GREEN};
                font-weight: 700;
                font-size: 0.95rem;
                line-height: 1;
            }}
            .why-text {{
                color: {CH_CHARCOAL};
            }}
            .why-current {{
                color: {CH_INK};
                font-weight: 700;
                font-variant-numeric: tabular-nums;
                margin-left: 0.6rem;
            }}
            .why-thr {{
                color: {CH_MUTED};
                font-size: 0.78rem;
                font-variant-numeric: tabular-nums;
                margin-left: 0.8rem;
                white-space: nowrap;
            }}

            /* 'Case activity' step list (inside the expander).
               Two-column grid: a marker column (dot + connecting line) and the
               step content. The dot is vertically aligned with the first line
               of the label; the line runs between dots through the gap. */
            .activity-list {{
                list-style: none;
                margin: 0.25rem 0 0 0;
                padding: 0;
            }}
            .activity-item {{
                display: grid;
                grid-template-columns: 18px 1fr;
                column-gap: 0.85rem;
                padding: 0.25rem 0;
            }}
            .activity-marker {{
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100%;
                min-height: 38px;
            }}
            .activity-dot {{
                width: 13px;
                height: 13px;
                border-radius: 50%;
                background: {CH_PAGE};
                border: 2px solid {CH_GREY_DARK};
                box-sizing: border-box;
                margin-top: 5px;
                flex-shrink: 0;
            }}
            .activity-dot-done {{
                background: {CH_GREEN};
                border-color: {CH_GREEN};
                box-shadow: 0 0 0 3px rgba(27,127,59,0.14);
            }}
            .activity-dot-pending {{
                background: {CH_PAGE};
                border-color: {CH_ORANGE};
                box-shadow: 0 0 0 3px rgba(199,120,0,0.12);
            }}
            .activity-dot-scheduled {{
                background: {CH_PAGE};
                border-color: {CH_GREY_DARK};
            }}
            .activity-line {{
                flex: 1 1 auto;
                width: 2px;
                background-image: linear-gradient({CH_GREY}, {CH_GREY} 50%, transparent 50%);
                background-size: 100% 6px;
                background-repeat: repeat-y;
                margin-top: 4px;
                min-height: 12px;
            }}
            .activity-item:last-child .activity-line {{ display: none; }}
            .activity-body {{
                padding-bottom: 0.5rem;
            }}
            .activity-label {{
                font-size: 0.88rem;
                color: {CH_INK};
                font-weight: 600;
                line-height: 1.35;
            }}
            .activity-detail {{
                font-size: 0.79rem;
                color: {CH_CHARCOAL};
                margin-top: 0.15rem;
                line-height: 1.4;
            }}
            .activity-ts {{
                font-size: 0.72rem;
                color: {CH_MUTED};
                margin-top: 0.15rem;
                letter-spacing: 0.02em;
            }}
            .activity-status-pending {{
                color: {CH_ORANGE};
                font-weight: 600;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-left: 0.5rem;
            }}
            .activity-status-scheduled {{
                color: {CH_MUTED};
                font-weight: 600;
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-left: 0.5rem;
            }}

            /* Supporting-concerns list inside a case card */
            .concerns-list {{
                margin: 0.7rem 0 0.1rem 0;
                padding: 0;
                list-style: none;
            }}
            .concerns-item {{
                padding: 0.45rem 0;
                border-top: 1px dashed {CH_GREY};
                font-size: 0.84rem;
                color: {CH_CHARCOAL};
            }}
            .concerns-item-title {{ font-weight: 600; color: {CH_INK}; }}
            .concerns-item-meta {{
                color: {CH_MUTED};
                font-size: 0.72rem;
                margin-top: 0.15rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="ch-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(
        f"<div class='section-gap'></div><div class='section-title'>{text}</div>",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, caption: str = "", colour: str | None = None) -> str:
    """Return HTML for one KPI card. Use inside st.markdown(unsafe_allow_html=True).

    colour: None | 'red' | 'orange' | 'green' tints the value text.
    """
    cls = ""
    if colour == "red":
        cls = " kpi-value-red"
    elif colour == "orange":
        cls = " kpi-value-orange"
    elif colour == "green":
        cls = " kpi-value-green"
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value{cls}">{value}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """


def kpi_row(cards: list[str]) -> None:
    """Render N KPI cards in equal-width columns."""
    cols = st.columns(len(cards), gap="medium")
    for col, card_html in zip(cols, cards):
        with col:
            st.markdown(card_html, unsafe_allow_html=True)


def severity_badge(severity: str) -> str:
    sev = severity.lower()
    return f'<span class="badge badge-{sev}">{sev}</span>'


def risk_band_html(band: str) -> str:
    return f'<span class="band-{band}">{band}</span>'


def automated_banner() -> str:
    """Small dark pill that flags an item as automation-generated."""
    return (
        '<span class="auto-banner">'
        '<span class="auto-banner-dot"></span>'
        "AUTOMATED"
        "</span>"
    )


def trigger_box(condition: str, policy_basis: str = "") -> str:
    """Render the literal trigger condition that fired, plus optional policy ref."""
    policy_line = (
        f'<div style="font-family:system-ui,sans-serif; font-size:0.72rem; '
        f'color:#8A8A85; margin-top:0.35rem;">Policy basis: {policy_basis}</div>'
        if policy_basis
        else ""
    )
    return (
        '<div class="trigger-label">RULE TRIGGER</div>'
        '<div class="trigger-box">'
        f"{condition}"
        f"{policy_line}"
        "</div>"
    )


def comms_strip(
    last_date: str,
    last_channel: str,
    last_status: str,
    count_30d: int,
    responded_30d: int,
) -> str:
    """Footer strip that shows recent outreach activity for a student."""
    if not last_date or count_30d == 0:
        return (
            '<div class="comms-strip">'
            "No outreach recorded in the last 30 days."
            "</div>"
        )
    no_reply = (responded_30d == 0) and (count_30d >= 2)
    warn_html = (
        '<span class="comms-strip-warn"> &nbsp;-&nbsp; NO RESPONSE</span>'
        if no_reply else ""
    )
    return (
        '<div class="comms-strip">'
        f"<strong>{count_30d}</strong> contact attempt(s) in last 30 days &middot; "
        f"<strong>{responded_30d}</strong> response(s){warn_html}<br>"
        f"Last: {last_channel} on {last_date} &middot; <em>{last_status}</em>"
        "</div>"
    )


def why_block_html(rows: list[dict[str, str]]) -> str:
    """Render the 'Why this case was created' checklist as a single HTML block.

    Each row is ``{"label": str, "current": str, "threshold": str}``.
    """
    if not rows:
        return ""
    items: list[str] = []
    for r in rows:
        items.append(
            '<div class="why-row">'
            '<span class="why-check">&#10003;</span>'
            f'<span class="why-text">{r["label"]}</span>'
            f'<span class="why-current">{r["current"]}</span>'
            f'<span class="why-thr">trigger: {r["threshold"]}</span>'
            "</div>"
        )
    return (
        '<div class="why-block">'
        '<div class="why-label">WHY THIS CASE WAS CREATED</div>'
        f'{"".join(items)}'
        "</div>"
    )


def activity_log_html(steps: list[dict]) -> str:
    """Render the case activity timeline as a vertical list."""
    if not steps:
        return "<div style='color:#8A8A85; font-size:0.85rem;'>No activity recorded.</div>"
    items: list[str] = []
    for s in steps:
        status = str(s.get("status", "done")).lower()
        dot_cls = {
            "done": "activity-dot activity-dot-done",
            "pending": "activity-dot activity-dot-pending",
            "scheduled": "activity-dot activity-dot-scheduled",
        }.get(status, "activity-dot activity-dot-done")
        ts = s.get("timestamp") or ""
        ts_html = f'<div class="activity-ts">{ts}</div>' if ts else ""
        status_html = ""
        if status == "pending":
            status_html = '<span class="activity-status-pending">pending</span>'
        elif status == "scheduled":
            status_html = '<span class="activity-status-scheduled">scheduled</span>'
        detail = s.get("detail") or ""
        detail_html = f'<div class="activity-detail">{detail}</div>' if detail else ""
        items.append(
            '<li class="activity-item">'
            '<div class="activity-marker">'
            f'<div class="{dot_cls}"></div>'
            '<div class="activity-line"></div>'
            '</div>'
            '<div class="activity-body">'
            f'<div class="activity-label">{s.get("label", "")}{status_html}</div>'
            f"{detail_html}"
            f"{ts_html}"
            '</div>'
            "</li>"
        )
    return f'<ul class="activity-list">{"".join(items)}</ul>'


def supporting_concerns_html(concerns: list[dict]) -> str:
    """Render the list of supporting concerns inside a consolidated case."""
    if not concerns:
        return ""
    items: list[str] = []
    for c in concerns:
        sev = c.get("severity", "low").lower()
        badge = severity_badge(sev)
        items.append(
            '<li class="concerns-item">'
            f'<span class="concerns-item-title">{c.get("rule_name", "")}</span>'
            f"{badge}<br>"
            f'<span style="font-size:0.83rem; color:{CH_CHARCOAL};">{c.get("rationale", "")}</span>'
            f'<div class="concerns-item-meta">'
            f"Action: {c.get('action', '')} &middot; "
            f"Routed to {c.get('route_to', '')} &middot; "
            f"{c.get('policy_basis', '')}"
            "</div>"
            "</li>"
        )
    return f'<ul class="concerns-list">{"".join(items)}</ul>'


def case_card(
    title_html: str,
    reason: str,
    action: str,
    severity: str,
    assigned_to: str = "",
    policy_basis: str = "",
) -> str:
    """Render a case card with severity-coloured left border.

    If ``assigned_to`` is empty, the right-side "Assigned to ..." aside is
    omitted - the caller is expected to render assigned-to (and any action
    buttons) in a separate Streamlit column next to the card.

    Output is intentionally a single-line HTML string so Streamlit's Markdown
    parser does not interpret indentation or blank lines as a code block.
    """
    sev = severity.lower()
    reason_block = (
        f'<div class="case-reason">{reason}</div>'
        if reason
        else ""
    )
    policy_block = (
        f'<div class="case-policy">Policy basis: {policy_basis}</div>'
        if policy_basis
        else ""
    )
    aside_block = ""
    if assigned_to:
        aside_block = (
            '<div class="case-aside">'
            "<div>Assigned to</div>"
            f'<div style="color:{CH_INK}; font-weight:600; font-size:0.9rem; '
            f'margin-top:0.2rem;">{assigned_to}</div>'
            "</div>"
        )

    return (
        f'<div class="case-card case-card-{sev}">'
        '<div class="case-row">'
        '<div class="case-main">'
        f'<div class="case-title">{title_html}</div>'
        f"{reason_block}"
        f'<div class="case-action">{action}</div>'
        f"{policy_block}"
        "</div>"
        f"{aside_block}"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Plotly chart defaults
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: Optional[int] = None, title: Optional[str] = None) -> go.Figure:
    layout_kwargs: dict = dict(
        paper_bgcolor=CH_PAGE,
        plot_bgcolor=CH_PAGE,
        font=dict(color=CH_CHARCOAL, family="system-ui, sans-serif", size=12),
        xaxis=dict(gridcolor=CH_GREY, zerolinecolor=CH_GREY, linecolor=CH_GREY),
        yaxis=dict(gridcolor=CH_GREY, zerolinecolor=CH_GREY, linecolor=CH_GREY),
    )
    if title:
        layout_kwargs["title"] = dict(
            text=title, font=dict(size=14, color=CH_INK), x=0.01, xanchor="left",
        )
        layout_kwargs["margin"] = dict(l=40, r=20, t=46, b=40)
    else:
        layout_kwargs["margin"] = dict(l=40, r=20, t=18, b=40)
    fig.update_layout(**layout_kwargs)
    if height is not None:
        fig.update_layout(height=height)
    return fig
