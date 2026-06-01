"""
University of Chester - Student Engagement Risk Dashboard (v3).

Reads only the precomputed CSVs and JSON written by the ML and intervention
pipelines. No model or training-time dependency at runtime.
"""

from __future__ import annotations

import streamlit as st

from lib.theme import inject_theme

import pages_impl.overview as page_overview
import pages_impl.students as page_students
import pages_impl.student_detail as page_student_detail
import pages_impl.modules as page_modules
import pages_impl.at_risk as page_at_risk
import pages_impl.interventions as page_interventions

st.set_page_config(
    page_title="Engagement Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


PAGES: dict = {
    "Overview": page_overview.render,
    "Students": page_students.render,
    "Student Profile": page_student_detail.render,
    "Modules": page_modules.render,
    "At-Risk Students": page_at_risk.render,
    "Interventions": page_interventions.render,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='padding: 0.8rem 0 1rem 0;'>
                <div style='font-size:1.1rem; font-weight:700; color:#1B1B17; line-height:1.15;'>
                    University of Chester
                </div>
                <div style='font-size:0.78rem; color:#8A8A85; margin-top:0.15rem;'>
                    Student Engagement Dashboard
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        choice = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.markdown("---")
        st.caption(
            "Prototype on synthetic data. Predictions and cases are produced by the ML pipeline "
            "and the policy-rule engine. No real student records are used."
        )

    PAGES[choice]()


if __name__ == "__main__":
    main()
