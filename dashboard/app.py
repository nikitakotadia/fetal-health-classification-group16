"""
Dashboard entry point - Group 16

Run: streamlit run dashboard/app.py

No separate landing/home page - the sidebar goes straight to the six content
pages via st.navigation, defaulting to Baseline Models.
"""

import streamlit as st

pages = [
    st.Page("pages/1_Baseline_Models.py", title="Baseline Models"),
    st.Page("pages/2_Hyperparameter_Tuning.py", title="Hyperparameter Tuning"),
    st.Page("pages/3_External_Validation.py", title="External Validation"),
    st.Page("pages/4_Pathological_Investigation.py", title="Pathological Investigation"),
    st.Page("pages/5_Suspect_Investigation.py", title="Suspect Investigation"),
    st.Page("pages/6_Conclusions.py", title="Conclusions"),
]

st.navigation(pages).run()
