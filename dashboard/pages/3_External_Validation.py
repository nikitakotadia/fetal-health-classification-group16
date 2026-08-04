"""
Dashboard page: CTU-UHB Validation
Owner: Nikita

Standalone: streamlit run dashboard/pages/3_External_Validation.py

Shows XGBoost trained and evaluated entirely within CTU-UHB (5-fold CV),
plus whether SMOTE helps the Pathological class. This is in-domain
retraining, not a literal cross-dataset transfer of the UCI model -- the
page says so explicitly rather than implying "external validation" in
the strict sense.

Source: outputs/evaluation/ctu_external_validation_comparison.csv,
        outputs/evaluation/ctu_smote_comparison.csv
        (07_ctu_external_validation.ipynb)
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="CTU-UHB Validation", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------
# Palette -- Midnight Executive (navy / ice blue), matches the report/deck
# ------------------------------------------------------------------

NAVY = "#1E2761"
PURPLE = "#6C3FC4"
BLUE = "#1C7293"
GREEN = "#1E8A5F"
RED = "#C0392B"
AMBER = "#C9930B"
MUTED = "#95A5A6"
INK = "#1E2761"
GRIDCOLOR = "#E7EAF3"

st.markdown(
    f"""
    <style>
    header[data-testid="stHeader"] {{ height: 0; min-height: 0; visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 0.8rem; padding-bottom: 0.4rem; padding-left: 1.3rem; padding-right: 1.3rem; max-width: 100%; }}
    div[data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}

    .dash-banner {{
        background: linear-gradient(90deg, {NAVY} 0%, {PURPLE} 100%);
        padding: 0.8rem 1.3rem; border-radius: 10px; margin-bottom: 0.7rem;
    }}
    .dash-banner .eyebrow {{ color: #CADCFC; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }}
    .dash-banner h1 {{ color: white; font-size: 1.3rem; font-weight: 700; margin: 0.1rem 0 0 0; }}
    .dash-banner p {{ color: #E4E1F5; font-size: 0.76rem; margin: 0.2rem 0 0 0; }}

    .kpi-row {{ display: flex; gap: 0.7rem; margin-bottom: 0.7rem; flex-wrap: wrap; }}
    .kpi-card {{ flex: 1; min-width: 150px; border-radius: 10px; padding: 0.6rem 0.9rem; color: white; box-shadow: 0 2px 8px rgba(30,39,97,0.14); }}
    .kpi-card .label {{ font-size: 0.62rem; opacity: 0.88; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.2rem; font-weight: 600; }}
    .kpi-card .value {{ font-size: 1.35rem; font-weight: 700; line-height: 1.1; }}
    .kpi-card .sub {{ font-size: 0.64rem; opacity: 0.88; margin-top: 0.2rem; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px !important; border-color: #E4E8F2 !important; box-shadow: 0 2px 8px rgba(30,39,97,0.07); }}
    .panel-title {{ font-weight: 700; font-size: 0.8rem; color: {NAVY}; margin-bottom: 0.2rem; }}
    .stCaption, small {{ font-size: 0.68rem !important; }}

    section[data-testid="stSidebar"] > div {{ display: flex; flex-direction: column; height: 100%; }}
    .sidebar-print-spacer {{ flex-grow: 1; }}
    @media print {{
        section[data-testid="stSidebar"] {{ display: none !important; }}
        header[data-testid="stHeader"] {{ display: none !important; }}
        .block-container {{ padding-top: 0 !important; }}
    }}

    /* Sidebar can no longer be collapsed -- always rendered open */
    section[data-testid="stSidebar"] {{
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
        transform: none !important;
        visibility: visible !important;
        margin-left: 0px !important;
    }}
    [data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
    [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-print-spacer"></div>', unsafe_allow_html=True)
with st.sidebar:
    components.html(
        f"""
        <button id="printPageBtn" style="
            display:block; width:100%; box-sizing:border-box;
            background:{NAVY}; color:white; border:none; border-radius:8px;
            padding:0.55rem 0.8rem; font-size:0.78rem; font-weight:700;
            cursor:pointer; box-shadow:0 2px 8px rgba(30,39,97,0.2);
            font-family:Arial, Helvetica, sans-serif;">
            🖨️ Print This Page
        </button>
        <script>
        document.getElementById('printPageBtn').addEventListener('click', function() {{
            window.parent.print();
        }});
        </script>
        """,
        height=46,
    )

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

@st.cache_data
def load_ctu_indomain_summary() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "ctu_external_validation_comparison.csv")
    return df.rename(columns={df.columns[0]: "Metric"})

@st.cache_data
def load_smote_comparison() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "ctu_smote_comparison.csv")
    return df.rename(columns={df.columns[0]: "Metric"})

summary = load_ctu_indomain_summary().set_index("Metric")
smote = load_smote_comparison().set_index("Metric")

def base_layout(fig, height=460, legend=True, xaxis_title="", yaxis_title=""):
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", color=INK, size=11),
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        margin=dict(t=8, b=44, l=54, r=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=10)) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR, zeroline=False, title=dict(text=xaxis_title, font=dict(size=10)))
    fig.update_yaxes(gridcolor=GRIDCOLOR, zeroline=False, title=dict(text=yaxis_title, font=dict(size=10)))
    return fig

# ------------------------------------------------------------------
# Banner + KPIs
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="dash-banner">
        <p class="eyebrow">Page 3</p>
        <h1>CTU-UHB Validation</h1>
        <p>In-domain, 5-fold CV retraining on CTU-UHB, with SMOTE comparison</p>
    </div>
    """,
    unsafe_allow_html=True,
)

roc = summary.loc["ROC-AUC", "Mean"]
mcc = summary.loc["MCC", "Mean"]
acc = summary.loc["Accuracy", "Mean"]
path_recall = summary.loc["Pathological Recall", "Mean"]

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card" style="background:{NAVY};">
            <div class="label">ROC-AUC (5-fold CV)</div><div class="value">{roc:.3f}</div>
            <div class="sub">Off chance level</div>
        </div>
        <div class="kpi-card" style="background:{BLUE};">
            <div class="label">MCC</div><div class="value">{mcc:.3f}</div>
            <div class="sub">Positive, but weak</div>
        </div>
        <div class="kpi-card" style="background:{PURPLE};">
            <div class="label">Accuracy</div><div class="value">{acc:.1%}</div>
            <div class="sub">vs. 96.2% on UCI</div>
        </div>
        <div class="kpi-card" style="background:{RED};">
            <div class="label">Pathological Recall</div><div class="value">{path_recall:.3f}</div>
            <div class="sub">0/13, no resampling</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">F1 by Class (mean ± std)</div>', unsafe_allow_html=True)
        classes = ["F1 - Normal", "F1 - Suspect", "F1 - Pathological"]
        labels = ["Normal", "Suspect", "Pathological"]
        values = [summary.loc[c, "Mean"] for c in classes]
        errors = [summary.loc[c, "Std"] for c in classes]
        fig = go.Figure(go.Bar(
            x=labels, y=values, error_y=dict(type="data", array=errors, color=INK),
            marker_color=[GREEN, AMBER, RED],
        ))
        base_layout(fig, legend=False, xaxis_title="Class", yaxis_title="F1 Score")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">No Resampling vs. SMOTE</div>', unsafe_allow_html=True)
        metrics_to_show = ["Accuracy", "Weighted F1", "F1 - Pathological", "Pathological Recall", "MCC"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="No Resampling", x=metrics_to_show,
                               y=[smote.loc[m, "No resampling"] for m in metrics_to_show], marker_color=MUTED))
        fig2.add_trace(go.Bar(name="With SMOTE", x=metrics_to_show,
                               y=[smote.loc[m, "With SMOTE"] for m in metrics_to_show], marker_color=PURPLE))
        base_layout(fig2, xaxis_title="Metric", yaxis_title="Score")
        fig2.update_layout(barmode="group")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
