"""
Dashboard page: Conclusions
Owner: Nikita

Standalone: streamlit run dashboard/pages/6_Conclusions.py

Maps this phase's findings back to the five dissertation objectives (Section
1.2, Preliminary Report), and synthesises the Pathological (CTU-UHB) vs.
Suspect (UCI) minority-class investigations run by Shreya and Mamatha.

Source: outputs/evaluation/model_comparison_full.csv,
        outputs/evaluation/ctu_external_validation_comparison.csv,
        outputs/evaluation/ctu_smote_comparison.csv,
        outputs/evaluation/suspect_threshold_sweep.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="Conclusions", page_icon="🎯", layout="wide")

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
    div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}

    .dash-banner {{
        background: linear-gradient(90deg, {NAVY} 0%, {PURPLE} 100%);
        padding: 0.75rem 1.3rem; border-radius: 10px; margin-bottom: 0.6rem;
    }}
    .dash-banner .eyebrow {{ color: #CADCFC; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin: 0; }}
    .dash-banner h1 {{ color: white; font-size: 1.25rem; font-weight: 700; margin: 0.1rem 0 0 0; }}
    .dash-banner p {{ color: #E4E1F5; font-size: 0.74rem; margin: 0.2rem 0 0 0; }}

    .obj-row {{ display: flex; gap: 0.55rem; margin-bottom: 0.6rem; flex-wrap: wrap; }}
    .obj-card {{ flex: 1; min-width: 155px; border-radius: 10px; padding: 0.55rem 0.75rem; color: white; box-shadow: 0 2px 8px rgba(30,39,97,0.14); }}
    .obj-card .label {{ font-size: 0.58rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.02em; margin-bottom: 0.15rem; font-weight: 700; }}
    .obj-card .value {{ font-size: 0.98rem; font-weight: 700; line-height: 1.15; }}
    .obj-card .sub {{ font-size: 0.6rem; opacity: 0.9; margin-top: 0.15rem; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px !important; border-color: #E4E8F2 !important; box-shadow: 0 2px 8px rgba(30,39,97,0.07); }}
    .panel-title {{ font-weight: 700; font-size: 0.8rem; color: {NAVY}; margin-bottom: 0.35rem; }}
    .stCaption, small {{ font-size: 0.64rem !important; }}

    .obj-list {{ display: flex; flex-direction: column; gap: 0.65rem; }}
    .obj-item {{ display: flex; align-items: center; gap: 0.8rem; background: #F7F8FC; border-radius: 8px; padding: 0.85rem 1rem; }}
    .obj-item .num {{ font-size: 1.05rem; font-weight: 700; color: {NAVY}; min-width: 1.4rem; }}
    .obj-item .name {{ font-size: 0.82rem; font-weight: 600; color: {NAVY}; min-width: 165px; }}
    .obj-item .pill {{ font-size: 0.64rem; font-weight: 700; color: white; padding: 0.22rem 0.6rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.02em; min-width: 70px; text-align: center; }}
    .obj-item .finding {{ font-size: 0.78rem; color: #4A4F63; flex: 1; }}

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
def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "model_comparison_full.csv")
    df.columns = [c.replace(chr(0x2013), "-").replace(chr(0x2014), "-") for c in df.columns]
    return df

@st.cache_data
def load_ctu_summary() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "ctu_external_validation_comparison.csv")
    return df.rename(columns={df.columns[0]: "Metric"}).set_index("Metric")

@st.cache_data
def load_smote() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "ctu_smote_comparison.csv")
    return df.rename(columns={df.columns[0]: "Metric"}).set_index("Metric")

@st.cache_data
def load_suspect_sweep() -> pd.DataFrame:
    return pd.read_csv(EVAL_DIR / "suspect_threshold_sweep.csv").sort_values("Threshold")

baseline_df = load_baseline()
ctu_summary = load_ctu_summary()
smote_df = load_smote()
suspect_df = load_suspect_sweep()

best_baseline = baseline_df.loc[baseline_df["Weighted F1"].idxmax()]
ctu_roc = ctu_summary.loc["ROC-AUC", "Mean"]
ctu_mcc = ctu_summary.loc["MCC", "Mean"]
best_suspect_f1 = suspect_df["F1"].max()
default_suspect_f1 = float(baseline_df.loc[baseline_df["Model"].eq("XGBoost"), "F1 - Suspect"].iloc[0])
path_f1_no_resample = smote_df.loc["F1 - Pathological", "No resampling"]
path_f1_smote = smote_df.loc["F1 - Pathological", "With SMOTE"]

def base_layout(fig, height=390, legend=True, xaxis_title="", yaxis_title=""):
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", color=INK, size=10),
        plot_bgcolor="white", paper_bgcolor="white", height=height,
        margin=dict(t=8, b=44, l=48, r=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=9)) if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR, zeroline=False, title=dict(text=xaxis_title, font=dict(size=10)))
    fig.update_yaxes(gridcolor=GRIDCOLOR, zeroline=False, title=dict(text=yaxis_title, font=dict(size=10)))
    return fig

# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="dash-banner">
        <p class="eyebrow">Page 6</p>
        <h1>Conclusions - Findings vs. Objectives</h1>
        <p>Five objectives, assessed against this phase's evidence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Objective status cards
# ------------------------------------------------------------------

st.markdown(
    f"""
    <div class="obj-row">
        <div class="obj-card" style="background:{GREEN};">
            <div class="label">Obj. 1</div>
            <div class="value">Achieved</div>
            <div class="sub">{best_baseline['Model']}, F1 {best_baseline['Weighted F1']:.2f}</div>
        </div>
        <div class="obj-card" style="background:{RED};">
            <div class="label">Obj. 2</div>
            <div class="value">Limited</div>
            <div class="sub">13-case ceiling</div>
        </div>
        <div class="obj-card" style="background:{GREEN};">
            <div class="label">Obj. 3</div>
            <div class="value">Achieved</div>
            <div class="sub">Macro F1 / MCC</div>
        </div>
        <div class="obj-card" style="background:{GREEN};">
            <div class="label">Obj. 4</div>
            <div class="value">Achieved</div>
            <div class="sub">SHAP applied</div>
        </div>
        <div class="obj-card" style="background:{AMBER};">
            <div class="label">Obj. 5</div>
            <div class="value">Limited</div>
            <div class="sub">{ctu_roc:.2f} vs. ≈0.50</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Detail table + minority-class synthesis chart
# ------------------------------------------------------------------

col1, col2 = st.columns([1.15, 1])

with col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Objective Summary</div>', unsafe_allow_html=True)
        summary_rows = [
            ("1", "Model comparison", "Achieved", GREEN, f"{best_baseline['Model']} best (F1 {best_baseline['Weighted F1']:.2f})"),
            ("2", "Pathological detection", "Limited", RED, "13 records is the ceiling"),
            ("3", "Imbalance-robust eval", "Achieved", GREEN, "Macro F1 / MCC used throughout"),
            ("4", "Interpretability", "Achieved", GREEN, "SHAP confirms weak signal"),
            ("5", "Generalisability", "Limited", AMBER, f"{ctu_roc:.2f} in-domain vs. ≈0.50 transfer"),
        ]
        items_html = "".join(
            f'<div class="obj-item"><div class="num">{num}</div>'
            f'<div class="name">{name}</div>'
            f'<div class="pill" style="background:{color};">{status}</div>'
            f'<div class="finding">{finding}</div></div>'
            for num, name, status, color, finding in summary_rows
        )
        st.markdown(f'<div class="obj-list">{items_html}</div>', unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Minority-Class F1: Pathological vs. Suspect</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Pathological<br>(no resample)", "Pathological<br>(SMOTE)", "Suspect<br>(default 0.50)", "Suspect<br>(tuned 0.30)"],
            y=[path_f1_no_resample, path_f1_smote, default_suspect_f1, best_suspect_f1],
            marker_color=[RED, RED, AMBER, GREEN],
            text=[f"{v:.3f}" for v in [path_f1_no_resample, path_f1_smote, default_suspect_f1, best_suspect_f1]],
            textposition="outside",
        ))
        base_layout(fig, height=390, legend=False, xaxis_title="Class / Technique", yaxis_title="F1 Score")
        fig.update_yaxes(range=[0, 1.0])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.container(border=True):
    st.markdown('<div class="panel-title">Synthesis</div>', unsafe_allow_html=True)
    st.caption(
        "Same class-imbalance approach, opposite outcome: Suspect (58 records) improved with tuning, "
        "Pathological (13 records) did not - the deciding factor is sample size, not method."
    )
