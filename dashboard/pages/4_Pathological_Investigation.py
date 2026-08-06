# Pathological Investigation (CTU-UHB) - Shreya
"""
Dashboard page: Pathological Investigation (CTU-UHB)
Owner: Shreya

Standalone: streamlit run dashboard/pages/4_Pathological_Investigation.py

Source: outputs/evaluation/ctu_threshold_sweep.csv,
        08_ctu_threshold_tuning.ipynb (thresholds 0.10-0.30, max P(Pathological)
        on a true Pathological case = 0.0748)
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="Pathological Investigation", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------
# Palette -- Midnight Executive (navy / ice blue)
# ------------------------------------------------------------------

NAVY = "#1E2761"
PURPLE = "#6C3FC4"
BLUE = "#1C7293"
RED = "#C0392B"
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

    /* Sidebar stays open and uncollapsible on desktop only -- on
       mobile/narrow screens it keeps Streamlit's normal collapsible
       behaviour (with its native reopen arrow) so it doesn't
       permanently cover the whole viewport. */
    @media (min-width: 769px) {{
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
    }}
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
def load_pathological_sweep() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "ctu_threshold_sweep.csv")
    return df.rename(columns={
        "Accuracy_mean": "Accuracy",
        "Macro F1_mean": "Macro F1",
        "Pathological Recall_mean": "Pathological Recall",
        "Pathological Precision_mean": "Pathological Precision",
        "F1 - Pathological_mean": "Pathological F1",
        "MCC_mean": "MCC",
    })

TRUE_PATHOLOGICAL_PROBS = [
    0.0748, 0.0423, 0.0215, 0.0097, 0.0069,
    0.0068, 0.0009, 0.0007, 0.0002, 0.0001,
    0.0001, 0.0001, 0.0000,
]

sweep_df = load_pathological_sweep()
KEY_COLS = ["Threshold", "Pathological Recall", "Pathological Precision", "Pathological F1", "MCC"]
display_cols = [c for c in KEY_COLS if c in sweep_df.columns]

MIN_THRESHOLD_TESTED = 0.10
MAX_TRUE_CONFIDENCE = max(TRUE_PATHOLOGICAL_PROBS)
N_PATHOLOGICAL = len(TRUE_PATHOLOGICAL_PROBS)

def base_layout(fig, height=280, legend=True, xaxis_title="", yaxis_title=""):
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
        <p class="eyebrow">Page 4</p>
        <h1>Pathological Investigation - CTU-UHB</h1>
        <p>Threshold sweep and confidence diagnosis for the minority Pathological class</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card" style="background:{PURPLE};">
            <div class="label">Pathological Records</div><div class="value">{N_PATHOLOGICAL}</div>
            <div class="sub">of 552 total</div>
        </div>
        <div class="kpi-card" style="background:{BLUE};">
            <div class="label">Thresholds Tested</div><div class="value">{MIN_THRESHOLD_TESTED:.2f}-0.30</div>
            <div class="sub">5-fold CV</div>
        </div>
        <div class="kpi-card" style="background:{RED};">
            <div class="label">Max Model Confidence</div><div class="value">{MAX_TRUE_CONFIDENCE:.4f}</div>
            <div class="sub">Below min. threshold</div>
        </div>
        <div class="kpi-card" style="background:{NAVY};">
            <div class="label">Result</div><div class="value">Flat Null</div>
            <div class="sub">0.0000 throughout</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sweep + Diagnosis
# ------------------------------------------------------------------

col1, col2, col3 = st.columns([1, 1, 1])

compact_cols = ["Threshold", "Pathological Recall", "Pathological Precision", "Pathological F1"]
compact_cols = [c for c in compact_cols if c in display_cols]

with col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Sweep Results</div>', unsafe_allow_html=True)
        st.dataframe(
            sweep_df[compact_cols].style.format(
                {c: ("{:.2f}" if c == "Threshold" else "{:.4f}") for c in compact_cols}
            ),
            use_container_width=True, hide_index=True, height=325,
            column_config={
                "Threshold": st.column_config.NumberColumn(width="small"),
                "Pathological Recall": st.column_config.NumberColumn("Recall", width="small"),
                "Pathological Precision": st.column_config.NumberColumn("Precision", width="small"),
                "Pathological F1": st.column_config.NumberColumn("F1", width="small"),
            },
        )
        st.caption("Flat null: Recall/Precision/F1 all 0.0000 at every threshold.")

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Recall / Precision vs. Threshold</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df["Threshold"], y=sweep_df["Pathological Recall"],
            mode="lines+markers", name="Recall",
            line=dict(color=NAVY, width=3), marker=dict(size=8),
            fill="tozeroy", fillcolor="rgba(30,39,97,0.12)",
        ))
        fig.add_trace(go.Scatter(
            x=sweep_df["Threshold"], y=sweep_df["Pathological Precision"],
            mode="lines+markers", name="Precision",
            line=dict(color=RED, width=3, dash="dash"), marker=dict(size=8, symbol="square"),
        ))
        base_layout(fig, height=380, xaxis_title="Decision Threshold", yaxis_title="Score")
        fig.update_yaxes(range=[-0.05, 1.05])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col3:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Model Confidence - 13 True Cases</div>', unsafe_allow_html=True)
        ranked = sorted(TRUE_PATHOLOGICAL_PROBS, reverse=True)
        bar_colors = [RED if p >= MIN_THRESHOLD_TESTED else MUTED for p in ranked]
        fig2 = go.Figure(go.Bar(x=list(range(1, len(ranked) + 1)), y=ranked, marker_color=bar_colors))
        fig2.add_hline(y=MIN_THRESHOLD_TESTED, line_dash="dash", line_color=INK,
                        annotation_text="Min. threshold", annotation_font_size=9)
        base_layout(fig2, height=380, legend=False,
                    xaxis_title="Case Rank (highest confidence first)",
                    yaxis_title="P(Pathological)")
        fig2.update_yaxes(range=[0, 0.15])
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.caption("Source: 08_ctu_threshold_tuning.ipynb (Shreya)")
