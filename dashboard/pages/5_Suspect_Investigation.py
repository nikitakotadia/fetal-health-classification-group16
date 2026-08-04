"""
Dashboard page: Suspect Investigation (UCI)
Owner: Mamatha

Standalone: streamlit run dashboard/pages/5_Suspect_Investigation.py

Interactive threshold sweep for the UCI Suspect class, showing how the
decision boundary changes precision, recall, F1, and predicted count.

Source: outputs/evaluation/suspect_threshold_sweep.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="Suspect Investigation", page_icon="🔎", layout="wide")

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


@st.cache_data
def load_suspect_sweep() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "suspect_threshold_sweep.csv")
    required = {"Threshold", "Precision", "Recall", "F1", "Predicted_Suspect_Count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.sort_values("Threshold").reset_index(drop=True)


sweep_df = load_suspect_sweep()
best_row = sweep_df.loc[sweep_df["F1"].idxmax()]
default_row = sweep_df.iloc[(sweep_df["Threshold"] - 0.50).abs().argsort()[:1]].iloc[0]
high_precision_row = sweep_df.loc[sweep_df["Precision"].idxmax()]
recall_gain_pp = (best_row["Recall"] - default_row["Recall"]) * 100
extra_flags = int(best_row["Predicted_Suspect_Count"] - default_row["Predicted_Suspect_Count"])


def base_layout(fig: go.Figure, height: int = 290, legend: bool = True,
                 xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", color=INK, size=11),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=8, b=46, l=54, r=16),
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=9))
            if legend else None
        ),
        showlegend=legend,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR, zeroline=False,
                      title=dict(text=xaxis_title, font=dict(size=10)))
    fig.update_yaxes(gridcolor=GRIDCOLOR, zeroline=False,
                      title=dict(text=yaxis_title, font=dict(size=10)))
    return fig


st.markdown(
    """
    <div class="dash-banner">
        <p class="eyebrow">Page 5</p>
        <h1>Suspect Investigation - UCI</h1>
        <p>Threshold sweep for the minority Suspect class</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_options = ["Precision", "Recall", "F1"]
selected_metric = st.selectbox("Focus metric", metric_options, index=2)
active_threshold = st.slider(
    "Selected decision threshold",
    min_value=float(sweep_df["Threshold"].min()),
    max_value=float(sweep_df["Threshold"].max()),
    value=float(best_row["Threshold"]),
    step=0.05,
)
active_row = sweep_df.loc[sweep_df["Threshold"].sub(active_threshold).abs().idxmin()]

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card" style="background:{NAVY};">
            <div class="label">Best F1 Threshold</div><div class="value">{best_row['Threshold']:.2f}</div>
            <div class="sub">Highest harmonic balance</div>
        </div>
        <div class="kpi-card" style="background:{PURPLE};">
            <div class="label">Selected Threshold</div><div class="value">{active_threshold:.2f}</div>
            <div class="sub">{selected_metric}: {active_row[selected_metric]:.3f}</div>
        </div>
        <div class="kpi-card" style="background:{AMBER};">
            <div class="label">Recall Gain vs. 0.50</div><div class="value">+{recall_gain_pp:.2f} pp</div>
            <div class="sub">Only {extra_flags:+d} extra Suspect flags</div>
        </div>
        <div class="kpi-card" style="background:{GREEN};">
            <div class="label">Predicted Suspect Count</div><div class="value">{active_row['Predicted_Suspect_Count']}</div>
            <div class="sub">Precision {active_row['Precision']:.3f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Precision-Recall-F1 Sweep</div>', unsafe_allow_html=True)
        fig = go.Figure()
        line_specs = [
            ("Precision", BLUE),
            ("Recall", AMBER),
            ("F1", PURPLE),
        ]
        for metric, color in line_specs:
            fig.add_trace(
                go.Scatter(
                    name=metric,
                    x=sweep_df["Threshold"],
                    y=sweep_df[metric],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f"Threshold %{{x:.2f}}<br>{metric}: %{{y:.4f}}<extra></extra>",
                )
            )
        fig.add_vline(
            x=active_threshold,
            line_dash="dash",
            line_color=RED,
            annotation_text=f"Threshold {active_threshold:.2f}",
            annotation_font_size=9,
        )
        fig.add_trace(
            go.Scatter(
                x=[active_threshold],
                y=[active_row[selected_metric]],
                mode="markers",
                marker=dict(color=RED, size=12, symbol="diamond"),
                name="Selected point",
                hovertemplate=f"Threshold {active_threshold:.2f}<br>{selected_metric}: {active_row[selected_metric]:.4f}<extra></extra>",
            )
        )
        base_layout(fig, xaxis_title="Decision Threshold", yaxis_title="Score")
        fig.update_xaxes(tickformat=".2f", dtick=0.05)
        fig.update_yaxes(range=[0.72, 0.98], tickformat=".2f")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Prediction Volume by Threshold</div>', unsafe_allow_html=True)
        bar_colors = [
            RED if abs(t - active_threshold) < 1e-9
            else NAVY if abs(t - 0.50) < 1e-9
            else MUTED
            for t in sweep_df["Threshold"]
        ]
        fig2 = go.Figure(
            go.Bar(
                x=sweep_df["Threshold"],
                y=sweep_df["Predicted_Suspect_Count"],
                marker_color=bar_colors,
                text=sweep_df["Predicted_Suspect_Count"],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Threshold %{x:.2f}<br>Predicted Suspect: %{y}<extra></extra>",
            )
        )
        base_layout(fig2, legend=False, xaxis_title="Decision Threshold", yaxis_title="Predicted Suspect Count")
        fig2.update_xaxes(tickformat=".2f", dtick=0.05)
        fig2.update_yaxes(range=[42, 56], dtick=2)
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

