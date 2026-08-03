# Baseline Models (UCI) — Gnaneshwar
"""
Dashboard page: Baseline Models (UCI)
Owner: Gnaneshwar

Standalone: streamlit run dashboard/pages/1_Baseline_Models.py

Interactive comparison of all seven UCI baseline models using overall metrics
and class-level F1 performance.

Source: outputs/evaluation/model_comparison_full.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="Baseline Models", page_icon="📊", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_baseline_results() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "model_comparison_full.csv")
    required = {
        "Model",
        "Accuracy",
        "Weighted F1",
        "ROC-AUC",
        "MCC",
        "F1 — Normal",
        "F1 — Suspect",
        "F1 — Pathological",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.sort_values("Rank").reset_index(drop=True)


baseline_df = load_baseline_results()


def base_layout(fig: go.Figure, height: int = 290, legend: bool = True) -> go.Figure:
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", color=INK, size=11),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=8, b=32, l=46, r=16),
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=9))
            if legend else None
        ),
        showlegend=legend,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR, zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=GRIDCOLOR, zeroline=False)
    return fig


st.markdown(
    """
    <div class="dash-banner">
        <p class="eyebrow">Page 1</p>
        <h1>Baseline Models — UCI</h1>
        <p>Seven classifiers compared on overall and class-level performance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_options = ["Accuracy", "Weighted F1", "ROC-AUC", "MCC"]
selected_metric = st.selectbox("Focus metric", metric_options, index=1)

ranked_df = baseline_df.sort_values(selected_metric, ascending=False).reset_index(drop=True)
best_overall = ranked_df.iloc[0]
best_suspect = baseline_df.loc[baseline_df["F1 — Suspect"].idxmax()]
best_pathological = baseline_df.loc[baseline_df["F1 — Pathological"].idxmax()]

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card" style="background:{NAVY};">
            <div class="label">Top Model</div><div class="value">{best_overall['Model']}</div>
            <div class="sub">{selected_metric}: {best_overall[selected_metric]:.3f}</div>
        </div>
        <div class="kpi-card" style="background:{BLUE};">
            <div class="label">Best Accuracy</div><div class="value">{baseline_df['Accuracy'].max():.2%}</div>
            <div class="sub">{baseline_df.loc[baseline_df['Accuracy'].idxmax(), 'Model']}</div>
        </div>
        <div class="kpi-card" style="background:{AMBER};">
            <div class="label">Best Suspect F1</div><div class="value">{best_suspect['F1 — Suspect']:.3f}</div>
            <div class="sub">{best_suspect['Model']}</div>
        </div>
        <div class="kpi-card" style="background:{GREEN};">
            <div class="label">Best Pathological F1</div><div class="value">{best_pathological['F1 — Pathological']:.3f}</div>
            <div class="sub">{best_pathological['Model']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Overall Metrics — Pick the Leaderboard View</div>', unsafe_allow_html=True)
        fig = go.Figure(
            go.Bar(
                x=ranked_df["Model"],
                y=ranked_df[selected_metric],
                marker_color=[NAVY, PURPLE, BLUE, GREEN, RED, AMBER, MUTED],
                text=ranked_df[selected_metric].round(3),
                textposition="outside",
                hovertemplate="%{x}<br>" + selected_metric + ": %{y:.4f}<extra></extra>",
            )
        )
        base_layout(fig, legend=False)
        fig.update_xaxes(tickangle=-24)
        fig.update_yaxes(range=[0.6, 1.0], tickformat=".2f")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Per-Class F1 by Model</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for column, label, color in [
            ("F1 — Normal", "Normal", GREEN),
            ("F1 — Suspect", "Suspect", AMBER),
            ("F1 — Pathological", "Pathological", RED),
        ]:
            fig2.add_trace(
                go.Scatter(
                    name=label,
                    x=baseline_df["Model"],
                    y=baseline_df[column],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f"%{{x}}<br>{label} F1: %{{y:.4f}}<extra></extra>",
                )
            )
        base_layout(fig2)
        fig2.update_yaxes(range=[0.6, 1.02], tickformat=".2f")
        fig2.update_xaxes(tickangle=-24)
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

st.caption(
    "XGBoost leads on the overall UCI benchmark, while the Suspect class remains the weakest spot across all seven models."
)
