"""
Dashboard page: Hyperparameter Tuning
Owner: Gnaneshwar

Standalone: streamlit run dashboard/pages/2_Hyperparameter_Tuning.py

Compares default and tuned XGBoost, LightGBM and CatBoost models. Improvements
and regressions are shown directly, including LightGBM's higher log-loss.

Source: outputs/evaluation/model_tuning_comparison.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "outputs" / "evaluation"

st.set_page_config(page_title="Hyperparameter Tuning", page_icon="🎛️", layout="wide")

# ------------------------------------------------------------------
# Palette -- Midnight Executive (matches the other dashboard pages)
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
def load_tuning_results() -> pd.DataFrame:
    df = pd.read_csv(EVAL_DIR / "model_tuning_comparison.csv")
    required = {
        "Model", "Version", "Accuracy", "F1_Weighted", "F1_Macro",
        "MCC", "ROC_AUC", "Log_Loss",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


tuning_df = load_tuning_results()
model_order = ["XGBoost", "LightGBM", "CatBoost"]
default_df = tuning_df[tuning_df["Version"].eq("Default")].set_index("Model").loc[model_order]
tuned_df = tuning_df[tuning_df["Version"].eq("Tuned")].set_index("Model").loc[model_order]


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


# ------------------------------------------------------------------
# Calculations for KPI cards and deltas
# ------------------------------------------------------------------

best_accuracy = tuned_df["Accuracy"].max()
best_accuracy_models = ", ".join(tuned_df.index[tuned_df["Accuracy"].eq(best_accuracy)])
best_macro_model = tuned_df["F1_Macro"].idxmax()
best_mcc_model = tuned_df["MCC"].idxmax()
lightgbm_logloss_pct = (
    (tuned_df.loc["LightGBM", "Log_Loss"] - default_df.loc["LightGBM", "Log_Loss"])
    / default_df.loc["LightGBM", "Log_Loss"]
) * 100

metric_deltas = pd.DataFrame(index=model_order)
metric_deltas["Accuracy"] = tuned_df["Accuracy"] - default_df["Accuracy"]
metric_deltas["Weighted F1"] = tuned_df["F1_Weighted"] - default_df["F1_Weighted"]
metric_deltas["Macro F1"] = tuned_df["F1_Macro"] - default_df["F1_Macro"]
metric_deltas["MCC"] = tuned_df["MCC"] - default_df["MCC"]
metric_deltas["ROC-AUC"] = tuned_df["ROC_AUC"] - default_df["ROC_AUC"]
metric_deltas["Log Loss"] = tuned_df["Log_Loss"] - default_df["Log_Loss"]

# ------------------------------------------------------------------
# Banner + KPIs
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="dash-banner">
        <p class="eyebrow">Page 2</p>
        <h1>Hyperparameter Tuning</h1>
        <p>Default vs. tuned XGBoost, LightGBM and CatBoost</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_options = ["Accuracy", "Weighted F1", "Macro F1", "MCC", "ROC-AUC", "Log Loss"]
selected_metric = st.selectbox("Focus metric", metric_options, index=1)
model_focus = st.selectbox("Model spotlight", ["All models", *model_order], index=0)

st.markdown(
    f"""
    <div class="kpi-row">
        <div class="kpi-card" style="background:{NAVY};">
            <div class="label">Top Tuned Accuracy</div><div class="value">{best_accuracy:.2%}</div>
            <div class="sub">Tie: {best_accuracy_models}</div>
        </div>
        <div class="kpi-card" style="background:{PURPLE};">
            <div class="label">Best Macro F1</div><div class="value">{tuned_df.loc[best_macro_model, 'F1_Macro']:.3f}</div>
            <div class="sub">{best_macro_model} tuned</div>
        </div>
        <div class="kpi-card" style="background:{GREEN};">
            <div class="label">Best MCC</div><div class="value">{tuned_df.loc[best_mcc_model, 'MCC']:.3f}</div>
            <div class="sub">{best_mcc_model} tuned</div>
        </div>
        <div class="kpi-card" style="background:{RED};">
            <div class="label">LightGBM Log-Loss</div><div class="value">+{lightgbm_logloss_pct:.1f}%</div>
            <div class="sub">Worse after tuning</div>
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
        st.markdown('<div class="panel-title">Default vs. Tuned - Selected Metric</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Default",
                x=model_order,
                y=default_df[selected_metric.replace("Weighted F1", "F1_Weighted").replace("Macro F1", "F1_Macro").replace("ROC-AUC", "ROC_AUC").replace("Log Loss", "Log_Loss")],
                marker_color=MUTED,
                text=[f"{v:.4f}" for v in default_df[selected_metric.replace("Weighted F1", "F1_Weighted").replace("Macro F1", "F1_Macro").replace("ROC-AUC", "ROC_AUC").replace("Log Loss", "Log_Loss")]],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.add_trace(
            go.Bar(
                name="Tuned",
                x=model_order,
                y=tuned_df[selected_metric.replace("Weighted F1", "F1_Weighted").replace("Macro F1", "F1_Macro").replace("ROC-AUC", "ROC_AUC").replace("Log Loss", "Log_Loss")],
                marker_color=PURPLE,
                text=[f"{v:.4f}" for v in tuned_df[selected_metric.replace("Weighted F1", "F1_Weighted").replace("Macro F1", "F1_Macro").replace("ROC-AUC", "ROC_AUC").replace("Log Loss", "Log_Loss")]],
                textposition="outside",
                cliponaxis=False,
            )
        )
        base_layout(fig, xaxis_title="Model", yaxis_title=selected_metric)
        fig.update_layout(barmode="group", bargap=0.28)
        if selected_metric == "Log Loss":
            fig.update_yaxes(range=[0.10, 0.28], tickformat=".3f")
        else:
            fig.update_yaxes(range=[0.87, 0.98], tickformat=".3f")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with col2:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Tuning Delta by Model</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        model_colors = {"XGBoost": NAVY, "LightGBM": BLUE, "CatBoost": PURPLE}
        selected_entries = model_order if model_focus == "All models" else [model_focus]
        for model in selected_entries:
            fig2.add_trace(
                go.Bar(
                    name=model,
                    x=[selected_metric],
                    y=[metric_deltas.loc[model, selected_metric]],
                    marker_color=model_colors[model],
                    text=[f"{metric_deltas.loc[model, selected_metric]:+.4f}"],
                    textposition="outside",
                    hovertemplate=f"{model}<br>{selected_metric} delta: %{{y:+.4f}}<extra></extra>",
                )
            )
        base_layout(fig2, legend=False, xaxis_title="Metric", yaxis_title="Tuned - Default")
        fig2.add_hline(y=0, line_color=INK, line_width=1)
        fig2.update_yaxes(tickformat="+.4f")
        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

