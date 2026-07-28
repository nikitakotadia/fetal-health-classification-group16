"""
Dashboard page: Pathological Threshold Tuning + Comparative Synthesis
Owner: Shreya

Drop this file into dashboard/pages/ once Nikita's skeleton exists, and
rename to match her numbering convention (e.g. 3_Threshold_Tuning.py).

Runs standalone right now for testing:
    pip install streamlit pandas matplotlib
    streamlit run 3_Pathological_Threshold_and_Synthesis.py

All numbers below are taken directly from 08_ctu_threshold_tuning.ipynb
and cross-checked against the Interim Report (Section 2.1). If a saved
CSV of the threshold sweep exists in the repo, this will use it instead
of the hardcoded fallback below -- see load_threshold_data().
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Pathological Threshold Tuning", layout="wide")

# ------------------------------------------------------------------
# Data -- real values from 08_ctu_threshold_tuning.ipynb
# ------------------------------------------------------------------

THRESHOLD_SWEEP_CSV = Path("outputs/evaluation/ctu_threshold_sweep.csv")

def load_threshold_data():
    """Load the real sweep results if the repo has them saved; otherwise
    fall back to the exact values already confirmed during development."""
    if THRESHOLD_SWEEP_CSV.exists():
        return pd.read_csv(THRESHOLD_SWEEP_CSV)
    return pd.DataFrame({
        "Threshold": [0.10, 0.15, 0.20, 0.25, 0.30],
        "Pathological Recall": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        "Pathological Precision": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        "Pathological F1": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        "Macro F1": [0.3502, 0.3500, 0.3495, 0.3495, 0.3495],
        "Accuracy": [0.6305, 0.6323, 0.6341, 0.6341, 0.6341],
        "MCC": [0.0891, 0.0889, 0.0844, 0.0844, 0.0844],
    })

# The model's own confidence on each of the 13 true Pathological cases,
# pooled across all 5 CV folds (highest to lowest)
TRUE_PATHOLOGICAL_PROBS = [
    0.0748, 0.0423, 0.0215, 0.0097, 0.0069,
    0.0068, 0.0009, 0.0007, 0.0002, 0.0001,
    0.0001, 0.0001, 0.0000,
]

MIN_THRESHOLD_TESTED = 0.10
MAX_TRUE_CONFIDENCE = max(TRUE_PATHOLOGICAL_PROBS)

# ------------------------------------------------------------------
# Page
# ------------------------------------------------------------------

st.title("Pathological Threshold Tuning — CTU-UHB")
st.caption("Owner: Shreya  |  Source: 08_ctu_threshold_tuning.ipynb")

st.markdown(
    """
Threshold tuning overrides the model's default decision rule — instead of
predicting whichever class has the highest probability, this tests: **"if
P(Pathological) ≥ threshold, predict Pathological regardless of rank."**
Five thresholds were swept, from 0.10 to 0.30, on the CTU-UHB in-domain
model (5-fold CV).
"""
)

# ------------------------------------------------------------------
# Section 1 — Sweep results
# ------------------------------------------------------------------

st.header("1. Threshold Sweep Results")

sweep_df = load_threshold_data()

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(
        sweep_df.style.format({
            "Threshold": "{:.2f}",
            "Pathological Recall": "{:.4f}",
            "Pathological Precision": "{:.4f}",
            "Pathological F1": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Accuracy": "{:.4f}",
            "MCC": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep_df["Threshold"], sweep_df["Pathological Recall"],
            marker="o", label="Pathological Recall", linewidth=2)
    ax.plot(sweep_df["Threshold"], sweep_df["Pathological Precision"],
            marker="s", label="Pathological Precision", linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Pathological Recall / Precision vs. Threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.warning(
    "**Flat null result.** Pathological Recall, Precision, and F1 are all "
    "0.0000 at every threshold tested — not a partial trade-off, a complete "
    "absence of detection at any setting in this range."
)

# ------------------------------------------------------------------
# Section 2 — Why: the diagnostic
# ------------------------------------------------------------------

st.header("2. Diagnosis — Why the Sweep Found Nothing")

st.markdown(
    f"""
Rather than stop at reporting a flat 0.0000, the model's actual confidence
on the 13 real Pathological cases (pooled across all 5 CV folds) was
checked directly.
"""
)

col3, col4 = st.columns([1, 1])

with col3:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(
        range(1, len(TRUE_PATHOLOGICAL_PROBS) + 1),
        sorted(TRUE_PATHOLOGICAL_PROBS, reverse=True),
        color=["#e74c3c" if p >= MIN_THRESHOLD_TESTED else "#95a5a6"
               for p in sorted(TRUE_PATHOLOGICAL_PROBS, reverse=True)],
    )
    ax2.axhline(MIN_THRESHOLD_TESTED, color="black", linestyle="--",
                linewidth=1.2, label=f"Lowest threshold tested ({MIN_THRESHOLD_TESTED})")
    ax2.set_xlabel("True Pathological case (ranked by model confidence)")
    ax2.set_ylabel("P(Pathological) assigned by model")
    ax2.set_title("Model confidence on the 13 real Pathological cases")
    ax2.legend()
    ax2.set_ylim(0, 0.15)
    st.pyplot(fig2)

with col4:
    st.metric(
        "Max confidence on a TRUE Pathological case",
        f"{MAX_TRUE_CONFIDENCE:.4f}",
        help="Across all 13 real Pathological records, pooled over 5 CV folds",
    )
    st.metric(
        "Lowest threshold tested",
        f"{MIN_THRESHOLD_TESTED:.2f}",
    )
    st.markdown(
        f"""
        The model's own maximum confidence on any real Pathological case
        (**{MAX_TRUE_CONFIDENCE:.4f}**) sits **below** the lowest threshold
        tested (**{MIN_THRESHOLD_TESTED:.2f}**). No threshold in the swept
        range could ever have caught these cases — the null result is a
        direct, explainable consequence of what the model learned, not a
        tuning failure.
        """
    )

st.info(
    "This corroborates the SHAP finding in `07_ctu_external_validation.ipynb`: "
    "Pathological's feature signal is 3–10x weaker than Normal/Suspect across "
    "every feature. Two independent methods — SHAP on the input side, this "
    "threshold sweep on the output side — reach the same conclusion."
)

# ------------------------------------------------------------------
# Section 3 — Comparative synthesis with Mamatha's Suspect work
# ------------------------------------------------------------------

st.header("3. Comparative Synthesis — Two Classes, Same Technique")

st.markdown(
    """
The same threshold-override technique was applied independently to two
different classes across two different datasets. Both concluded threshold
tuning does not improve detection — but for opposite underlying reasons.
"""
)

comparison_df = pd.DataFrame({
    "": ["Dataset", "Target class", "Test set size (class)",
         "Threshold range tested", "Why that range",
         "Result", "Underlying reason"],
    "Pathological (Shreya)": [
        "CTU-UHB",
        "Pathological",
        "13 records",
        "0.10 – 0.30",
        "Confidence ceiling (0.0748) is below even the lowest threshold — "
        "no reason to test higher",
        "Flat null (0.0000 Recall/Precision/F1 throughout)",
        "No learnable signal at all — model never learned this class",
    ],
    "Suspect (Mamatha)": [
        "UCI",
        "Suspect",
        "58 records",
        "0.30 – 0.70",
        "Well-separated probabilities justified sweeping around the "
        "standard 0.5 default",
        "No meaningful improvement; small increase in false positives",
        "Already near-optimal at default — nothing left to gain",
    ],
})

st.table(comparison_df.set_index(""))

st.success(
    "**Synthesis:** the technique itself behaves rationally in both cases — "
    "it simply has nothing to work with on Pathological, and nothing left "
    "to improve on Suspect. The bottleneck for Pathological detection is "
    "data volume (13 records), not model tuning or decision-rule choice."
)

st.caption(
    "Sources: 08_ctu_threshold_tuning.ipynb (Shreya), "
    "09_Threshold_Investigation.ipynb (Mamatha), "
    "07_ctu_external_validation.ipynb (SHAP corroboration, Nikita)"
)