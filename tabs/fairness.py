import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_score, recall_score

from config import TARGET
from state import is_trained

FAIRNESS_METRICS = [
    "Actual attrition rate",
    "Predicted positive rate",
    "Recall / true positive rate",
    "False positive rate",
    "Precision",
]


def _group_metrics(group):
    """Compute the five subgroup metrics for one group."""
    actual = group[TARGET].astype(int)
    predicted = group["_predicted"].astype(int)

    positives = int(actual.sum())
    negatives = len(actual) - positives

    return {
        "Employees": len(group),
        "Actual attrition rate": actual.mean(),
        "Predicted positive rate": predicted.mean(),
        "Recall / true positive rate": (
            recall_score(actual, predicted, zero_division=0)
            if positives > 0 else np.nan
        ),
        "False positive rate": (
            ((predicted == 1) & (actual == 0)).sum() / negatives
            if negatives > 0 else np.nan
        ),
        "Precision": precision_score(actual, predicted, zero_division=0),
    }


def _build_fairness_table(model, subgroup_column, threshold):
    X_test = st.session_state.X_test

    test_df = X_test.copy()
    test_df[TARGET] = st.session_state.y_test.to_numpy()
    test_df["_group"] = test_df[subgroup_column].fillna("Missing").astype(str)

    probabilities = model.predict_proba(X_test)[:, 1]
    test_df["_predicted"] = (probabilities >= threshold).astype(int)

    rows = []
    for group_name, group in test_df.groupby("_group", dropna=False):
        rows.append({"Group": str(group_name), **_group_metrics(group)})

    return pd.DataFrame(rows)


def render(data):
    st.header("Fairness and subgroup evaluation")

    st.warning(
        "Subgroup performance comparisons are diagnostic, not proof of "
        "fairness. Protected characteristics and proxy variables require "
        "careful legal, ethical, and organizational review."
    )

    if not is_trained():
        st.info("Train the models in the Model Lab tab to enable evaluation.")
        return

    subgroup_options = [
        c for c in data.X.columns if data.df[c].nunique(dropna=True) <= 20
    ]
    if not subgroup_options:
        st.info("No suitable subgroup columns were found.")
        return

    model_name = st.selectbox(
        "Model for subgroup evaluation",
        options=list(st.session_state.trained_models.keys()),
        key="fairness_model",
    )
    subgroup_column = st.selectbox(
        "Group employees by", options=subgroup_options, key="fairness_group"
    )
    threshold = st.slider(
        "Positive prediction threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        key="fairness_threshold",
    )

    fairness_df = _build_fairness_table(
        st.session_state.trained_models[model_name], subgroup_column, threshold
    )

    st.subheader("Subgroup metrics")
    st.dataframe(fairness_df.round(3), use_container_width=True, hide_index=True)

    metric = st.selectbox("Metric to visualize", options=FAIRNESS_METRICS)

    fig = px.bar(
        fairness_df, x="Group", y=metric, hover_data=["Employees"],
        title=f"{metric} by {subgroup_column}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download subgroup evaluation",
        data=fairness_df.to_csv(index=False).encode("utf-8"),
        file_name="hr_attrition_fairness_metrics.csv",
        mime="text/csv",
    )
