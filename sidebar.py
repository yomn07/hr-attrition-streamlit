from dataclasses import dataclass

import streamlit as st

from config import TUNING_METHODS


@dataclass
class Settings:
    test_size: int
    cv_folds: int
    scoring: str
    tuning_method: str
    search_iterations: int


def render_sidebar() -> Settings:
    st.sidebar.header("⚙️ Settings")

    test_size = st.sidebar.slider(
        "Test set percentage", min_value=10, max_value=40, value=20, step=5
    )
    cv_folds = st.sidebar.selectbox(
        "Cross-validation folds", options=[3, 4, 5], index=2
    )
    scoring = st.sidebar.selectbox(
        "Model tuning metric",
        options=["roc_auc", "average_precision", "f1", "recall", "accuracy"],
        index=0,
    )

    st.sidebar.divider()
    tuning_method = st.sidebar.radio("Hyperparameter selection", options=TUNING_METHODS)

    search_iterations = st.sidebar.slider(
        "Randomized search iterations per model",
        min_value=3, max_value=20, value=8,
        disabled=(tuning_method != TUNING_METHODS[0]),
    )

    st.sidebar.caption("Changing the training settings requires retraining the models.")

    return Settings(test_size, cv_folds, scoring, tuning_method, search_iterations)
