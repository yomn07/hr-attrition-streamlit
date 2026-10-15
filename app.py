import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from data import load_raw_data, prepare_data
from training import split_data, train_and_tune
from evaluation import evaluate_models

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")
st.title("📊 HR Employee Attrition Analytics")

try:
    data = prepare_data(load_raw_data())
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

if st.button("Quick test (3 iterations, 3 folds)"):
    X_train, X_test, y_train, y_test = split_data(data.X, data.y, 20)
    models, cv_results = train_and_tune(
        X_train, y_train,
        data.numeric_features, data.categorical_features,
        cv_folds=3, scoring="roc_auc", search_iterations=3,
    )
    metrics, _, _ = evaluate_models(models, X_test, y_test)
    st.dataframe(metrics.round(3), hide_index=True)
