import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from data import load_raw_data, prepare_data
from sidebar import render_sidebar
from state import init_state
from tabs import overview, eda, model_lab, prediction, fairness

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")

st.title("📊 HR Employee Attrition Analytics")
st.caption(
    "Explore employee attrition, compare seven machine-learning models, "
    "estimate attrition risk, and evaluate potential business impact."
)

try:
    data = prepare_data(load_raw_data())
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

init_state()
settings = render_sidebar()

tab_overview, tab_eda, tab_models, tab_prediction, tab_fairness = st.tabs([
    "📋 Overview",
    "🔎 Explore Data",
    "🤖 Model Lab",
    "🎯 Prediction & Impact",
    "⚖️ Fairness",
])

with tab_overview:
    overview.render(data)

with tab_eda:
    eda.render(data)

with tab_models:
    model_lab.render(data, settings)

with tab_prediction:
    prediction.render(data)

with tab_fairness:
    fairness.render(data)

st.divider()
st.caption(
    "HR Attrition Analytics | IBM HR Attrition dataset | "
    "For educational and analytical use. Validate performance, privacy, "
    "legal compliance, and fairness before any real-world use."
)
