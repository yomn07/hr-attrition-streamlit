import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from data import load_raw_data, prepare_data
from sidebar import render_sidebar
from state import init_state
from tabs import overview, eda, model_lab

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")
st.title("📊 HR Employee Attrition Analytics")

try:
    data = prepare_data(load_raw_data())
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

init_state()
settings = render_sidebar()

tab_overview, tab_eda, tab_models = st.tabs(
    ["📋 Overview", "🔎 Explore Data", "🤖 Model Lab"]
)

with tab_overview:
    overview.render(data)

with tab_eda:
    eda.render(data)

with tab_models:
    model_lab.render(data, settings)
