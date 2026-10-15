import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from data import load_raw_data, prepare_data
from sidebar import render_sidebar
from state import init_state, is_trained

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")
st.title("📊 HR Employee Attrition Analytics")

try:
    data = prepare_data(load_raw_data())
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

init_state()
settings = render_sidebar()

st.write("Settings:", settings)
st.write("Models trained?", is_trained())
