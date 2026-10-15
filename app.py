import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from data import load_raw_data, prepare_data

st.set_page_config(page_title="HR Attrition Analytics", page_icon="📊", layout="wide")
st.title("📊 HR Employee Attrition Analytics")

try:
    data = prepare_data(load_raw_data())
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

st.write(data.df.shape, data.numeric_features[:3], data.categorical_features[:3])
