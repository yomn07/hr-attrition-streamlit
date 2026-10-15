import pandas as pd
import plotly.express as px
import streamlit as st

from config import TARGET


def render(data):
    df, X, y = data.df, data.X, data.y

    st.header("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees", f"{len(df):,}")
    col2.metric("Attrition cases", f"{int(y.sum()):,}")
    col3.metric("Attrition rate", f"{y.mean() * 100:.1f}%")
    col4.metric("Predictor columns", f"{X.shape[1]}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Target distribution")
    target_counts = (
        df[TARGET].map({0: "Stayed", 1: "Left"})
        .value_counts()
        .rename_axis("Employee outcome")
        .reset_index(name="Employees")
    )
    fig = px.pie(
        target_counts,
        names="Employee outcome",
        values="Employees",
        hole=0.45,
        title="Employee attrition distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data quality")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=False)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True)

    st.download_button(
        "Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ibm_hr_attrition_cleaned.csv",
        mime="text/csv",
    )
