import plotly.express as px
import streamlit as st

from config import TARGET

OUTCOME = "Employee outcome"
COLORS = {"Stayed": "#1f77b4", "Left": "#d62728"}


def _render_class_distribution(df):
    st.subheader("Class distribution")

    stayed = int((df[TARGET] == 0).sum())
    left = int((df[TARGET] == 1).sum())
    total = stayed + left

    counts = {
        OUTCOME: ["Stayed", "Left"],
        "Employees": [stayed, left],
        "Label": [
            f"{stayed:,} ({stayed / total:.1%})",
            f"{left:,} ({left / total:.1%})",
        ],
    }

    fig = px.bar(
        counts, x=OUTCOME, y="Employees", color=OUTCOME, text="Label",
        color_discrete_map=COLORS,
        title="Employees who stayed vs. left",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"The classes are imbalanced: about {stayed / left:.1f} employees "
        "stayed for every 1 who left. This is why accuracy is a misleading "
        "metric here."
    )


def render(data):
    df, X = data.df, data.X
    numeric_features = data.numeric_features

    plot_df = df.copy()
    plot_df[OUTCOME] = plot_df[TARGET].map({0: "Stayed", 1: "Left"})

    st.header("Exploratory data analysis")

    _render_class_distribution(df)

    col_left, col_right = st.columns(2)

    with col_left:
        selected_column = st.selectbox("Choose a column", options=X.columns.tolist())

        if selected_column in numeric_features:
            as_percent = st.toggle(
                "Show % within each class",
                value=True,
                help="Counts hide the 'Left' group because it is much smaller. "
                     "Percent compares the shapes of the two distributions.",
            )
            fig = px.histogram(
                plot_df, x=selected_column, color=OUTCOME,
                barmode="overlay", marginal="box",
                histnorm="percent" if as_percent else None,
                color_discrete_map=COLORS,
                title=f"{selected_column} distribution by attrition",
            )
        else:
            grouped = (
                plot_df.groupby([selected_column, OUTCOME], dropna=False)
                .size().reset_index(name="Employees")
            )
            fig = px.bar(
                grouped, x=selected_column, y="Employees",
                color=OUTCOME, barmode="group",
                color_discrete_map=COLORS,
                title=f"Attrition by {selected_column}",
            )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Attrition rate by category")

        group_options = [c for c in X.columns if df[c].nunique() <= 20]

        if group_options:
            group_column = st.selectbox("Group employees by", options=group_options)

            rate_data = (
                df.groupby(group_column, dropna=False)[TARGET]
                .agg(["mean", "count"]).reset_index()
            )
            rate_data["Attrition rate (%)"] = rate_data["mean"] * 100

            fig = px.bar(
                rate_data, x=group_column, y="Attrition rate (%)",
                hover_data=["count"],
                title=f"Attrition rate by {group_column}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No low-cardinality columns found.")

    st.subheader("Numeric correlations")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        fig = px.imshow(
            numeric_df.corr(), text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Numeric feature correlation matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric feature by outcome")
    if numeric_features:
        numeric_choice = st.selectbox(
            "Select a numeric feature", options=numeric_features,
            key="eda_numeric_choice",
        )
        fig = px.box(
            plot_df, x=OUTCOME, y=numeric_choice, color=OUTCOME,
            points="outliers", color_discrete_map=COLORS,
            title=f"{numeric_choice} by employee outcome",
        )
        st.plotly_chart(fig, use_container_width=True)
