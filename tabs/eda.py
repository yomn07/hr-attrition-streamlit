import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from preprocessing import make_preprocessor
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

def _render_scatter(plot_df, numeric_features):
    st.subheader("Scatter plot (nuage de points)")

    if len(numeric_features) < 2:
        st.info("At least two numeric features are needed.")
        return

    col1, col2, col3 = st.columns(3)
    x_col = col1.selectbox(
        "X axis", numeric_features,
        index=numeric_features.index("Age") if "Age" in numeric_features else 0,
        key="scatter_x",
    )
    y_col = col2.selectbox(
        "Y axis", numeric_features,
        index=(numeric_features.index("MonthlyIncome")
               if "MonthlyIncome" in numeric_features else 1),
        key="scatter_y",
    )
    jitter = col3.slider(
        "Jitter", 0.0, 1.0, 0.3, step=0.1,
        help="Many columns hold whole numbers, so dots stack on top of each "
             "other. Jitter adds a small random shift to reveal them.",
    )

    data = plot_df[[x_col, y_col, OUTCOME]].copy()

    # Draw "Stayed" first so the rarer "Left" dots stay visible on top
    data = data.sort_values(OUTCOME, ascending=False)

    if jitter > 0:
        rng = np.random.default_rng(42)
        for col in (x_col, y_col):
            step = data[col].diff().abs().replace(0, np.nan).min()
            step = step if step == step else 1  # fall back if NaN
            data[col] = data[col] + rng.uniform(-jitter, jitter, len(data)) * step

    fig = px.scatter(
        data, x=x_col, y=y_col, color=OUTCOME,
        color_discrete_map=COLORS,
        category_orders={OUTCOME: ["Stayed", "Left"]},
        opacity=0.6, marginal_x="box", marginal_y="box",
        title=f"{y_col} vs {x_col}",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pca_scatter(data):
    st.subheader("All features in 2D (PCA)")

    preprocessor = make_preprocessor(
        data.numeric_features, data.categorical_features, scale_numeric=True
    )
    transformed = preprocessor.fit_transform(data.X)

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(transformed)
    explained = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame({
        "PC1": points[:, 0],
        "PC2": points[:, 1],
        OUTCOME: data.y.map({0: "Stayed", 1: "Left"}).to_numpy(),
    }).sort_values(OUTCOME, ascending=False)

    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color=OUTCOME,
        color_discrete_map=COLORS,
        category_orders={OUTCOME: ["Stayed", "Left"]},
        opacity=0.6,
        title=f"PCA projection ({explained:.0%} of the variance kept)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each dot is one employee, built from all features combined. If red "
        "and blue dots are mixed together, no simple boundary separates the "
        "classes, which is why the models top out around 0.80 to 0.83 AUC."
    )
def render(data):
    df, X = data.df, data.X
    numeric_features = data.numeric_features

    plot_df = df.copy()
    plot_df[OUTCOME] = plot_df[TARGET].map({0: "Stayed", 1: "Left"})

    st.header("Exploratory data analysis")

    _render_class_distribution(df)
    _render_scatter(plot_df, numeric_features)
    _render_pca_scatter(data)
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
