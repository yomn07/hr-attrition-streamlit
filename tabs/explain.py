import numpy as np
import streamlit as st

from config import RANDOM_STATE
from explainability import plots
from explainability.shap_utils import (
    compute_pdp, compute_shap, explainer_kind, mean_abs, shap_frame,
)
from state import is_trained


def _render_compute(model_name, model, data):
    X_test = st.session_state.X_test
    kind = explainer_kind(model.named_steps["classifier"])
    rows = X_test

    if kind == "kernel":
        st.warning(
            "This model type has no fast SHAP method, so it explains a "
            "sample of employees and can take a minute or two."
        )
        n_rows = st.slider(
            "Employees to explain", 20, min(150, len(X_test)), 50, step=10,
            key="explain_rows",
        )
        rows = X_test.sample(n=n_rows, random_state=RANDOM_STATE)

    if st.button("Compute SHAP explanations", type="primary"):
        try:
            with st.spinner("Computing SHAP values..."):
                st.session_state.shap_cache[model_name] = compute_shap(
                    model, st.session_state.X_train, rows
                )
        except Exception as exc:
            st.error(f"SHAP failed: {exc}")

    result = st.session_state.shap_cache.get(model_name)
    if result is None:
        st.info("Click the button to compute explanations for this model.")
    return result


def _render_global(result, data):
    st.subheader("Global importance: what the model relies on")

    grouped = st.toggle(
        "Combine one-hot columns into the original feature", value=True
    )
    importance = mean_abs(shap_frame(
        result, data.numeric_features, data.categorical_features, grouped
    ))
    st.plotly_chart(
        plots.global_bar(importance, result["units"]), use_container_width=True
    )

    st.subheader("Direction of effect")
    ungrouped = shap_frame(
        result, data.numeric_features, data.categorical_features, False
    )
    st.plotly_chart(
        plots.beeswarm(ungrouped, result["X_transformed"]),
        use_container_width=True,
    )
    st.caption(
        "Each dot is an employee. Dots on the right push that employee's "
        "risk up; dots on the left push it down. Red means a high value "
        "of the feature, blue a low value."
    )


def _render_local(result, model, data):
    st.subheader("Local explanation: why this employee?")

    X_rows = result["X_original"]
    probabilities = model.predict_proba(X_rows)[:, 1]
    y_test = st.session_state.y_test

    order = np.argsort(-probabilities).tolist()  # highest risk first

    def describe(i):
        actual = "left" if y_test.loc[X_rows.index[i]] == 1 else "stayed"
        return (f"Row {X_rows.index[i]}: predicted {probabilities[i]:.0%}, "
                f"actually {actual}")

    position = st.selectbox(
        "Employee (sorted by predicted risk)", order, format_func=describe
    )

    st.plotly_chart(
        plots.local_bar(result, position, data.numeric_features),
        use_container_width=True,
    )

    total = float(result["values"][position].sum())
    st.caption(
        f"Base value {result['base']:.3f} + contributions {total:+.3f} "
        f"= {result['base'] + total:.3f} ({result['units']}). "
        "The base value is the model's average output."
    )


def _render_pdp(model, data):
    st.subheader("Partial dependence: how a feature changes risk")

    feature = st.selectbox(
        "Numeric feature", data.numeric_features, key="pdp_feature"
    )
    pdp = compute_pdp(model, st.session_state.X_test, feature)
    st.plotly_chart(plots.pdp_line(pdp, feature), use_container_width=True)
    st.caption(
        "Average predicted attrition probability as this feature changes "
        "while all other features stay as they are."
    )


def render(data):
    st.header("Explainable AI")

    if not is_trained():
        st.info("Train the models in the Model Lab tab first.")
        return

    if st.session_state.get("shap_cache") is None:
        st.session_state.shap_cache = {}

    with st.expander("How to read this tab"):
        st.markdown(
            "- **SHAP** splits a model's output into one contribution per "
            "feature.\n"
            "- **Positive** contribution: pushes toward *leaving*. "
            "**Negative**: pushes toward *staying*.\n"
            "- Units depend on the model: *probability* for Random Forest, "
            "Decision Tree, SVM and KNN, and *log-odds* for XGBoost, "
            "Gradient Boosting and Logistic Regression. The sign and "
            "ranking mean the same in both.\n"
            "- Explanations show what the **model** uses, not what "
            "**causes** attrition."
        )

    model_name = st.selectbox(
        "Model to explain",
        list(st.session_state.trained_models.keys()),
        key="explain_model",
    )
    model = st.session_state.trained_models[model_name]

    result = _render_compute(model_name, model, data)

    if result is not None:
        _render_global(result, data)
        _render_local(result, model, data)

    _render_pdp(model, data)
