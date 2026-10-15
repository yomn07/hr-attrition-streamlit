import pandas as pd
import streamlit as st
from explainability.plots import local_bar
from explainability.shap_utils import compute_shap
from state import is_trained

LAST_INPUT_KEY = "last_prediction_input"


# ------------------------------------------------------------
# Model choice
# ------------------------------------------------------------
def _get_prediction_model(selected_name):
    """Return (model, use_calibrated) based on the user's choice."""
    calibrated_name = st.session_state.calibrated_model_name
    use_calibrated = False

    if st.session_state.calibrated_model is not None:
        use_calibrated = st.checkbox(
            f"Use calibrated probabilities ({calibrated_name})",
            value=(selected_name == calibrated_name),
        )

    if use_calibrated and selected_name == calibrated_name:
        return st.session_state.calibrated_model
    return st.session_state.trained_models[selected_name]


# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
def _render_input_form(data):
    """Draw one input per feature. Returns (values, submitted)."""
    X = data.X
    values = {}

    with st.form("employee_prediction_form"):
        cols = st.columns(3)

        for index, column in enumerate(X.columns):
            with cols[index % 3]:
                if column in data.categorical_features:
                    options = sorted(X[column].dropna().astype(str).unique())
                    values[column] = st.selectbox(
                        column, options=options, key=f"prediction_{column}"
                    )
                else:
                    series = pd.to_numeric(X[column], errors="coerce").dropna()
                    default = float(series.median()) if not series.empty else 0.0
                    low = float(series.min()) if not series.empty else 0.0
                    high = float(series.max()) if not series.empty else 100.0

                    values[column] = st.number_input(
                        column,
                        min_value=low, max_value=high, value=default,
                        key=f"prediction_{column}",
                    )

        submitted = st.form_submit_button("Predict attrition risk", type="primary")

    return values, submitted


def _build_input_df(values, data):
    """Restore the same numeric and categorical types as the training data."""
    input_df = pd.DataFrame([values], columns=data.X.columns)

    for col in data.numeric_features:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    for col in data.categorical_features:
        input_df[col] = input_df[col].astype(str)

    return input_df


# ------------------------------------------------------------
# Result display
# ------------------------------------------------------------
def _render_result(model, input_df, threshold):
    probability = float(model.predict_proba(input_df)[0, 1])
    is_high_risk = probability >= threshold

    col1, col2 = st.columns(2)
    col1.metric("Estimated attrition probability", f"{probability:.1%}")
    col2.metric("Risk classification", "Higher risk" if is_high_risk else "Lower risk")

    st.progress(min(max(probability, 0.0), 1.0))

    if is_high_risk:
        st.warning(
            "This employee's estimated risk is above the selected threshold. "
            "Treat this as a signal for supportive review, not a decision "
            "about the employee."
        )
    else:
        st.success(
            "This employee's estimated risk is below the selected threshold. "
            "A low score does not mean attrition is impossible."
        )

    st.caption(
        "This is a statistical estimate, not a certainty. Use predictions "
        "to guide voluntary, supportive retention conversations, not "
        "hiring, firing, promotion, or other consequential decisions."
    )


# ------------------------------------------------------------
# Business impact
# ------------------------------------------------------------
def _render_business_impact(data):
    st.subheader("Business impact scenario")

    col1, col2, col3 = st.columns(3)

    with col1:
        workforce_size = st.number_input(
            "Employees in workforce",
            min_value=1, max_value=1_000_000, value=len(data.df),
        )
    with col2:
        attrition_rate = st.slider(
            "Expected attrition rate (%)",
            min_value=0.0, max_value=100.0,
            value=float(round(data.y.mean() * 100, 1)), step=0.5,
        )
    with col3:
        replacement_cost = st.number_input(
            "Estimated replacement cost per departure",
            min_value=0.0, value=15000.0, step=1000.0,
        )

    departures = workforce_size * attrition_rate / 100
    st.metric(
        "Estimated annual replacement cost",
        f"{departures * replacement_cost:,.0f}",
    )

    reduction = st.slider(
        "Hypothetical reduction in departures (%)",
        min_value=0, max_value=50, value=10, step=5,
    )

    avoided = departures * reduction / 100

    col1, col2 = st.columns(2)
    col1.metric("Hypothetical departures avoided", f"{avoided:,.1f}")
    col2.metric("Potential gross savings", f"{avoided * replacement_cost:,.0f}")

    st.caption(
        "Scenario estimates only. They assume the selected reduction is "
        "achieved and exclude intervention costs, uncertainty, and other "
        "business effects."
    )

def _render_explanation(model_name, input_df, data):
    st.subheader("Why this prediction?")

    if not st.toggle("Explain this prediction (SHAP)", key="explain_toggle"):
        return

    model = st.session_state.trained_models[model_name]
    cache_key = (model_name, input_df.to_json())
    cached = st.session_state.get("last_explanation")

    if cached is None or cached[0] != cache_key:
        with st.spinner("Computing SHAP values..."):
            result = compute_shap(model, st.session_state.X_train, input_df)
        st.session_state.last_explanation = (cache_key, result)
    else:
        result = cached[1]

    st.plotly_chart(
        local_bar(result, 0, data.numeric_features), use_container_width=True
    )
    st.caption(
        "Red bars raise this employee's risk, blue bars lower it. This "
        "explains the selected model's raw score, before any calibration."
    )
# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def render(data):
    st.header("Employee attrition prediction")

    if not is_trained():
        st.info("Open the Model Lab tab and train the seven models first.")
        return

    selected_name = st.selectbox(
        "Prediction model",
        options=list(st.session_state.trained_models.keys()),
    )
    model = _get_prediction_model(selected_name)

    threshold = st.slider(
        "Attrition alert threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        key="prediction_threshold",
    )

    st.subheader("Enter employee characteristics")
    values, submitted = _render_input_form(data)

    # Remember the last submitted employee so the result survives
    # reruns (for example when the threshold slider moves).
    if submitted:
        st.session_state[LAST_INPUT_KEY] = _build_input_df(values, data)

    if st.session_state.get(LAST_INPUT_KEY) is not None:
        _render_result(model, st.session_state[LAST_INPUT_KEY], threshold)
        _render_explanation(selected_name, st.session_state[LAST_INPUT_KEY], data)
    st.divider()
    _render_business_impact(data)
