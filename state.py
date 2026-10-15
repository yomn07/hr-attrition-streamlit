import streamlit as st

# Everything the Model Lab produces, stored in one place.
STATE_DEFAULTS = {
    "trained_models": None,
    "cv_results": None,
    "metrics": None,
    "probabilities": None,
    "predictions": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "calibrated_model": None,
    "calibrated_model_name": None,
    "shap_cache": None,
}


def init_state():
    """Create any missing session_state keys without overwriting existing ones."""
    for key, default in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def save_training_results(
    trained_models, cv_results, metrics, probabilities, predictions,
    X_train, X_test, y_train, y_test,
):
    """Store a fresh training run and clear the old calibrated model."""
    st.session_state.trained_models = trained_models
    st.session_state.cv_results = cv_results
    st.session_state.metrics = metrics
    st.session_state.probabilities = probabilities
    st.session_state.predictions = predictions
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.calibrated_model = None
    st.session_state.calibrated_model_name = None
    st.session_state.shap_cache = {}


def is_trained() -> bool:
    return st.session_state.trained_models is not None
