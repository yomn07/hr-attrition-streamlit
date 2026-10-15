import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_curve

from config import MODEL_NAMES
from evaluation import evaluate_models, safe_auc
from state import save_training_results
from training import split_data, train_and_tune

IMPORTANCE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Gradient Boosting",
    "Decision Tree",
]


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def _render_training(data, settings):
    st.markdown(
        "All models use the same stratified train/test split. "
        "Hyperparameters are selected through randomized cross-validation."
    )

    with st.expander("Models included", expanded=True):
        st.markdown("\n".join(f"- **{name}**" for name in MODEL_NAMES))

    if not st.button(
        "🚀 Train and tune all seven models",
        type="primary",
        use_container_width=True,
    ):
        return

    X_train, X_test, y_train, y_test = split_data(
        data.X, data.y, settings.test_size
    )

    try:
        with st.spinner("Training models. This may take a few minutes..."):
            trained_models, cv_results = train_and_tune(
                X_train, y_train,
                data.numeric_features, data.categorical_features,
                settings.cv_folds, settings.scoring, settings.search_iterations,
            )
            metrics, probabilities, predictions = evaluate_models(
                trained_models, X_test, y_test
            )

        save_training_results(
            trained_models, cv_results, metrics, probabilities, predictions,
            X_train, X_test, y_train, y_test,
        )
        st.success("All seven models were trained successfully.")

    except Exception as exc:
        st.error(f"Training failed: {exc}")


# ------------------------------------------------------------
# Metrics table and comparison chart
# ------------------------------------------------------------
def _render_metrics():
    metrics = st.session_state.metrics

    st.subheader("Test-set performance")
    st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

    metric_choice = st.selectbox(
        "Compare models using",
        options=[
            "ROC-AUC", "F1", "Recall", "Precision",
            "Accuracy", "Average Precision",
        ],
        index=0,
    )

    fig = px.bar(
        metrics.sort_values(metric_choice, ascending=True),
        x=metric_choice, y="Model", orientation="h",
        title=f"Model comparison: {metric_choice}",
        text_auto=".3f",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-validation search results")
    st.dataframe(
        st.session_state.cv_results,
        use_container_width=True, hide_index=True,
    )

    st.download_button(
        "Download model comparison",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="hr_attrition_model_comparison.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# ROC curves
# ------------------------------------------------------------
def _render_roc():
    st.subheader("ROC curves")

    y_test = st.session_state.y_test
    fig = go.Figure()

    for name, probs in st.session_state.probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = safe_auc(y_test, probs)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})"
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash"), name="Random baseline",
    ))
    fig.update_layout(
        title="ROC curve comparison",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend_title="Model",
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------
def _render_confusion():
    st.subheader("Confusion matrix")

    name = st.selectbox(
        "Select model for confusion matrix",
        options=list(st.session_state.trained_models.keys()),
    )

    cm = confusion_matrix(
        st.session_state.y_test,
        st.session_state.predictions[name],
        labels=[0, 1],
    )

    fig = px.imshow(
        cm, text_auto=True,
        x=["Predicted stayed", "Predicted left"],
        y=["Actually stayed", "Actually left"],
        labels={"x": "Prediction", "y": "Actual outcome", "color": "Employees"},
        title=f"Confusion matrix: {name}",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Built-in feature importance (SHAP will come later)
# ------------------------------------------------------------
def _render_importance():
    st.subheader("Feature importance")

    name = st.selectbox(
        "Choose model for feature importance",
        options=IMPORTANCE_MODELS,
        key="importance_model",
    )

    model = st.session_state.trained_models[name]
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    try:
        # "numeric__Age" -> "Age"
        feature_names = [
            f.split("__", 1)[-1] for f in preprocessor.get_feature_names_out()
        ]

        if hasattr(classifier, "feature_importances_"):
            values = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            values = np.abs(classifier.coef_[0])
        else:
            st.info("Feature importance is not available for this classifier.")
            return

        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": values})
            .sort_values("Importance", ascending=False)
            .head(20)
        )

        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title=f"Top features: {name}",
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.info(f"Could not calculate feature importance: {exc}")


# ------------------------------------------------------------
# Probability calibration
# ------------------------------------------------------------
def _build_calibrator(base_model, method):
    """Support both newer and older scikit-learn versions."""
    try:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, method=method, cv=3)


def _render_calibration():
    st.subheader("Probability calibration")
    st.write(
        "Calibration can make predicted probabilities more reliable. "
        "The calibrated model is fitted using training data only."
    )

    name = st.selectbox(
        "Choose model to calibrate",
        options=list(st.session_state.trained_models.keys()),
        key="calibration_model",
    )
    method = st.radio(
        "Calibration method", options=["sigmoid", "isotonic"], horizontal=True
    )

    if st.button("Calibrate selected model"):
        base_model = clone(st.session_state.trained_models[name])
        try:
            calibrated = _build_calibrator(base_model, method)
            with st.spinner("Calibrating probabilities..."):
                calibrated.fit(st.session_state.X_train, st.session_state.y_train)

            st.session_state.calibrated_model = calibrated
            st.session_state.calibrated_model_name = name
            st.success(f"{name} calibrated successfully.")
        except Exception as exc:
            st.error(f"Calibration failed: {exc}")

    if st.session_state.calibrated_model is None:
        return

    y_test = st.session_state.y_test
    calibrated_probs = st.session_state.calibrated_model.predict_proba(
        st.session_state.X_test
    )[:, 1]
    original_probs = st.session_state.probabilities[
        st.session_state.calibrated_model_name
    ]

    comparison = pd.DataFrame([
        {
            "Version": f"Original ({st.session_state.calibrated_model_name})",
            "ROC-AUC": safe_auc(y_test, original_probs),
            "Brier score": brier_score_loss(y_test, original_probs),
        },
        {
            "Version": "Calibrated",
            "ROC-AUC": safe_auc(y_test, calibrated_probs),
            "Brier score": brier_score_loss(y_test, calibrated_probs),
        },
    ])
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Lower Brier score is better. Calibration does not "
        "guarantee improved discrimination or fairness."
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def render(data, settings):
    st.header("Train and compare seven machine-learning models")

    _render_training(data, settings)

    if st.session_state.metrics is None:
        return

    _render_metrics()
    _render_roc()
    _render_confusion()
    _render_importance()
    _render_calibration()
