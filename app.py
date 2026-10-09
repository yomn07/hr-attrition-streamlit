
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import hr_features  # Required by the saved model, if referenced during serialization

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "attrition_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files are missing. Check that artifacts/ contains "
        "attrition_model.joblib and model_metadata.json."
    )
    st.stop()

try:
    saved_model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


# =========================================================
# GENERAL HELPERS
# =========================================================

def get_expected_features(estimator):
    """Get the original input feature names from the fitted estimator."""
    features = getattr(estimator, "feature_names_in_", None)

    if features is None:
        return None

    return list(features)


def get_positive_class(classes):
    """Identify the attrition class in common binary encodings."""
    classes = list(classes)

    for cls in classes:
        if isinstance(cls, str) and cls.strip().lower() in {
            "yes", "true", "attrition"
        }:
            return cls

    for cls in classes:
        if not isinstance(cls, str) and cls == 1:
            return cls

    raise ValueError(
        "Cannot identify the attrition class. "
        f"Model classes are {classes}. Expected 1 or 'Yes'."
    )


def get_negative_class(classes, positive_class):
    remaining = [cls for cls in classes if cls != positive_class]

    if len(remaining) != 1:
        raise ValueError("This dashboard requires a binary classifier.")

    return remaining[0]


def normalize_target(y, classes):
    """Convert common Yes/No or 0/1 labels to the model's class labels."""
    y = pd.Series(y).reset_index(drop=True)
    classes = list(classes)

    if len(classes) != 2:
        raise ValueError("Only binary classification is supported.")

    positive = get_positive_class(classes)
    negative = get_negative_class(classes, positive)

    normalized = []

    for value in y:
        if pd.isna(value):
            raise ValueError("The Attrition column contains missing labels.")

        # First preserve exact matches to the fitted model's classes.
        if value in classes:
            normalized.append(value)
            continue

        label = str(value).strip().lower()

        if label in {"yes", "y", "true", "1", "1.0", "attrition"}:
            normalized.append(positive)
        elif label in {
            "no", "n", "false", "0", "0.0", "no attrition"
        }:
            normalized.append(negative)
        else:
            raise ValueError(
                f"Unrecognized Attrition label: {value!r}. "
                f"Model classes: {classes}."
            )

    return pd.Series(normalized)


def prepare_features(data, estimator):
    """Validate feature columns and preserve the training column order."""
    expected = get_expected_features(estimator)

    if not expected:
        raise ValueError(
            "The model does not expose feature_names_in_. "
            "The input feature schema must be configured explicitly."
        )

    missing = [column for column in expected if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[expected].copy()

    if X.empty:
        raise ValueError("The CSV contains no employee records.")

    if X.isna().any().any():
        missing_values = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Missing values were found in: {missing_values}. "
            "Correct them before continuing."
        )

    return X


def get_probabilities(estimator, X):
    """Return the positive-class probabilities and the class labels."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            "This model does not support predict_proba()."
        )

    classes = list(estimator.classes_)
    positive = get_positive_class(classes)
    negative = get_negative_class(classes, positive)

    probabilities = estimator.predict_proba(X)
    positive_index = classes.index(positive)

    return probabilities[:, positive_index], positive, negative


def evaluate_model(estimator, X, y, threshold):
    """Calculate metrics and predictions at a chosen threshold."""
    scores, positive, negative = get_probabilities(estimator, X)

    y_binary = (np.asarray(y) == positive).astype(int)
    predicted_binary = (scores >= threshold).astype(int)

    predictions = np.where(
        predicted_binary == 1, positive, negative
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[negative, positive],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Balanced accuracy": balanced_accuracy_score(y, predictions),
        "Precision": precision_score(
            y_binary, predicted_binary, zero_division=0
        ),
        "Recall": recall_score(
            y_binary, predicted_binary, zero_division=0
        ),
        "F1-score": f1_score(
            y_binary, predicted_binary, zero_division=0
        ),
        "Specificity": specificity,
        "ROC-AUC": roc_auc_score(y_binary, scores),
        "Average precision": average_precision_score(y_binary, scores),
    }

    return metrics, cm, scores, predictions, positive, negative


def plot_confusion_matrix(cm, negative, positive):
    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(negative), str(positive)],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    fig.tight_layout()

    return fig


def find_logistic_regression(estimator):
    """
    Find the LogisticRegression estimator and its parameter prefix.
    Supports a standalone LogisticRegression or a Pipeline.
    """
    if isinstance(estimator, LogisticRegression):
        return estimator, ""

    if not hasattr(estimator, "get_params"):
        return None, None

    params = estimator.get_params(deep=True)

    for name, value in params.items():
        if isinstance(value, LogisticRegression):
            prefix = name.rsplit("__", 1)[0]
            return value, prefix + "__"

    return None, None


st.title("📊 HR Attrition Analytics")
st.caption("IBM HR Analytics · Supervised Classification")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Selected model",
    metadata.get("selected_model", "Logistic Regression"),
)
col2.metric(
    "scikit-learn version",
    metadata.get("sklearn_version", "Unknown"),
)
col3.metric("Model type", "Supervised binary classification")

st.info(
    "Predictions are estimates, not certainties. This dashboard is for "
    "exploring model performance and workplace trends, not for making "
    "automated employment decisions."
)

# The saved model remains untouched. A tuned model is kept only for this
# running Streamlit session.
if "tuned_model" not in st.session_state:
    st.session_state.tuned_model = None

if "tuning_results" not in st.session_state:
    st.session_state.tuning_results = None


saved_or_tuned = st.radio(
    "Model to use for predictions and evaluation",
    options=["Saved model", "Tuned model"],
    horizontal=True,
    disabled=st.session_state.tuned_model is None,
)

if saved_or_tuned == "Tuned model" and st.session_state.tuned_model is not None:
    active_model = st.session_state.tuned_model
else:
    active_model = saved_model


tab_predict, tab_eval, tab_tune = st.tabs(
    [
        "🔮 Predictions",
        "📈 Evaluation & Curves",
        "⚙️ Hyperparameter Tuning",
    ]
)

with tab_predict:
    st.header("Batch Attrition Prediction")

    st.write(
        "Upload a CSV containing the same predictor columns used during "
        "training. The Attrition column is optional here."
    )

    prediction_file = st.file_uploader(
        "Upload employee data",
        type=["csv"],
        key="prediction_file",
    )

    prediction_threshold = st.slider(
        "Attrition classification threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="prediction_threshold",
        help=(
            "An employee is predicted to have attrition when the model's "
            "estimated attrition probability is at least this threshold."
        ),
    )

    if prediction_file is not None:
        try:
            prediction_data = pd.read_csv(prediction_file)
            X_prediction = prepare_features(
                prediction_data, active_model
            )

            st.write(f"Records loaded: **{len(X_prediction):,}**")

            if st.button(
                "Generate predictions",
                type="primary",
                key="predict_button",
            ):
                scores, positive, negative = get_probabilities(
                    active_model, X_prediction
                )

                predicted_classes = np.where(
                    scores >= prediction_threshold,
                    positive,
                    negative,
                )

                results = prediction_data.copy()
                results["Attrition_Probability"] = scores
                results["Predicted_Attrition"] = [
                    "Yes" if cls == positive else "No"
                    for cls in predicted_classes
                ]
                results["Classification_Threshold"] = (
                    prediction_threshold
                )

                st.session_state.prediction_results = results

        except Exception as exc:
            st.error(f"Could not prepare predictions: {exc}")

    if "prediction_results" in st.session_state:
        results = st.session_state.prediction_results

        st.subheader("Prediction results")

        yes_count = (
            results["Predicted_Attrition"] == "Yes"
        ).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Employees scored", f"{len(results):,}")
        c2.metric("Predicted attrition", f"{yes_count:,}")
        c3.metric(
            "Predicted attrition rate",
            f"{yes_count / len(results):.1%}" if len(results) else "N/A",
        )

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="attrition_predictions.csv",
            mime="text/csv",
            key="download_predictions",
        )


# =========================================================
# TAB 2: EVALUATION AND CURVES
# =========================================================

with tab_eval:
    st.header("Model Evaluation")

    st.write(
        "Upload a labeled evaluation CSV. It must contain the actual "
        "Attrition column and the model's required predictor columns."
    )

    st.warning(
        "Use held-out validation or test data that was not used to train "
        "the selected model. Metrics computed on training data can be "
        "overly optimistic."
    )

    evaluation_file = st.file_uploader(
        "Upload labeled evaluation CSV",
        type=["csv"],
        key="evaluation_file",
    )

    evaluation_threshold = st.slider(
        "Evaluation threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="evaluation_threshold",
    )

    if evaluation_file is not None:
        try:
            evaluation_data = pd.read_csv(evaluation_file)

            if "Attrition" not in evaluation_data.columns:
                raise ValueError(
                    "The evaluation CSV must contain an Attrition column."
                )

            X_eval = prepare_features(
                evaluation_data.drop(columns=["Attrition"]),
                active_model,
            )

            y_eval = normalize_target(
                evaluation_data["Attrition"],
                active_model.classes_,
            )

            if y_eval.nunique() != 2:
                raise ValueError(
                    "The evaluation data must contain both target classes."
                )

            (
                metrics,
                cm,
                scores,
                predictions,
                positive,
                negative,
            ) = evaluate_model(
                active_model,
                X_eval,
                y_eval,
                evaluation_threshold,
            )

            st.subheader("Evaluation metrics")

            metric_cols = st.columns(4)

            metric_names = [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "Specificity",
                "Balanced accuracy",
                "ROC-AUC",
                "Average precision",
            ]

            for index, name in enumerate(metric_names):
                metric_cols[index % 4].metric(
                    name,
                    f"{metrics[name]:.3f}",
                )

            st.caption(
                f"Metrics use threshold {evaluation_threshold:.2f}. "
                "ROC-AUC and average precision use probabilities, so "
                "they do not depend on the selected classification threshold."
            )

            st.subheader("Confusion Matrix")

            fig_cm = plot_confusion_matrix(
                cm, negative, positive
            )
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            tn, fp, fn, tp = cm.ravel()

            a, b, c, d = st.columns(4)
            a.metric("True negatives", f"{tn:,}")
            b.metric("False positives", f"{fp:,}")
            c.metric("False negatives", f"{fn:,}")
            d.metric("True positives", f"{tp:,}")

            st.subheader("ROC Curve")

            y_binary = (
                np.asarray(y_eval) == positive
            ).astype(int)

            fpr, tpr, _ = roc_curve(y_binary, scores)
            auc_value = roc_auc_score(y_binary, scores)

            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            ax_roc.plot(
                fpr,
                tpr,
                label=f"Model (AUC = {auc_value:.3f})",
            )
            ax_roc.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random classifier",
            )
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic")
            ax_roc.legend()
            ax_roc.grid(alpha=0.25)
            fig_roc.tight_layout()

            st.pyplot(fig_roc)
            plt.close(fig_roc)

            st.subheader("Precision-Recall Curve")

            precision_values, recall_values, _ = (
                precision_recall_curve(y_binary, scores)
            )

            fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
            ax_pr.plot(recall_values, precision_values)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title(
                f"Precision-Recall Curve "
                f"(Average Precision = {metrics['Average precision']:.3f})"
            )
            ax_pr.grid(alpha=0.25)
            fig_pr.tight_layout()

            st.pyplot(fig_pr)
            plt.close(fig_pr)

            st.subheader("Classification Report")

            report = classification_report(
                y_eval,
                predictions,
                labels=[negative, positive],
                target_names=[
                    f"No attrition ({negative})",
                    f"Attrition ({positive})",
                ],
                output_dict=True,
                zero_division=0,
            )

            st.dataframe(
                pd.DataFrame(report).transpose().round(3),
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Could not evaluate the model: {exc}")


# =========================================================
# TAB 3: HYPERPARAMETER TUNING
# =========================================================

with tab_tune:
    st.header("Logistic Regression Hyperparameter Tuning")

    st.write(
        "Upload a labeled TRAINING CSV to tune the Logistic Regression "
        "model using stratified cross-validation. The original saved "
        "model artifact will not be overwritten."
    )

    tuning_file = st.file_uploader(
        "Upload labeled training CSV",
        type=["csv"],
        key="tuning_file",
    )

    st.markdown("### Search configuration")

    col_cv, col_scoring = st.columns(2)

    with col_cv:
        cv_folds = st.selectbox(
            "Cross-validation folds",
            options=[3, 5],
            index=0,
            help="Every fold must contain both target classes.",
        )

    with col_scoring:
        tuning_scoring = st.selectbox(
            "Selection metric",
            options=["roc_auc", "f1", "balanced_accuracy"],
            index=0,
        )

    st.caption(
        "The search tests different C and class-weight settings. "
        "C controls regularization strength: smaller values apply "
        "stronger regularization."
    )

    if tuning_file is not None:
        try:
            tuning_data = pd.read_csv(tuning_file)

            if "Attrition" not in tuning_data.columns:
                raise ValueError(
                    "The training CSV must contain an Attrition column."
                )

            X_train = prepare_features(
                tuning_data.drop(columns=["Attrition"]),
                saved_model,
            )

            y_train = normalize_target(
                tuning_data["Attrition"],
                saved_model.classes_,
            )

            if y_train.nunique() != 2:
                raise ValueError(
                    "Training data must contain both target classes."
                )

            class_counts = y_train.value_counts()

            if class_counts.min() < cv_folds:
                raise ValueError(
                    f"The smallest class has {class_counts.min()} rows, "
                    f"but {cv_folds}-fold cross-validation was selected. "
                    "Use fewer folds or provide more training examples."
                )

            st.write(f"Training rows: **{len(X_train):,}**")
            st.write("Target distribution:")
            st.dataframe(
                class_counts.rename("Count").to_frame(),
                use_container_width=True,
            )

            logistic_regression, prefix = find_logistic_regression(
                saved_model
            )

            if logistic_regression is None:
                st.error(
                    "Could not find a LogisticRegression estimator in "
                    "the saved model. This tuning section is designed "
                    "for your Logistic Regression pipeline."
                )
            else:
                params = logistic_regression.get_params()

                param_grid = {
                    f"{prefix}C": [0.01, 0.1, 1.0, 10.0, 100.0],
                }

                if "class_weight" in params:
                    param_grid[f"{prefix}class_weight"] = [
                        None, "balanced"
                    ]

                if st.button(
                    "Run cross-validated hyperparameter search",
                    type="primary",
                    key="tune_button",
                ):
                    try:
                        with st.spinner(
                            "Training candidate models with "
                            "stratified cross-validation..."
                        ):
                            cv = StratifiedKFold(
                                n_splits=cv_folds,
                                shuffle=True,
                                random_state=42,
                            )

                            search = GridSearchCV(
                                estimator=clone(saved_model),
                                param_grid=param_grid,
                                scoring=tuning_scoring,
                                cv=cv,
                                refit=True,
                                n_jobs=-1,
                                error_score="raise",
                                return_train_score=True,
                            )

                            search.fit(X_train, y_train)

                        st.session_state.tuned_model = (
                            search.best_estimator_
                        )

                        results = pd.DataFrame(
                            search.cv_results_
                        ).sort_values(
                            "rank_test_score"
                        )

                        st.session_state.tuning_results = results

                        st.success(
                            "Tuning completed. Select 'Tuned model' "
                            "at the top of the page to use this model."
                        )

                    except Exception as exc:
                        st.error(f"Hyperparameter search failed: {exc}")

        except Exception as exc:
            st.error(f"Could not prepare training data: {exc}")

    if st.session_state.tuning_results is not None:
        results = st.session_state.tuning_results

        st.subheader("Best configuration")

        best_row = results.iloc[0]

        st.json(
            {
                "Best parameters": best_row["params"],
                "Mean cross-validation score": float(
                    best_row["mean_test_score"]
                ),
                "Scoring metric": tuning_scoring,
            }
        )

        st.subheader("All tested configurations")

        columns_to_show = [
            "rank_test_score",
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
        ]

        available_columns = [
            column for column in columns_to_show
            if column in results.columns
        ]

        st.dataframe(
            results[available_columns].rename(
                columns={
                    "rank_test_score": "Rank",
                    "params": "Parameters",
                    "mean_test_score": "Mean CV score",
                    "std_test_score": "CV score std",
                    "mean_train_score": "Mean training score",
                }
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download tuning results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="logistic_regression_tuning_results.csv",
            mime="text/csv",
            key="download_tuning",
        )

        if st.button(
            "Clear tuned model and use the saved model",
            key="clear_tuned_model",
        ):
            st.session_state.tuned_model = None
            st.session_state.tuning_results = None
            st.rerun()


st.divider()
st.caption(
    "HR Attrition Analytics · Model estimates should support "
    "responsible analysis, not automated employment decisions."
)
