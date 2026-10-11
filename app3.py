
from __future__ import annotations

import warnings
from io import BytesIO
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


TARGET = "Attrition"

# These fields are constant, administrative, or identifiers rather than
# useful employee-level predictors.
DROP_FEATURES = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

SENSITIVE_FEATURES = [
    "Gender",
    "Age",
    "Department",
    "MaritalStatus",
    "JobLevel",
]

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Loading the IBM HR Attrition dataset...")
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}. "
            "Check that the CSV is committed to your Git repository."
        )
    return pd.read_csv(DATA_PATH)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column != TARGET and column not in DROP_FEATURES
    ]


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric = df[feature_columns].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        column for column in feature_columns if column not in numeric
    ]

    return numeric, categorical


def make_pipeline(
    X: pd.DataFrame,
    classifier: Any,
) -> Pipeline:
    numeric, categorical = prepare_features(X, X.columns.tolist())

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------
# Five-model tuning
# ---------------------------------------------------------------------

def model_search_spaces() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                random_state=42,
            ),
            {
                "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [None, 5, 10, 16],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", 0.7],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            {
                "classifier__n_estimators": [150, 250, 400],
                "classifier__max_depth": [None, 5, 10, 16],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", 0.7],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [75, 125, 200],
                "classifier__learning_rate": [0.025, 0.05, 0.1],
                "classifier__max_depth": [1, 2, 3],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(random_state=42),
            {
                "classifier__max_iter": [100, 175, 250],
                "classifier__learning_rate": [0.03, 0.07, 0.1],
                "classifier__max_leaf_nodes": [7, 15, 31],
                "classifier__l2_regularization": [0.0, 0.5, 1.0],
            },
        ),
    }


def train_and_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iterations: int,
    cv_folds: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    results = {}
    rows = []

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )

    spaces = model_search_spaces()

    progress = st.progress(0.0)
    status = st.empty()

    for index, (name, (classifier, params)) in enumerate(spaces.items()):
        status.write(f"**Tuning {name}** ({index + 1}/{len(spaces)})")

        pipeline = make_pipeline(X_train, classifier)

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=iterations,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )

        search.fit(X_train, y_train)

        results[name] = {
            "search": search,
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_auc": float(search.best_score_),
        }

        rows.append(
            {
                "Model": name,
                "Best CV ROC-AUC": search.best_score_,
                "Best parameters": str(search.best_params_),
            }
        )

        progress.progress((index + 1) / len(spaces))

    status.success("All five models have been tuned.")
    progress.empty()

    table = pd.DataFrame(rows).sort_values(
        "Best CV ROC-AUC", ascending=False
    )

    return results, table


def positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    if "Yes" not in classes:
        raise ValueError("The fitted model does not contain the Yes class.")

    return probabilities[:, classes.index("Yes")]


def metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    truth = (pd.Series(y_true).astype(str).to_numpy() == "Yes").astype(int)
    predicted = (probabilities >= threshold).astype(int)

    return {
        "ROC-AUC": roc_auc_score(truth, probabilities),
        "Accuracy": accuracy_score(truth, predicted),
        "Precision": precision_score(truth, predicted, zero_division=0),
        "Recall": recall_score(truth, predicted, zero_division=0),
        "F1": f1_score(truth, predicted, zero_division=0),
        "Brier score": brier_score_loss(truth, probabilities),
    }


# ---------------------------------------------------------------------
# Evaluation and diagnostics
# ---------------------------------------------------------------------

def plot_confusion(y_true, probabilities, threshold):
    truth = (pd.Series(y_true).astype(str).to_numpy() == "Yes").astype(int)
    predicted = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(truth, predicted, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted stay", "Predicted leave"],
        yticklabels=["Actual stay", "Actual leave"],
        ax=ax,
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual outcome")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def plot_calibration(y_true, probabilities):
    truth = (pd.Series(y_true).astype(str).to_numpy() == "Yes").astype(int)

    fraction_positive, mean_predicted = calibration_curve(
        truth,
        probabilities,
        n_bins=8,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted attrition probability")
    ax.set_ylabel("Observed attrition rate")
    ax.set_title("Calibration / reliability curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(model: Pipeline, top_n: int = 15):
    classifier = model.named_steps["classifier"]
    preprocess = model.named_steps["preprocess"]

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
    else:
        return None

    try:
        names = preprocess.get_feature_names_out()
    except Exception:
        names = [f"Feature {i + 1}" for i in range(len(importance))]

    table = pd.DataFrame(
        {"Feature": names, "Importance": importance}
    ).sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        data=table,
        x="Importance",
        y="Feature",
        ax=ax,
    )
    ax.set_title(f"Top {len(table)} model features")
    fig.tight_layout()
    return fig


def fairness_report(
    df: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    group_column: str,
) -> pd.DataFrame:
    actual = (pd.Series(y_true).astype(str).to_numpy() == "Yes").astype(int)
    predicted = (probabilities >= threshold).astype(int)

    groups = df[group_column].reset_index(drop=True)
    records = []

    for group in groups.dropna().unique():
        mask = (groups == group).to_numpy()
        group_actual = actual[mask]
        group_predicted = predicted[mask]

        positives = int(group_actual.sum())
        negatives = int(len(group_actual) - positives)

        tpr = (
            float(((group_predicted == 1) & (group_actual == 1)).sum() / positives)
            if positives else np.nan
        )
        fpr = (
            float(((group_predicted == 1) & (group_actual == 0)).sum() / negatives)
            if negatives else np.nan
        )

        records.append(
            {
                "Group": str(group),
                "Employees": int(mask.sum()),
                "Observed attrition rate": float(group_actual.mean()),
                "Flagged rate": float(group_predicted.mean()),
                "True positive rate": tpr,
                "False positive rate": fpr,
            }
        )

    report = pd.DataFrame(records)
    if report.empty:
        return report

    max_flagged = report["Flagged rate"].max()
    min_flagged = report["Flagged rate"].min()

    report["Flagged-rate gap vs. highest"] = max_flagged - report["Flagged rate"]
    report["Flagged-rate ratio vs. highest"] = np.where(
        max_flagged > 0,
        report["Flagged rate"] / max_flagged,
        np.nan,
    )

    report.attrs["tpr_gap"] = (
        report["True positive rate"].max()
        - report["True positive rate"].min()
    )
    report.attrs["fpr_gap"] = (
        report["False positive rate"].max()
        - report["False positive rate"].min()
    )
    report.attrs["flagged_rate_gap"] = max_flagged - min_flagged

    return report.sort_values("Employees", ascending=False)


# ---------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------

st.title("📊 HR Attrition Analytics")
st.caption(
    "Explore employee attrition, compare tuned models, estimate business "
    "impact, and inspect group-level fairness metrics."
)

st.info(
    "This app supports workforce planning and human review. A model score "
    "is uncertain, is not a causal explanation, and should not be used as "
    "the sole basis for employment decisions."
)

try:
    df = load_dataset()
except Exception as exc:
    st.error(str(exc))
    st.stop()

feature_columns = get_feature_columns(df)

if df[TARGET].nunique() != 2:
    st.error("Expected a binary Yes/No Attrition target.")
    st.stop()

# Target is encoded as strings for interpretable positive-class handling.
y = df[TARGET].astype(str)
X = df[feature_columns].copy()

# Remove constant columns, if any.
constant_columns = [
    col for col in X.columns if X[col].nunique(dropna=False) <= 1
]
X = X.drop(columns=constant_columns)
feature_columns = X.columns.tolist()

with st.sidebar:
    st.header("Dataset")
    st.write(f"**Rows:** {len(df):,}")
    st.write(f"**Columns:** {df.shape[1]:,}")
    st.write(f"**Model features:** {len(feature_columns):,}")
    st.write(f"**Missing cells:** {int(df.isna().sum().sum()):,}")

    st.divider()
    st.header("Model settings")

    test_size = st.slider(
        "Test-set proportion",
        min_value=0.15,
        max_value=0.35,
        value=0.25,
        step=0.05,
    )
    cv_folds = st.selectbox("Cross-validation folds", [3, 4, 5], index=0)
    iterations = st.select_slider(
        "Randomized tuning iterations per model",
        options=[3, 5, 8, 12],
        value=5,
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=10000,
        value=42,
        step=1,
    )

    st.caption(
        "More iterations can improve tuning but increase training time."
    )

tab_eda, tab_models, tab_predict, tab_impact, tab_fairness = st.tabs(
    [
        "🔎 EDA",
        "🧪 Model lab",
        "🎯 Prediction",
        "💼 Business impact",
        "⚖️ Fairness",
    ]
)

# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------

with tab_eda:
    st.header("Exploratory data analysis")

    attrition_rate = (y == "Yes").mean()
    total_attrition = int((y == "Yes").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{len(df):,}")
    c2.metric("Attrition cases", f"{total_attrition:,}")
    c3.metric("Observed attrition", f"{attrition_rate:.1%}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    left, right = st.columns(2)

    with left:
        st.subheader("Attrition distribution")
        counts = y.value_counts().reindex(["No", "Yes"], fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(
            x=counts.index,
            y=counts.values,
            ax=ax,
        )
        ax.set_xlabel("Attrition")
        ax.set_ylabel("Employees")
        ax.set_title("Stay vs. leave")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Missingness by column")
        missing = (
            df.isna().sum()
            .rename("Missing cells")
            .reset_index(names="Column")
        )
        missing = missing[missing["Missing cells"] > 0]
        if missing.empty:
            st.success("No missing cells found in the loaded dataset.")
        else:
            st.dataframe(
                missing.sort_values("Missing cells", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Attrition rate by category")
    category_candidates = [
        c for c in ["Department", "JobRole", "Gender", "MaritalStatus",
                    "OverTime", "BusinessTravel", "JobSatisfaction"]
        if c in df.columns
    ]

    if category_candidates:
        chosen_category = st.selectbox(
            "Choose a category",
            category_candidates,
            key="eda_category",
        )
        grouped = (
            df.assign(_attrition=(y == "Yes").astype(int))
            .groupby(chosen_category, dropna=False)["_attrition"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        grouped = grouped.rename(
            columns={
                "_attrition": "Attrition rate",
                "mean": "Attrition rate",
                "count": "Employees",
            }
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            data=grouped,
            x="Attrition rate",
            y=chosen_category,
            ax=ax,
        )
        ax.set_xlim(0, min(1, max(0.2, grouped["Attrition rate"].max() * 1.15)))
        ax.set_title(f"Observed attrition by {chosen_category}")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(grouped, use_container_width=True, hide_index=True)

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

    st.subheader("Numeric feature distributions")
    if numeric_columns:
        selected_numeric = st.selectbox(
            "Choose a numeric feature",
            numeric_columns,
            index=numeric_columns.index("Age") if "Age" in numeric_columns else 0,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=df,
            x=selected_numeric,
            hue=TARGET,
            multiple="layer",
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Numeric correlations")
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Preview dataset"):
        st.dataframe(df.head(50), use_container_width=True)

    st.download_button(
        "Download loaded dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ibm_hr_attrition.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Model lab
# ---------------------------------------------------------------------

with tab_models:
    st.header("Five-model training and calibration")
    st.write(
        "A stratified holdout set is kept separate from hyperparameter "
        "tuning. Models are ranked by cross-validated ROC-AUC."
    )

    st.warning(
        "The holdout set is used for evaluation and should not be repeatedly "
        "used to select a threshold or make model-development decisions."
    )

    if st.button("🚀 Tune all five models", type="primary", key="train_models"):
        with st.spinner("Splitting data and tuning models..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=float(test_size),
                stratify=y,
                random_state=int(seed),
            )

            model_results, tuning_table = train_and_tune(
                X_train,
                y_train,
                iterations=int(iterations),
                cv_folds=int(cv_folds),
            )

            ranked_names = sorted(
                model_results,
                key=lambda name: model_results[name]["cv_auc"],
                reverse=True,
            )
            best_name = ranked_names[0]
            best_model = model_results[best_name]["model"]

            # Calibration is fit using training data only. The test set
            # remains untouched until evaluation.
            calibrated_model = CalibratedClassifierCV(
                estimator=clone(best_model),
                method="sigmoid",
                cv=int(cv_folds),
            )
            calibrated_model.fit(X_train, y_train)

            raw_probabilities = positive_probabilities(best_model, X_test)
            calibrated_probabilities = positive_probabilities(
                calibrated_model, X_test
            )

            metrics = []
            for name in ranked_names:
                model = model_results[name]["model"]
                probabilities = positive_probabilities(model, X_test)
                row = {
                    "Model": name,
                    "CV ROC-AUC": model_results[name]["cv_auc"],
                    **metrics_at_threshold(y_test, probabilities, 0.5),
                }
                metrics.append(row)

            raw_metrics = metrics_at_threshold(
                y_test, raw_probabilities, 0.5
            )
            calibrated_metrics = metrics_at_threshold(
                y_test, calibrated_probabilities, 0.5
            )

            st.session_state["hr_results"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "models": model_results,
                "tuning_table": tuning_table,
                "best_name": best_name,
                "best_model": best_model,
                "calibrated_model": calibrated_model,
                "raw_probabilities": raw_probabilities,
                "calibrated_probabilities": calibrated_probabilities,
                "model_metrics": pd.DataFrame(metrics),
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
            }

    results = st.session_state.get("hr_results")

    if results is None:
        st.info(
            "Click **Tune all five models** to train, evaluate, and calibrate "
            "the classifiers. Training may take a few minutes."
        )
    else:
        st.success(f"Best cross-validated model: {results['best_name']}")

        st.subheader("Hyperparameter search results")
        st.dataframe(
            results["tuning_table"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Holdout performance")
        st.dataframe(
            results["model_metrics"].round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Calibration comparison")
        calibration_comparison = pd.DataFrame(
            [
                {
                    "Model": results["best_name"] + " (uncalibrated)",
                    **results["raw_metrics"],
                },
                {
                    "Model": results["best_name"] + " (calibrated)",
                    **results["calibrated_metrics"],
                },
            ]
        )
        st.dataframe(
            calibration_comparison.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Lower Brier score indicates better probabilistic accuracy. "
            "Calibration does not guarantee good discrimination or fairness."
        )

        st.subheader("Calibrated model diagnostics")
        left, right = st.columns(2)
        with left:
            st.pyplot(
                plot_confusion(
                    results["y_test"],
                    results["calibrated_probabilities"],
                    0.5,
                )
            )
        with right:
            st.pyplot(
                plot_calibration(
                    results["y_test"],
                    results["calibrated_probabilities"],
                )
            )

        st.subheader("Feature importance / coefficient magnitude")
        importance_fig = plot_feature_importance(results["best_model"])
        if importance_fig is not None:
            st.pyplot(importance_fig)
            plt.close(importance_fig)
        else:
            st.info(
                "This classifier does not expose native feature importances."
            )

        st.download_button(
            "Download holdout model metrics",
            data=results["model_metrics"].to_csv(index=False).encode("utf-8"),
            file_name="hr_model_metrics.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------
# Individual prediction
# ---------------------------------------------------------------------

with tab_predict:
    st.header("Employee attrition-risk estimate")
    st.write(
        "Choose an existing employee record as a starting point, then edit "
        "the available features. The score is a statistical estimate, not "
        "a certainty or a causal assessment."
    )

    results = st.session_state.get("hr_results")

    if results is None:
        st.info("Train the five models in the Model lab tab first.")
    else:
        model_choice = st.radio(
            "Prediction model",
            ["Calibrated best model", "Uncalibrated best model"],
            horizontal=True,
        )

        source_index = st.selectbox(
            "Starting employee record",
            options=list(range(len(df))),
            format_func=lambda i: (
                f"Record {i + 1}"
                + (
                    f" — Employee #{df.iloc[i]['EmployeeNumber']}"
                    if "EmployeeNumber" in df.columns else ""
                )
            ),
        )

        original = X.iloc[source_index].copy()
        edited: dict[str, Any] = {}

        with st.form("prediction_form"):
            st.subheader("Employee attributes")
            columns = st.columns(3)

            for i, column in enumerate(feature_columns):
                with columns[i % 3]:
                    series = X[column]

                    if pd.api.types.is_numeric_dtype(series):
                        median = float(series.median())
                        minimum = float(series.min())
                        maximum = float(series.max())
                        value = original[column]

                        if pd.isna(value):
                            value = median

                        if minimum == maximum:
                            edited[column] = value
                        else:
                            integer_like = (
                                pd.api.types.is_integer_dtype(series)
                                or np.all(
                                    np.isclose(
                                        series.dropna() % 1,
                                        0,
                                    )
                                )
                            )

                            if integer_like:
                                edited[column] = st.number_input(
                                    column,
                                    min_value=int(minimum),
                                    max_value=int(maximum),
                                    value=int(value),
                                    step=1,
                                )
                            else:
                                edited[column] = st.number_input(
                                    column,
                                    min_value=minimum,
                                    max_value=maximum,
                                    value=float(value),
                                    format="%.3f",
                                )
                    else:
                        choices = sorted(
                            series.dropna().astype(str).unique().tolist()
                        )
                        current = str(original[column])
                        if current not in choices and choices:
                            current = choices[0]

                        edited[column] = st.selectbox(
                            column,
                            choices,
                            index=choices.index(current) if choices else 0,
                        )

            submitted = st.form_submit_button(
                "Estimate attrition probability",
                type="primary",
            )

        if submitted:
            input_df = pd.DataFrame(
                [edited],
                columns=feature_columns,
            )

            if model_choice == "Calibrated best model":
                model = results["calibrated_model"]
            else:
                model = results["best_model"]

            probability = float(
                positive_probabilities(model, input_df)[0]
            )

            st.subheader("Prediction result")
            c1, c2 = st.columns([1, 2])

            with c1:
                st.metric("Estimated attrition probability", f"{probability:.1%}")

            with c2:
                st.progress(min(max(probability, 0.0), 1.0))
                st.write(
                    "This probability reflects the fitted model and the "
                    "historical dataset. It should not be interpreted as a "
                    "validated individual forecast."
                )

            if probability >= 0.5:
                st.warning(
                    "The model score is at or above the illustrative 50% "
                    "threshold. Review context with a qualified human."
                )
            else:
                st.success(
                    "The model score is below the illustrative 50% threshold. "
                    "A lower score does not mean attrition is impossible."
                )

            st.caption(
                "Do not use this score to deny employment opportunities, "
                "discipline employees, or make other consequential decisions "
                "without appropriate validation, governance, and review."
            )


# ---------------------------------------------------------------------
# Business impact scenario
# ---------------------------------------------------------------------

with tab_impact:
    st.header("Business impact scenario")
    st.write(
        "Estimate potential financial implications under user-defined "
        "assumptions. This is a scenario calculator, not a causal estimate "
        "of savings from deploying the model."
    )

    results = st.session_state.get("hr_results")

    col1, col2, col3 = st.columns(3)
    with col1:
        headcount = st.number_input(
            "Workforce size",
            min_value=1,
            max_value=10_000_000,
            value=int(len(df)),
            step=10,
        )
        annual_attrition = st.slider(
            "Expected annual attrition rate",
            min_value=0.0,
            max_value=1.0,
            value=float((y == "Yes").mean()),
            step=0.01,
        )
    with col2:
        replacement_cost = st.number_input(
            "Cost per departure",
            min_value=0.0,
            value=15000.0,
            step=1000.0,
            help="Enter costs in your chosen currency.",
        )
        intervention_cost = st.number_input(
            "Intervention cost per employee",
            min_value=0.0,
            value=500.0,
            step=50.0,
        )
    with col3:
        outreach_fraction = st.slider(
            "Share of workforce targeted",
            min_value=0.01,
            max_value=1.0,
            value=0.20,
            step=0.01,
        )
        effectiveness = st.slider(
            "Assumed attrition reduction among targeted leavers",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="A hypothetical assumption; it is not learned from this dataset.",
        )

    expected_departures = headcount * annual_attrition
    targeted_employees = int(round(headcount * outreach_fraction))
    expected_targeted_departures = expected_departures * outreach_fraction
    departures_avoided = expected_targeted_departures * effectiveness

    gross_avoided_cost = departures_avoided * replacement_cost
    intervention_spend = targeted_employees * intervention_cost
    net_impact = gross_avoided_cost - intervention_spend
    roi = (
        net_impact / intervention_spend
        if intervention_spend > 0 else np.nan
    )

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Expected annual departures", f"{expected_departures:,.0f}")
    b.metric("Targeted employees", f"{targeted_employees:,}")
    c.metric("Hypothetical avoided departures", f"{departures_avoided:,.1f}")
    d.metric("Estimated net impact", f"{net_impact:,.0f}")

    st.write(f"**Gross avoided replacement cost:** {gross_avoided_cost:,.2f}")
    st.write(f"**Intervention spend:** {intervention_spend:,.2f}")
    st.write(
        "**Scenario ROI:** "
        + (f"{roi:.1%}" if np.isfinite(roi) else "Not defined")
    )

    st.caption(
        "Currency is unspecified and should be consistent across all inputs. "
        "The calculation assumes targeted employees have the same baseline "
        "attrition rate as the workforce. Real intervention effectiveness "
        "requires prospective evaluation or a credible causal design."
    )

    if results is not None:
        st.subheader("Model-informed targeting illustration")
        threshold = st.slider(
            "Illustrative risk threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="impact_threshold",
        )
        probs = results["calibrated_probabilities"]
        holdout = results["X_test"].copy()
        flagged = int((probs >= threshold).sum())

        st.write(
            f"At this threshold, **{flagged:,} of {len(holdout):,}** holdout "
            "records would be flagged by the calibrated model."
        )
        st.caption(
            "This is a holdout-set count, not a projection of the number of "
            "employees to target in a future workforce."
        )


# ---------------------------------------------------------------------
# Fairness diagnostics
# ---------------------------------------------------------------------

with tab_fairness:
    st.header("Fairness and subgroup performance")
    st.write(
        "These diagnostics compare error rates and prediction rates across "
        "groups in the holdout data. They are descriptive checks, not proof "
        "that a model is fair or legally compliant."
    )

    results = st.session_state.get("hr_results")

    if results is None:
        st.info("Train the five models in the Model lab tab first.")
    else:
        fairness_threshold = st.slider(
            "Classification threshold for fairness checks",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="fairness_threshold",
        )

        available_groups = [
            column for column in SENSITIVE_FEATURES
            if column in df.columns
        ]

        if not available_groups:
            st.warning("No configured subgroup columns are available.")
        else:
            selected_group = st.selectbox(
                "Subgroup feature",
                available_groups,
            )

            test_indices = results["X_test"].index
            holdout_df = df.loc[test_indices].reset_index(drop=True)
            holdout_y = results["y_test"].reset_index(drop=True)
            holdout_probs = results["calibrated_probabilities"]

            report = fairness_report(
                holdout_df,
                holdout_y,
                holdout_probs,
                fairness_threshold,
                selected_group,
            )

            if report.empty:
                st.warning("No subgroup results could be calculated.")
            else:
                f1, f2, f3 = st.columns(3)
                f1.metric(
                    "Flagged-rate gap",
                    f"{report.attrs['flagged_rate_gap']:.1%}",
                )
                f2.metric(
                    "True-positive-rate gap",
                    (
                        f"{report.attrs['tpr_gap']:.1%}"
                        if np.isfinite(report.attrs["tpr_gap"])
                        else "Unavailable"
                    ),
                )
                f3.metric(
                    "False-positive-rate gap",
                    (
                        f"{report.attrs['fpr_gap']:.1%}"
                        if np.isfinite(report.attrs["fpr_gap"])
                        else "Unavailable"
                    ),
                )

                st.dataframe(
                    report.round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                fig, ax = plt.subplots(figsize=(9, 5))
                plot_data = report.melt(
                    id_vars="Group",
                    value_vars=[
                        "Observed attrition rate",
                        "Flagged rate",
                    ],
                    var_name="Measure",
                    value_name="Rate",
                )
                sns.barplot(
                    data=plot_data,
                    x="Group",
                    y="Rate",
                    hue="Measure",
                    ax=ax,
                )
                ax.set_ylim(0, 1)
                ax.set_title(f"Rates by {selected_group}")
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.download_button(
                    "Download fairness report",
                    data=report.to_csv(index=False).encode("utf-8"),
                    file_name=f"fairness_{selected_group.lower()}.csv",
                    mime="text/csv",
                )

                st.markdown(

                )

st.divider()
st.caption(
    "IBM HR Attrition Analytics • Educational decision-support prototype • "
    "Validate data provenance, model quality, security, privacy, and fairness "
    "before any real-world use."
)
