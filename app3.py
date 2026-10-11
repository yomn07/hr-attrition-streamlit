
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Attrition Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 HR Employee Attrition Analytics")
st.caption(
    "Explore employee attrition, compare seven machine-learning models, "
    "estimate attrition risk, and evaluate potential business impact."
)

RANDOM_STATE = 42
TARGET = "Attrition"
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

DATA_URLS = [
    (
        "https://raw.githubusercontent.com/"
        "IBM/employee-attrition-aif360/master/data/"
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ),
    (
        "https://raw.githubusercontent.com/"
        "IBM/employee-attrition-aif360/main/data/"
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ),
]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading the IBM HR Attrition dataset...")
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        try:
            response = requests.get(DATA_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(BytesIO(response.content))
        except Exception as exc:
            raise RuntimeError(
                "Could not load the IBM HR Attrition dataset. "
                "Check your internet connection or place the CSV beside app.py."
            ) from exc

    if TARGET not in df.columns:
        raise ValueError(f"Dataset is missing the required column: {TARGET}")

    if not set(df[TARGET].dropna().astype(str).unique()).issubset({"Yes", "No"}):
        raise ValueError("Attrition must contain only Yes/No values.")

    return df

# ============================================================
# DATA PREPARATION
# ============================================================
try:
    df = load_data()
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.info(
        "The application downloads the dataset automatically, "
        "so an internet connection is required on first load."
    )
    st.stop()
# Remove columns that are identifiers or have only one unique value.
# This prevents the models from learning meaningless ID patterns.
drop_columns = []

for col in df.columns:
    if col == TARGET:
        continue

    if df[col].nunique(dropna=False) <= 1:
        drop_columns.append(col)

# EmployeeNumber is an identifier, not a meaningful predictor.
if "EmployeeNumber" in df.columns:
    drop_columns.append("EmployeeNumber")

df = df.drop(columns=list(set(drop_columns)), errors="ignore")

# Normalize the binary target.
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0, "1": 1, "0": 0})
)

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

test_size = st.sidebar.slider(
    "Test set percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

cv_folds = st.sidebar.selectbox(
    "Cross-validation folds",
    options=[3, 4, 5],
    index=2,
)

scoring = st.sidebar.selectbox(
    "Model tuning metric",
    options=[
        "roc_auc",
        "average_precision",
        "f1",
        "recall",
        "accuracy",
    ],
    index=0,
)

search_iterations = st.sidebar.slider(
    "Randomized search iterations per model",
    min_value=3,
    max_value=20,
    value=8,
)

st.sidebar.caption(
    "Changing the training settings requires retraining the models."
)


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor(scale_numeric=False):
    """
    Impute missing values and encode categorical features.
    Scale numeric features for distance- and margin-based models.
    """

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            make_one_hot_encoder(),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# seven MODEL DEFINITIONS
# ============================================================

def model_search_spaces():
    """
    Return the seven classifiers and their hyperparameter spaces.

    All preprocessing is inside each Pipeline so that the
    cross-validation folds do not share imputed or scaled data.
    """

    models = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        {
            "classifier__C": [
                0.01, 0.1, 0.5, 1.0, 3.0, 10.0
            ],
            "classifier__solver": ["lbfgs"],
        },
    )

    # 2. Random Forest
    models["Random Forest"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]),
        {
            "classifier__n_estimators": [
                100, 200, 300
            ],
            "classifier__max_depth": [
                None, 8, 12, 20
            ],
            "classifier__min_samples_leaf": [
                1, 2, 4
            ],
            "classifier__max_features": [
                "sqrt", 0.7
            ],
        },
    )
        # 7. XGBoost
    models["XGBoost"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]),
        {
            "classifier__n_estimators": [
                100, 200, 300
            ],
            "classifier__max_depth": [
                2, 3, 5, 7
            ],
            "classifier__learning_rate": [
                0.01, 0.03, 0.05, 0.1
            ],
            "classifier__subsample": [
                0.7, 0.85, 1.0
            ],
            "classifier__colsample_bytree": [
                0.7, 0.85, 1.0
            ],
            "classifier__min_child_weight": [
                1, 3, 5
            ],
            "classifier__reg_lambda": [
                1, 5, 10
            ],
        },
    )
    # 3. Gradient Boosting
    models["Gradient Boosting"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        {
            "classifier__n_estimators": [
                50, 100, 150, 200
            ],
            "classifier__learning_rate": [
                0.01, 0.03, 0.05, 0.1
            ],
            "classifier__max_depth": [
                1, 2, 3
            ],
            "classifier__min_samples_leaf": [
                1, 2, 4
            ],
            "classifier__subsample": [
                0.7, 0.85, 1.0
            ],
        },
    )

    # 4. Support Vector Machine
    models["Support Vector Machine"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        {
            "classifier__C": [
                0.1, 0.5, 1.0, 3.0, 10.0
            ],
            "classifier__kernel": [
                "rbf", "linear"
            ],
            "classifier__gamma": [
                "scale", "auto"
            ],
        },
    )

    # 5. K-Nearest Neighbors
    models["K-Nearest Neighbors"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                KNeighborsClassifier(
                    n_jobs=1,
                ),
            ),
        ]),
        {
            "classifier__n_neighbors": [
                3, 5, 7, 9, 13, 17
            ],
            "classifier__weights": [
                "uniform", "distance"
            ],
            "classifier__p": [1, 2],
        },
    )

    # 6. Decision Tree
    models["Decision Tree"] = (
        Pipeline([
            (
                "preprocessor",
                make_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]),
        {
            "classifier__max_depth": [
                None, 3, 5, 8, 12, 20
            ],
            "classifier__min_samples_split": [
                2, 5, 10, 20
            ],
            "classifier__min_samples_leaf": [
                1, 2, 4, 8
            ],
            "classifier__criterion": [
                "gini", "entropy"
            ],
        },
    )

    return models


# ============================================================
# TRAINING AND HYPERPARAMETER TUNING
# ============================================================

def safe_auc(y_true, probabilities):
    """Return ROC-AUC, or NaN if it cannot be calculated."""
    try:
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return np.nan


def train_and_tune(
    X_train,
    y_train,
    cv_folds,
    scoring,
    search_iterations,
):
    models = model_search_spaces()

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_models = {}
    cv_results = []

    progress = st.progress(0)
    status = st.empty()

    for index, (name, (pipeline, parameters)) in enumerate(
        models.items()
    ):
        status.write(
            f"Training model {index + 1}/{len(models)}: **{name}**"
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameters,
            n_iter=search_iterations,
            scoring=scoring,
            cv=cv,
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_models[name] = search.best_estimator_

        cv_results.append({
            "Model": name,
            "Best CV score": search.best_score_,
            "Best parameters": str(search.best_params_),
        })

        progress.progress(
            (index + 1) / len(models)
        )

    status.success("All seven models have been trained and tuned.")

    return best_models, pd.DataFrame(cv_results)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_models(models, X_test, y_test):
    rows = []
    probabilities = {}
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # Convert decision scores to a probability-like scale
        # for ROC-AUC only when predict_proba is unavailable.
        if not hasattr(model, "predict_proba"):
            from sklearn.preprocessing import MinMaxScaler
            y_prob = MinMaxScaler().fit_transform(
                np.asarray(y_prob).reshape(-1, 1)
            ).ravel()

        probabilities[name] = y_prob
        predictions[name] = y_pred

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "ROC-AUC": safe_auc(y_test, y_prob),
            "Average Precision": average_precision_score(
                y_test, y_prob
            ),
            "Brier Score": (
                brier_score_loss(y_test, y_prob)
                if np.min(y_prob) >= 0 and np.max(y_prob) <= 1
                else np.nan
            ),
        })

    metrics = pd.DataFrame(rows).sort_values(
        "ROC-AUC",
        ascending=False,
    )

    return metrics, probabilities, predictions


# ============================================================
# SESSION STATE
# ============================================================

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "cv_results" not in st.session_state:
    st.session_state.cv_results = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "X_train" not in st.session_state:
    st.session_state.X_train = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_train" not in st.session_state:
    st.session_state.y_train = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "calibrated_model" not in st.session_state:
    st.session_state.calibrated_model = None

if "calibrated_model_name" not in st.session_state:
    st.session_state.calibrated_model_name = None


# ============================================================
# NAVIGATION
# ============================================================

tab_overview, tab_eda, tab_models, tab_prediction, tab_fairness = (
    st.tabs([
        "📋 Overview",
        "🔎 Explore Data",
        "🤖 Model Lab",
        "🎯 Prediction & Impact",
        "⚖️ Fairness",
    ])
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:
    st.header("Dataset overview")

    total_employees = len(df)
    attrition_count = int(y.sum())
    attrition_rate = float(y.mean() * 100)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Employees", f"{total_employees:,}")
    col2.metric("Attrition cases", f"{attrition_count:,}")
    col3.metric("Attrition rate", f"{attrition_rate:.1f}%")
    col4.metric("Predictor columns", f"{X.shape[1]}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Target distribution")

    target_counts = (
        df[TARGET]
        .map({0: "Stayed", 1: "Left"})
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
        "Data type": [
            str(df[col].dtype) for col in df.columns
        ],
        "Missing values": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "Unique values": [
            int(df[col].nunique(dropna=False))
            for col in df.columns
        ],
    })

    st.dataframe(quality, use_container_width=True)

    st.download_button(
        "Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ibm_hr_attrition_cleaned.csv",
        mime="text/csv",
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

with tab_eda:
    st.header("Exploratory data analysis")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        selected_column = st.selectbox(
            "Choose a column",
            options=X.columns.tolist(),
        )

        if selected_column in numeric_features:
            fig = px.histogram(
                df,
                x=selected_column,
                color=TARGET,
                barmode="overlay",
                marginal="box",
                title=f"{selected_column} distribution by attrition",
                labels={TARGET: "Attrition (0 = No, 1 = Yes)"},
            )
        else:
            grouped = (
                df.groupby(
                    [selected_column, TARGET],
                    dropna=False,
                )
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                grouped,
                x=selected_column,
                y="Employees",
                color=TARGET,
                barmode="group",
                title=f"Attrition by {selected_column}",
            )

        st.plotly_chart(fig, use_container_width=True)

    with eda_col2:
        st.subheader("Attrition rate by category")

        categorical_options = [
            col for col in X.columns
            if df[col].nunique() <= 20
        ]

        if categorical_options:
            group_column = st.selectbox(
                "Group employees by",
                options=categorical_options,
            )

            rate_data = (
                df.groupby(group_column, dropna=False)[TARGET]
                .agg(["mean", "count"])
                .reset_index()
            )

            rate_data["Attrition rate (%)"] = (
                rate_data["mean"] * 100
            )

            fig = px.bar(
                rate_data,
                x=group_column,
                y="Attrition rate (%)",
                hover_data=["count"],
                title=f"Attrition rate by {group_column}",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No low-cardinality categorical columns found.")

    st.subheader("Numeric correlations")

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric feature correlation matrix",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attrition rate by numeric feature")

    if numeric_features:
        numeric_choice = st.selectbox(
            "Select a numeric feature",
            options=numeric_features,
            key="eda_numeric_choice",
        )

        box_data = df.copy()
        box_data["Employee outcome"] = box_data[TARGET].map({
            0: "Stayed",
            1: "Left",
        })

        fig = px.box(
            box_data,
            x="Employee outcome",
            y=numeric_choice,
            color="Employee outcome",
            points="outliers",
            title=f"{numeric_choice} by employee outcome",
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODEL LAB
# ============================================================

with tab_models:
    st.header("Train and compare seven machine-learning models")

    st.markdown(
        """
        All models use the same stratified train/test split.
        Hyperparameters are selected through randomized cross-validation.
        """
    )

    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Gradient Boosting",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Decision Tree",
    ]

    with st.expander("Models included", expanded=True):
        st.markdown(
            "\n".join(f"- **{name}**" for name in model_names)
        )

    if st.button(
        "🚀 Train and tune all seven models",
        type="primary",
        use_container_width=True,
    ):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        try:
            with st.spinner(
                "Training models. This may take a few minutes..."
            ):
                trained_models, cv_results = train_and_tune(
                    X_train,
                    y_train,
                    cv_folds,
                    scoring,
                    search_iterations,
                )

                metrics, probabilities, predictions = (
                    evaluate_models(
                        trained_models,
                        X_test,
                        y_test,
                    )
                )

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

            st.success("All seven models were trained successfully.")

        except Exception as exc:
            st.error(f"Training failed: {exc}")

    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics

        st.subheader("Test-set performance")

        display_metrics = metrics.copy()

        numeric_metric_cols = display_metrics.select_dtypes(
            include=["number"]
        ).columns

        display_metrics[numeric_metric_cols] = (
            display_metrics[numeric_metric_cols].round(4)
        )

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True,
        )

        metric_choice = st.selectbox(
            "Compare models using",
            options=[
                "ROC-AUC",
                "F1",
                "Recall",
                "Precision",
                "Accuracy",
                "Average Precision",
            ],
            index=0,
        )

        chart_metrics = metrics.sort_values(
            metric_choice,
            ascending=True,
        )

        fig = px.bar(
            chart_metrics,
            x=metric_choice,
            y="Model",
            orientation="h",
            title=f"Model comparison: {metric_choice}",
            text_auto=".3f",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cross-validation search results")
        st.dataframe(
            st.session_state.cv_results,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download model comparison",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="hr_attrition_model_comparison.csv",
            mime="text/csv",
        )

        st.subheader("ROC curves")

        y_test = st.session_state.y_test
        fig = go.Figure()

        for name, probabilities in (
            st.session_state.probabilities.items()
        ):
            fpr, tpr, _ = roc_curve(
                y_test,
                probabilities,
            )

            auc = safe_auc(y_test, probabilities)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={auc:.3f})",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Random baseline",
            )
        )

        fig.update_layout(
            title="ROC curve comparison",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend_title="Model",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion matrix")

        chosen_model_name = st.selectbox(
            "Select model for confusion matrix",
            options=list(
                st.session_state.trained_models.keys()
            ),
        )

        y_pred = st.session_state.predictions[
            chosen_model_name
        ]

        cm = confusion_matrix(
            st.session_state.y_test,
            y_pred,
            labels=[0, 1],
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted stayed", "Predicted left"],
            y=["Actually stayed", "Actually left"],
            labels={
                "x": "Prediction",
                "y": "Actual outcome",
                "color": "Employees",
            },
            title=f"Confusion matrix: {chosen_model_name}",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature importance")

        importance_name = st.selectbox(
            "Choose model for feature importance",
            options=[
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "Gradient Boosting",
                "Decision Tree",
            ],
            key="importance_model",
        )

        importance_model = (
            st.session_state.trained_models[
                importance_name
            ]
        )

        classifier = importance_model.named_steps["classifier"]
        preprocessor = importance_model.named_steps["preprocessor"]

        try:
            feature_names = preprocessor.get_feature_names_out()

            if hasattr(classifier, "feature_importances_"):
                importance_values = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importance_values = np.abs(
                    classifier.coef_[0]
                )
            else:
                importance_values = None

            if importance_values is not None:
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance_values,
                }).sort_values(
                    "Importance",
                    ascending=False,
                ).head(20)

                fig = px.bar(
                    importance_df.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title=f"Top features: {importance_name}",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )
            else:
                st.info(
                    "Feature importance is not directly available "
                    "for this classifier."
                )

        except Exception as exc:
            st.info(f"Could not calculate feature importance: {exc}")

        st.subheader("Probability calibration")

        st.write(
            "Calibration can make predicted probabilities more reliable. "
            "The calibrated model is fitted using training data only."
        )

        calibration_name = st.selectbox(
            "Choose model to calibrate",
            options=list(
                st.session_state.trained_models.keys()
            ),
            key="calibration_model",
        )

        calibration_method = st.radio(
            "Calibration method",
            options=["sigmoid", "isotonic"],
            horizontal=True,
        )

        if st.button("Calibrate selected model"):
            base_model = clone(
                st.session_state.trained_models[
                    calibration_name
                ]
            )

            try:
                try:
                    calibrated = CalibratedClassifierCV(
                        estimator=base_model,
                        method=calibration_method,
                        cv=3,
                    )
                except TypeError:
                    calibrated = CalibratedClassifierCV(
                        base_estimator=base_model,
                        method=calibration_method,
                        cv=3,
                    )

                with st.spinner("Calibrating probabilities..."):
                    calibrated.fit(
                        st.session_state.X_train,
                        st.session_state.y_train,
                    )

                st.session_state.calibrated_model = calibrated
                st.session_state.calibrated_model_name = (
                    calibration_name
                )

                st.success(
                    f"{calibration_name} calibrated successfully."
                )

            except Exception as exc:
                st.error(f"Calibration failed: {exc}")

        if st.session_state.calibrated_model is not None:
            calibrated = st.session_state.calibrated_model

            calibrated_probs = calibrated.predict_proba(
                st.session_state.X_test
            )[:, 1]

            original_probs = calibrated_name_probs = (
                st.session_state.probabilities[
                    st.session_state.calibrated_model_name
                ]
            )

            calibration_metrics = pd.DataFrame([
                {
                    "Version": "Original model",
                    "ROC-AUC": safe_auc(
                        st.session_state.y_test,
                        original_probs,
                    ),
                    "Brier score": brier_score_loss(
                        st.session_state.y_test,
                        original_probs,
                    ),
                },
                {
                    "Version": "Calibrated model",
                    "ROC-AUC": safe_auc(
                        st.session_state.y_test,
                        calibrated_probs,
                    ),
                    "Brier score": brier_score_loss(
                        st.session_state.y_test,
                        calibrated_probs,
                    ),
                },
            ])

            st.dataframe(
                calibration_metrics.round(4),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Lower Brier score is better. Calibration does not "
                "guarantee improved discrimination or fairness."
            )


# ============================================================
# PREDICTION AND BUSINESS IMPACT
# ============================================================

with tab_prediction:
    st.header("Employee attrition prediction")

    if st.session_state.trained_models is None:
        st.info(
            "Open the Model Lab tab and train the seven models first."
        )
    else:
        selected_model_name = st.selectbox(
            "Prediction model",
            options=list(
                st.session_state.trained_models.keys()
            ),
        )

        use_calibrated = False

        if st.session_state.calibrated_model is not None:
            use_calibrated = st.checkbox(
                (
                    "Use calibrated probabilities "
                    f"({st.session_state.calibrated_model_name})"
                ),
                value=(
                    selected_model_name
                    == st.session_state.calibrated_model_name
                ),
            )

        st.subheader("Enter employee characteristics")

        with st.form("employee_prediction_form"):
            input_values = {}

            cols = st.columns(3)

            for index, column in enumerate(X.columns):
                container = cols[index % 3]

                with container:
                    if column in categorical_features:
                        options = sorted(
                            X[column]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if not options:
                            input_values[column] = ""
                        else:
                            input_values[column] = st.selectbox(
                                column,
                                options=options,
                                key=f"prediction_{column}",
                            )

                    else:
                        series = pd.to_numeric(
                            X[column],
                            errors="coerce",
                        ).dropna()

                        default_value = (
                            float(series.median())
                            if not series.empty
                            else 0.0
                        )

                        min_value = (
                            float(series.min())
                            if not series.empty
                            else 0.0
                        )

                        max_value = (
                            float(series.max())
                            if not series.empty
                            else 100.0
                        )

                        if min_value == max_value:
                            input_values[column] = default_value
                            st.number_input(
                                column,
                                value=default_value,
                                disabled=True,
                                key=f"prediction_{column}",
                            )
                        else:
                            input_values[column] = st.number_input(
                                column,
                                min_value=min_value,
                                max_value=max_value,
                                value=default_value,
                                key=f"prediction_{column}",
                            )

            predict_clicked = st.form_submit_button(
                "Predict attrition risk",
                type="primary",
            )

        if predict_clicked:
            input_df = pd.DataFrame(
                [input_values],
                columns=X.columns,
            )

            # Restore the same numeric and categorical types.
            for col in numeric_features:
                input_df[col] = pd.to_numeric(
                    input_df[col],
                    errors="coerce",
                )

            for col in categorical_features:
                input_df[col] = input_df[col].astype(str)

            if (
                use_calibrated
                and st.session_state.calibrated_model is not None
                and selected_model_name
                == st.session_state.calibrated_model_name
            ):
                prediction_model = (
                    st.session_state.calibrated_model
                )
            else:
                prediction_model = (
                    st.session_state.trained_models[
                        selected_model_name
                    ]
                )

            probability = float(
                prediction_model.predict_proba(
                    input_df
                )[0, 1]
            )

            threshold = st.slider(
                "Attrition alert threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="prediction_threshold",
            )

            prediction = int(probability >= threshold)

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Estimated attrition probability",
                f"{probability:.1%}",
            )

            result_col2.metric(
                "Risk classification",
                "Higher risk" if prediction else "Lower risk",
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction:
                st.warning(
                    "This employee's estimated risk is above the "
                    "selected threshold. Treat this as a signal for "
                    "supportive review, not a decision about the employee."
                )
            else:
                st.success(
                    "This employee's estimated risk is below the "
                    "selected threshold. A low score does not mean "
                    "attrition is impossible."
                )

            st.caption(
                "This is a statistical estimate, not a certainty. "
                "Use predictions to guide voluntary, supportive "
                "retention conversations—not hiring, firing, promotion, "
                "or other consequential decisions."
            )

        st.divider()
        st.subheader("Business impact scenario")

        impact_col1, impact_col2, impact_col3 = st.columns(3)

        with impact_col1:
            workforce_size = st.number_input(
                "Employees in workforce",
                min_value=1,
                max_value=1_000_000,
                value=len(df),
            )

        with impact_col2:
            expected_attrition_rate = st.slider(
                "Expected attrition rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(round(y.mean() * 100, 1)),
                step=0.5,
            )

        with impact_col3:
            replacement_cost = st.number_input(
                "Estimated replacement cost per departure",
                min_value=0.0,
                value=15000.0,
                step=1000.0,
            )

        current_departures = (
            workforce_size * expected_attrition_rate / 100
        )

        st.metric(
            "Estimated annual replacement cost",
            f"{current_departures * replacement_cost:,.0f}",
        )

        retention_improvement = st.slider(
            "Hypothetical reduction in departures (%)",
            min_value=0,
            max_value=50,
            value=10,
            step=5,
        )

        avoided_departures = (
            current_departures
            * retention_improvement
            / 100
        )

        estimated_savings = (
            avoided_departures * replacement_cost
        )

        impact_col1, impact_col2 = st.columns(2)

        impact_col1.metric(
            "Hypothetical departures avoided",
            f"{avoided_departures:,.1f}",
        )

        impact_col2.metric(
            "Potential gross savings",
            f"{estimated_savings:,.0f}",
        )

        st.caption(
            "Scenario estimates only. They assume the selected reduction "
            "is achieved and exclude intervention costs, uncertainty, "
            "and other business effects."
        )


# ============================================================
# FAIRNESS CHECK
# ============================================================

with tab_fairness:
    st.header("Fairness and subgroup evaluation")

    st.warning(
        "Subgroup performance comparisons are diagnostic, not proof "
        "of fairness. Protected characteristics and proxy variables "
        "require careful legal, ethical, and organizational review."
    )

    if st.session_state.trained_models is None:
        st.info(
            "Train the models in the Model Lab tab to enable evaluation."
        )
    else:
        subgroup_options = [
            col for col in X.columns
            if df[col].nunique(dropna=True) <= 20
        ]

        if not subgroup_options:
            st.info("No suitable subgroup columns were found.")
        else:
            fairness_model_name = st.selectbox(
                "Model for subgroup evaluation",
                options=list(
                    st.session_state.trained_models.keys()
                ),
                key="fairness_model",
            )

            subgroup_column = st.selectbox(
                "Group employees by",
                options=subgroup_options,
                key="fairness_group",
            )

            fairness_threshold = st.slider(
                "Positive prediction threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                key="fairness_threshold",
            )

            fairness_test = st.session_state.X_test.copy()
            fairness_test[TARGET] = (
                st.session_state.y_test.to_numpy()
            )

            fairness_test["_group"] = (
                fairness_test[subgroup_column]
                .fillna("Missing")
                .astype(str)
                .values
            )

            fairness_model = (
                st.session_state.trained_models[
                    fairness_model_name
                ]
            )

            fairness_probs = fairness_model.predict_proba(
                st.session_state.X_test
            )[:, 1]

            fairness_test["_probability"] = fairness_probs
            fairness_test["_predicted"] = (
                fairness_probs >= fairness_threshold
            ).astype(int)

            fairness_rows = []

            for group_name, group in fairness_test.groupby(
                "_group",
                dropna=False,
            ):
                actual = group[TARGET].astype(int)
                predicted = group["_predicted"].astype(int)

                positives = int(actual.sum())
                negatives = int(len(actual) - positives)

                fairness_rows.append({
                    "Group": str(group_name),
                    "Employees": len(group),
                    "Actual attrition rate": actual.mean(),
                    "Predicted positive rate": predicted.mean(),
                    "Recall / true positive rate": (
                        recall_score(
                            actual,
                            predicted,
                            zero_division=0,
                        )
                        if positives > 0
                        else np.nan
                    ),
                    "False positive rate": (
                        (
                            (
                                (predicted == 1)
                                & (actual == 0)
                            ).sum()
                        ) / negatives
                        if negatives > 0
                        else np.nan
                    ),
                    "Precision": precision_score(
                        actual,
                        predicted,
                        zero_division=0,
                    ),
                })

            fairness_df = pd.DataFrame(fairness_rows)

            st.subheader("Subgroup metrics")

            st.dataframe(
                fairness_df.round(3),
                use_container_width=True,
                hide_index=True,
            )

            fairness_metric = st.selectbox(
                "Metric to visualize",
                options=[
                    "Actual attrition rate",
                    "Predicted positive rate",
                    "Recall / true positive rate",
                    "False positive rate",
                    "Precision",
                ],
            )

            fig = px.bar(
                fairness_df,
                x="Group",
                y=fairness_metric,
                hover_data=["Employees"],
                title=f"{fairness_metric} by {subgroup_column}",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.download_button(
                "Download subgroup evaluation",
                data=fairness_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="hr_attrition_fairness_metrics.csv",
                mime="text/csv",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Attrition Analytics | IBM HR Attrition dataset | "
    "For educational and analytical use. Validate performance, "
    "privacy, legal compliance, and fairness before any real-world use."
)
