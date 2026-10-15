import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

TREE_TYPES = (
    RandomForestClassifier,
    GradientBoostingClassifier,
    DecisionTreeClassifier,
    XGBClassifier,
)
LOG_ODDS_TYPES = (GradientBoostingClassifier, XGBClassifier, LogisticRegression)


def explainer_kind(classifier):
    """Pick the SHAP method that fits the model family."""
    if isinstance(classifier, TREE_TYPES):
        return "tree"
    if isinstance(classifier, LogisticRegression):
        return "linear"
    return "kernel"  # SVM, KNN: slow but works for any model


def output_units(classifier):
    """Boosting and logistic models explain log-odds; the rest explain probability."""
    return "log-odds" if isinstance(classifier, LOG_ODDS_TYPES) else "probability"


def _feature_names(model):
    names = model.named_steps["preprocessor"].get_feature_names_out()
    return [n.split("__", 1)[-1] for n in names]  # "numeric__Age" -> "Age"


def _transform(model, X):
    values = model.named_steps["preprocessor"].transform(X)
    return pd.DataFrame(np.asarray(values), columns=_feature_names(model), index=X.index)


def _class1(values):
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def _base_value(expected):
    expected = np.asarray(expected).ravel()
    return float(expected[-1])


def compute_shap(model, X_background, X_explain, kernel_nsamples=200):

    classifier = model.named_steps["classifier"]
    kind = explainer_kind(classifier)

    background = _transform(model, X_background)
    explained = _transform(model, X_explain)

    if kind == "tree":
        explainer = shap.TreeExplainer(classifier)
        raw = explainer.shap_values(explained.values, check_additivity=False)
    elif kind == "linear":
        explainer = shap.LinearExplainer(classifier, background.values)
        raw = explainer.shap_values(explained.values)
    else:
        summary = shap.kmeans(background.values, 20)  # 20 representative employees
        explainer = shap.KernelExplainer(
            lambda a: classifier.predict_proba(a)[:, 1], summary
        )
        raw = explainer.shap_values(
            explained.values, nsamples=kernel_nsamples, silent=True
        )

    return {
        "values": _class1(raw),
        "base": _base_value(explainer.expected_value),
        "features": list(explained.columns),
        "X_transformed": explained,
        "X_original": X_explain,
        "units": output_units(classifier),
        "kind": kind,
    }


def shap_frame(result, numeric_features, categorical_features, grouped):
    frame = pd.DataFrame(
        result["values"],
        columns=result["features"],
        index=result["X_original"].index,
    )
    if not grouped:
        return frame

    def original_name(name):
        if name in numeric_features:
            return name
        matches = [c for c in categorical_features if name.startswith(c + "_")]
        return max(matches, key=len) if matches else name

    return frame.T.groupby(original_name).sum().T


def mean_abs(shap_df):
    return shap_df.abs().mean().sort_values(ascending=False)


def compute_pdp(model, X, feature, grid_resolution=20):
    """Average predicted attrition probability as one numeric feature varies."""
    # partial_dependence writes decimal grid values into the column,
    # which fails on integer columns. Floats can hold both.
    X = X.copy()
    int_columns = X.select_dtypes(include="integer").columns
    X[int_columns] = X[int_columns].astype(float)

    result = partial_dependence(
        model, X, [feature], grid_resolution=grid_resolution, kind="average"
    )
    grid = result["grid_values"][0] if "grid_values" in result else result["values"][0]
    return pd.DataFrame({feature: grid, "Attrition probability": result["average"][0]})
