from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE
from preprocessing import make_preprocessor


def _pipeline(classifier, numeric_features, categorical_features, scale=False):
    """Preprocessing lives inside the pipeline so CV folds never leak."""
    return Pipeline([
        ("preprocessor", make_preprocessor(
            numeric_features, categorical_features, scale_numeric=scale
        )),
        ("classifier", classifier),
    ])


def model_search_spaces(numeric_features, categorical_features):
    """Return {name: (pipeline, hyperparameter_space)} for the seven models."""
    nf, cf = numeric_features, categorical_features
    models = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = (
        _pipeline(
            LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            nf, cf, scale=True,
        ),
        {
            "classifier__C": [0.01, 0.1, 0.5, 1.0, 3.0, 10.0],
            "classifier__solver": ["lbfgs"],
        },
    )

    # 2. Random Forest
    models["Random Forest"] = (
        _pipeline(
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=1
            ),
            nf, cf,
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 8, 12, 20],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.7],
        },
    )

    # 3. XGBoost
    models["XGBoost"] = (
        _pipeline(
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                objective="binary:logistic",
                n_jobs=1,
                verbosity=0,
            ),
            nf, cf,
        ),
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [2, 3, 5, 7],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__reg_lambda": [1, 5, 10],
        },
    )

    # 4. Gradient Boosting
    models["Gradient Boosting"] = (
        _pipeline(
            GradientBoostingClassifier(random_state=RANDOM_STATE), nf, cf
        ),
        {
            "classifier__n_estimators": [50, 100, 150, 200],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "classifier__max_depth": [1, 2, 3],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__subsample": [0.7, 0.85, 1.0],
        },
    )

    # 5. Support Vector Machine
    models["Support Vector Machine"] = (
        _pipeline(
            SVC(probability=True, class_weight="balanced",
                random_state=RANDOM_STATE),
            nf, cf, scale=True,
        ),
        {
            "classifier__C": [0.1, 0.5, 1.0, 3.0, 10.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"],
        },
    )

    # 6. K-Nearest Neighbors
    models["K-Nearest Neighbors"] = (
        _pipeline(KNeighborsClassifier(n_jobs=1), nf, cf, scale=True),
        {
            "classifier__n_neighbors": [3, 5, 7, 9, 13, 17],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    )

    # 7. Decision Tree
    models["Decision Tree"] = (
        _pipeline(
            DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight="balanced"
            ),
            nf, cf,
        ),
        {
            "classifier__max_depth": [None, 3, 5, 8, 12, 20],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__criterion": ["gini", "entropy"],
        },
    )

    return models
