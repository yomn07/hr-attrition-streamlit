import math

import pandas as pd
import streamlit as st
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from config import RANDOM_STATE, TUNING_METHODS
from models import model_search_spaces

RANDOM, GRID, MANUAL = TUNING_METHODS


def split_data(X, y, test_size_percent):
    return train_test_split(
        X, y,
        test_size=test_size_percent / 100,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def space_size(parameters):
    """Number of combinations in a hyperparameter grid."""
    return math.prod(len(v) for v in parameters.values())


def clean_params(params):
    """{'classifier__C': 1.0} -> {'C': 1.0}"""
    return {k.replace("classifier__", ""): v for k, v in params.items()}


def format_params(params):
    return ", ".join(f"{k}={v}" for k, v in clean_params(params).items())


def train_and_tune(
    X_train, y_train,
    numeric_features, categorical_features,
    cv_folds, scoring, method,
    search_iterations=8, manual_params=None,
    selected_models=None,
):
    all_models = model_search_spaces(numeric_features, categorical_features)
    models = {
        name: spec for name, spec in all_models.items()
        if selected_models is None or name in selected_models
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    best_models, rows = {}, []
    progress = st.progress(0)
    status = st.empty()

    for index, (name, (pipeline, parameters)) in enumerate(models.items()):
        status.write(f"Training model {index + 1}/{len(models)}: **{name}** ({method})")
        size = space_size(parameters)

        if method == GRID:
            search = GridSearchCV(
                pipeline, parameters, scoring=scoring, cv=cv,
                refit=True, n_jobs=-1, error_score="raise",
            )
            search.fit(X_train, y_train)
            model, score, params, tried = (
                search.best_estimator_, search.best_score_, search.best_params_, size,
            )

        elif method == RANDOM:
            n_iter = min(search_iterations, size)
            search = RandomizedSearchCV(
                pipeline, parameters, n_iter=n_iter, scoring=scoring, cv=cv,
                refit=True, random_state=RANDOM_STATE, n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            model, score, params, tried = (
                search.best_estimator_, search.best_score_, search.best_params_, n_iter,
            )

        else:  # MANUAL: no search, just the values the user chose
            params = manual_params[name]
            pipeline.set_params(**params)
            scores = cross_val_score(
                pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1
            )
            pipeline.fit(X_train, y_train)
            model, score, tried = pipeline, scores.mean(), 1

        best_models[name] = model
        rows.append({
            "Model": name,
            "Method": method,
            "Combinations tried": tried,
            "Search space size": size,
            "CV score": score,
            "Parameters": format_params(params),
            "_params": clean_params(params),  # dict, used for the JSON view
        })
        progress.progress((index + 1) / len(models))

    status.success("All seven models have been trained.")
    return best_models, pd.DataFrame(rows)
