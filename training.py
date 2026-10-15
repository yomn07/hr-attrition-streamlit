import pandas as pd
import streamlit as st
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from config import RANDOM_STATE
from models import model_search_spaces


def split_data(X, y, test_size_percent):
    return train_test_split(
        X, y,
        test_size=test_size_percent / 100,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_and_tune(
    X_train, y_train,
    numeric_features, categorical_features,
    cv_folds, scoring, search_iterations,
):
    models = model_search_spaces(numeric_features, categorical_features)

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
    )

    best_models = {}
    cv_results = []

    progress = st.progress(0)
    status = st.empty()

    for index, (name, (pipeline, parameters)) in enumerate(models.items()):
        status.write(f"Training model {index + 1}/{len(models)}: **{name}**")

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

        progress.progress((index + 1) / len(models))

    status.success("All seven models have been trained and tuned.")
    return best_models, pd.DataFrame(cv_results)
