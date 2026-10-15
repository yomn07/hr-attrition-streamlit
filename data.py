from dataclasses import dataclass

import pandas as pd
import streamlit as st

from config import DATA_PATH, TARGET


@dataclass
class Dataset:
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    numeric_features: list
    categorical_features: list


@st.cache_data(show_spinner="Loading the IBM HR Attrition dataset...")
def load_raw_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found. Place '{DATA_PATH.name}' in the same "
            "folder as app.py."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Dataset is missing the required column: {TARGET}")

    if not set(df[TARGET].dropna().astype(str).unique()).issubset({"Yes", "No"}):
        raise ValueError("Attrition must contain only Yes/No values.")

    return df


def prepare_data(df: pd.DataFrame) -> Dataset:
    df = df.copy()
    drop_columns = [
        col for col in df.columns
        if col != TARGET and df[col].nunique(dropna=False) <= 1
    ]
    if "EmployeeNumber" in df.columns:
        drop_columns.append("EmployeeNumber")
    df = df.drop(columns=list(set(drop_columns)), errors="ignore")
    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return Dataset(df, X, y, numeric_features, categorical_features)
