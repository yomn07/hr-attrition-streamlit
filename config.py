from pathlib import Path

RANDOM_STATE = 42
TARGET = "Attrition"
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Gradient Boosting",
    "Support Vector Machine",
    "K-Nearest Neighbors",
    "Decision Tree",
]
