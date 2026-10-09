
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Import the module referenced by the saved model.
# Keep hr_features.py in the repository root.
import hr_features  # noqa: F401

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
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

st.title("HR Attrition Analytics")
st.caption("IBM HR Analytics · Supervised Classification")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing from the artifacts folder.")
    st.stop()

model = load_model()
metadata = load_metadata()

st.success("The trained model is loaded successfully.")

col1, col2 = st.columns(2)
col1.metric("Selected model", metadata["selected_model"])
col2.metric("scikit-learn version", metadata["sklearn_version"])

st.info(
    "Predictions are estimates, not certainties. "
    "Use them to investigate workplace trends and support employees, "
    "not to make automated employment decisions."
)

st.subheader("Batch attrition prediction")
st.write(
    "Upload a CSV containing the same predictor columns used during "
    "training. The app will check the columns before making predictions."
)

uploaded_file = st.file_uploader(
    "Upload employee data",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Remove the outcome column if the uploaded dataset includes it.
        data = data.drop(columns=["Attrition"], errors="ignore")

        expected = list(getattr(model, "feature_names_in_", []))

        if not expected:
            st.error(
                "The model does not expose its original input column names. "
                "We need to configure the input schema before prediction."
            )
            st.stop()

        missing = [col for col in expected if col not in data.columns]

        if missing:
            st.error("The CSV is missing required columns:")
            st.write(missing)
            st.stop()

        # Keep the original training column order.
        X = data[expected].copy()

        if X.empty:
            st.warning("The uploaded CSV contains no employee records.")
            st.stop()

        if X.isna().any().any():
            st.error(
                "Missing values were found. Please correct them before "
                "uploading. Do not replace unknown values with arbitrary values."
            )
            st.stop()

        if st.button("Predict attrition", type="primary"):
            predictions = model.predict(X)

            probabilities = model.predict_proba(X)
            classes = list(model.classes_)

            if 1 not in classes:
                st.error("The model does not contain the expected positive class.")
                st.stop()

            positive_index = classes.index(1)

            results = data.copy()
            results["Predicted_Attrition"] = [
                "Yes" if p == 1 else "No" for p in predictions
            ]
            results["Attrition_Probability"] = probabilities[
                :, positive_index
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download prediction results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="attrition_predictions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not process this file: {exc}")
