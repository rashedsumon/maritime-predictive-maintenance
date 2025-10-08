# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import (
    load_data,
    basic_cleaning,
    engineer_features,
    prepare_target,
    get_feature_matrix,
)
from src.model_utils import load_model, train_and_evaluate, predict_single
from src.viz_utils import plot_feature_distribution, plot_confusion_matrix


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Marine Engine Predictive Maintenance",
    layout="wide",
    page_icon="🚢",
)

DATA_PATH_KAGGLE = "/kaggle/input/preventive-maintenance-for-marine-engines/marine_engine_data.csv"
LOCAL_DATA_PATH = "data/marine_engine_data.csv"
MODEL_PATH = "models/maintenance_model.joblib"
METRICS_PATH = "models/metrics_summary.json"


# ---------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load dataset from local or Kaggle path."""
    if os.path.exists(LOCAL_DATA_PATH):
        path = LOCAL_DATA_PATH
    elif os.path.exists(DATA_PATH_KAGGLE):
        path = DATA_PATH_KAGGLE
    else:
        st.error("❌ Dataset not found in either local or Kaggle path.")
        st.stop()
    return load_data(path)


# ---------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------
def main():
    st.title("🚢 Marine Engine — Predictive Maintenance Dashboard")
    st.markdown(
        "This dashboard predicts **maintenance status** from engine telemetry data. "
        "The target variable is `maintenance_status`."
    )

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------
    df = load_dataset()
    if df is None or df.empty:
        st.error("Dataset is empty or failed to load. Please check the file path.")
        st.stop()

    # Optionally preview dataset
    if st.checkbox("Show raw dataset", value=False):
        st.dataframe(df.head(200), use_container_width=True)

    # -----------------------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------------------
    df_clean = basic_cleaning(df)
    df_feat = engineer_features(df_clean)
    df_feat, target_encoder = prepare_target(df_feat, target_col="maintenance_status")

    # -----------------------------------------------------------------
    # Model section
    # -----------------------------------------------------------------
    st.sidebar.header("⚙️ Model / Training Controls")

    model = load_model()
    if model is None:
        st.sidebar.warning("⚠️ No valid model found at `models/maintenance_model.joblib`.")
        if st.sidebar.button("Train model now (may take a few minutes)"):
            with st.spinner("Training model..."):
                X, y = get_feature_matrix(df_feat, target_col="maintenance_status")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                pipeline, metrics = train_and_evaluate(X, y, save=True)

                # Save metrics
                os.makedirs("models", exist_ok=True)
                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f, indent=2)

                st.success("✅ Training complete — model saved successfully.")
                st.json(metrics)
                model = pipeline
    else:
        st.sidebar.success("✅ Loaded saved model.")
        if st.sidebar.button("Retrain model"):
            with st.spinner("Retraining model..."):
                X, y = get_feature_matrix(df_feat, target_col="maintenance_status")
                valid = y.notna()
                X, y = X.loc[valid], y.loc[valid]
                pipeline, metrics = train_and_evaluate(X, y, save=True)

                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f, indent=2)

                st.success("✅ Retraining complete — model updated.")
                st.json(metrics)
                model = pipeline

    # -----------------------------------------------------------------
    # Exploratory Data Analysis
    # -----------------------------------------------------------------
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns([1, 1])

    with col1:
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            feat = st.selectbox("Select numeric feature to visualize", num_cols, index=0)
            fig = plot_feature_distribution(df_feat, feat)
            st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("#### Summary Statistics")
        st.write(df_feat.describe().loc[["mean", "std", "min", "75%", "max"]])

    # -----------------------------------------------------------------
    # Metrics display
    # -----------------------------------------------------------------
    if os.path.exists(METRICS_PATH):
        st.header("📈 Saved Model Metrics")
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        st.json(metrics)

    # -----------------------------------------------------------------
    # Single-sample prediction
    # -----------------------------------------------------------------
    st.header("🔍 Single-record Prediction")
    X_all, y_all = get_feature_matrix(df_feat, target_col="maintenance_status")

    if X_all.empty:
        st.error("No valid samples available for prediction.")
        st.stop()

    sample_index = st.selectbox("Select a sample index", X_all.index.tolist()[:200])
    sample = X_all.loc[[sample_index]]
    st.write("Input features for selected sample:")
    st.dataframe(sample.T, use_container_width=True)

    if model is not None:
        try:
            pred, proba = predict_single(model, sample)
            st.success(f"**Predicted maintenance_status (encoded):** {pred}")
            if proba is not None:
                st.info(f"Predicted probability (class=1): {proba:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("No model available. Train or upload one to enable predictions.")

    # -----------------------------------------------------------------
    # Footer / Notes
    # -----------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        "💡 **Notes:** This dashboard is a template for predictive maintenance. "
        "For production, migrate training to batch pipelines, store models in artifact registries, "
        "and implement monitoring for drift and reliability."
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
