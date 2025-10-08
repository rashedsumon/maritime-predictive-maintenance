# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

from src.data_utils import load_data, basic_cleaning, engineer_features, prepare_target, get_feature_matrix
from src.model_utils import load_model, train_and_evaluate, predict_single, MODEL_PATH
from src.viz_utils import plot_feature_distribution, plot_confusion_matrix
import joblib
import json

st.set_page_config(page_title="Marine Engine Predictive Maintenance", layout="wide")

DATA_PATH_KAGGLE = "/kaggle/input/preventive-maintenance-for-marine-engines/marine_engine_data.csv"
LOCAL_DATA_PATH = "data/marine_engine_data.csv"

@st.cache_data(show_spinner=False)
def load_dataset():
    # prefer local copy if exists, otherwise use kaggle path
    if os.path.exists(LOCAL_DATA_PATH):
        path = LOCAL_DATA_PATH
    else:
        path = DATA_PATH_KAGGLE
    df = load_data(path)
    return df

def main():
    st.title("Marine Engine — Predictive Maintenance Dashboard")
    st.markdown("Predict maintenance status from engine telemetry. Uses `maintenance_status` as target.")

    df = load_dataset()

    # show dataset
    if st.checkbox("Show raw dataset", value=False):
        st.dataframe(df.head(200))

    # Basic cleaning & features
    df_clean = basic_cleaning(df)
    df_feat = engineer_features(df_clean)
    df_feat, target_encoder = prepare_target(df_feat, target_col='maintenance_status')

    st.sidebar.header("Model / Training")
    model = load_model()  # will return None if not found

    if model is None:
        st.sidebar.warning("No saved model found at `models/maintenance_model.joblib`.")
        if st.sidebar.button("Train model now (may take a while)"):
            with st.spinner("Training model..."):
                X, y = get_feature_matrix(df_feat, target_col='maintenance_status')
                # Remove rows with NaN target
                valid = y.notna()
                X = X.loc[valid]
                y = y.loc[valid]
                pipeline, metrics = train_and_evaluate(X, y, save=True)
                st.success("Training complete — model saved.")
                st.json(metrics)
                model = pipeline
    else:
        st.sidebar.success("Loaded saved model.")
        if st.sidebar.button("Retrain model"):
            with st.spinner("Retraining..."):
                X, y = get_feature_matrix(df_feat, target_col='maintenance_status')
                valid = y.notna()
                X = X.loc[valid]
                y = y.loc[valid]
                pipeline, metrics = train_and_evaluate(X, y, save=True)
                st.success("Retraining complete.")
                st.json(metrics)
                model = pipeline

    # EDA
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns([1, 1])
    with col1:
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            feat = st.selectbox("Choose numeric feature to view distribution", num_cols, index=0)
            fig = plot_feature_distribution(df_feat, feat)
            st.pyplot(fig)
    with col2:
        st.markdown("Summary statistics")
        st.write(df_feat.describe().loc[['mean','std','min','75%','max']])

    # Model evaluation (if metrics exist)
    if os.path.exists("models/metrics_summary.json"):
        st.header("Saved Model Metrics")
        with open("models/metrics_summary.json") as f:
            metrics = json.load(f)
        st.json(metrics)

    # Single-sample inference
    st.header("Single-record prediction")
    X_all, y_all = get_feature_matrix(df_feat, target_col='maintenance_status')
    # choose a sample
    sample_index = st.selectbox("Pick a sample from dataset (index)", X_all.index.tolist()[:200])
    sample = X_all.loc[[sample_index]]
    st.write("Input features:")
    st.dataframe(sample.T)

    if model is not None:
        pred, proba = predict_single(model, sample)
        st.write("Predicted maintenance_status (encoded):", int(pred[0]))
        if proba is not None:
            st.write("Predicted probability (class=1):", float(proba))
    else:
        st.info("No model available. Train or upload a model to run predictions.")

    st.markdown("---")
    st.markdown("**Notes:** The pipeline here is a template. For production, move long-running training jobs to batch or scheduled pipelines, keep models in artifact stores, and add robust monitoring.")

if __name__ == "__main__":
    main()
