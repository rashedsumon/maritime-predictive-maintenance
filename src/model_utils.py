# src/model_utils.py
import os
import joblib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "maintenance_model.joblib")

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def ensure_model_dir() -> None:
    """Ensure the model directory exists."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def build_pipeline() -> Pipeline:
    """
    Build a modeling pipeline for numeric features.
    (Categorical handling should occur upstream.)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline = Pipeline([
        ("numeric", numeric_pipeline),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )),
    ])
    return pipeline


# ---------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------
def train_and_evaluate(X, y, save: bool = True) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train and evaluate a RandomForest pipeline.
    Returns: (fitted_pipeline, metrics_dict)
    """
    X = X.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train.values, y_train.values)

    preds = pipeline.predict(X_test.values)
    proba = pipeline.predict_proba(X_test.values)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
    }

    if save:
        ensure_model_dir()
        try:
            joblib.dump(pipeline, MODEL_PATH)
            print(f"✅ Model saved successfully to {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")

    return pipeline, metrics


# ---------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------
def load_model(path: Optional[str] = None) -> Optional[Pipeline]:
    """
    Load a saved model safely.
    Returns None if file is missing or corrupted.
    """
    if path is None:
        path = MODEL_PATH

    if not os.path.exists(path):
        print(f"⚠️ Model file not found at: {path}")
        return None

    try:
        model = joblib.load(path)
        print(f"✅ Model loaded successfully from {path}")
        return model
    except EOFError:
        print(f"❌ Model file at {path} appears to be incomplete or corrupted (EOFError).")
        return None
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def predict_single(model, X_single):
    """
    Predict for a single sample (1D array, list, or 1-row DataFrame).
    Returns: (prediction, probability)
    """
    if model is None:
        raise ValueError("No valid model provided for prediction.")

    if hasattr(X_single, "values") and X_single.shape[0] == 1:
        arr = X_single.values.reshape(1, -1)
    else:
        arr = np.asarray(X_single).reshape(1, -1)

    pred = model.predict(arr)
    proba = model.predict_proba(arr)[:, 1] if hasattr(model, "predict_proba") else None

    return pred[0], (float(proba[0]) if proba is not None else None)
