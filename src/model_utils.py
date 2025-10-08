# src/model_utils.py
import os
from typing import Tuple, Optional
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "maintenance_model.joblib")

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)

def build_pipeline() -> Pipeline:
    """Build a modeling pipeline for numeric features. Categorical handled via get_dummies upstream."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    pipeline = Pipeline([
        ('numeric', numeric_pipeline),
        # final estimator
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
    ])
    return pipeline

def train_and_evaluate(X, y, save: bool = True) -> Tuple[Pipeline, dict]:
    """Train a model and evaluate. Returns pipeline (fitted with numeric transformer applied to all columns) and metrics dict."""
    # ensure arrays
    X = X.copy()
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # build pipeline suitable for whole numeric matrix
    pipeline = build_pipeline()
    # fit
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
        joblib.dump(pipeline, MODEL_PATH)
    return pipeline, metrics

def load_model(path: Optional[str] = None):
    """Load a saved model."""
    if path is None:
        path = MODEL_PATH
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_single(model, X_single):
    """Predict for a single sample (1D or DataFrame)."""
    arr = X_single.values.reshape(1, -1) if hasattr(X_single, 'values') and X_single.shape[0] == 1 else np.asarray(X_single).reshape(1, -1)
    pred = model.predict(arr)
    proba = model.predict_proba(arr)[:, 1] if hasattr(model, 'predict_proba') else None
    return pred, (proba[0] if proba is not None else None)
