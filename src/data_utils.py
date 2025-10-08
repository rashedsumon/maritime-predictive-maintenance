# `src/data_utils.py`
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame. Path could be Kaggle path or local path."""
    df = pd.read_csv(path)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: parse timestamps, drop exact duplicates, unify column names."""
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # drop duplicates
    df = df.drop_duplicates()
    # fix obvious numeric columns (coerce)
    numeric_cols = ['engine_temp', 'oil_pressure', 'fuel_consumption',
                    'vibration_level', 'rpm', 'engine_load',
                    'coolant_temp', 'exhaust_temp', 'running_period',
                    'fuel_consumption_per_hour']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: rolling statistics per engine, basic derived features."""
    df = df.copy()
    # Ensure timestamp sorting per engine
    if 'timestamp' in df.columns and 'engine_id' in df.columns:
        df = df.sort_values(['engine_id', 'timestamp'])
        # rolling window features (last 3 observations)
        roll_cols = ['engine_temp', 'oil_pressure', 'vibration_level', 'rpm', 'engine_load']
        for col in roll_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_3'] = df.groupby('engine_id')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_rolling_std_3'] = df.groupby('engine_id')[col].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    # simple derived metrics:
    if 'fuel_consumption' in df.columns and 'running_period' in df.columns:
        df['fuel_per_minute'] = df['fuel_consumption'] / df['running_period'].replace({0: np.nan})
    # time-based features
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df

def prepare_target(df: pd.DataFrame, target_col='maintenance_status') -> Tuple[pd.DataFrame, LabelEncoder]:
    """Prepare target column: convert to numeric if categorical."""
    df = df.copy()
    le = None
    if target_col in df.columns:
        if df[target_col].dtype == 'object' or not np.issubdtype(df[target_col].dtype, np.number):
            le = LabelEncoder()
            df[target_col] = df[target_col].astype(str).fillna('unknown')
            df[target_col] = le.fit_transform(df[target_col])
    return df, le

def get_feature_matrix(df: pd.DataFrame, target_col='maintenance_status') -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y for modelling by selecting numeric and encoded categorical features."""
    df = df.copy()
    # Drop identifiers and timestamp
    drop_cols = ['timestamp', 'engine_id']
    drop_cols = [c for c in drop_cols if c in df.columns]
    # decide on features: numeric + some categorical encoded with get_dummies
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_col in numeric.columns:
        y = numeric[target_col]
        X_numeric = numeric.drop(columns=[target_col])
    else:
        y = None
        X_numeric = numeric
    # categorical features
    cat_cols = ['engine_type', 'fuel_type', 'manufacturer', 'failure_mode']
    cats = [c for c in cat_cols if c in df.columns]
    X_cat = pd.get_dummies(df[cats].astype(str).fillna('missing'), drop_first=True) if cats else pd.DataFrame(index=df.index)
    X = pd.concat([X_numeric, X_cat], axis=1)
    # drop id/time derived columns if present in X
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
    return X, y
