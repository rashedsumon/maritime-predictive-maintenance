# train.py
from src.data_utils import load_data, basic_cleaning, engineer_features, prepare_target, get_feature_matrix
from src.model_utils import train_and_evaluate
import os
import json

DATA_PATH = "/kaggle/input/preventive-maintenance-for-marine-engines/marine_engine_data.csv"

def main():
    df = load_data(DATA_PATH)
    df = basic_cleaning(df)
    df = engineer_features(df)
    df, le = prepare_target(df, target_col='maintenance_status')
    X, y = get_feature_matrix(df, target_col='maintenance_status')
    pipeline, metrics = train_and_evaluate(X, y, save=True)
    # save metrics summary
    os.makedirs('models', exist_ok=True)
    with open('models/metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training complete. Model saved to models/maintenance_model.joblib")
    print("Metrics saved to models/metrics_summary.json")

if __name__ == "__main__":
    main()
