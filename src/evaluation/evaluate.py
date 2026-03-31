"""
Model evaluation script for the SageMaker pipeline.
Loads trained model + test data, computes metrics, writes evaluation.json.
The pipeline's ConditionStep reads accuracy from this JSON to decide
whether to register the model.

Works both locally and on SageMaker Processing jobs.
"""

import os
import sys
import json
import tarfile
import logging
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PathConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker Processing paths (defaults to repo-relative paths for local testing)
MODEL_PATH = os.environ.get("SM_CHANNEL_MODEL", str(PathConfig.models()))
TEST_PATH = os.environ.get("SM_CHANNEL_TEST", str(PathConfig.data_processed()))
OUTPUT_PATH = os.environ.get("SM_OUTPUT_DIR", str(PathConfig.evaluation_output()))


def extract_model(model_dir: str):
    """Extract model.tar.gz if running on SageMaker."""
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        logger.info(f"Extracting {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
    return model_dir


def load_model(model_dir: str):
    """Load trained XGBoost model and scaler."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    logger.info("Model and scaler loaded successfully.")
    return model, scaler


def load_test_data(test_dir: str):
    """Load test CSV (expects a 'Churn' target column)."""
    # Find the CSV file in the test directory
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {test_dir}")
    filepath = os.path.join(test_dir, csv_files[0])
    logger.info(f"Loading test data from {filepath}")
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def evaluate(model, scaler, X, y) -> dict:
    """Run evaluation and return metrics dictionary."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    acc: float = float(accuracy_score(y, y_pred))
    auc: float = float(roc_auc_score(y, y_proba))
    f1: float = float(f1_score(y, y_pred))
    prec: float = float(precision_score(y, y_pred))
    rec: float = float(recall_score(y, y_pred))

    metrics = {
        "accuracy": {"value": round(acc, 4)},
        "roc_auc": {"value": round(auc, 4)},
        "f1_score": {"value": round(f1, 4)},
        "precision": {"value": round(prec, 4)},
        "recall": {"value": round(rec, 4)},
    }

    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
    logger.info("\n" + classification_report(y, y_pred))
    return metrics


def write_evaluation_report(metrics: dict, output_dir: str):
    """Write evaluation.json — consumed by the SageMaker pipeline PropertyFile."""
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "metrics": metrics,
        "dataset_size": int(sum(1 for _ in metrics)),  # placeholder
    }
    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {output_path}")


def main():
    logger.info("Starting model evaluation...")

    # Extract model artifacts (SageMaker bundles as tar.gz)
    extract_model(MODEL_PATH)

    # Load
    model, scaler = load_model(MODEL_PATH)
    X_test, y_test = load_test_data(TEST_PATH)
    logger.info(f"Test set: {len(y_test)} samples")

    # Evaluate
    metrics = evaluate(model, scaler, X_test, y_test)

    # Write report for pipeline
    write_evaluation_report(metrics, OUTPUT_PATH)

    # Print metrics in SageMaker-parseable format
    for name, val in metrics.items():
        print(f"{name}: {val['value']}")

    logger.info("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
