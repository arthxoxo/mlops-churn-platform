"""
Training script for Customer Churn Prediction.
Works locally AND on SageMaker Training Jobs (same code).
"""

import os
import json
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    # Data paths (SageMaker injects these automatically)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "models"))
    # MLflow
    parser.add_argument("--mlflow-tracking-uri", type=str, default="./mlruns")
    parser.add_argument("--experiment-name", type=str, default="churn-prediction")
    return parser.parse_args()


def load_data(data_path: str):
    """Load processed training data."""
    filepath = os.path.join(data_path, "train.csv")
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_model(X_train, y_train, params: dict):
    """Train model using XGBoost when available, otherwise sklearn fallback."""
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        model._model_backend = "xgboost"
        return model
    except Exception as exc:
        logger.warning("XGBoost unavailable (%s). Falling back to RandomForestClassifier.", exc)
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        model._model_backend = "sklearn"
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }
    logger.info(f"Metrics: {metrics}")
    logger.info("\n" + classification_report(y_test, y_pred))
    return metrics


def save_model(model, scaler, feature_names, output_dir: str, metrics: dict):
    """Save model artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    logger.info(f"Model saved to {output_dir}")


def main():
    args = parse_args()

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
        }
        mlflow.log_params(params)

        # Load data
        X, y = load_data(args.train)
        feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        logger.info("Training model...")
        model = train_model(X_train_scaled, y_train, params)

        # Evaluate
        metrics = evaluate_model(model, X_test_scaled, y_test)
        mlflow.log_metrics(metrics)

        # Log model to MLflow using the matching flavor.
        if getattr(model, "_model_backend", "xgboost") == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save artifacts locally
        save_model(model, scaler, feature_names, args.output_dir, metrics)

        # Quality gate — fail the job if accuracy is too low
        if metrics["accuracy"] < 0.75:
            raise ValueError(f"Model accuracy {metrics['accuracy']} below threshold 0.75. Failing pipeline.")

        logger.info("Training complete!")
        return metrics


if __name__ == "__main__":
    main()
