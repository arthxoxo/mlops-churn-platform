"""
Training script for Customer Churn Prediction.
Works locally AND on SageMaker Training Jobs (same code).
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
)
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PathConfig, ensure_safe_environment

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
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", str(PathConfig.data_processed())))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR", str(PathConfig.models())))
    # MLflow
    parser.add_argument("--mlflow-tracking-uri", type=str, default=PathConfig.mlflow_tracking())
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


def train_model(X_train, y_train, params: dict, X_val=None, y_val=None):
    """Train model using XGBoost when available, otherwise sklearn fallback."""
    try:
        import xgboost as xgb

        base = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params.get("colsample_bytree", 0.9),
            "min_child_weight": params.get("min_child_weight", 1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "scale_pos_weight": params.get("scale_pos_weight", 1.0),
        }

        candidates = [
            base,
            {
                **base,
                "n_estimators": int(base["n_estimators"] * 1.6),
                "learning_rate": max(base["learning_rate"] * 0.7, 0.03),
                "max_depth": min(base["max_depth"] + 1, 8),
            },
            {
                **base,
                "n_estimators": base["n_estimators"] + 80,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "max_depth": max(base["max_depth"] - 1, 4),
            },
        ]

        best_model = None
        best_score = -1.0

        for idx, hp in enumerate(candidates, start=1):
            model = xgb.XGBClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                learning_rate=hp["learning_rate"],
                subsample=hp["subsample"],
                colsample_bytree=hp["colsample_bytree"],
                min_child_weight=hp["min_child_weight"],
                reg_lambda=hp["reg_lambda"],
                objective="binary:logistic",
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="auc",
                scale_pos_weight=hp["scale_pos_weight"],
                random_state=42,
            )
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                score = accuracy_score(y_val, model.predict(X_val))
                logger.info("Candidate %s validation accuracy: %.4f", idx, score)
            else:
                model.fit(X_train, y_train, verbose=False)
                score = 0.0

            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError("No XGBoost model candidate was trained.")

        best_model._model_backend = "xgboost"
        if X_val is not None and y_val is not None:
            logger.info("Selected XGBoost candidate with validation accuracy %.4f", best_score)
        return best_model
    except Exception as exc:
        logger.warning("XGBoost unavailable (%s). Falling back to RandomForestClassifier.", exc)
        rf_candidates = [
            {
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            },
            {
                "n_estimators": params["n_estimators"] + 120,
                "max_depth": max(params["max_depth"] + 2, 8),
                "min_samples_split": 4,
                "min_samples_leaf": 2,
            },
            {
                "n_estimators": params["n_estimators"] + 80,
                "max_depth": None,
                "min_samples_split": 6,
                "min_samples_leaf": 2,
            },
        ]

        best_model = None
        best_score = -1.0
        for idx, hp in enumerate(rf_candidates, start=1):
            model = RandomForestClassifier(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_split=hp["min_samples_split"],
                min_samples_leaf=hp["min_samples_leaf"],
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            if X_val is not None and y_val is not None:
                score = accuracy_score(y_val, model.predict(X_val))
                logger.info("RF candidate %s validation accuracy: %.4f", idx, score)
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_model = model

        if best_model is None:
            raise RuntimeError("No RandomForest model candidate was trained.")

        best_model._model_backend = "sklearn"
        if X_val is not None and y_val is not None:
            logger.info("Selected RandomForest candidate with validation accuracy %.4f", best_score)
        return best_model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }
    logger.info(f"Metrics: {metrics}")
    logger.info("\n" + classification_report(y_test, y_pred))
    return metrics


def save_model(model, scaler, feature_names, output_dir: str, metrics: dict):
    """Save model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    logger.info(f"Model saved to {output_dir}")


def main():
    # Ensure safe environment before running
    ensure_safe_environment()
    
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
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        }
        mlflow.log_params(params)

        # Load data
        X, y = load_data(args.train)
        feature_names = list(X.columns)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=y_train_full,
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Train
        logger.info("Training model...")
        model = train_model(X_train_scaled, y_train, params, X_val=X_val_scaled, y_val=y_val)

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
