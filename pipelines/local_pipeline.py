"""
Local ML Pipeline — Free-tier alternative to SageMaker Pipelines.
Orchestrates: Preprocess → Train → Evaluate → Upload to S3.

All processing runs locally (or in GitHub Actions CI).
Model artifacts are stored in S3 (free tier: 5GB).

Run:
  python pipelines/local_pipeline.py                 # full pipeline
  python pipelines/local_pipeline.py --skip-upload    # local only, no S3
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
S3_BUCKET = os.environ.get("S3_BUCKET", "")
MODEL_DIR = os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models"))
DATA_RAW = str(PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
DATA_PROCESSED = str(PROJECT_ROOT / "data" / "processed")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))


def run_step(name: str, command: list[str]) -> int:
    """Run a pipeline step as a subprocess."""
    logger.info(f"{'─' * 60}")
    logger.info(f"Step: {name}")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"{'─' * 60}")
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        logger.error(f"Step '{name}' failed with exit code {result.returncode}")
        sys.exit(1)
    logger.info(f"✅ {name} completed successfully.\n")
    return result.returncode


def step_preprocess():
    """Step 1: Data preprocessing."""
    run_step("Preprocessing", [sys.executable, "src/data/preprocess.py"])


def step_train():
    """Step 2: Model training with MLflow tracking."""
    run_step("Training", [
        sys.executable, "src/training/train.py",
        "--output-dir", MODEL_DIR,
        "--mlflow-tracking-uri", "./mlruns",
        "--experiment-name", "churn-prediction",
    ])


def step_evaluate() -> dict:
    """Step 3: Model evaluation (quality gate)."""
    # Set up paths for the evaluation script
    eval_output = str(PROJECT_ROOT / "evaluation_output")
    os.makedirs(eval_output, exist_ok=True)

    env = os.environ.copy()
    env["SM_CHANNEL_MODEL"] = MODEL_DIR
    env["SM_CHANNEL_TEST"] = DATA_PROCESSED
    env["SM_OUTPUT_DIR"] = eval_output

    logger.info(f"{'─' * 60}")
    logger.info("Step: Evaluation (Quality Gate)")
    logger.info(f"{'─' * 60}")

    result = subprocess.run(
        [sys.executable, "src/evaluation/evaluate.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    if result.returncode != 0:
        logger.error("Evaluation step failed.")
        sys.exit(1)

    # Read evaluation results
    eval_path = os.path.join(eval_output, "evaluation.json")
    with open(eval_path) as f:
        evaluation = json.load(f)

    accuracy = evaluation["metrics"]["accuracy"]["value"]
    logger.info(f"Model accuracy: {accuracy:.4f} (threshold: {ACCURACY_THRESHOLD})")

    if accuracy < ACCURACY_THRESHOLD:
        logger.error(
            f"❌ QUALITY GATE FAILED — accuracy {accuracy:.4f} < {ACCURACY_THRESHOLD}. "
            "Model will NOT be promoted."
        )
        sys.exit(1)

    logger.info("✅ Quality gate passed!\n")
    return evaluation


def step_upload_to_s3(evaluation: dict):
    """Step 4: Upload model artifacts to S3 (free tier: 5GB)."""
    if not S3_BUCKET:
        logger.warning("S3_BUCKET not set. Skipping upload. Model saved locally.")
        return

    try:
        import boto3

        s3 = boto3.client("s3")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"models/{timestamp}"

        # Upload model artifacts
        model_dir = Path(MODEL_DIR)
        for filepath in model_dir.iterdir():
            if filepath.is_file():
                s3_key = f"{s3_prefix}/{filepath.name}"
                s3.upload_file(str(filepath), S3_BUCKET, s3_key)
                logger.info(f"Uploaded: s3://{S3_BUCKET}/{s3_key}")

        # Upload evaluation report
        eval_key = f"{s3_prefix}/evaluation.json"
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=eval_key,
            Body=json.dumps(evaluation, indent=2),
            ContentType="application/json",
        )
        logger.info(f"Uploaded: s3://{S3_BUCKET}/{eval_key}")

        # Update "latest" pointer
        s3.put_object(
            Bucket=S3_BUCKET,
            Key="models/latest.json",
            Body=json.dumps({"path": s3_prefix, "timestamp": timestamp}, indent=2),
            ContentType="application/json",
        )
        logger.info(f"✅ Model artifacts uploaded to s3://{S3_BUCKET}/{s3_prefix}")

    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        logger.warning("Model saved locally only.")


def step_register_model(evaluation: dict):
    """Step 5: Register model in MLflow registry (local, free)."""
    try:
        import mlflow

        mlflow.set_tracking_uri("./mlruns")

        # Find the latest run
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("churn-prediction")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                result = mlflow.register_model(model_uri, "xgboost-churn")
                logger.info(f"✅ Model registered: {result.name} v{result.version}")
                return

        logger.warning("No MLflow runs found. Skipping registration.")
    except Exception as e:
        logger.warning(f"MLflow registration skipped: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local MLOps Pipeline")
    parser.add_argument("--skip-upload", action="store_true", help="Skip S3 upload")
    parser.add_argument("--skip-register", action="store_true", help="Skip MLflow model registration")
    args = parser.parse_args()

    logger.info("🚀 Starting MLOps Pipeline (local execution)")
    logger.info(f"   Model dir: {MODEL_DIR}")
    logger.info(f"   S3 bucket: {S3_BUCKET or '(not set)'}")
    logger.info(f"   Accuracy threshold: {ACCURACY_THRESHOLD}")
    logger.info("")

    # Step 1: Preprocess
    step_preprocess()

    # Step 2: Train
    step_train()

    # Step 3: Evaluate (quality gate)
    evaluation = step_evaluate()

    # Step 4: Upload to S3
    if not args.skip_upload:
        step_upload_to_s3(evaluation)

    # Step 5: Register model
    if not args.skip_register:
        step_register_model(evaluation)

    logger.info("=" * 60)
    logger.info("🎉 Pipeline completed successfully!")
    logger.info("=" * 60)

    # Print summary
    metrics = evaluation.get("metrics", {})
    for name, val in metrics.items():
        logger.info(f"   {name}: {val['value']}")


if __name__ == "__main__":
    main()
