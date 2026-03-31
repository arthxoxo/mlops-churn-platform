"""
Data drift monitoring using Evidently AI.
Run on a schedule (e.g., daily) to detect drift in production data.
Sends alert to SNS if drift is detected.
"""

import os
import sys
import json
import boto3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import PathConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "your-mlops-bucket")
DRIFT_THRESHOLD = 0.2  # Alert if drift score > 20%


def load_reference_data() -> pd.DataFrame:
    """Load training data as reference."""
    return pd.read_csv(PathConfig.data_processed() / "train.csv")


def load_production_data() -> pd.DataFrame:
    """Load recent production predictions from S3."""
    s3 = boto3.client("s3")
    today = datetime.now().strftime("%Y-%m-%d")
    key = f"production-logs/{today}/predictions.csv"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_csv(obj["Body"])
    except Exception as e:
        logger.warning(f"Could not load production data: {e}. Using sample.")
        # For local testing, return a sample of reference data with noise
        ref = load_reference_data()
        sample = ref.sample(500, random_state=42).copy()
        # Add some artificial drift for demo
        sample["tenure"] = sample["tenure"] * 1.5
        return sample


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Run Evidently drift report and return summary."""
    feature_cols = [c for c in reference.columns if c != "Churn"]

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
    )

    # Save HTML report
    os.makedirs("monitoring/reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"monitoring/reports/drift_{timestamp}.html"
    report.save_html(report_path)
    logger.info(f"Drift report saved: {report_path}")

    # Extract drift summary
    result = report.as_dict()
    drift_metric = result["metrics"][1]["result"]
    return {
        "timestamp": timestamp,
        "dataset_drift_detected": drift_metric.get("dataset_drift", False),
        "drift_share": drift_metric.get("drift_share", 0.0),
        "number_of_drifted_columns": drift_metric.get("number_of_drifted_columns", 0),
        "report_path": report_path,
    }


def send_alert(summary: dict):
    """Send SNS alert if drift is detected."""
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not set. Skipping alert.")
        return

    sns = boto3.client("sns")
    message = f"""
🚨 DATA DRIFT DETECTED — Churn Prediction Model

Timestamp: {summary['timestamp']}
Drift Share: {summary['drift_share']:.1%}
Drifted Columns: {summary['number_of_drifted_columns']}

Action Required: Review production data and consider retraining.
Report: s3://{S3_BUCKET}/monitoring/reports/drift_{summary['timestamp']}.html
    """
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="[MLOps Alert] Data Drift Detected",
        Message=message,
    )
    logger.info("Alert sent via SNS.")


def main():
    logger.info("Starting drift monitoring...")

    reference = load_reference_data()
    current = load_production_data()

    logger.info(f"Reference data: {len(reference)} rows")
    logger.info(f"Current data: {len(current)} rows")

    summary = run_drift_report(reference, current)
    logger.info(f"Drift summary: {json.dumps(summary, indent=2)}")

    if summary["dataset_drift_detected"] or summary["drift_share"] > DRIFT_THRESHOLD:
        logger.warning("⚠️  Drift detected! Sending alert...")
        send_alert(summary)
    else:
        logger.info("✅ No significant drift detected.")

    return summary


if __name__ == "__main__":
    main()
