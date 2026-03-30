"""
AWS Lambda handler for scheduled drift monitoring.
Triggered by CloudWatch Events on a cron schedule.

Environment Variables (set in Lambda configuration):
  S3_BUCKET         – bucket for data and reports
  SNS_TOPIC_ARN     – alert topic
  DRIFT_THRESHOLD   – drift share threshold (default 0.20)
"""

import os
import json
import logging
import boto3
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get("S3_BUCKET", "your-mlops-bucket")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.20"))

s3 = boto3.client("s3")
sns = boto3.client("sns")
cloudwatch = boto3.client("cloudwatch")


def load_csv_from_s3(key: str) -> pd.DataFrame:
    """Download a CSV from S3 and return as DataFrame."""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(obj["Body"])


def compute_drift_score(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Lightweight drift detection (no Evidently dependency in Lambda).
    Compares column means/stds between reference and current data.
    Returns drift share = fraction of columns with significant shift.
    """
    feature_cols = [c for c in reference.columns if c != "Churn"]
    drifted = 0

    for col in feature_cols:
        if reference[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            ref_mean = reference[col].mean()
            ref_std = reference[col].std()
            cur_mean = current[col].mean()
            # Column is "drifted" if current mean is > 2 std away from reference
            if ref_std > 0 and abs(cur_mean - ref_mean) / ref_std > 2.0:
                drifted += 1

    drift_share = drifted / max(len(feature_cols), 1)
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_share": round(drift_share, 4),
        "drifted_columns": drifted,
        "total_columns": len(feature_cols),
        "dataset_drift_detected": drift_share > DRIFT_THRESHOLD,
    }


def publish_metric(drift_share: float):
    """Push drift score to CloudWatch for dashboards & alarms."""
    cloudwatch.put_metric_data(
        Namespace="MLOps/ChurnModel",
        MetricData=[
            {
                "MetricName": "DriftShare",
                "Value": drift_share,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "Model", "Value": "xgboost-churn"},
                ],
            }
        ],
    )
    logger.info(f"Published DriftShare={drift_share} to CloudWatch.")


def send_alert(summary: dict):
    """Send SNS alert when drift is detected."""
    if not SNS_TOPIC_ARN:
        logger.warning("SNS_TOPIC_ARN not configured. Skipping alert.")
        return

    message = (
        "🚨 DATA DRIFT DETECTED — Churn Prediction Model\n\n"
        f"Timestamp: {summary['timestamp']}\n"
        f"Drift Share: {summary['drift_share']:.1%}\n"
        f"Drifted Columns: {summary['drifted_columns']} / {summary['total_columns']}\n\n"
        "Action Required: Review production data and consider retraining.\n"
    )
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="[MLOps Alert] Data Drift Detected",
        Message=message,
    )
    logger.info("SNS alert sent.")


def handler(event, context):
    """
    Lambda entry point.
    Expects reference data at s3://{BUCKET}/processed/train/train.csv
    and production data at s3://{BUCKET}/production-logs/{today}/predictions.csv
    """
    logger.info(f"Event: {json.dumps(event)}")

    try:
        # Load reference data
        reference = load_csv_from_s3("processed/train/train.csv")
        logger.info(f"Reference data loaded: {len(reference)} rows")
    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        return {"statusCode": 500, "body": f"Reference data error: {e}"}

    try:
        # Load today's production data
        today = datetime.utcnow().strftime("%Y-%m-%d")
        current = load_csv_from_s3(f"production-logs/{today}/predictions.csv")
        logger.info(f"Production data loaded: {len(current)} rows")
    except Exception as e:
        logger.warning(f"No production data for today: {e}")
        return {"statusCode": 200, "body": "No production data available yet."}

    # Compute drift
    summary = compute_drift_score(reference, current)
    logger.info(f"Drift summary: {json.dumps(summary)}")

    # Push metric to CloudWatch
    publish_metric(summary["drift_share"])

    # Save result to S3
    result_key = f"monitoring/drift-results/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=result_key,
        Body=json.dumps(summary, indent=2),
        ContentType="application/json",
    )

    # Alert if drifted
    if summary["dataset_drift_detected"]:
        logger.warning("⚠️ Drift detected — sending alert.")
        send_alert(summary)

    return {
        "statusCode": 200,
        "body": json.dumps(summary),
    }
