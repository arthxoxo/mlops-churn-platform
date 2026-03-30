"""
CloudWatch infrastructure setup for the MLOps Churn Platform.

Creates:
  1. CloudWatch alarms for model accuracy & drift detection
  2. CloudWatch Events rule to trigger Lambda drift monitor daily
  3. SNS topic for alerts

Run: python infra/cloudwatch_alarms.py
"""

import os
import json
import boto3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-east-1")
SNS_TOPIC_NAME = os.environ.get("SNS_TOPIC_NAME", "mlops-alerts")
LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "mlops-drift-monitor")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "")

cloudwatch = boto3.client("cloudwatch", region_name=REGION)
events = boto3.client("events", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)
lambda_client = boto3.client("lambda", region_name=REGION)


def create_sns_topic() -> str:
    """Create or get the SNS topic for MLOps alerts."""
    response = sns.create_topic(
        Name=SNS_TOPIC_NAME,
        Tags=[
            {"Key": "Project", "Value": "mlops-churn-platform"},
            {"Key": "Environment", "Value": "production"},
        ],
    )
    topic_arn = response["TopicArn"]
    logger.info(f"SNS topic ready: {topic_arn}")
    return topic_arn


def create_drift_alarm(sns_topic_arn: str):
    """
    Alarm: triggers when DriftShare metric exceeds threshold.
    The Lambda drift handler publishes this metric after each run.
    """
    cloudwatch.put_metric_alarm(
        AlarmName="mlops-drift-share-high",
        AlarmDescription="Data drift share exceeds 20% — model may need retraining",
        Namespace="MLOps/ChurnModel",
        MetricName="DriftShare",
        Dimensions=[{"Name": "Model", "Value": "xgboost-churn"}],
        Statistic="Maximum",
        Period=86400,  # 24 hours
        EvaluationPeriods=1,
        Threshold=0.20,
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[sns_topic_arn],
        OKActions=[sns_topic_arn],
        TreatMissingData="notBreaching",
        Tags=[{"Key": "Project", "Value": "mlops-churn-platform"}],
    )
    logger.info("Created alarm: mlops-drift-share-high")


def create_accuracy_alarm(sns_topic_arn: str):
    """
    Alarm: triggers when model accuracy drops below threshold.
    Expects the serving layer to push ModelAccuracy metrics.
    """
    cloudwatch.put_metric_alarm(
        AlarmName="mlops-model-accuracy-low",
        AlarmDescription="Model accuracy below 80% — quality gate violation",
        Namespace="MLOps/ChurnModel",
        MetricName="ModelAccuracy",
        Dimensions=[{"Name": "Model", "Value": "xgboost-churn"}],
        Statistic="Average",
        Period=3600,  # 1 hour
        EvaluationPeriods=3,
        Threshold=0.80,
        ComparisonOperator="LessThanThreshold",
        AlarmActions=[sns_topic_arn],
        TreatMissingData="notBreaching",
        Tags=[{"Key": "Project", "Value": "mlops-churn-platform"}],
    )
    logger.info("Created alarm: mlops-model-accuracy-low")


def create_api_latency_alarm(sns_topic_arn: str):
    """
    Alarm: triggers when API P99 latency exceeds 2 seconds.
    Requires custom metrics from the FastAPI serving layer.
    """
    cloudwatch.put_metric_alarm(
        AlarmName="mlops-api-latency-high",
        AlarmDescription="API P99 latency exceeds 2 seconds",
        Namespace="MLOps/ChurnModel",
        MetricName="PredictLatencyP99",
        Dimensions=[{"Name": "Endpoint", "Value": "predict"}],
        Statistic="p99",
        Period=300,  # 5 minutes
        EvaluationPeriods=3,
        Threshold=2000,  # 2 seconds in ms
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[sns_topic_arn],
        TreatMissingData="notBreaching",
        Tags=[{"Key": "Project", "Value": "mlops-churn-platform"}],
    )
    logger.info("Created alarm: mlops-api-latency-high")


def create_daily_schedule():
    """
    CloudWatch Events rule to trigger the Lambda drift monitor daily at 8 AM UTC.
    """
    rule_name = "mlops-daily-drift-check"

    events.put_rule(
        Name=rule_name,
        ScheduleExpression="cron(0 8 * * ? *)",
        State="ENABLED",
        Description="Trigger drift monitoring Lambda daily at 8 AM UTC",
        Tags=[{"Key": "Project", "Value": "mlops-churn-platform"}],
    )

    # Get Lambda ARN
    if ACCOUNT_ID:
        lambda_arn = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{LAMBDA_FUNCTION_NAME}"
    else:
        try:
            func = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
            lambda_arn = func["Configuration"]["FunctionArn"]
        except Exception as e:
            logger.warning(f"Could not find Lambda function: {e}")
            logger.info(f"Skipping target for rule {rule_name}. Add manually after deploying Lambda.")
            return

    events.put_targets(
        Rule=rule_name,
        Targets=[
            {
                "Id": "drift-monitor-target",
                "Arn": lambda_arn,
                "Input": json.dumps({"source": "cloudwatch-scheduled"}),
            }
        ],
    )

    # Grant Events permission to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId="AllowCloudWatchInvoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{rule_name}",
        )
    except lambda_client.exceptions.ResourceConflictException:
        logger.info("Permission already exists.")

    logger.info(f"Created schedule: {rule_name} → {LAMBDA_FUNCTION_NAME}")


def create_dashboard(sns_topic_arn: str):
    """Create a CloudWatch dashboard for model observability."""
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "Data Drift Share",
                    "metrics": [
                        ["MLOps/ChurnModel", "DriftShare", "Model", "xgboost-churn"],
                    ],
                    "period": 86400,
                    "stat": "Maximum",
                    "region": REGION,
                    "annotations": {
                        "horizontal": [{"value": 0.2, "label": "Threshold"}],
                    },
                },
            },
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "Model Accuracy",
                    "metrics": [
                        ["MLOps/ChurnModel", "ModelAccuracy", "Model", "xgboost-churn"],
                    ],
                    "period": 3600,
                    "stat": "Average",
                    "region": REGION,
                    "annotations": {
                        "horizontal": [{"value": 0.8, "label": "Quality Gate"}],
                    },
                },
            },
            {
                "type": "metric",
                "x": 0, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "title": "API Predict Latency (P99)",
                    "metrics": [
                        ["MLOps/ChurnModel", "PredictLatencyP99", "Endpoint", "predict"],
                    ],
                    "period": 300,
                    "stat": "p99",
                    "region": REGION,
                },
            },
            {
                "type": "alarm",
                "x": 12, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "title": "Active Alarms",
                    "alarms": [
                        f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:alarm:mlops-drift-share-high",
                        f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:alarm:mlops-model-accuracy-low",
                        f"arn:aws:cloudwatch:{REGION}:{ACCOUNT_ID}:alarm:mlops-api-latency-high",
                    ],
                },
            },
        ],
    }

    cloudwatch.put_dashboard(
        DashboardName="MLOps-Churn-Platform",
        DashboardBody=json.dumps(dashboard_body),
    )
    logger.info("Created CloudWatch dashboard: MLOps-Churn-Platform")


def main():
    logger.info("Setting up CloudWatch monitoring infrastructure...")

    # 1. SNS topic
    topic_arn = create_sns_topic()

    # 2. Alarms
    create_drift_alarm(topic_arn)
    create_accuracy_alarm(topic_arn)
    create_api_latency_alarm(topic_arn)

    # 3. Scheduled trigger
    create_daily_schedule()

    # 4. Dashboard
    create_dashboard(topic_arn)

    logger.info("✅ CloudWatch infrastructure setup complete!")
    logger.info(f"   SNS Topic: {topic_arn}")
    logger.info(f"   Dashboard: https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards:name=MLOps-Churn-Platform")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Subscribe your email to the SNS topic for alerts")
    logger.info("  2. Deploy the Lambda function (lambda/drift_handler.py)")
    logger.info("  3. Verify the dashboard in the AWS Console")


if __name__ == "__main__":
    main()
