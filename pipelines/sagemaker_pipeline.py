"""
SageMaker Pipeline — Full MLOps pipeline:
  1. Preprocessing
  2. Training
  3. Evaluation (with quality gate)
  4. Model registration
  
Run: python pipelines/sagemaker_pipeline.py --run
"""

import os
import json
import boto3
import argparse
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost import XGBoost
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.workflow.functions import JsonGet


# ── Config ───────────────────────────────────────────────────────────────────

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("S3_BUCKET", "your-mlops-bucket")
ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole")
PIPELINE_NAME = "churn-prediction-pipeline"
MODEL_PACKAGE_GROUP = "churn-prediction-models"


def get_session():
    boto_session = boto3.Session(region_name=REGION)
    return sagemaker.Session(boto_session=boto_session)


def build_pipeline(sagemaker_session):
    # ── Pipeline parameters (can be overridden at runtime) ───────────────────
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.80)
    n_estimators = ParameterInteger(name="NEstimators", default_value=100)
    max_depth = ParameterInteger(name="MaxDepth", default_value=6)
    learning_rate = ParameterFloat(name="LearningRate", default_value=0.1)

    # ── Step 1: Preprocessing ────────────────────────────────────────────────
    sklearn_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )

    preprocessing_step = ProcessingStep(
        name="ChurnPreprocessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/raw/telco-churn.csv",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{BUCKET}/processed/train",
            ),
        ],
        code="src/data/preprocess.py",
    )

    # ── Step 2: Training ─────────────────────────────────────────────────────
    xgb_estimator = XGBoost(
        entry_point="src/training/train.py",
        framework_version="1.7-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "n-estimators": n_estimators,
            "max-depth": max_depth,
            "learning-rate": learning_rate,
        },
        output_path=f"s3://{BUCKET}/models",
        metric_definitions=[
            {"Name": "accuracy", "Regex": r"accuracy: ([0-9\.]+)"},
            {"Name": "roc_auc", "Regex": r"roc_auc: ([0-9\.]+)"},
        ],
    )

    training_step = TrainingStep(
        name="ChurnTraining",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    # ── Step 3: Evaluation ───────────────────────────────────────────────────
    evaluation_processor = SKLearnProcessor(
        framework_version="1.0-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=ROLE,
        sagemaker_session=sagemaker_session,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluation_step = ProcessingStep(
        name="ChurnEvaluation",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=f"s3://{BUCKET}/processed/train",
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{BUCKET}/evaluation",
            )
        ],
        code="src/evaluation/evaluate.py",
        property_files=[evaluation_report],
    )

    # ── Step 4: Register model (only if accuracy passes threshold) ───────────
    model = Model(
        image_uri=xgb_estimator.training_image_uri(),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=ROLE,
    )

    register_step = ModelStep(
        name="RegisterChurnModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["application/json"],
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status="PendingManualApproval",  # Change to Approved to auto-deploy
            model_metrics={},
        ),
    )

    # ── Step 5: Condition gate ────────────────────────────────────────────────
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="metrics.accuracy.value",
        ),
        right=accuracy_threshold,
    )

    condition_step = ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[condition],
        if_steps=[register_step],
        else_steps=[],  # Could add a notification step here
    )

    # ── Assemble pipeline ─────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[accuracy_threshold, n_estimators, max_depth, learning_rate],
        steps=[preprocessing_step, training_step, evaluation_step, condition_step],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute the pipeline after creating it")
    parser.add_argument("--upsert-only", action="store_true", help="Only create/update pipeline, don't run")
    args = parser.parse_args()

    session = get_session()
    pipeline = build_pipeline(session)

    print("Upserting pipeline...")
    pipeline.upsert(role_arn=ROLE)
    print(f"Pipeline '{PIPELINE_NAME}' created/updated successfully.")

    if args.run:
        print("Starting pipeline execution...")
        execution = pipeline.start()
        print(f"Pipeline execution started: {execution.arn}")
        print("Waiting for completion (this may take 20-30 minutes)...")
        execution.wait()
        print("Pipeline execution complete!")
        print(json.dumps(execution.list_steps(), indent=2))


if __name__ == "__main__":
    main()
