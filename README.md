# 🚀 Full MLOps Platform — Customer Churn Prediction

A production-grade MLOps platform built on AWS, demonstrating the full ML lifecycle:
data preprocessing → training → evaluation → deployment → monitoring.

## Architecture

```
GitHub Push
    │
    ▼
GitHub Actions CI/CD
    ├── Lint & Test (ruff + pytest)
    ├── Build & Push Docker → ECR
    └── Trigger SageMaker Pipeline
            │
            ├── Step 1: Preprocessing  (SKLearn Processor)
            ├── Step 2: Training       (XGBoost on SageMaker)
            ├── Step 3: Evaluation     (Quality Gate: accuracy ≥ 80%)
            └── Step 4: Register Model (SageMaker Model Registry)
                            │
                            ▼
                   SageMaker Endpoint (Real-time serving)
                            │
                            ▼
                   FastAPI wrapper → API Gateway
                            │
                            ▼
                   Monitoring (Evidently + CloudWatch + SNS)
                            │
                            ▼
                   Lambda Drift Monitor (scheduled daily)
                            │
                            ▼
                   CloudWatch Dashboard + Alarms → SNS Alerts
```

## Tech Stack

| Layer | Tool |
|---|---|
| Experiment Tracking | MLflow |
| Data Storage | AWS S3 |
| Pipeline Orchestration | SageMaker Pipelines |
| CI/CD | GitHub Actions |
| Model Registry | SageMaker Model Registry |
| Serving | SageMaker Endpoints + FastAPI |
| Monitoring | Evidently AI + CloudWatch |
| Alerting | AWS SNS |
| Scheduled Tasks | AWS Lambda + CloudWatch Events |
| Containerization | Docker + ECR |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/mlops-churn-platform
cd mlops-churn-platform
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials and configuration
```

### 3. Download dataset

Download from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Run locally (Phase 1)

```bash
# Preprocess
python src/data/preprocess.py

# Train with MLflow tracking
python src/training/train.py

# View MLflow UI
mlflow ui  # open http://localhost:5000

# Start inference server
MODEL_DIR=models uvicorn src.serving.app:app --reload
# open http://localhost:8000/docs
```

### 5. Run tests

```bash
pip install -r requirements-dev.txt

# Lint
ruff check src/ pipelines/ monitoring/ lambda/ infra/ tests/

# Unit & integration tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 6. Configure AWS (Phase 2)

```bash
# Set environment variables (or use .env file)
export AWS_REGION=us-east-1
export S3_BUCKET=your-unique-bucket-name
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole

# Upload data to S3
aws s3 cp data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv s3://$S3_BUCKET/raw/telco-churn.csv

# Build and push Docker image
aws ecr create-repository --repository-name mlops-churn-platform
docker build -t mlops-churn-platform .
docker tag mlops-churn-platform:latest YOUR_ECR_URI/mlops-churn-platform:latest
docker push YOUR_ECR_URI/mlops-churn-platform:latest
```

### 7. Run SageMaker Pipeline (Phase 3)

```bash
# Create + run pipeline
python pipelines/sagemaker_pipeline.py --run
```

### 8. Deploy Lambda Drift Monitor (Phase 4)

```bash
# Package Lambda function
cd lambda
pip install -r requirements.txt -t package/
cp drift_handler.py package/
cd package && zip -r ../drift-monitor.zip . && cd ..

# Create Lambda function
aws lambda create-function \
  --function-name mlops-drift-monitor \
  --runtime python3.10 \
  --handler drift_handler.handler \
  --zip-file fileb://drift-monitor.zip \
  --role $SAGEMAKER_ROLE_ARN \
  --timeout 300 \
  --memory-size 512 \
  --environment "Variables={S3_BUCKET=$S3_BUCKET,SNS_TOPIC_ARN=your-topic-arn,DRIFT_THRESHOLD=0.20}"
```

### 9. Set up CloudWatch monitoring (Phase 5)

```bash
# Create alarms, dashboard, and daily schedule
export AWS_ACCOUNT_ID=your-account-id
python infra/cloudwatch_alarms.py
```

This creates:
- **SNS Topic** for email/Slack alerts
- **CloudWatch Alarms**: drift share > 20%, accuracy < 80%, API latency P99 > 2s
- **CloudWatch Events Rule**: triggers Lambda daily at 8 AM UTC
- **CloudWatch Dashboard**: real-time model observability

### 10. GitHub Secrets (for CI/CD)

Add these in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret |
| `S3_BUCKET` | Your S3 bucket name |
| `SAGEMAKER_ROLE_ARN` | Your SageMaker role ARN |

---

## Project Structure

```
mlops-churn-platform/
├── src/
│   ├── data/
│   │   └── preprocess.py       # Data cleaning & feature engineering
│   ├── training/
│   │   └── train.py            # XGBoost training + MLflow logging
│   ├── evaluation/
│   │   └── evaluate.py         # Model evaluation with quality gate
│   └── serving/
│       └── app.py              # FastAPI inference server
├── pipelines/
│   └── sagemaker_pipeline.py   # SageMaker Pipeline definition
├── monitoring/
│   └── drift_monitor.py        # Evidently drift detection (local)
├── lambda/
│   ├── drift_handler.py        # AWS Lambda drift monitor (serverless)
│   └── requirements.txt        # Lambda dependencies
├── infra/
│   └── cloudwatch_alarms.py    # CloudWatch alarms, dashboard, schedules
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml  # GitHub Actions CI/CD
├── tests/
│   ├── test_preprocess.py      # Preprocessing unit tests
│   ├── test_train.py           # Training unit tests
│   └── test_app.py             # FastAPI integration tests
├── configs/
│   └── config.yaml             # Centralized config
├── data/
│   ├── raw/                    # Raw dataset (gitignored)
│   └── processed/              # Preprocessed data (gitignored)
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## Key MLOps Concepts Demonstrated

- ✅ **Experiment tracking** with MLflow
- ✅ **Automated retraining** on code push via GitHub Actions
- ✅ **Quality gates** — pipeline fails if accuracy drops below threshold
- ✅ **Model registry** — versioned models with approval workflow
- ✅ **Containerized serving** — Docker + FastAPI
- ✅ **Data drift detection** — Evidently AI + Lambda scheduled monitoring
- ✅ **CloudWatch observability** — dashboards, alarms, custom metrics
- ✅ **Alerting pipeline** — SNS notifications for drift/accuracy/latency
- ✅ **Infrastructure as Code** — SageMaker Pipelines + CloudWatch setup in Python
- ✅ **CI/CD** — lint, test, build, deploy on every push

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (default: us-east-1) |
| `S3_BUCKET` | S3 bucket for data and artifacts |
| `SAGEMAKER_ROLE_ARN` | IAM role for SageMaker |
| `SNS_TOPIC_ARN` | SNS topic for alerts |
| `MODEL_DIR` | Local model directory (default: models) |
| `MLFLOW_TRACKING_URI` | MLflow URI (default: ./mlruns) |
