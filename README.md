# 🚀 Full MLOps Platform — Customer Churn Prediction

A production-grade MLOps platform demonstrating the full ML lifecycle:
data preprocessing → training → evaluation → deployment → monitoring.

**100% AWS Free Tier** — no SageMaker, no paid instances.

## Architecture

```
GitHub Push
    │
    ▼
GitHub Actions CI/CD
    ├── Lint & Test (ruff + pytest)
    └── Train → Evaluate (quality gate) → Upload to S3
                    │
                    ▼
              MLflow (local experiment tracking + model registry)
                    │
                    ▼
              S3 (model artifact storage — free tier: 5GB)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   FastAPI (local)    Lambda + API Gateway
   Development         Production serving
                      (free tier: 1M req/mo)
                              │
                              ▼
              Lambda drift monitor (daily cron)
                              │
                              ▼
              CloudWatch dashboard + alarms → SNS alerts
```

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Experiment Tracking | MLflow (local) | Free |
| Data Storage | AWS S3 | Free tier (5GB) |
| Pipeline Orchestration | Local Python + GitHub Actions | Free |
| CI/CD | GitHub Actions | Free (public repos) |
| Model Registry | MLflow Model Registry | Free |
| Serving (dev) | FastAPI + Docker | Free |
| Serving (prod) | AWS Lambda + API Gateway | Free tier (1M req/mo) |
| Monitoring | Evidently AI + CloudWatch | Free tier (10 alarms) |
| Alerting | AWS SNS | Free tier (1M pub/mo) |
| Scheduled Tasks | Lambda + CloudWatch Events | Free tier |

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/mlops-churn-platform
cd mlops-churn-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Download dataset

Download from Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Place the CSV at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 4. Run the full pipeline locally

```bash
# Runs: preprocess → train → evaluate → register model
python pipelines/local_pipeline.py --skip-upload

# With S3 upload (requires AWS credentials)
export S3_BUCKET=your-bucket
python pipelines/local_pipeline.py
```

### 5. Or run each step individually

```bash
# Preprocess
python src/data/preprocess.py

# Train with MLflow tracking
python src/training/train.py

# View MLflow UI
mlflow ui  # open http://localhost:5000

# Start local inference server
MODEL_DIR=models uvicorn src.serving.app:app --reload
# open http://localhost:8000/docs
```

### 6. Run tests

```bash
pip install -r requirements-dev.txt

# Lint
ruff check src/ pipelines/ monitoring/ tests/

# Unit & integration tests
pytest tests/ -v --tb=short

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 7. Configure AWS (free tier only)

```bash
export AWS_REGION=us-east-1
export S3_BUCKET=your-unique-bucket-name

# Create S3 bucket
aws s3 mb s3://$S3_BUCKET

# Upload data to S3
aws s3 cp data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv s3://$S3_BUCKET/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 8. Deploy Lambda inference (serverless serving)

```bash
# Package Lambda function with model
cd lambda
pip install -r requirements-serving.txt -t package/
cp serve_handler.py package/
cp -r ../models/ package/models/
cd package && zip -r ../serve.zip . && cd ..

# Create Lambda function
aws lambda create-function \
  --function-name mlops-churn-predict \
  --runtime python3.10 \
  --handler serve_handler.handler \
  --zip-file fileb://serve.zip \
  --role $LAMBDA_ROLE_ARN \
  --timeout 30 \
  --memory-size 256

# Add API Gateway trigger (optional, for REST endpoint)
# See: https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html
```

### 8b. Deploy FastAPI directly on Lambda (recommended)

This deploys the same app from src/serving/app.py to Lambda using Mangum.

```bash
export AWS_REGION=us-east-1
export LAMBDA_ROLE_ARN=arn:aws:iam::<account-id>:role/<lambda-exec-role>
export FUNCTION_NAME=mlops-churn-fastapi

bash lambda/deploy_fastapi_lambda.sh
```

Then expose it with one of these options:

1. API Gateway HTTP API integration to Lambda
2. Lambda Function URL

Quick Function URL setup:

```bash
aws lambda create-function-url-config \
  --function-name "$FUNCTION_NAME" \
  --auth-type NONE \
  --region "$AWS_REGION"

aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal '*' \
  --function-url-auth-type NONE \
  --region "$AWS_REGION"
```

After that, open /docs on the Function URL to access Swagger.

### 8c. Required IAM + ECR permissions for image-based Lambda deploys

If your workflow fails with `AccessDeniedException` during `CreateFunction` and mentions ECR access, the CI principal cannot validate/update the ECR repository policy.

At minimum, grant the CI principal these actions on the target ECR repository:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowReadWriteEcrRepositoryPolicy",
      "Effect": "Allow",
      "Action": [
        "ecr:GetRepositoryPolicy",
        "ecr:SetRepositoryPolicy"
      ],
      "Resource": "arn:aws:ecr:ap-south-1:<ACCOUNT_ID>:repository/<ECR_REPOSITORY>"
    }
  ]
}
```

Also ensure the ECR repository policy allows Lambda pulls:

```json
{
  "Version": "2008-10-17",
  "Statement": [
    {
      "Sid": "LambdaECRImageRetrievalPolicy",
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Condition": {
        "StringLike": {
          "aws:sourceArn": "arn:aws:lambda:ap-south-1:<ACCOUNT_ID>:function:*"
        },
        "StringEquals": {
          "aws:sourceAccount": "<ACCOUNT_ID>"
        }
      }
    }
  ]
}
```

Optional but recommended for CI image build/push flows:

- `ecr:DescribeRepositories`
- `ecr:GetAuthorizationToken`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`

### 9. Deploy Lambda drift monitor

```bash
cd lambda
pip install -r requirements.txt -t drift-package/
cp drift_handler.py drift-package/
cd drift-package && zip -r ../drift.zip . && cd ..

aws lambda create-function \
  --function-name mlops-drift-monitor \
  --runtime python3.10 \
  --handler drift_handler.handler \
  --zip-file fileb://drift.zip \
  --role $LAMBDA_ROLE_ARN \
  --timeout 300 \
  --memory-size 512 \
  --environment "Variables={S3_BUCKET=$S3_BUCKET,DRIFT_THRESHOLD=0.20}"
```

### 10. Set up CloudWatch monitoring

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
python infra/cloudwatch_alarms.py
```

This creates:
- **SNS Topic** for email/Slack alerts
- **CloudWatch Alarms**: drift share > 20%, accuracy < 80%, API latency P99 > 2s
- **CloudWatch Events Rule**: triggers Lambda daily at 8 AM UTC
- **CloudWatch Dashboard**: real-time model observability

### 11. GitHub Secrets (for CI/CD)

Add in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Your IAM key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret |
| `S3_BUCKET` | Your S3 bucket name |

Optional (enable full zero-manual AWS rollout):

| Secret | Value |
|---|---|
| `AWS_ROLE_TO_ASSUME` | Optional OIDC role ARN (recommended) |
| `LAMBDA_DEPLOY_ENABLED` | `true` to auto-deploy Lambda functions |
| `SERVING_FUNCTION_NAME` | e.g. `mlops-churn-fastapi` |
| `DRIFT_FUNCTION_NAME` | e.g. `mlops-drift-monitor` |
| `LAMBDA_ROLE_ARN` | Required only when creating functions |
| `MONITORING_SETUP_ENABLED` | `true` to auto-apply CloudWatch/SNS resources |
| `SNS_TOPIC_NAME` | e.g. `mlops-alerts` |

The workflow runs on push to `main` and also daily (`0 6 * * *`), so model refresh and production rollout can happen automatically without manual steps.

---

## Project Structure

```
mlops-churn-platform/
├── src/
│   ├── data/
│   │   └── preprocess.py          # Data cleaning & feature engineering
│   ├── training/
│   │   └── train.py               # XGBoost training + MLflow logging
│   ├── evaluation/
│   │   └── evaluate.py            # Model evaluation with quality gate
│   └── serving/
│       └── app.py                 # FastAPI inference server (local dev)
├── pipelines/
│   ├── local_pipeline.py          # Full pipeline orchestrator (free)
│   └── sagemaker_pipeline.py      # SageMaker pipeline (reference only)
├── lambda/
│   ├── serve_handler.py           # Lambda inference (serverless serving)
│   ├── drift_handler.py           # Lambda drift monitor (scheduled)
│   ├── requirements.txt           # Drift monitor deps
│   └── requirements-serving.txt   # Serving deps
├── monitoring/
│   └── drift_monitor.py           # Evidently drift detection (local)
├── infra/
│   └── cloudwatch_alarms.py       # CloudWatch alarms, dashboard, schedules
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml     # GitHub Actions CI/CD
├── tests/
│   ├── test_preprocess.py         # Preprocessing unit tests
│   ├── test_train.py              # Training unit tests
│   └── test_app.py                # FastAPI integration tests
├── configs/
│   └── config.yaml                # Centralized config
├── data/
│   ├── raw/                       # Raw dataset (gitignored)
│   └── processed/                 # Preprocessed data (gitignored)
├── Dockerfile                     # Container for local serving
├── requirements.txt
├── requirements-dev.txt
├── conftest.py                    # Pytest root config
├── .env.example
└── README.md
```

---

## AWS Free Tier Usage

| Service | Free Tier Limit | Our Usage |
|---|---|---|
| S3 | 5GB, 20k GET/mo | Model artifacts + data (~50MB) |
| Lambda | 1M requests, 400k GB-sec/mo | Inference + drift monitoring |
| CloudWatch | 10 metrics, 10 alarms | 3 alarms, 1 dashboard |
| SNS | 1M publishes/mo | Alert notifications |
| API Gateway | 1M calls/mo (12 months) | REST endpoint for Lambda |

---

## Key MLOps Concepts Demonstrated

- ✅ **Experiment tracking** with MLflow
- ✅ **Automated retraining** on code push + daily schedule via GitHub Actions
- ✅ **Quality gates** — pipeline fails if accuracy drops below threshold
- ✅ **Model registry** — versioned models with MLflow
- ✅ **Local serving** — Docker + FastAPI
- ✅ **Serverless serving** — Lambda + API Gateway (free tier)
- ✅ **Data drift detection** — Evidently AI + Lambda scheduled monitoring
- ✅ **CloudWatch observability** — dashboards, alarms, custom metrics
- ✅ **Alerting pipeline** — SNS notifications for drift/accuracy/latency
- ✅ **CI/CD** — lint, test, train, deploy, and monitoring bootstrap automatically
- ✅ **100% free tier** — no paid AWS services
