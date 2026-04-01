#!/usr/bin/env bash
set -euo pipefail

# Deploy FastAPI app to AWS Lambda using zip packaging + Mangum adapter.
# Usage:
#   export AWS_REGION=us-east-1
#   export LAMBDA_ROLE_ARN=arn:aws:iam::<account-id>:role/<lambda-role>
#   export FUNCTION_NAME=mlops-churn-fastapi
#   bash lambda/deploy_fastapi_lambda.sh

AWS_REGION="${AWS_REGION:-us-east-1}"
FUNCTION_NAME="${FUNCTION_NAME:-mlops-churn-fastapi}"
RUNTIME="python3.10"
HANDLER="fastapi_handler.handler"
TIMEOUT="30"
MEMORY_SIZE="1024"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAMBDA_DIR="$ROOT_DIR/lambda"
PKG_DIR="$LAMBDA_DIR/fastapi-package"
ZIP_PATH="$LAMBDA_DIR/fastapi.zip"

rm -rf "$PKG_DIR" "$ZIP_PATH"
mkdir -p "$PKG_DIR"

python -m pip install -r "$LAMBDA_DIR/requirements-serving.txt" -t "$PKG_DIR"
cp "$LAMBDA_DIR/fastapi_handler.py" "$PKG_DIR/"
cp -R "$ROOT_DIR/src" "$PKG_DIR/src"
cp -R "$ROOT_DIR/models" "$PKG_DIR/models"

(
  cd "$PKG_DIR"
  zip -qr "$ZIP_PATH" .
)

if [[ -z "${S3_BUCKET:-}" ]]; then
  echo "S3_BUCKET is required for Lambda code upload."
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d%H%M%S)"
CODE_S3_KEY="lambda-artifacts/${FUNCTION_NAME}/fastapi-${TIMESTAMP}.zip"
echo "Uploading Lambda package to s3://${S3_BUCKET}/${CODE_S3_KEY}"
aws s3 cp "$ZIP_PATH" "s3://${S3_BUCKET}/${CODE_S3_KEY}" --region "$AWS_REGION" >/dev/null

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Updating existing Lambda function: $FUNCTION_NAME"
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --s3-bucket "$S3_BUCKET" \
    --s3-key "$CODE_S3_KEY" \
    --region "$AWS_REGION" >/dev/null

  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --handler "$HANDLER" \
    --runtime "$RUNTIME" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "Variables={MODEL_DIR=models}" \
    --region "$AWS_REGION" >/dev/null
else
  echo "Creating new Lambda function: $FUNCTION_NAME"
  if [[ -z "${LAMBDA_ROLE_ARN:-}" ]]; then
    echo "LAMBDA_ROLE_ARN is required to create function $FUNCTION_NAME."
    exit 1
  fi
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" \
    --handler "$HANDLER" \
    --code "S3Bucket=$S3_BUCKET,S3Key=$CODE_S3_KEY" \
    --role "$LAMBDA_ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "Variables={MODEL_DIR=models}" \
    --region "$AWS_REGION" >/dev/null
fi

echo "Deployment complete for $FUNCTION_NAME in $AWS_REGION"
echo "Tip: attach API Gateway (HTTP API) or enable Lambda Function URL for public access."
