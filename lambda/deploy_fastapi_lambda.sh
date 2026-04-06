#!/usr/bin/env bash
set -euo pipefail

# Deploy FastAPI app to AWS Lambda as a CONTAINER IMAGE (ECR-backed).
# This avoids Lambda zip limits (50 MB zipped / 250 MB unzipped).
# Usage:
#   export AWS_REGION=ap-south-1
#   export LAMBDA_ROLE_ARN=arn:aws:iam::<account-id>:role/<lambda-role>
#   export FUNCTION_NAME=mlops-churn-fastapi
#   export ECR_REPOSITORY=mlops-churn-fastapi
#   bash lambda/deploy_fastapi_lambda.sh

AWS_REGION="${AWS_REGION:-ap-south-1}"
FUNCTION_NAME="${FUNCTION_NAME:-mlops-churn-fastapi}"
ECR_REPOSITORY="${ECR_REPOSITORY:-$FUNCTION_NAME}"
TIMEOUT="30"
MEMORY_SIZE="1024"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCKERFILE_PATH="$ROOT_DIR/lambda/Dockerfile.fastapi"

if [[ -z "${LAMBDA_ROLE_ARN:-}" ]]; then
  echo "LAMBDA_ROLE_ARN is required for Lambda create/update."
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")"
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="$(date +%Y%m%d%H%M%S)"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "Ensuring ECR repository exists: ${ECR_REPOSITORY}"
if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "ECR repository not found or access denied: ${ECR_REPOSITORY}"
  echo "Create it once in AWS Console (or grant ecr:CreateRepository), then rerun CI."
  exit 1
fi

echo "Configuring ECR repository policy for Lambda image pulls"
REPO_POLICY_FILE="$(mktemp)"
cat > "$REPO_POLICY_FILE" <<EOF
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
          "aws:sourceArn": "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:*"
        },
        "StringEquals": {
          "aws:sourceAccount": "${ACCOUNT_ID}"
        }
      ]
    }
  ]
}
EOF

set_policy_ok="false"
if aws ecr set-repository-policy \
  --repository-name "$ECR_REPOSITORY" \
  --policy-text "file://$REPO_POLICY_FILE" \
  --region "$AWS_REGION" >/dev/null 2>&1; then
  set_policy_ok="true"
fi

if [[ "$set_policy_ok" != "true" ]]; then
  echo "Could not update ECR repository policy automatically (missing ecr:SetRepositoryPolicy?)."
  echo "Checking whether an existing policy already allows Lambda pulls..."

  existing_policy="$(aws ecr get-repository-policy --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION" --query 'policyText' --output text 2>/dev/null || true)"
  if [[ "$existing_policy" == *"lambda.amazonaws.com"* ]] && [[ "$existing_policy" == *"ecr:BatchGetImage"* ]] && [[ "$existing_policy" == *"ecr:GetDownloadUrlForLayer"* ]]; then
    echo "Existing ECR policy appears to allow Lambda image pulls. Continuing."
  else
    echo "ERROR: ECR policy does not allow Lambda to pull image from ${ECR_REPOSITORY}."
    echo "Grant CI identity these permissions, then rerun:"
    echo "  - ecr:GetRepositoryPolicy"
    echo "  - ecr:SetRepositoryPolicy"
    echo "Or manually set repository policy to include lambda.amazonaws.com with actions:"
    echo "  - ecr:BatchGetImage"
    echo "  - ecr:GetDownloadUrlForLayer"
    echo
    echo "One-time manual fix (run with an admin AWS profile):"
    echo "  export AWS_REGION=${AWS_REGION}"
    echo "  export ECR_REPOSITORY=${ECR_REPOSITORY}"
    echo "  export ACCOUNT_ID=${ACCOUNT_ID}"
    echo "  cat > /tmp/ecr-lambda-policy.json <<'JSON'"
    echo "  {"
    echo "    \"Version\": \"2008-10-17\"," 
    echo "    \"Statement\": ["
    echo "      {"
    echo "        \"Sid\": \"LambdaECRImageRetrievalPolicy\"," 
    echo "        \"Effect\": \"Allow\"," 
    echo "        \"Principal\": { \"Service\": \"lambda.amazonaws.com\" },"
    echo "        \"Action\": ["
    echo "          \"ecr:BatchCheckLayerAvailability\"," 
    echo "          \"ecr:BatchGetImage\"," 
    echo "          \"ecr:GetDownloadUrlForLayer\""
    echo "        ],"
    echo "        \"Condition\": {"
    echo "          \"StringLike\": { \"aws:sourceArn\": \"arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:*\" },"
    echo "          \"StringEquals\": { \"aws:sourceAccount\": \"${ACCOUNT_ID}\" }"
    echo "        }"
    echo "      }"
    echo "    ]"
    echo "  }"
    echo "  JSON"
    echo "  aws ecr set-repository-policy --repository-name \"$ECR_REPOSITORY\" --policy-text file:///tmp/ecr-lambda-policy.json --region \"$AWS_REGION\""
    rm -f "$REPO_POLICY_FILE"
    exit 1
  fi
fi
rm -f "$REPO_POLICY_FILE"

echo "Logging into ECR: ${ECR_REGISTRY}"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null

echo "Building image: ${IMAGE_URI}"
docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_URI" "$ROOT_DIR"

echo "Pushing image: ${IMAGE_URI}"
docker push "$IMAGE_URI" >/dev/null

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  package_type="$(aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" --query 'Configuration.PackageType' --output text)"
  if [[ "$package_type" != "Image" ]]; then
    echo "Function ${FUNCTION_NAME} exists as package type '${package_type}'."
    echo "Delete it once (or use a new function name) to migrate to Image package type."
    exit 1
  fi

  echo "Updating existing image-based Lambda function: ${FUNCTION_NAME}"
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" >/dev/null

  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "Variables={MODEL_DIR=models}" \
    --region "$AWS_REGION" >/dev/null
else
  echo "Creating new image-based Lambda function: ${FUNCTION_NAME}"
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code "ImageUri=$IMAGE_URI" \
    --role "$LAMBDA_ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --environment "Variables={MODEL_DIR=models}" \
    --region "$AWS_REGION" >/dev/null
fi

echo "Deployment complete for ${FUNCTION_NAME} (${IMAGE_URI}) in ${AWS_REGION}"
echo "Tip: attach API Gateway (HTTP API) or enable Lambda Function URL for public access."
