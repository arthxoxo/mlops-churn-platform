"""
AWS Lambda inference handler — serverless model serving (free tier).
Replaces SageMaker Endpoints with Lambda + API Gateway.

Free tier: 1M requests/month + 400,000 GB-seconds compute.

Note: Lambda packages are self-contained, so MODEL_DIR typically points to a
relative path within the zip. Set MODEL_DIR env var when deploying if needed.

Deploy:
  cd lambda
  pip install -r requirements-serving.txt -t package/
  cp serve_handler.py package/
  cp -r ../models/ package/models/
  cd package && zip -r ../serve.zip . && cd ..

  aws lambda create-function \
    --function-name mlops-churn-predict \
    --runtime python3.10 \
    --handler serve_handler.handler \
    --zip-file fileb://serve.zip \
    --role $LAMBDA_ROLE_ARN \
    --timeout 30 \
    --memory-size 256 \
    --environment Variables={MODEL_DIR=models}
"""

import os
import json
import logging
import numpy as np
import joblib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load model at cold start (reused across invocations)
# For Lambda, MODEL_DIR defaults to 'models' (packaged in zip)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
model = None
scaler = None
feature_names = None


def load_model():
    """Load model artifacts once at cold start."""
    global model, scaler, feature_names
    if model is None:
        model = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
            feature_names = json.load(f)
        logger.info("Model loaded successfully.")


def handler(event, context):
    """
    Lambda handler for churn prediction.

    Accepts API Gateway proxy events or direct invocation.

    Input (via API Gateway POST /predict):
      {"features": [0, 12, 1, 0, ...]}

    Input (direct invocation):
      {"features": [0, 12, 1, 0, ...]}
      or
      {"instances": [[0, 12, ...], [1, 5, ...]]}  # batch
    """
    load_model()

    try:
        # Parse body (API Gateway wraps in 'body' field)
        if "body" in event:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        else:
            body = event

        # Health check
        if event.get("httpMethod") == "GET" or body.get("action") == "health":
            response_body = {
                "status": "ok",
                "model_loaded": model is not None,
                "feature_names": feature_names,
            }
            return _response(200, response_body)

        # Single prediction
        if "features" in body:
            X = np.array(body["features"]).reshape(1, -1)
            X_scaled = scaler.transform(X)
            pred = int(model.predict(X_scaled)[0])
            proba = float(model.predict_proba(X_scaled)[0][1])
            response_body = {
                "churn": pred,
                "churn_probability": round(proba, 4),
                "label": "Will Churn" if pred == 1 else "Will Stay",
            }
            return _response(200, response_body)

        # Batch prediction
        if "instances" in body:
            X = np.array(body["instances"])
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled).tolist()
            probas = model.predict_proba(X_scaled)[:, 1].tolist()
            response_body = {
                "predictions": [
                    {
                        "churn": int(p),
                        "churn_probability": round(float(pr), 4),
                        "label": "Will Churn" if p == 1 else "Will Stay",
                    }
                    for p, pr in zip(preds, probas)
                ]
            }
            return _response(200, response_body)

        return _response(400, {"error": "Provide 'features' or 'instances' in request body."})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return _response(500, {"error": str(e)})


def _response(status_code: int, body: dict) -> dict:
    """Format response for API Gateway."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
