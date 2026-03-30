"""
FastAPI inference server.
Run locally: uvicorn src.serving.app:app --reload
Or deploy as Docker container.
"""

import os
import json
import logging
from typing import List
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state
model_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    if all(key in model_artifacts for key in ("model", "scaler", "feature_names", "metrics")):
        logger.info("Using preloaded model artifacts.")
        yield
        model_artifacts.clear()
        return

    model_dir = os.environ.get("MODEL_DIR", "models")
    try:
        model_artifacts["model"] = joblib.load(os.path.join(model_dir, "model.joblib"))
        model_artifacts["scaler"] = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        with open(os.path.join(model_dir, "feature_names.json")) as f:
            model_artifacts["feature_names"] = json.load(f)
        with open(os.path.join(model_dir, "metrics.json")) as f:
            model_artifacts["metrics"] = json.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    model_artifacts.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="MLOps Portfolio — Real-time churn prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0, 12, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 0, 29.85, 29.85]
            }
        }


class PredictResponse(BaseModel):
    churn: int
    churn_probability: float
    label: str


class BatchPredictRequest(BaseModel):
    instances: List[List[float]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    metrics: dict


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in model_artifacts,
        "metrics": model_artifacts.get("metrics", {}),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if "model" not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = np.array(request.features).reshape(1, -1)
        X_scaled = model_artifacts["scaler"].transform(X)
        pred = int(model_artifacts["model"].predict(X_scaled)[0])
        proba = float(model_artifacts["model"].predict_proba(X_scaled)[0][1])
        return {
            "churn": pred,
            "churn_probability": round(proba, 4),
            "label": "Will Churn" if pred == 1 else "Will Stay",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if "model" not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = np.array(request.instances)
        X_scaled = model_artifacts["scaler"].transform(X)
        preds = model_artifacts["model"].predict(X_scaled).tolist()
        probas = model_artifacts["model"].predict_proba(X_scaled)[:, 1].tolist()
        return {
            "predictions": [
                {"churn": p, "churn_probability": round(pr, 4)}
                for p, pr in zip(preds, probas)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/features")
def get_features():
    return {"feature_names": model_artifacts.get("feature_names", [])}
