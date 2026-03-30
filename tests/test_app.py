"""
Integration tests for the FastAPI inference server.
Uses a mock model so no actual model files are needed.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


# Mock model artifacts before import
def _create_mock_artifacts():
    """Create mock model + scaler that behave like real ones."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.1] * 19])

    return {
        "model": mock_model,
        "scaler": mock_scaler,
        "feature_names": [f"feature_{i}" for i in range(19)],
        "metrics": {"accuracy": 0.85, "roc_auc": 0.90, "f1_score": 0.82},
    }


@pytest.fixture
def client():
    """Create a test client with mocked model artifacts."""
    from src.serving.app import app, model_artifacts

    # Inject mock artifacts
    mock = _create_mock_artifacts()
    model_artifacts.update(mock)

    with TestClient(app) as c:
        yield c

    model_artifacts.clear()


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_includes_metrics(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "accuracy" in data["metrics"]


class TestPredictEndpoint:
    def test_single_predict(self, client):
        payload = {"features": [0.0] * 19}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "churn" in data
        assert "churn_probability" in data
        assert "label" in data
        assert data["churn"] in [0, 1]

    def test_predict_label_matches_prediction(self, client):
        payload = {"features": [0.0] * 19}
        resp = client.post("/predict", json=payload)
        data = resp.json()
        if data["churn"] == 1:
            assert data["label"] == "Will Churn"
        else:
            assert data["label"] == "Will Stay"

    def test_predict_missing_features(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422  # Pydantic validation error


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client):
        # Override mock to return batch results
        from src.serving.app import model_artifacts

        model_artifacts["model"].predict.return_value = np.array([0, 1])
        model_artifacts["model"].predict_proba.return_value = np.array(
            [[0.9, 0.1], [0.3, 0.7]]
        )
        model_artifacts["scaler"].transform.return_value = np.array(
            [[0.1] * 19, [0.2] * 19]
        )

        payload = {"instances": [[0.0] * 19, [1.0] * 19]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 2


class TestFeaturesEndpoint:
    def test_get_features(self, client):
        resp = client.get("/features")
        assert resp.status_code == 200
        data = resp.json()
        assert "feature_names" in data
        assert len(data["feature_names"]) == 19
