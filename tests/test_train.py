"""
Unit tests for training utilities.
Uses synthetic data — validates model creation and metric computation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.training.train import train_model, evaluate_model


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic binary classification dataset."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.rand(n) * 10,
        "feature_4": np.random.randint(0, 5, n),
    })
    # Create a target with some signal
    y = pd.Series(
        ((X["feature_1"] + X["feature_3"] > 5).astype(int)).values,
        name="Churn",
    )
    return X, y


@pytest.fixture
def train_test_data(synthetic_dataset):
    """Split and scale synthetic data."""
    X, y = synthetic_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


class TestTrainModel:
    def test_model_trains_successfully(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        model = train_model(X_train, y_train, params)
        assert model is not None

    def test_model_can_predict(self, train_test_data):
        X_train, X_test, y_train, _ = train_test_data
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        model = train_model(X_train, y_train, params)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

    def test_model_predict_proba(self, train_test_data):
        X_train, X_test, y_train, _ = train_test_data
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        model = train_model(X_train, y_train, params)
        probas = model.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        assert np.all(probas >= 0) and np.all(probas <= 1)


class TestEvaluateModel:
    def test_evaluate_returns_metrics(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        model = train_model(X_train, y_train, params)
        metrics = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "f1_score" in metrics

    def test_metrics_in_valid_range(self, train_test_data):
        X_train, X_test, y_train, y_test = train_test_data
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
        }
        model = train_model(X_train, y_train, params)
        metrics = evaluate_model(model, X_test, y_test)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of range"
