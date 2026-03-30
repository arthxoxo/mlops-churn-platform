"""
Unit tests for data preprocessing pipeline.
Uses synthetic data — no Kaggle download required.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import clean, encode_categoricals


@pytest.fixture
def raw_dataframe():
    """Create a synthetic DataFrame mimicking the Telco Churn schema."""
    return pd.DataFrame({
        "customerID": ["0001", "0002", "0003", "0004", "0005"],
        "gender": ["Female", "Male", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 0, 1, 0, 1],
        "Partner": ["Yes", "No", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [1, 34, 2, 45, 8],
        "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "One year"],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": ["29.85", "1889.5", " ", "1840.75", "151.65"],
        "Churn": ["Yes", "No", "Yes", "No", "No"],
    })


class TestClean:
    def test_drops_customer_id(self, raw_dataframe):
        result = clean(raw_dataframe)
        assert "customerID" not in result.columns

    def test_total_charges_numeric(self, raw_dataframe):
        result = clean(raw_dataframe)
        assert result["TotalCharges"].dtype in [np.float64, np.float32]

    def test_fills_missing_total_charges(self, raw_dataframe):
        result = clean(raw_dataframe)
        assert result["TotalCharges"].isna().sum() == 0

    def test_churn_is_binary(self, raw_dataframe):
        result = clean(raw_dataframe)
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_churn_encoding_correct(self, raw_dataframe):
        result = clean(raw_dataframe)
        # "Yes" → 1, "No" → 0
        assert result["Churn"].iloc[0] == 1  # was "Yes"
        assert result["Churn"].iloc[1] == 0  # was "No"


class TestEncodeCategoricals:
    def test_no_object_columns_after_encoding(self, raw_dataframe):
        cleaned = clean(raw_dataframe)
        result = encode_categoricals(cleaned)
        assert len(result.select_dtypes(include="object").columns) == 0

    def test_shape_preserved(self, raw_dataframe):
        cleaned = clean(raw_dataframe)
        result = encode_categoricals(cleaned)
        assert result.shape[0] == cleaned.shape[0]
        assert result.shape[1] == cleaned.shape[1]

    def test_all_numeric(self, raw_dataframe):
        cleaned = clean(raw_dataframe)
        result = encode_categoricals(cleaned)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"
