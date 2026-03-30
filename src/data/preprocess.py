"""
Data preprocessing for Telco Churn dataset.
Run this before training. Outputs clean CSV to data/processed/.
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "data/processed"


def load_raw(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop customerID (not useful)
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has some spaces — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    logger.info(f"Encoding categorical columns: {cat_cols}")

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def save(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")


def main():
    df = load_raw(RAW_PATH)
    logger.info(f"Raw data shape: {df.shape}")

    df = clean(df)
    df = encode_categoricals(df)

    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")

    save(df, OUTPUT_DIR)
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
