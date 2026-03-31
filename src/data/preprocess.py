"""
Data preprocessing for Telco Churn dataset.
Run this before training. Outputs clean CSV to data/processed/.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PathConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_raw_path() -> Path:
    """Get the raw data file path."""
    raw_dir = PathConfig.data_raw()
    return raw_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def get_output_dir() -> Path:
    """Get the processed data output directory."""
    return PathConfig.data_processed()




def load_raw(path: Path | str) -> pd.DataFrame:
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


def save(df: pd.DataFrame, output_dir: Path | str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "train.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")


def main():
    raw_path = get_raw_path()
    output_dir = get_output_dir()
    
    df = load_raw(raw_path)
    logger.info(f"Raw data shape: {df.shape}")

    df = clean(df)
    df = encode_categoricals(df)

    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Churn rate: {df['Churn'].mean():.2%}")

    save(df, output_dir)
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
