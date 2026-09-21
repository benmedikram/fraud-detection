from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features (no fitting needed, so no leakage)."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] // 3600) % 24
    return df


def split(df: pd.DataFrame):
    """Stratified 70/15/15 train/val/test split."""
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["Class"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["Class"], random_state=RANDOM_STATE
    )
    return train, val, test


def main() -> None:
    df = add_features(clean(load_raw()))
    train, val, test = split(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print("train", train.shape, "val", val.shape, "test", test.shape)


if __name__ == "__main__":
    main()