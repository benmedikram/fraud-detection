import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
INFO_PATH = MODEL_DIR / "model_info.json"

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same transformation used in training (src/data.py), kept in sync manually."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] // 3600) % 24
    return df


class FraudModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        with open(THRESHOLD_PATH) as f:
            self.threshold = json.load(f)["threshold"]
        with open(INFO_PATH) as f:
            info = json.load(f)
        self.features = info["features"]

    def predict_one(self, raw_values: list[float]) -> dict:
        """raw_values: [Time, V1, ..., V28, Amount] — 30 raw values."""
        if len(raw_values) != len(RAW_COLUMNS):
            raise ValueError(f"Expected {len(RAW_COLUMNS)} values, got {len(raw_values)}")

        row = pd.DataFrame([raw_values], columns=RAW_COLUMNS)
        row = add_features(row)
        X = row[self.features]

        proba = float(self.model.predict_proba(X)[0, 1])
        return {
            "fraud_probability": proba,
            "is_fraud": proba >= self.threshold,
            "threshold_used": self.threshold,
        }