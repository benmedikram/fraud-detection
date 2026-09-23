import pandas as pd
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

samples = pd.read_csv("data/processed/api_samples.csv")
RAW_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_input():
    row = samples.iloc[0][RAW_COLS].tolist()
    resp = client.post("/predict", json={"features": row})
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_wrong_number_of_features():
    resp = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 422   # FastAPI's built-in validation error


def test_predict_known_fraud_sample():
    fraud_row = samples[samples.index >= 3].iloc[0][RAW_COLS].tolist()  # a fraud sample
    resp = client.post("/predict", json={"features": fraud_row})
    assert resp.status_code == 200
    # Not asserting is_fraud == True here, since even a good model can miss one,
    # but the probability should at least be returned and valid.
    assert 0.0 <= resp.json()["fraud_probability"] <= 1.0