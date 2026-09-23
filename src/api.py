from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import FraudModel, RAW_COLUMNS

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a credit card transaction is fraudulent.",
    version="1.0.0",
)

fraud_model = FraudModel()   # loaded once at startup, reused for every request


class Transaction(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 raw values in order: Time, V1..V28, Amount",
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok", "threshold": fraud_model.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        return fraud_model.predict_one(transaction.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))