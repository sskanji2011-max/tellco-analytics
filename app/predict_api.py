"""Prediction API for the satisfaction model.

Run:  uvicorn app.predict_api:app --host 0.0.0.0 --port 8000
Test: curl -X POST localhost:8000/predict -H "Content-Type: application/json" \\
        -d '{"Sessions": 2, "Duration_ms": 1.2e8, "Total_Data": 7e8, "Avg_TCP": 5e6, "Avg_RTT": 40, "Avg_Throughput": 3000}'
"""
import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/satisfaction_model.joblib")
bundle = joblib.load(MODEL_PATH)
app = FastAPI(title="TellCo satisfaction model")


class Customer(BaseModel):
    Sessions: float
    Duration_ms: float
    Total_Data: float
    Avg_TCP: float
    Avg_RTT: float
    Avg_Throughput: float


@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["name"], "code_version": bundle["code_version"]}


@app.post("/predict")
def predict(c: Customer):
    row = pd.DataFrame([[c.Sessions, c.Duration_ms, c.Total_Data, c.Avg_TCP, c.Avg_RTT, c.Avg_Throughput]],
                       columns=bundle["features"])
    return {"satisfaction_score": float(bundle["model"].predict(row)[0]), "model": bundle["name"]}
