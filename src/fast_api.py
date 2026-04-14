from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# ---------------- INITIALIZE APP ----------------
app = FastAPI(title="FactoryGuard AI API", version="1.0")

# ---------------- LOAD MODEL & THRESHOLD ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../factorygaurd_ai/models/lightgbm_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "../factorygaurd_ai/models/lightgbm_threshold.pkl")

try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or threshold: {e}")

# ---------------- INPUT SCHEMA ----------------
class MachineData(BaseModel):
    Torque_Nm: float
    Tool_wear_min: float
    Rotational_speed_rpm: float
    Process_temperature_K: float
    Air_temperature_K: float
    temp_lag_1: float
    temp_lag_2: float
    temp_mean_6: float
    temp_std_6: float
    temp_ema_6: float
    torque_lag_1: float
    torque_std_6: float
    torque_mean_6: float
    Type_L: int
    Type_M: int

# ---------------- ROOT ENDPOINT ----------------
@app.get("/")
def home():
    return {
        "message": "FactoryGuard AI API is running 🚀",
        "model": "LightGBM",
        "status": "healthy"
    }

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "OK"}

# ---------------- PREDICTION ENDPOINT ----------------
@app.post("/predict")
def predict(data: MachineData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Ensure column order matches training (important!)
        expected_columns = [
            "Torque_Nm", "Tool_wear_min", "Rotational_speed_rpm",
            "Process_temperature_K", "Air_temperature_K",
            "temp_lag_1", "temp_lag_2", "temp_mean_6",
            "temp_std_6", "temp_ema_6",
            "torque_lag_1", "torque_std_6", "torque_mean_6",
            "Type_L", "Type_M"
        ]

        df = df[expected_columns]

        # Predict probability
        prob = model.predict_proba(df)[:, 1][0]

        # Apply threshold
        prediction = int(prob > threshold)

        # Response
        return {
            "failure_probability": round(float(prob), 4),
            "prediction": prediction,
            "threshold_used": round(float(threshold), 4),
            "status": "FAILURE" if prediction == 1 else "NORMAL"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))