from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import time

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../factorygaurd_ai/models/lightgbm_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "../factorygaurd_ai/models/lightgbm_threshold.pkl")

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# ---------------- EXPECTED COLUMNS ----------------
expected_columns = [
    "Torque_Nm", "Tool_wear_min", "Rotational_speed_rpm",
    "Process_temperature_K", "Air_temperature_K",
    "temp_lag_1", "temp_lag_2", "temp_mean_6",
    "temp_std_6", "temp_ema_6",
    "torque_lag_1", "torque_std_6", "torque_mean_6",
    "Type_L", "Type_M"
]

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        start_time = time.time()

        data = request.json
        df = pd.DataFrame([data])
        df = df[expected_columns]

        prob = model.predict_proba(df)[:, 1][0]
        prediction = int(prob > threshold)

        latency = (time.time() - start_time) * 1000

        return jsonify({
            "failure_probability": round(float(prob), 4),
            "prediction": prediction,
            "status": "FAILURE" if prediction else "NORMAL",
            "latency_ms": round(latency, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)