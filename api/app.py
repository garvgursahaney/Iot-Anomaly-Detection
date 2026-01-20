from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "anomaly_model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_PATH = os.path.join(DATA_DIR, "prediction_log.csv")

# Load model
model = joblib.load(MODEL_PATH)

# Prepare data folder and log file
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=[
        "timestamp", "temperature", "humidity", "sound",
        "anomaly_score", "is_anomaly"
    ])
    df.to_csv(LOG_PATH, index=False)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    sound = float(data["sound"])

    X = np.array([[temperature, humidity, sound]])

    score = model.decision_function(X)[0]
    prediction = model.predict(X)[0]

    is_anomaly = True if prediction == -1 else False

    # Result logging
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "sound": sound,
        "anomaly_score": score,
        "is_anomaly": is_anomaly
    }

    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)

    return jsonify({
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    })

if __name__ == "__main__":
    app.run(debug=True)
