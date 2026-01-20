import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

# Reproducibility
np.random.seed(42)

# Generation of NORMAL training data
n_samples = 5000

temperature = np.random.normal(70, 5, n_samples)
humidity = np.random.normal(40, 10, n_samples)
sound = np.random.normal(30, 5, n_samples)

X = np.column_stack([temperature, humidity, sound])

# Actual model training
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Absolute path to save the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "anomaly_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Perfect! Model trained and saved onto:", MODEL_PATH)
