import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "data", "prediction_log.csv")

df = pd.read_csv(LOG_PATH)

if len(df) == 0:
    print("No data yet.")
    exit()

plt.figure()
plt.plot(df["anomaly_score"])
plt.title("Anomaly Score Over Time")
plt.xlabel("Prediction Index")
plt.ylabel("Anomaly Score")
plt.show()
