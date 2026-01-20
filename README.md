# IoT Anomaly Detection System

This project successfuly integrates a streaming anomaly detection system using:
- Python
- Flask
- Scikit-learn
- Isolation Forest


## Project Structure

- model/ -> model training
- api/ -> REST API service
- simulator/ -> streaming data simulator
- monitoring/ -> monitoring & plotting
- data/ -> logs

## How to Run

### 1. Install dependencies (If not already installed)

pip install -r requirements.txt

### 2. Train the model

python model/train_model.py

### 3. Run the API

python api/app.py

### 4. Start the simulator (in a new terminal preferably)

python simulator/sensor_simulator.py

### 5. Run monitoring (if required)

python monitoring/plot_monitoring.py

## API

POST /predict

Input:
{
  "temperature": 70,
  "humidity": 40,
  "sound": 30
}

Output:
{
  "anomaly_score": -0.5,
  "is_anomaly": true
}

## Reproducibility

Clone the repository from GitHub and make sure to follow each step properly.
