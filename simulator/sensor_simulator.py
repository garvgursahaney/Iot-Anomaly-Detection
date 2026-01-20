import requests
import random
import time

API_URL = "http://127.0.0.1:5000/predict"

def generate_sensor_data():
    # Normal values
    temperature = random.gauss(70, 5)
    humidity = random.gauss(40, 10)
    sound = random.gauss(30, 5)

    # Randomly introduce anomalies
    if random.random() < 0.1:
        temperature += random.choice([30, -30])
        sound += random.choice([40, -40])

    return {
        "temperature": temperature,
        "humidity": humidity,
        "sound": sound
    }

print("Complete. Sensor simulator started...")

while True:
    data = generate_sensor_data()
    response = requests.post(API_URL, json=data)

    print("Sent:", data)
    print("Received:", response.json())
    print("-" * 50)

    time.sleep(1)
