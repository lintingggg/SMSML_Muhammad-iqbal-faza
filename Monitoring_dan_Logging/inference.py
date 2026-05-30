import requests
import time
import random

url = "http://localhost:8000/predict"

dummy_payload = {
    "inputs": [[0.1, 0.2, 0.3, 0.4, 0.5]] 
}

print("=== Memulai Simulasi Trafik ke Model ===")

request_ke = 1
while True:
    try:
        response = requests.post(url, json=dummy_payload)
        print(f"Request ke-{request_ke} | Status: {response.status_code}")
    except Exception as e:
        print(f"Request ke-{request_ke} gagal: Sensor belum nyala!")
    
    request_ke += 1

    time.sleep(random.uniform(0.5, 2.0))