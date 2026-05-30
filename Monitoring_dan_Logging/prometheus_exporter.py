import time
import requests
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest

app = Flask(__name__)

# Counter: Menghitung total request masuk
REQUEST_COUNT = Counter('model_requests_total', 'Total request ke model ML', ['method'])

# Histogram: Mengukur lama waktu respons model
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Waktu respons model ML')

# Gauge: Menghitung request yang sedang berjalan saat ini
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Jumlah request yang sedang diproses')

@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/predict', methods=['POST'])
def predict():
    
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        response = requests.post("http://localhost:5001/invocations", json=request.json)
        result = response.json()
    except Exception as e:
        result = {"status": "error", "message": "Model belum siap atau error koneksi"}
    
    REQUEST_COUNT.labels(method='POST').inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    time.sleep(0.5)
    ACTIVE_REQUESTS.dec()
    
    return jsonify(result)

if __name__ == '__main__':
    print("Sensor Prometheus nyala di port 8000...")
    app.run(host='0.0.0.0', port=8000)