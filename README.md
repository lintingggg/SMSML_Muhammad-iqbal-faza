# Movie Feelings Recommender: MLOps Pipeline

Repositori ini berisi implementasi alur MLOps untuk sistem rekomendasi film berbasis kemiripan vektor emosi. Dataset yang digunakan sudah melalui tahap preprocessing dan memuat 1.500 data film dengan 150 fitur numerik bertema emosi/perasaan. Model utama memakai `NearestNeighbors` dari scikit-learn untuk mencari film yang paling mirip berdasarkan jarak antarf fitur.

Selain eksperimen model, project ini juga menyertakan bukti integrasi dengan DagsHub/MLflow, workflow CI terpisah, model serving lokal, monitoring Prometheus, dashboard Grafana, dan alerting.

## Ringkasan Project

| Area | Isi |
|---|---|
| Use case | Rekomendasi film berdasarkan kemiripan fitur emosi |
| Model | `NearestNeighbors` / KNN recommender |
| Tracking | MLflow yang terhubung ke DagsHub |
| Tuning | Kombinasi `n_neighbors` dan metrik jarak |
| Serving | Flask wrapper pada port `8000`, meneruskan prediksi ke MLflow serving port `5001` |
| Monitoring | Prometheus scrape endpoint `/metrics` |
| Visualisasi | Grafana dashboard dan rules alerting |

## Struktur Repositori

```text
SMSML_Muhammad-iqbal-faza/
|-- Membangun_model/
|   |-- movie_feelings_dataset_preprocessing.csv  # Dataset bersih untuk training recommender
|   |-- modelling.py                              # Baseline experiment dengan NearestNeighbors
|   |-- modelling_tuning.py                       # Tuning KNN dan logging artefak ke MLflow
|   |-- requirements.txt                          # Dependensi eksperimen model
|   |-- DagsHub.txt                               # Link project DagsHub
|   |-- model_metadata.txt                        # Ringkasan hasil training
|   |-- sample_recommendation.csv                 # Contoh output rekomendasi
|   |-- tuning_summary.json                       # Rekap hasil kombinasi tuning
|   |-- tuning_plot.png                           # Visualisasi hasil tuning
|   |-- screenshoot_dashboard.jpg                 # Bukti dashboard eksperimen
|   `-- screenshoot_artifak.jpg                   # Bukti artefak model
|
|-- Monitoring_dan_Logging/
|   |-- prometheus_exporter.py                    # Flask API dan exporter metrik Prometheus
|   |-- inference.py                              # Script simulasi trafik request
|   |-- prometheus.yml                            # Konfigurasi scrape Prometheus
|   |-- bukti_serving.png                         # Dokumentasi model/API serving
|   |-- bukti_serving_2.png                       # Dokumentasi tambahan serving
|   |-- 4.bukti monitoring Prometheus/            # Screenshot query metrik di Prometheus
|   |-- 5.bukti monitoring Grafana/               # Screenshot panel dashboard Grafana
|   `-- 6.bukti alerting Grafana/                 # Screenshot rules dan notifikasi alert
|
|-- Eksperimen_SML_Iqbal.txt                      # Link repositori eksperimen
|-- Workflow-CI.txt                               # Link repositori CI workflow
`-- README.md
```

## Link Terkait

| Komponen | URL |
|---|---|
| Repositori utama | `https://github.com/lintingggg/SMSML_Muhammad-iqbal-faza` |
| Eksperimen SML | `https://github.com/lintingggg/Eksperimen_SML_Iqbal` |
| Workflow CI | `https://github.com/lintingggg/Workflow-CI` |
| DagsHub / MLflow tracking | `https://dagshub.com/lintinggg/SMSML_Muhammad-iqbal-faza` |

## Dataset

File dataset utama berada di [Membangun_model/movie_feelings_dataset_preprocessing.csv](Membangun_model/movie_feelings_dataset_preprocessing.csv). Dataset ini memiliki:

| Detail | Nilai |
|---|---|
| Jumlah baris data film | 1.500 |
| Jumlah kolom total | 152 |
| Kolom non-fitur | `imdb_id`, `title_year` |
| Jumlah fitur numerik model | 150 |
| Contoh fitur | `f1_skepticism`, `f1_serenity`, `f1_fear`, `f1_disgust`, `f1_love`, `f1_tension` |

Pada script training, kolom identitas seperti `imdb_id` dan `title_year` dikeluarkan dari fitur. Model hanya memakai kolom numerik agar representasi film dapat dihitung menggunakan jarak kemiripan.

## Eksperimen Model

### Baseline

Baseline dijalankan melalui [Membangun_model/modelling.py](Membangun_model/modelling.py). Script ini:

- Membaca dataset preprocessing.
- Menghapus kolom non-fitur.
- Memilih fitur bertipe numerik.
- Mengaktifkan `mlflow.autolog()`.
- Melatih model `NearestNeighbors(n_neighbors=5, metric="cosine")`.
- Mencatat run ke experiment `Movie_Recommendation_Baseline`.

### Hyperparameter Tuning

Eksperimen tuning berada di [Membangun_model/modelling_tuning.py](Membangun_model/modelling_tuning.py). Skenario tuning mencoba beberapa kombinasi jumlah tetangga dan metrik jarak.

| Parameter | Nilai |
|---|---|
| `n_neighbors` | `3`, `5`, `7` |
| `metric` | `cosine`, `euclidean` |
| Metrik evaluasi | `mean_distance` |
| Experiment MLflow | `Movie_Recommendation_Tuning` |

Hasil tuning yang tersimpan di [Membangun_model/tuning_summary.json](Membangun_model/tuning_summary.json):

| `n_neighbors` | `metric` | `mean_distance` |
|---:|---|---:|
| 3 | `cosine` | 0.1969 |
| 5 | `cosine` | 0.2588 |
| 7 | `cosine` | 0.2940 |
| 3 | `euclidean` | 1.2489 |
| 5 | `euclidean` | 1.5727 |
| 7 | `euclidean` | 1.7377 |

Artefak tambahan yang dicatat:

- [Membangun_model/model_metadata.txt](Membangun_model/model_metadata.txt), berisi informasi jumlah data, jumlah fitur, algoritma, dan skenario tuning.
- [Membangun_model/sample_recommendation.csv](Membangun_model/sample_recommendation.csv), berisi contoh indeks film terdekat beserta skor jaraknya.
- [Membangun_model/tuning_plot.png](Membangun_model/tuning_plot.png), berisi visualisasi hasil tuning.

## Menjalankan Eksperimen

Jalankan perintah berikut dari root repositori.

```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py
```

Untuk menjalankan tuning:

```bash
cd Membangun_model
python modelling_tuning.py
```

Catatan: script sudah menginisialisasi koneksi DagsHub melalui `dagshub.init(...)`, sehingga environment perlu memiliki akses yang sesuai jika ingin mengirim run ke remote tracking.

## Workflow CI

Workflow CI disimpan pada repositori terpisah: `https://github.com/lintingggg/Workflow-CI`.

Secara garis besar, pipeline CI digunakan untuk mengotomatisasi proses berikut:

| Tahap | Tujuan |
|---|---|
| Setup environment | Menyiapkan runner dan dependensi Python |
| Training automation | Menjalankan script model secara otomatis |
| MLflow artifact handling | Menyimpan artefak hasil training |
| Docker build | Membuat image model/service |
| Docker push | Mengirim image ke registry jika konfigurasi secret tersedia |

## Serving dan Exporter Prometheus

File [Monitoring_dan_Logging/prometheus_exporter.py](Monitoring_dan_Logging/prometheus_exporter.py) menjalankan Flask server di port `8000`.

Endpoint yang tersedia:

| Endpoint | Method | Fungsi |
|---|---|---|
| `/predict` | `POST` | Menerima payload inferensi dan meneruskan request ke MLflow model serving di `http://localhost:5001/invocations` |
| `/metrics` | `GET` | Mengekspos metrik dalam format Prometheus |

Metrik custom yang dibuat:

| Metrik | Tipe | Deskripsi |
|---|---|---|
| `model_requests_total` | Counter | Total request prediksi yang masuk |
| `model_request_latency_seconds` | Histogram | Waktu pemrosesan request |
| `model_active_requests` | Gauge | Jumlah request yang sedang diproses |

Selain metrik custom, library `prometheus_client` juga mengekspos metrik runtime Python/process seperti penggunaan CPU, memory, GC, open file descriptors, dan waktu start process.

## Menjalankan Serving dan Monitoring

Pastikan model MLflow sudah disajikan pada port `5001`. Contoh perintah umum:

```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --no-conda
```

Jalankan exporter/API:

```bash
cd Monitoring_dan_Logging
python prometheus_exporter.py
```

Jalankan Prometheus dengan konfigurasi dari repo:

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v "${PWD}/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus
```

Jalankan Grafana:

```bash
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

Kirim trafik dummy untuk menguji metrik:

```bash
python inference.py
```

Konfigurasi scrape Prometheus berada di [Monitoring_dan_Logging/prometheus.yml](Monitoring_dan_Logging/prometheus.yml):

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'model_recsys'
    static_configs:
      - targets: ['host.docker.internal:8000']
```

## Bukti Monitoring

Screenshot Prometheus berada di folder [Monitoring_dan_Logging/4.bukti monitoring Prometheus](Monitoring_dan_Logging/4.bukti%20monitoring%20Prometheus), meliputi:

- `model_requests_total`
- `model_request_latency_seconds`
- `model_active_requests`

Screenshot Grafana berada di folder [Monitoring_dan_Logging/5.bukti monitoring Grafana](Monitoring_dan_Logging/5.bukti%20monitoring%20Grafana). Panel yang didokumentasikan mencakup:

- `model_requests_total`
- `model_request_latency_seconds`
- `model_active_requests`
- `process_cpu_seconds_total`
- `process_resident_memory_bytes`
- `python_gc_objects_collected_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_open_fds`
- `process_start_time_seconds`

## Alerting Grafana

Bukti alerting disimpan di [Monitoring_dan_Logging/6.bukti alerting Grafana](Monitoring_dan_Logging/6.bukti%20alerting%20Grafana). Rules yang ditampilkan berfokus pada:

| Alert | Sinyal yang Dipantau |
|---|---|
| Antrean request tinggi | `model_active_requests` |
| Penggunaan virtual memory tinggi | `process_virtual_memory_bytes` |
| Trafik request tinggi | `model_requests_total` |

Folder tersebut juga berisi screenshot notifikasi untuk masing-masing kondisi alert.

## Tech Stack

| Kebutuhan | Teknologi |
|---|---|
| Bahasa | Python |
| Data handling | pandas |
| Model ML | scikit-learn `NearestNeighbors` |
| Experiment tracking | MLflow, DagsHub |
| Visualisasi eksperimen | matplotlib |
| API serving | Flask |
| Traffic simulation | requests |
| Monitoring | Prometheus |
| Dashboard dan alert | Grafana |
| CI/CD | GitHub Actions, Docker |

## Catatan

Payload pada [Monitoring_dan_Logging/inference.py](Monitoring_dan_Logging/inference.py) masih berupa dummy input untuk simulasi trafik. Jika ingin dipakai untuk rekomendasi aktual, bentuk input perlu disesuaikan dengan jumlah fitur model, yaitu 150 fitur numerik.
