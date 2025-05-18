from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time
import pandas as pd
import pickle

# Path ke model Anda (gunakan path yang sama dari inference.py)
MODEL_PATH = "D:/LASKAR AI/Eksperimen_SML_Kholish-Fauzan/Membangun_model/mlruns/493502173447380538/69efbe0b442342b18946d79ec12b3d70/artifacts/best_random_forest_model/model.pkl"

# Mapping label prediksi
label_mapping = {0: "Sehat", 1: "Tidak sehat"}

# Inisialisasi metrik Prometheus
total_requests = Counter('model_requests_total', 'Total number of prediction requests.')
prediction_latency_seconds = Summary('model_prediction_latency_seconds', 'Time spent processing prediction requests.')
healthy_predictions = Counter('model_healthy_predictions_total', 'Total number of healthy predictions.')
unhealthy_predictions = Counter('model_unhealthy_predictions_total', 'Total number of unhealthy predictions.')
model_status = Gauge('model_up', 'Status of the model serving (1 for up, 0 for down).')

# Load model
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    model_status.set(1)  # Set model status to up
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model_status.set(0)  # Set model status to down

def predict(data):
    """Simulasi fungsi prediksi yang memanggil model dan mengupdate metrik."""
    total_requests.inc()
    start_time = time.time()
    try:
        prediction = model.predict(data)
        latency = time.time() - start_time
        prediction_latency_seconds.observe(latency)
        predicted_label = [label_mapping[p] for p in prediction][0] # Ambil label pertama
        if predicted_label == "Sehat":
            healthy_predictions.inc()
        else:
            unhealthy_predictions.inc()
        return predicted_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"

if __name__ == '__main__':
    # Start Prometheus HTTP server
    start_http_server(8000)  # Server akan berjalan di port 8000
    print("Prometheus exporter berjalan di http://localhost:8000/metrics")

    # Simulasi permintaan prediksi setiap beberapa detik
    while True:
        # Contoh data input (sesuaikan dengan fitur model Anda)
        sample_data = pd.DataFrame({
            'Gender_Female': [random.choice([0.0, 1.0])],
            'Gender_Male': [random.choice([0.0, 1.0])],
            'Education Level_BA': [random.choice([0.0, 1.0])],
            'Education Level_BSc': [random.choice([0.0, 1.0])],
            'Education Level_BTech': [random.choice([0.0, 1.0])],
            'Education Level_Class 10': [random.choice([0.0, 1.0])],
            'Education Level_Class 11': [random.choice([0.0, 1.0])],
            'Education Level_Class 12': [random.choice([0.0, 1.0])],
            'Education Level_Class 8': [random.choice([0.0, 1.0])],
            'Education Level_Class 9': [random.choice([0.0, 1.0])],
            'Education Level_MA': [random.choice([0.0, 1.0])],
            'Education Level_MSc': [random.choice([0.0, 1.0])],
            'Education Level_MTech': [random.choice([0.0, 1.0])],
            'Anxious Before Exams_No': [random.choice([0.0, 1.0])],
            'Anxious Before Exams_Yes': [random.choice([0.0, 1.0])],
            'Academic Performance Change_Declined': [random.choice([0.0, 1.0])],
            'Academic Performance Change_Improved': [random.choice([0.0, 1.0])],
            'Academic Performance Change_Same': [random.choice([0.0, 1.0])],
            'Age_binned_15-18': [random.choice([0.0, 1.0])],
            'Age_binned_19-22': [random.choice([0.0, 1.0])],
            'Age_binned_23-26': [random.choice([0.0, 1.0])],
            'Screen Time (hrs/day)': [random.uniform(0, 5)],
            'Sleep Duration (hrs)': [random.uniform(5, 10)],
            'Physical Activity (hrs/week)': [random.uniform(0, 10)]
        })
        result = predict(sample_data)
        print(f"Prediksi: {result}")
        time.sleep(5) # Simulasi permintaan setiap 5 detik