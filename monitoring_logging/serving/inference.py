import mlflow
import pandas as pd
import random
import time
import os
import pickle  # Import library pickle

# Path langsung ke file model.pkl
MODEL_PATH = "D:/LASKAR AI/Eksperimen_SML_Kholish-Fauzan/Membangun_model/mlruns/493502173447380538/69efbe0b442342b18946d79ec12b3d70/artifacts/best_random_forest_model/model.pkl"

# Mapping label prediksi
label_mapping = {0: "Sehat", 1: "Tidak sehat"}

# Load model menggunakan pickle (karena kita langsung menunjuk ke file .pkl)
try:
    with open(MODEL_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Model berhasil dimuat dari path: {MODEL_PATH}")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    exit()

def predict(model, data):
    """Fungsi untuk melakukan prediksi menggunakan model."""
    try:
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        print(f"Error saat melakukan prediksi: {e}")
        return None

if __name__ == "__main__":
    # Data input sesuai urutan fitur
    sample_data = pd.DataFrame({
        'Gender_Female': [1.0],
        'Gender_Male': [0.0],
        'Education Level_BA': [0.0],
        'Education Level_BSc': [0.0],
        'Education Level_BTech': [0.0],
        'Education Level_Class 10': [0.0],
        'Education Level_Class 11': [0.0],
        'Education Level_Class 12': [0.0],
        'Education Level_Class 8': [0.0],
        'Education Level_Class 9': [1.0],
        'Education Level_MA': [0.0],
        'Education Level_MSc': [0.0],
        'Education Level_MTech': [0.0],
        'Anxious Before Exams_No': [1.0],
        'Anxious Before Exams_Yes': [0.0],
        'Academic Performance Change_Declined': [0.0],
        'Academic Performance Change_Improved': [0.0],
        'Academic Performance Change_Same': [1.0],
        'Age_binned_15-18': [1.0],
        'Age_binned_19-22': [0.0],
        'Age_binned_23-26': [0.0],
        'Screen Time (hrs/day)': [0.09190680554883035],
        'Sleep Duration (hrs)': [-0.2340312064687379],
        'Physical Activity (hrs/week)': [-1.0245033953453102]
    })

    if loaded_model:
        prediction = predict(loaded_model, sample_data)
        if prediction is not None:
            # Terjemahkan hasil prediksi ke label
            predicted_label = [label_mapping[p] for p in prediction]
            print("Hasil Prediksi:", predicted_label)