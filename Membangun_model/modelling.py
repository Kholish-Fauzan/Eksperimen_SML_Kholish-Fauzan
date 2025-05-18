import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Aktifkan autologging MLflow untuk scikit-learn
mlflow.sklearn.autolog()

# Definisikan nama folder dan file dataset
data_folder = 'mental_health_preprocessing'
X_train_path = f'{data_folder}/X_train.csv'
X_test_path = f'{data_folder}/X_test.csv'
y_train_path = f'{data_folder}/y_train.csv'
y_test_path = f'{data_folder}/y_test.csv'

def load_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """Fungsi untuk memuat data dari file CSV."""
    try:
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None, None, None

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Fungsi untuk melatih dan mengevaluasi model RandomForestClassifier."""
    if X_train is None:
        return None

    # Ubah y_train dan y_test menjadi array 1 dimensi jika berbentuk DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train['MH_final'].values.ravel()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test['MH_final'].values.ravel()

    with mlflow.start_run():
        # Melatih model (MLflow autolog akan mencatat parameter)
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)

        # Prediksi
        y_pred = model_rf.predict(X_test)

        # Evaluasi metrik (MLflow autolog akan mencatat ini)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        print("\nEvaluasi Model (pada test set):")
        print(f"Akurasi   : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # MLflow autolog akan mencatat model sebagai artefak

        return model_rf

if __name__ == "__main__":
    # Set nama eksperimen MLflow
    mlflow.set_experiment("Mental Health Prediction")

    # Memuat data
    X_train, X_test, y_train, y_test = load_data(X_train_path, X_test_path, y_train_path, y_test_path)

    # Melatih dan mengevaluasi model
    trained_model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    if trained_model:
        print("\nModel dan metrik telah dicatat ke MLflow Tracking UI (secara lokal).")
        print("Jalankan 'mlflow ui' di terminal untuk melihatnya.")