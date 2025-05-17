import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path):
    """
    Fungsi untuk memuat data dan melakukan preprocessing.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None, None

    if 'Name' in df.columns:
        df.drop('Name', axis=1, inplace=True)
    df = df[df['Gender'].isin(['Male', 'Female'])]
    df['MH_from_stress'] = df['Stress Level'].map({'Low': 0, 'Medium': 1, 'High': 1})
    bins = [14, 18, 22, 26, 30]
    labels = ['15-18', '19-22', '23-26', '27-30']
    df['Age_binned'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    df['MH_final'] = df.apply(lambda row: 1 if sum([row['Screen Time (hrs/day)'] <= 9.0, row['Sleep Duration (hrs)'] >= 6.4, row['Physical Activity (hrs/week)'] >= 4.5, row['Age_binned'] in ['19-22', '23-26']]) >= 3 else 0, axis=1)

    target_col = 'MH_final'
    categorical_cols = ['Gender', 'Education Level', 'Anxious Before Exams', 'Academic Performance Change', 'Age_binned']
    numerical_cols = ['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)']
    feature_cols = categorical_cols + numerical_cols

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print("Nama kolom di X sebelum ColumnTransformer:", X.columns.tolist())  # Tambahan debugging

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender', 'Education Level', 'Anxious Before Exams', 'Academic Performance Change', 'Age_binned']),
            ('num', StandardScaler(), numerical_cols)
        ])

    X_processed = preprocessor.fit_transform(X)

    # Debugging: Cetak nama fitur setelah One-Hot Encoding
    ohe_categories = preprocessor.named_transformers_['cat'].categories_
    ohe_feature_names = []
    for i, col in enumerate(['Gender', 'Education Level', 'Anxious Before Exams', 'Academic Performance Change', 'Age_binned']):
        ohe_feature_names.extend([f'{col}_{cat}' for cat in ohe_categories[i]])
    print("Nama fitur setelah One-Hot Encoding (debug):", ohe_feature_names)

    all_feature_names = list(ohe_feature_names) + numerical_cols

    return X_processed, y, all_feature_names

if __name__ == "__main__":
    data_directory = '/root/.cache/kagglehub/datasets/utkarshsharma11r/student-mental-health-analysis/versions/1'
    data_filename = 'Student Mental Health Analysis During Online Learning.csv'
    data_path = f'{data_directory}/{data_filename}'

    X_processed, y, feature_names = load_and_preprocess_data(data_path)

    if X_processed is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_train_df = pd.DataFrame(y_train, columns=['MH_final'])
        y_test_df = pd.DataFrame(y_test, columns=['MH_final'])

        X_train_df.to_csv('X_train.csv', index=False)
        X_test_df.to_csv('X_test.csv', index=False)
        y_train_df.to_csv('y_train.csv', index=False)
        y_test_df.to_csv('y_test.csv', index=False)

        print("Data berhasil dimuat, diproses, dibagi, dan disimpan ke file CSV:")
        print("- X_train.csv")
        print("- X_test.csv")
        print("- y_train.csv")
        print("- y_test.csv")