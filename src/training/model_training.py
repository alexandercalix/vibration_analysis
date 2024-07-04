import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import zipfile

def load_data(good_data_file, bad_data_file):
    good_data = pd.read_csv(good_data_file, header=None)
    bad_data = pd.read_csv(bad_data_file, header=None)
    
    good_data['label'] = 0
    bad_data['label'] = 1
    
    data = pd.concat([good_data, bad_data])
    
    X = data.drop(columns=['label'])
    y = data['label']
    
    return X, y

def extract_features(df, window_size=10):
    features = []
    for start in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[start:start + window_size]
        feature_row = {
            'mean_x': window[0].mean(),
            'mean_y': window[1].mean(),
            'mean_z': window[2].mean(),
            'std_x': window[0].std(),
            'std_y': window[1].std(),
            'std_z': window[2].std(),
            'max_x': window[0].max(),
            'max_y': window[1].max(),
            'max_z': window[2].max(),
            'min_x': window[0].min(),
            'min_y': window[1].min(),
            'min_z': window[2].min(),
            'kurtosis_x': window[0].kurt(),
            'kurtosis_y': window[1].kurt(),
            'kurtosis_z': window[2].kurt(),
            'skewness_x': window[0].skew(),
            'skewness_y': window[1].skew(),
            'skewness_z': window[2].skew()
        }
        features.append(feature_row)
    
    features_df = pd.DataFrame(features)
    return features_df

def extract_features_and_labels(df, labels, window_size=10):
    features = extract_features(df, window_size)
    num_windows = len(features)
    feature_labels = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_labels = labels.iloc[start:end]
        feature_labels.append(window_labels.mode()[0])
    
    feature_labels = pd.Series(feature_labels)
    return features, feature_labels

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    return model

def export_model(model, export_file):
    model_file = "vibration_model.pkl"
    joblib.dump(model, model_file)
    
    with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(model_file)
    
    print(f"Modelo exportado y comprimido como {export_file}")

# Código para probar la carga y el entrenamiento
if __name__ == "__main__":
    good_data_file = 'good_data.csv'
    bad_data_file = 'bad_data.csv'
    
    X, y = load_data(good_data_file, bad_data_file)
    X_features, y_features = extract_features_and_labels(X, y)
    
    model = train_model(X_features, y_features)
    
    export_file = 'vibration_model.zip'
    export_model(model, export_file)
    print("Modelo entrenado y exportado correctamente.")
