import pandas as pd
import joblib
import zipfile
import os

class FaultDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.feature_names = [
            "mean_x", "mean_y", "mean_z",
            "std_x", "std_y", "std_z",
            "max_x", "max_y", "max_z",
            "min_x", "min_y", "min_z",
            "kurtosis_x", "kurtosis_y", "kurtosis_z",
            "skewness_x", "skewness_y", "skewness_z"
        ]

    def load_model(self):
        with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
            zip_ref.extractall('model')
        model_path = os.path.join('model', 'vibration_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model

    def extract_features(self, data):
        df = pd.DataFrame(data, columns=["x", "y", "z"])
        features = {
            "mean_x": df["x"].mean(),
            "mean_y": df["y"].mean(),
            "mean_z": df["z"].mean(),
            "std_x": df["x"].std(),
            "std_y": df["y"].std(),
            "std_z": df["z"].std(),
            "max_x": df["x"].max(),
            "max_y": df["y"].max(),
            "max_z": df["z"].max(),
            "min_x": df["x"].min(),
            "min_y": df["y"].min(),
            "min_z": df["z"].min(),
            "kurtosis_x": df["x"].kurtosis(),
            "kurtosis_y": df["y"].kurtosis(),
            "kurtosis_z": df["z"].kurtosis(),
            "skewness_x": df["x"].skew(),
            "skewness_y": df["y"].skew(),
            "skewness_z": df["z"].skew()
        }
        features_df = pd.DataFrame([features], columns=self.feature_names)
        print("Extracted features:", features_df)
        return features_df

    def detect_fault(self, features):
        try:
            print("Features for prediction:", features)
            prediction = self.model.predict(features)
            print("Prediction result:", prediction)
            return prediction
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
