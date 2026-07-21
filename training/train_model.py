"""
Model training script for Driver Fatigue & Drowsiness Detection System.

Generates/loads a feature dataset and trains a RandomForestClassifier to predict
driver state ('Alert', 'Drowsy', 'Fatigued', 'Yawning').
Saves trained model to models/fatigue_rf_model.pkl and dataset to training/dataset.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

FEATURE_NAMES = [
    "ear_current",
    "ear_mean",
    "ear_min",
    "mar_current",
    "mar_mean",
    "mar_max",
    "perclos",
    "blink_rate",
    "eye_closure_duration",
    "head_pitch_var",
    "head_yaw_var",
    "head_roll_var",
    "yawn_count"
]

CLASSES = ["Alert", "Drowsy", "Fatigued", "Yawning"]


def generate_synthetic_dataset(num_samples_per_class: int = 500) -> pd.DataFrame:
    """Synthesizes realistic feature vectors for 4 driver fatigue classes."""
    np.random.seed(42)
    data = []

    for _ in range(num_samples_per_class):
        # 1. ALERT Class
        ear_curr = np.random.normal(0.31, 0.03)
        ear_mean = np.random.normal(0.31, 0.02)
        ear_min = np.random.normal(0.26, 0.02)
        mar_curr = np.random.normal(0.20, 0.04)
        mar_mean = np.random.normal(0.20, 0.03)
        mar_max = np.random.normal(0.28, 0.04)
        perclos = np.random.uniform(0.0, 0.08)
        blink_rate = np.random.uniform(10.0, 22.0)
        eye_closure = 0.0
        pitch_var = np.random.uniform(0.5, 3.0)
        yaw_var = np.random.uniform(0.5, 3.0)
        roll_var = np.random.uniform(0.2, 1.5)
        yawns = 0

        data.append([
            ear_curr, ear_mean, ear_min, mar_curr, mar_mean, mar_max,
            perclos, blink_rate, eye_closure, pitch_var, yaw_var, roll_var, yawns, "Alert"
        ])

        # 2. DROWSY Class
        ear_curr = np.random.normal(0.20, 0.03)
        ear_mean = np.random.normal(0.21, 0.02)
        ear_min = np.random.normal(0.14, 0.03)
        mar_curr = np.random.normal(0.22, 0.04)
        mar_mean = np.random.normal(0.22, 0.03)
        mar_max = np.random.normal(0.32, 0.05)
        perclos = np.random.uniform(0.20, 0.50)
        blink_rate = np.random.uniform(20.0, 40.0)  # frequent sluggish blinks
        eye_closure = np.random.uniform(0.4, 1.2)
        pitch_var = np.random.uniform(4.0, 12.0)  # slight head nods
        yaw_var = np.random.uniform(2.0, 8.0)
        roll_var = np.random.uniform(1.0, 4.0)
        yawns = np.random.choice([0, 1])

        data.append([
            ear_curr, ear_mean, ear_min, mar_curr, mar_mean, mar_max,
            perclos, blink_rate, eye_closure, pitch_var, yaw_var, roll_var, yawns, "Drowsy"
        ])

        # 3. FATIGUED Class
        ear_curr = np.random.normal(0.13, 0.03)
        ear_mean = np.random.normal(0.15, 0.02)
        ear_min = np.random.normal(0.08, 0.02)
        mar_curr = np.random.normal(0.24, 0.05)
        mar_mean = np.random.normal(0.24, 0.04)
        mar_max = np.random.normal(0.35, 0.06)
        perclos = np.random.uniform(0.55, 0.95)
        blink_rate = np.random.uniform(5.0, 15.0)  # microsleeps (fewer full blinks, long closure)
        eye_closure = np.random.uniform(1.5, 4.0)
        pitch_var = np.random.uniform(8.0, 25.0)  # heavy head drooping
        yaw_var = np.random.uniform(5.0, 18.0)
        roll_var = np.random.uniform(3.0, 10.0)
        yawns = np.random.choice([1, 2, 3])

        data.append([
            ear_curr, ear_mean, ear_min, mar_curr, mar_mean, mar_max,
            perclos, blink_rate, eye_closure, pitch_var, yaw_var, roll_var, yawns, "Fatigued"
        ])

        # 4. YAWNING Class
        ear_curr = np.random.normal(0.26, 0.04)
        ear_mean = np.random.normal(0.27, 0.03)
        ear_min = np.random.normal(0.20, 0.04)
        mar_curr = np.random.normal(0.68, 0.08)
        mar_mean = np.random.normal(0.55, 0.06)
        mar_max = np.random.normal(0.75, 0.08)
        perclos = np.random.uniform(0.05, 0.25)
        blink_rate = np.random.uniform(12.0, 25.0)
        eye_closure = np.random.uniform(0.0, 0.8)
        pitch_var = np.random.uniform(3.0, 10.0)  # head tilt back during yawn
        yaw_var = np.random.uniform(1.0, 5.0)
        roll_var = np.random.uniform(1.0, 4.0)
        yawns = np.random.randint(1, 5)

        data.append([
            ear_curr, ear_mean, ear_min, mar_curr, mar_mean, mar_max,
            perclos, blink_rate, eye_closure, pitch_var, yaw_var, roll_var, yawns, "Yawning"
        ])

    columns = FEATURE_NAMES + ["label"]
    df = pd.DataFrame(data, columns=columns)
    # Clip numerical features to realistic lower bounds
    for col in FEATURE_NAMES:
        df[col] = df[col].clip(lower=0.0)
    return df


def train_and_save_model(data_path: str = None, model_output_path: str = None):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if model_output_path is None:
        model_dir = os.path.join(project_root, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_output_path = os.path.join(model_dir, "fatigue_rf_model.pkl")

    dataset_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

    if data_path and os.path.exists(data_path):
        print(f"[TRAINING] Loading custom dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("[TRAINING] Synthesizing bootstrap training dataset...")
        df = generate_synthetic_dataset(num_samples_per_class=600)
        df.to_csv(dataset_csv_path, index=False)
        print(f"[TRAINING] Dataset saved to {dataset_csv_path}")

    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[TRAINING] Fitting RandomForestClassifier(n_estimators=100, max_depth=10)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[TRAINING SUCCESS] Model Accuracy on Test Set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    print("\n--- FEATURE IMPORTANCES ---")
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importances:
        print(f"  - {feat:22s}: {imp:.4f}")
    print("---------------------------\n")

    joblib.dump(clf, model_output_path)
    print(f"[TRAINING] Model serialized and saved to: {model_output_path}")


if __name__ == "__main__":
    train_and_save_model()
