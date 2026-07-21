"""
ML Predictor Module for Driver Fatigue & Drowsiness Detection System.

Loads Scikit-Learn RandomForestClassifier model from models/fatigue_rf_model.pkl.
Provides model prediction, class confidence probabilities, fatigue score, and risk assessment.
Includes a robust rule-based fallback if the model pickle is not yet available.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Suppress noisy deprecation warnings from protobuf and sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")


class FatiguePredictor:
    FEATURE_ORDER = [
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

    def __init__(self, model_path: str = None):
        self.model = None
        self.using_fallback = False

        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models", "fatigue_rf_model.pkl")

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[PREDICTOR] Successfully loaded RandomForestClassifier from {model_path}")
                if hasattr(self.model, "feature_importances_"):
                    print("[PREDICTOR] --- Random Forest Feature Importances ---")
                    for name, imp in zip(self.FEATURE_ORDER, self.model.feature_importances_):
                        print(f"  - {name:22s}: {imp:.4f}")
                    print("--------------------------------------------------")
            except Exception as e:
                print(f"[PREDICTOR WARNING] Failed to load model file ({e}). Using rule-based fallback.")
                self.using_fallback = True
        else:
            print(f"[PREDICTOR WARNING] Model file not found at '{model_path}'. Using rule-based fallback.")
            self.using_fallback = True

    def _rule_based_predict(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float], float, str, str]:
        """
        Rule-based heuristic fallback if RF model pickle is missing.
        """
        ear = features.get("ear_current", 0.3)
        mar = features.get("mar_current", 0.2)
        perclos = features.get("perclos", 0.0)
        eye_duration = features.get("eye_closure_duration", 0.0)
        pitch_var = features.get("head_pitch_var", 0.0)

        # Rule evaluation
        if mar > 0.55 or features.get("mar_max", 0.2) > 0.65:
            pred_class = "Yawning"
            fatigue_score = min(40.0 + perclos * 60.0, 85.0)
            risk = "MEDIUM"
            msg = "Yawn detected. Stay alert!"
            probs = {"Alert": 0.1, "Drowsy": 0.2, "Fatigued": 0.1, "Yawning": 0.6}
        elif eye_duration > 1.5 or perclos > 0.5 or (ear < 0.15 and pitch_var > 10.0):
            pred_class = "Fatigued"
            fatigue_score = min(80.0 + eye_duration * 10.0, 100.0)
            risk = "CRITICAL"
            msg = "CRITICAL FATIGUE DETECTED! Pull over safely."
            probs = {"Alert": 0.0, "Drowsy": 0.15, "Fatigued": 0.80, "Yawning": 0.05}
        elif eye_duration > 0.5 or perclos > 0.25 or ear < 0.20:
            pred_class = "Drowsy"
            fatigue_score = min(50.0 + perclos * 50.0, 79.0)
            risk = "HIGH"
            msg = "DROWSINESS DETECTED! Prolonged eye closure."
            probs = {"Alert": 0.1, "Drowsy": 0.70, "Fatigued": 0.15, "Yawning": 0.05}
        else:
            pred_class = "Alert"
            fatigue_score = max(perclos * 100.0, 5.0)
            risk = "LOW"
            msg = "Driver state normal."
            probs = {"Alert": 0.90, "Drowsy": 0.05, "Fatigued": 0.02, "Yawning": 0.03}

        return pred_class, probs, round(fatigue_score, 1), risk, msg

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs ML prediction (or rule fallback) on input aggregate feature dict.
        Returns class, probabilities, fatigue score (0-100), risk level, and message.
        """
        if self.using_fallback or self.model is None:
            pred_class, probs, fatigue_score, risk, msg = self._rule_based_predict(features)
        else:
            # Construct feature vector as DataFrame matching trained column schema
            feature_vector = pd.DataFrame(
                [[features.get(k, 0.0) for k in self.FEATURE_ORDER]],
                columns=self.FEATURE_ORDER
            )
            pred_class = self.model.predict(feature_vector)[0]
            prob_array = self.model.predict_proba(feature_vector)[0]
            
            # Map probabilities to class dict
            model_classes = list(self.model.classes_)
            probs = {c: round(float(prob_array[model_classes.index(c)]), 3) if c in model_classes else 0.0 for c in self.CLASSES}

            # Derive fatigue score (0-100) from weighted probabilities + PERCLOS + eye closure
            drowsy_prob = probs.get("Drowsy", 0.0)
            fatigued_prob = probs.get("Fatigued", 0.0)
            yawning_prob = probs.get("Yawning", 0.0)
            perclos = features.get("perclos", 0.0)
            eye_duration = features.get("eye_closure_duration", 0.0)

            base_score = (drowsy_prob * 60.0) + (fatigued_prob * 95.0) + (yawning_prob * 45.0)
            bonus_score = (perclos * 25.0) + min(eye_duration * 15.0, 30.0)
            fatigue_score = min(max(base_score + bonus_score, 0.0), 100.0)

            # Categorize Risk Level & Alert Message
            if fatigue_score >= 80.0 or pred_class == "Fatigued" or eye_duration >= 2.0:
                risk = "CRITICAL"
                msg = "CRITICAL FATIGUE! Immediate rest required."
            elif fatigue_score >= 55.0 or pred_class == "Drowsy" or eye_duration >= 0.8:
                risk = "HIGH"
                msg = "DROWSINESS DETECTED! Take a break."
            elif fatigue_score >= 35.0 or pred_class == "Yawning" or features.get("mar_current", 0.0) > 0.55:
                risk = "MEDIUM"
                msg = "Yawning / Signs of fatigue observed."
            else:
                risk = "LOW"
                msg = "Driver state optimal."

        return {
            "prediction_class": pred_class,
            "probabilities": probs,
            "fatigue_score": round(fatigue_score, 1),
            "risk_level": risk,
            "alert_message": msg,
            "using_fallback": self.using_fallback
        }
