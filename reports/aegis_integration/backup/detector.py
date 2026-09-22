"""
backend/detection/detector.py
==============================
Canonical 46-Feature IoT Intrusion Detector for AEGIS-IoT.
Uses:
- cic_iot_xgboost_model.pkl (46 Features, Frozen Production Model)
- label_encoder.pkl (Classes: DDoS, DoS, Spoofing)
- Strict 46-Feature Validator (offline_feature_validator.py)
- Raw float32 features (NO SCALER)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Union

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from offline_feature_validator import validate_46_features, CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(ROOT_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "label_encoder.pkl")
CALIB_PATH = os.path.join(ROOT_DIR, "calibrator_temperature.pkl")

# Canonical 46 Features
FEATURE_COLUMNS = CANONICAL_46_FEATURES
CLASS_NAMES = ["DDoS", "DoS", "Spoofing"]


class IoTIntrusionDetector:
    """
    Production detector using the verified 46-feature XGBoost model on raw telemetry,
    with calibrated temperature scaling when calibrator_temperature.pkl is available.
    """
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 encoder_path: str = ENCODER_PATH,
                 calib_path: str = CALIB_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Canonical 46-feature model not found at: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found at: {encoder_path}")

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.calibrator = joblib.load(calib_path) if os.path.exists(calib_path) else None
        self.booster = self.model.get_booster() if self.calibrator is not None else None
        self.feature_columns = list(self.model.feature_names_in_)
        self.classes = list(self.encoder.classes_)
        self.class_names = {i: name for i, name in enumerate(self.classes)}

    def format_input(self, raw_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Validates input against all 10 strict invariants.
        Returns a canonical 46-column float32 DataFrame.
        """
        return validate_46_features(raw_input)

    def predict(self, raw_features: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """
        Runs offline inference on validated 46 raw features in microseconds.
        """
        df_input = self.format_input(raw_features)
        
        # Predict class & calibrated probabilities
        if self.calibrator is not None:
            dmat = xgb.DMatrix(df_input)
            logits = self.booster.predict(dmat, output_margin=True)
            probabilities = self.calibrator.predict_proba(logits)[0]
            pred_idx = int(np.argmax(probabilities))
        else:
            pred_idx = int(self.model.predict(df_input)[0])
            probabilities = self.model.predict_proba(df_input)[0]

        confidence = float(probabilities[pred_idx])
        attack_type = self.class_names.get(pred_idx, "Unknown")
        prob_dict = {self.class_names[i]: float(probabilities[i]) for i in range(len(probabilities))}

        return {
            "predicted_class": pred_idx,
            "attack_type": attack_type,
            "confidence": confidence,
            "probabilities": prob_dict,
            "formatted_df": df_input,
            "raw_features": df_input.iloc[0].to_dict()
        }
