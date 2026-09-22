"""
Production-Ready Calibrated Inference Pipeline for 46-Feature IoT Intrusion Detection.

Provides:
1. Standalone Python Class: CalibratedIDSInferencePipeline
2. Functional API: predict_single(feature_dict_or_series)
3. CLI Interface with --test-sample and --input-json support.

Output schema:
{
    "attack_type": "DDoS",
    "confidence": 0.942,
    "probabilities": {
        "DDoS": 0.942,
        "DoS": 0.051,
        "Spoofing": 0.007
    }
}
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import softmax

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
DEFAULT_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DEFAULT_PIPELINE_PATH = os.path.join(BASE_DIR, "calibrated_xgboost_final.pkl")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")


class CalibratedIDSInferencePipeline:
    """
    Inference pipeline pairing the frozen 46-feature XGBoost model with the calibrated probability layer.
    """
    def __init__(self,
                 model_path=DEFAULT_MODEL_PATH,
                 encoder_path=DEFAULT_ENCODER_PATH,
                 calibration_path=DEFAULT_PIPELINE_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(f"Calibrated pipeline artifact not found: {calibration_path}. Run run_calibration_experiments.py first.")

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.calib_meta = joblib.load(calibration_path)

        self.features = list(self.model.feature_names_in_)
        self.classes = list(self.encoder.classes_)
        self.best_method = self.calib_meta.get("best_method_key", "temperature")
        self.booster = self.model.get_booster()

    def predict_one(self, input_data):
        """
        Accepts dict, pd.Series, pd.DataFrame, or 1D list/array of 46 features.
        Returns formatted dictionary with calibrated confidence.
        """
        # Convert input to DataFrame with exact column ordering
        if isinstance(input_data, dict):
            # Verify missing features
            missing = [f for f in self.features if f not in input_data]
            if missing:
                raise ValueError(f"Missing required features in input: {missing}")
            df = pd.DataFrame([{f: float(input_data[f]) for f in self.features}])
        elif isinstance(input_data, pd.Series):
            df = pd.DataFrame([{f: float(input_data[f]) for f in self.features}])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data[self.features].copy()
        elif isinstance(input_data, (list, np.ndarray)):
            if len(input_data) != len(self.features):
                raise ValueError(f"Expected {len(self.features)} features, received {len(input_data)}")
            df = pd.DataFrame([input_data], columns=self.features)
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

        # Ensure float32 dtype
        df = df.astype(np.float32)

        # Calibrated Prediction Logic
        if self.best_method == "temperature":
            dmat = xgb.DMatrix(df)
            logits = self.booster.predict(dmat, output_margin=True)
            calib_obj = self.calib_meta["calibrator_temperature"]
            calib_probs = calib_obj.predict_proba(logits)[0]
        elif self.best_method == "sigmoid":
            raw_probs = self.model.predict_proba(df)
            calib_obj = self.calib_meta["calibrator_sigmoid"]
            calib_probs = calib_obj.predict_proba(raw_probs)[0]
        elif self.best_method == "isotonic":
            raw_probs = self.model.predict_proba(df)
            calib_obj = self.calib_meta["calibrator_isotonic"]
            calib_probs = calib_obj.predict_proba(raw_probs)[0]
        else:
            raw_probs = self.model.predict_proba(df)
            calib_probs = raw_probs[0]

        # Normalization and sanity check
        calib_probs = np.maximum(calib_probs, 0.0)
        calib_probs = calib_probs / np.sum(calib_probs)

        pred_idx = int(np.argmax(calib_probs))
        attack_type = self.classes[pred_idx]
        confidence = float(calib_probs[pred_idx])

        prob_dict = {
            self.classes[i]: round(float(calib_probs[i]), 4)
            for i in range(len(self.classes))
        }

        return {
            "attack_type": attack_type,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        }


# Global Singleton Pipeline instance
_PIPELINE = None

def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = CalibratedIDSInferencePipeline()
    return _PIPELINE

def predict_single(input_data):
    """
    Quick helper for single sample inference.
    """
    pipeline = get_pipeline()
    return pipeline.predict_one(input_data)


def main():
    parser = argparse.ArgumentParser(description="Calibrated XGBoost Intrusion Detection Inference CLI")
    parser.add_argument("--test-sample", action="store_true", help="Run inference on a sample from the test dataset")
    parser.add_argument("--input-json", type=str, help="JSON string or file path containing 46 feature values")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to test from testing dataset")
    args = parser.parse_args()

    pipeline = CalibratedIDSInferencePipeline()

    if args.test_sample or (not args.input_json and not args.test_sample):
        if not os.path.exists(TEST_CSV_PATH):
            print(f"Error: Test dataset not found at {TEST_CSV_PATH}")
            sys.exit(1)
            
        df_test = pd.read_csv(TEST_CSV_PATH)
        idx = max(0, min(args.sample_idx, len(df_test) - 1))
        sample_row = df_test.iloc[idx]
        true_label = sample_row["attack_class"]
        
        feature_dict = {f: float(sample_row[f]) for f in pipeline.features}
        result = pipeline.predict_one(feature_dict)
        
        print("\n" + "=" * 60)
        print("    CALIBRATED INFERENCE RESULT (TEST SAMPLE)")
        print("=" * 60)
        print(f"Sample Index in Test Set : {idx}")
        print(f"Ground Truth Label       : {true_label}")
        print(f"Calibrated Prediction    :\n")
        print(json.dumps(result, indent=4))
        print("=" * 60)
        
    elif args.input_json:
        if os.path.exists(args.input_json):
            with open(args.input_json, "r") as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(args.input_json)
            
        result = pipeline.predict_one(input_data)
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
