"""
Standalone Inference Pipeline with Temperature Scaling for 46-Feature IoT Intrusion Detection.

Explicitly loads:
- cic_iot_xgboost_model.pkl
- label_encoder.pkl
- calibrator_temperature.pkl

Does NOT load or use:
- calibrated_xgboost_final.pkl
- calibrator_isotonic.pkl
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
TEMP_CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")


class TemperatureScaledInferencePipeline:
    """
    Inference pipeline explicitly using the frozen 46-feature XGBoost model
    and Temperature Scaling calibrator.
    """
    def __init__(self,
                 model_path=MODEL_PATH,
                 encoder_path=ENCODER_PATH,
                 temp_calib_path=TEMP_CALIB_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        if not os.path.exists(temp_calib_path):
            raise FileNotFoundError(f"Temperature calibrator not found: {temp_calib_path}")

        # Explicitly load only the required artifacts
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.calib_temp = joblib.load(temp_calib_path)

        self.features = list(self.model.feature_names_in_)
        self.classes = list(self.encoder.classes_)
        self.booster = self.model.get_booster()

        self.ddos_idx = self.classes.index("DDoS")
        self.dos_idx = self.classes.index("DoS")
        self.spoof_idx = self.classes.index("Spoofing")

    def predict_one(self, input_df_or_series):
        """
        Accepts a 1-row DataFrame, Series, or dict.
        Returns dict with raw and temperature-scaled probabilities.
        """
        if isinstance(input_df_or_series, dict):
            df = pd.DataFrame([{f: float(input_df_or_series[f]) for f in self.features}])
        elif isinstance(input_df_or_series, pd.Series):
            df = pd.DataFrame([{f: float(input_df_or_series[f]) for f in self.features}])
        elif isinstance(input_df_or_series, pd.DataFrame):
            df = input_df_or_series[self.features].copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_df_or_series)}")

        df = df.astype(np.float32)

        # 1. Raw XGBoost probabilities
        raw_probs = self.model.predict_proba(df)[0]
        raw_pred_idx = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[raw_pred_idx])

        # 2. Extract booster logits & apply Temperature Scaling
        dmat = xgb.DMatrix(df)
        logits = self.booster.predict(dmat, output_margin=True)
        temp_probs = self.calib_temp.predict_proba(logits)[0]
        temp_pred_idx = int(np.argmax(temp_probs))
        temp_conf = float(temp_probs[temp_pred_idx])

        return {
            "raw_probabilities": {
                "DDoS": float(raw_probs[self.ddos_idx]),
                "DoS": float(raw_probs[self.dos_idx]),
                "Spoofing": float(raw_probs[self.spoof_idx])
            },
            "raw_predicted_class": self.classes[raw_pred_idx],
            "raw_confidence": raw_conf,
            "temperature_probabilities": {
                "DDoS": float(temp_probs[self.ddos_idx]),
                "DoS": float(temp_probs[self.dos_idx]),
                "Spoofing": float(temp_probs[self.spoof_idx])
            },
            "calibrated_predicted_class": self.classes[temp_pred_idx],
            "calibrated_confidence": temp_conf
        }


def format_and_print_sample(sample_idx, actual_class, result):
    print("============================================================")
    print("TEMPERATURE-SCALED INFERENCE RESULT")
    print("============================================================")
    print()
    print(f"Sample Index: {sample_idx}")
    print(f"Ground Truth: {actual_class}")
    print()
    print("Raw XGBoost:")
    print(f"DDoS:     {result['raw_probabilities']['DDoS']:.8f}")
    print(f"DoS:      {result['raw_probabilities']['DoS']:.8f}")
    print(f"Spoofing: {result['raw_probabilities']['Spoofing']:.8f}")
    print()
    print("Temperature Scaled:")
    print(f"DDoS:     {result['temperature_probabilities']['DDoS']:.8f}")
    print(f"DoS:      {result['temperature_probabilities']['DoS']:.8f}")
    print(f"Spoofing: {result['temperature_probabilities']['Spoofing']:.8f}")
    print()
    print("Final Prediction:")
    print(f"Attack Type: {result['calibrated_predicted_class']}")
    print(f"Confidence:  {result['calibrated_confidence']:.8f}")
    print()
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Inference using XGBoost and Temperature Scaling Calibrator")
    parser.add_argument("--test-sample", action="store_true", help="Run inference on a sample from Testing_data/testing_data.csv")
    parser.add_argument("--sample-idx", type=int, default=0, help="Row index of the sample in testing_data.csv (0 to 4999)")
    parser.add_argument("--input-json", type=str, default=None, help="Path to JSON file with 46 features or JSON string")
    args = parser.parse_args()

    pipeline = TemperatureScaledInferencePipeline()

    if args.test_sample:
        if not os.path.exists(TEST_CSV_PATH):
            raise FileNotFoundError(f"Testing dataset not found: {TEST_CSV_PATH}")

        df_test = pd.read_csv(TEST_CSV_PATH)
        if args.sample_idx < 0 or args.sample_idx >= len(df_test):
            raise IndexError(f"sample-idx {args.sample_idx} is out of bounds (0 to {len(df_test)-1})")

        row = df_test.iloc[args.sample_idx]
        target_col = 'attack_class' if 'attack_class' in df_test.columns else 'AttackCategory'
        actual_class = str(row[target_col])

        result = pipeline.predict_one(row)
        format_and_print_sample(args.sample_idx, actual_class, result)

    elif args.input_json:
        if os.path.exists(args.input_json):
            with open(args.input_json, "r") as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(args.input_json)

        result = pipeline.predict_one(input_data)
        format_and_print_sample("Custom JSON Input", "N/A", result)

    else:
        # Default run on sample 0
        df_test = pd.read_csv(TEST_CSV_PATH)
        row = df_test.iloc[0]
        target_col = 'attack_class' if 'attack_class' in df_test.columns else 'AttackCategory'
        actual_class = str(row[target_col])
        result = pipeline.predict_one(row)
        format_and_print_sample(0, actual_class, result)


if __name__ == "__main__":
    main()
