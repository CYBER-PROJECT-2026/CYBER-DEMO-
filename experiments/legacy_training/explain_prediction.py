"""
Individual Prediction Explainability Script using SHAP for 46-Feature XGBoost IDS.

Features:
1. Loads frozen 46-feature XGBoost model (cic_iot_xgboost_model.pkl).
2. Loads label encoder (label_encoder.pkl) and Temperature Calibrator (calibrator_temperature.pkl).
3. Computes raw XGBoost probabilities and Temperature-calibrated confidence.
4. Generates exact TreeExplainer SHAP values for the predicted class.
5. Displays top contributing features with magnitude, direction, and actual feature values.
"""

import os
import sys
import json
import argparse
import joblib
import shap
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


class SHAPIndividualExplainer:
    def __init__(self,
                 model_path=MODEL_PATH,
                 encoder_path=ENCODER_PATH,
                 temp_calib_path=TEMP_CALIB_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        if not os.path.exists(temp_calib_path):
            raise FileNotFoundError(f"Temperature calibrator not found: {temp_calib_path}")

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.calib_temp = joblib.load(temp_calib_path)

        self.features = list(self.model.feature_names_in_)
        self.classes = list(self.encoder.classes_)
        self.booster = self.model.get_booster()

        # Initialize SHAP TreeExplainer on booster
        self.explainer = shap.TreeExplainer(self.model)

    def explain_sample(self, sample_series_or_df, top_n=10):
        if isinstance(sample_series_or_df, dict):
            df = pd.DataFrame([{f: float(sample_series_or_df[f]) for f in self.features}])
        elif isinstance(sample_series_or_df, pd.Series):
            df = pd.DataFrame([{f: float(sample_series_or_df[f]) for f in self.features}])
        elif isinstance(sample_series_or_df, pd.DataFrame):
            df = sample_series_or_df[self.features].copy()
        else:
            raise TypeError(f"Unsupported input type: {type(sample_series_or_df)}")

        df = df.astype(np.float32)

        # 1. Raw probabilities & Logits
        raw_probs = self.model.predict_proba(df)[0]
        dmat = xgb.DMatrix(df)
        logits = self.booster.predict(dmat, output_margin=True)

        # 2. Temperature Scaled probabilities
        temp_probs = self.calib_temp.predict_proba(logits)[0]
        pred_idx = int(np.argmax(temp_probs))
        pred_class = self.classes[pred_idx]
        confidence = float(temp_probs[pred_idx])

        # 3. Compute SHAP values
        sv = self.explainer.shap_values(df)  # Shape (1, 46, 3)
        if sv.ndim == 3:
            class_shap = sv[0, :, pred_idx]
        elif isinstance(sv, list):
            class_shap = sv[pred_idx][0]
        else:
            class_shap = sv[0]

        # Extract base value for predicted class
        base_val = float(self.explainer.expected_value[pred_idx]) if hasattr(self.explainer.expected_value, '__getitem__') else float(self.explainer.expected_value)

        # 4. Rank features by absolute SHAP value
        feature_vals = df.iloc[0].values
        abs_shap = np.abs(class_shap)
        ranked_indices = np.argsort(abs_shap)[::-1]

        contributions = []
        for rank, idx in enumerate(ranked_indices[:top_n], start=1):
            feat_name = self.features[idx]
            val = float(feature_vals[idx])
            s_val = float(class_shap[idx])
            direction = f"Pushes prediction TOWARD {pred_class} (+)" if s_val > 0 else f"Pushes prediction AWAY from {pred_class} (-)"
            contributions.append({
                "rank": rank,
                "feature": feat_name,
                "feature_value": val,
                "shap_value": s_val,
                "direction": direction,
                "raw_direction": "POSITIVE" if s_val > 0 else "NEGATIVE"
            })

        return {
            "predicted_class": pred_class,
            "predicted_class_idx": pred_idx,
            "confidence": confidence,
            "raw_probabilities": {self.classes[i]: float(raw_probs[i]) for i in range(len(self.classes))},
            "temperature_probabilities": {self.classes[i]: float(temp_probs[i]) for i in range(len(self.classes))},
            "base_value": base_val,
            "contributions": contributions,
            "all_shap_values": class_shap,
            "feature_values": feature_vals
        }


def print_explanation_report(sample_idx, actual_class, result, top_n=10):
    print("============================================================")
    print("SHAP IDS EXPLANATION")
    print("============================================================")
    print()
    print(f"Sample Index:          {sample_idx}")
    print(f"Ground Truth:          {actual_class}")
    print(f"Predicted Attack:      {result['predicted_class']}")
    print(f"Calibrated Confidence: {result['confidence']:.8f} ({result['confidence']*100:.4f}%)")
    print()
    print("Probabilities:")
    for cname, p in result['temperature_probabilities'].items():
        print(f"  {cname:<10s}: {p:.8f}")
    print()
    print(f"Top {top_n} SHAP Contributions for Predicted Class [{result['predicted_class']}]:")
    print("-" * 60)
    for item in result['contributions']:
        sign = "+" if item['shap_value'] > 0 else ""
        print(f"{item['rank']:2d}. Feature:       {item['feature']}")
        print(f"    SHAP value:    {sign}{item['shap_value']:.6f}")
        print(f"    Direction:     {item['direction']}")
        print(f"    Feature value: {item['feature_value']:.6f}")
        print()
    print("============================================================")
    print("Explanation Semantics:")
    print(f"- Positive SHAP (+): Feature pushes log-odds TOWARD {result['predicted_class']}.")
    print(f"- Negative SHAP (-): Feature pushes log-odds AWAY from {result['predicted_class']}.")
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Individual SHAP Explanation for 46-Feature XGBoost IDS")
    parser.add_argument("--sample-idx", type=int, default=100, help="Row index from Testing_data/testing_data.csv")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top SHAP features to display")
    parser.add_argument("--input-json", type=str, default=None, help="Path to JSON file with 46 features or JSON string")
    args = parser.parse_args()

    explainer = SHAPIndividualExplainer()

    if args.input_json:
        if os.path.exists(args.input_json):
            with open(args.input_json, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(args.input_json)
        res = explainer.explain_sample(data, top_n=args.top_n)
        print_explanation_report("Custom JSON Input", "N/A", res, top_n=args.top_n)
    else:
        if not os.path.exists(TEST_CSV_PATH):
            raise FileNotFoundError(f"Test dataset not found: {TEST_CSV_PATH}")
        df_test = pd.read_csv(TEST_CSV_PATH)
        if args.sample_idx < 0 or args.sample_idx >= len(df_test):
            raise IndexError(f"sample-idx {args.sample_idx} is out of range (0 to {len(df_test)-1})")
        row = df_test.iloc[args.sample_idx]
        target_col = 'attack_class' if 'attack_class' in df_test.columns else 'AttackCategory'
        actual_class = str(row[target_col])
        res = explainer.explain_sample(row, top_n=args.top_n)
        print_explanation_report(args.sample_idx, actual_class, res, top_n=args.top_n)


if __name__ == "__main__":
    main()
