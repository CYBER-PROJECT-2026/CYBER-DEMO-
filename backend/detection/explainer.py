"""
backend/detection/explainer.py
==============================
TreeSHAP Explainer for Canonical 46-Feature XGBoost Model.
Computes exact Shapley attributions directly on raw 46 features without scaling.
"""

import os
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
from .detector import IoTIntrusionDetector, FEATURE_COLUMNS


class TreeSHAPExplainer:
    def __init__(self, detector: IoTIntrusionDetector):
        self.detector = detector
        # Initialize SHAP TreeExplainer on the canonical 46-feature XGBoost model
        self.explainer = shap.TreeExplainer(self.detector.model)
        self.feature_names = self.detector.feature_columns

    def explain(self,
                raw_input_df: pd.DataFrame,
                predicted_class: int,
                raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes TreeSHAP values for the canonical 46 features.
        Returns exact positive (attack-driving) and negative (mitigating) contributing factors.
        """
        shap_values_raw = self.explainer.shap_values(raw_input_df)
        
        # Handle multiclass shap output format
        if isinstance(shap_values_raw, list):
            class_shap = shap_values_raw[predicted_class][0]
        elif len(shap_values_raw.shape) == 3:
            class_shap = shap_values_raw[0, :, predicted_class]
        else:
            class_shap = shap_values_raw[0]

        # Extract base value if available
        if hasattr(self.explainer, 'expected_value') and self.explainer.expected_value is not None:
            if hasattr(self.explainer.expected_value, '__getitem__'):
                base_val = float(self.explainer.expected_value[predicted_class])
            else:
                base_val = float(self.explainer.expected_value)
        else:
            base_val = 0.0

        # Build feature importance attribution list
        attributions: List[Dict[str, Any]] = []
        for name, shap_val in zip(self.feature_names, class_shap):
            raw_val = float(raw_features.get(name, 0.0))
            attributions.append({
                "feature": name,
                "raw_value": round(raw_val, 4),
                "shap_value": round(float(shap_val), 6),
                "direction": "POSITIVE" if shap_val > 0 else "NEGATIVE",
                "impact": "INCREASES_RISK" if shap_val > 0 else "DECREASES_RISK",
                "abs_importance": abs(float(shap_val))
            })

        # Sort by absolute SHAP importance
        attributions.sort(key=lambda x: x["abs_importance"], reverse=True)
        
        # Positive drivers (why it flagged this attack)
        top_positive_drivers = [a for a in attributions if a["shap_value"] > 0][:5]
        top_negative_drivers = [a for a in attributions if a["shap_value"] < 0][:3]
        
        predicted_attack_name = self.detector.class_names.get(predicted_class, "Attack")
        summary_sentences = []
        for d in top_positive_drivers[:3]:
            summary_sentences.append(f"{d['feature']} (val: {d['raw_value']}) pushed decision toward {predicted_attack_name} (SHAP: +{d['shap_value']:.4f})")
        
        narrative = ". ".join(summary_sentences) if summary_sentences else "Distributed flow feature attribution."

        return {
            "base_value": round(base_val, 4),
            "top_attributions": attributions[:10],
            "top_drivers": top_positive_drivers,
            "top_opposing": top_negative_drivers,
            "narrative_explanation": narrative
        }
