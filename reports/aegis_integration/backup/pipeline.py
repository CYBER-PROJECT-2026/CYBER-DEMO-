"""
backend/engine/pipeline.py
==========================
Unified End-to-End Pipeline for 46-Feature IoT Intrusion Detection System
with Temperature Scaling, TreeSHAP Explainability, Adaptive Hybrid RAG,
and 5-Agent Grounded Reasoning.

Flow:
46 Features -> XGBoost IDS -> Temperature Scaling -> TreeSHAP -> MultiAgentOrchestrator -> Final JSON
"""

import os
import sys
import joblib
import shap
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Union, List, Optional
from backend.agents.agent_system import MultiAgentOrchestrator
from backend.rag.adaptive_hybrid_rag import AdaptiveHybridRAG
from backend.detection.abstention_gate import OODAbstentionGate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from offline_feature_validator import validate_46_features, FeatureValidationError, CANONICAL_46_FEATURES

MODEL_PATH = os.path.join(BASE_DIR, "cic_iot_xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
CALIB_PATH = os.path.join(BASE_DIR, "calibrator_temperature.pkl")


class CompleteIDSPipeline:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 encoder_path: str = ENCODER_PATH,
                 calib_path: str = CALIB_PATH):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibrator not found at {calib_path}")

        # Load frozen artifacts
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.calib_temp = joblib.load(calib_path)

        self.features = list(self.model.feature_names_in_)
        self.classes = list(self.encoder.classes_)
        self.booster = self.model.get_booster()

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Initialize Adaptive Hybrid RAG and Multi-Agent Orchestrator
        self.rag = AdaptiveHybridRAG()
        self.orchestrator = MultiAgentOrchestrator(self.rag)
        self.abstention_gate = OODAbstentionGate()

    def extract_features(self, input_data: Union[pd.Series, pd.DataFrame, dict, list, np.ndarray]) -> pd.DataFrame:
        """
        Extracts and strictly validates all 46 canonical features against the production model schema.
        Enforces 10 strict invariants (zero missing features, zero extra columns, no silent zero-filling, float32 conversion).
        Raises FeatureValidationError if any of the 46 canonical features are missing or invalid.
        """
        return validate_46_features(input_data)

    def run_inference_and_reasoning(self,
                                    input_data: Union[pd.Series, pd.DataFrame, dict],
                                    top_n_shap: int = 10) -> Dict[str, Any]:
        """
        Executes complete pipeline:
        1. 46-feature XGBoost prediction
        2. Temperature Scaling calibration
        3. TreeSHAP computation for predicted class
        4. Structured Multi-Agent orchestration (5 agents)
        5. Grounded LLM synthesis
        """
        t_start = time.perf_counter()
        df = self.extract_features(input_data)

        # 1. XGBoost Logits & Raw Probabilities
        t0 = time.perf_counter()
        raw_probs = self.model.predict_proba(df)[0]
        dmat = xgb.DMatrix(df)
        logits = self.booster.predict(dmat, output_margin=True)
        t_ml = round((time.perf_counter() - t0) * 1000, 3)

        # 2. Temperature Scaling
        t0 = time.perf_counter()
        temp_probs = self.calib_temp.predict_proba(logits)[0]
        pred_idx = int(np.argmax(temp_probs))
        pred_class = self.classes[pred_idx]
        confidence = float(temp_probs[pred_idx])
        t_calib = round((time.perf_counter() - t0) * 1000, 3)

        prob_dict = {self.classes[i]: float(temp_probs[i]) for i in range(len(self.classes))}

        # 3. TreeSHAP Calculation
        t0 = time.perf_counter()
        sv = self.explainer.shap_values(df)
        if sv.ndim == 3:
            class_shap = sv[0, :, pred_idx]
        elif isinstance(sv, list):
            class_shap = sv[pred_idx][0]
        else:
            class_shap = sv[0]

        feature_vals = df.iloc[0].values
        abs_shap = np.abs(class_shap)
        ranked_indices = np.argsort(abs_shap)[::-1]

        contributions = []
        for rank, idx in enumerate(ranked_indices[:top_n_shap], start=1):
            feat_name = self.features[idx]
            val = float(feature_vals[idx])
            s_val = float(class_shap[idx])
            direction = "POSITIVE" if s_val > 0 else "NEGATIVE"
            contributions.append({
                "rank": rank,
                "feature": feat_name,
                "feature_value": val,
                "shap_value": s_val,
                "direction": direction
            })
        t_shap = round((time.perf_counter() - t0) * 1000, 3)

        # 4. External OOD & Confidence Abstention Gate
        t0 = time.perf_counter()
        gate_eval = self.abstention_gate.evaluate(
            {
                "attack_type": pred_class,
                "predicted_class": pred_class,
                "confidence": confidence,
                "probabilities": prob_dict
            },
            df
        )
        t_gate = round((time.perf_counter() - t0) * 1000, 3)

        # 5. Assemble Structured IDS Alert Input (Part 3 format + Phase 14 extensions)
        feature_dict = {self.features[i]: float(feature_vals[i]) for i in range(len(self.features))}
        ids_payload = {
            "attack_type": gate_eval["advisory_class"] if gate_eval["is_abstained"] else pred_class,
            "predicted_class": pred_class,
            "decision_state": gate_eval["decision_state"],
            "is_abstained": gate_eval["is_abstained"],
            "abstention_reasons": gate_eval["abstention_reasons"],
            "abstention_summary": gate_eval["abstention_summary"],
            "advisory_class": gate_eval["advisory_class"],
            "trusted_class": gate_eval["trusted_class"],
            "calibrated_confidence": confidence,
            "confidence": confidence,
            "top2_margin": gate_eval["top2_margin"],
            "cc_mahalanobis_distance": gate_eval["cc_mahalanobis_distance"],
            "is_atypical_distance": gate_eval["is_atypical_distance"],
            "is_low_confidence": gate_eval["is_low_confidence"],
            "automated_blocking_allowed": gate_eval["automated_blocking_allowed"],
            "advisory_note": gate_eval["advisory_note"],
            "class_probabilities": prob_dict,
            "probabilities": prob_dict,
            "features": feature_dict,
            "shap_evidence": contributions
        }

        # 6. Multi-Agent Reasoning & Grounded Synthesis
        agent_result = self.orchestrator.process_ids_alert(ids_payload)
        agent_result["gate_evaluation"] = gate_eval
        t_total = round((time.perf_counter() - t_start) * 1000, 3)

        # Append timing breakdown
        agent_result["_latency_breakdown_ms"] = {
            "ml_inference": t_ml,
            "temperature_calibration": t_calib,
            "shap_explainability": t_shap,
            "ood_abstention_gate": t_gate,
            "multi_agent_reasoning": agent_result["_agent_telemetry"]["total_pipeline_ms"],
            "end_to_end_total": t_total
        }

        return agent_result
