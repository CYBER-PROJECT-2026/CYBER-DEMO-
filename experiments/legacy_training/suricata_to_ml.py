import json
import time
import os
import joblib
import xgboost as xgb
import pandas as pd

# ============================================================
# PATHS
# ============================================================

EVE_FILE = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\suricata_logs\eve.json"

MODEL_FILE = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\models\xgboost_smote_tomek.json"

SCALER_FILE = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\preprocessed_objects\robust_scaler.pkl"


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

print("Loading XGBoost model...")

model = xgb.XGBClassifier()
model.load_model(MODEL_FILE)

print("XGBoost model loaded.")


# ============================================================
# LOAD SCALER
# ============================================================

print("Loading RobustScaler...")

scaler = joblib.load(SCALER_FILE)

print("RobustScaler loaded.")


# ============================================================
# CLASS MAPPING
# ============================================================

class_names = {
    0: "DDoS",
    1: "DoS",
    2: "Spoofing"
}


# ============================================================
# SURICATA MONITOR
# ============================================================

print()
print("========================================")
print("   SURICATA -> XGBOOST IDS BRIDGE")
print("========================================")
print("Waiting for Suricata alerts...")
print()


# Start reading new entries only
with open(EVE_FILE, "r", encoding="utf-8") as f:

    f.seek(0, os.SEEK_END)

    while True:

        line = f.readline()

        if not line:
            time.sleep(0.5)
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Only process Suricata alerts
        if event.get("event_type") != "alert":
            continue

        # ----------------------------------------------------
        # Extract Suricata information
        # ----------------------------------------------------

        src_ip = event.get("src_ip")
        src_port = event.get("src_port")
        dest_ip = event.get("dest_ip")
        dest_port = event.get("dest_port")
        proto = event.get("proto")

        alert = event.get("alert", {})

        signature_id = alert.get("signature_id")
        signature = alert.get("signature")
        severity = alert.get("severity")
        action = alert.get("action")

        # ----------------------------------------------------
        # Display Suricata alert
        # ----------------------------------------------------

        print("========================================")
        print("        SURICATA IDS ALERT")
        print("========================================")
        print("Source IP       :", src_ip)
        print("Source Port     :", src_port)
        print("Destination IP  :", dest_ip)
        print("Destination Port:", dest_port)
        print("Protocol        :", proto)
        print("Signature ID    :", signature_id)
        print("Alert           :", signature)
        print("Severity        :", severity)
        print("Action          :", action)

        # ----------------------------------------------------
        # Simple ML feature construction
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # These are placeholder values because a Suricata
        # alert does not contain all 17 features used during
        # XGBoost training.
        #
        # We are testing the complete technical connection
        # first.
        # ----------------------------------------------------

        features = [
            "flow_duration",
            "Protocol Type",
            "Duration",
            "Rate",
            "Srate",
            "Drate",
            "fin_flag_number",
            "syn_flag_number",
            "rst_flag_number",
            "psh_flag_number",
            "ack_flag_number",
            "syn_count",
            "fin_count",
            "TCP",
            "UDP",
            "Tot sum",
            "Weight"
        ]

        # Default values
        feature_values = {
            "flow_duration": 0,
            "Protocol Type": 6 if proto == "TCP" else 17,
            "Duration": 0,
            "Rate": 1,
            "Srate": 1,
            "Drate": 0,
            "fin_flag_number": 0,
            "syn_flag_number": 1 if proto == "TCP" else 0,
            "rst_flag_number": 0,
            "psh_flag_number": 0,
            "ack_flag_number": 0,
            "syn_count": 1 if proto == "TCP" else 0,
            "fin_count": 0,
            "TCP": 1 if proto == "TCP" else 0,
            "UDP": 1 if proto == "UDP" else 0,
            "Tot sum": 74,
            "Weight": 1
        }

        X = pd.DataFrame(
            [[feature_values[f] for f in features]],
            columns=features
        )

        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        X_scaled = scaler.transform(X)

        # ----------------------------------------------------
        # XGBoost prediction
        # ----------------------------------------------------

        prediction = model.predict(X_scaled)

        predicted_class = int(prediction[0])

        predicted_name = class_names.get(
            predicted_class,
            "Unknown"
        )

        # ----------------------------------------------------
        # Display ML result
        # ----------------------------------------------------

        print()
        print("========================================")
        print("          XGBOOST PREDICTION")
        print("========================================")
        print("Predicted Class :", predicted_class)
        print("Predicted Attack:", predicted_name)
        print("========================================")
        print()