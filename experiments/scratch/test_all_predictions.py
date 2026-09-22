import pandas as pd
import joblib
import xgboost as xgb

# 1. Load dataset
df = pd.read_csv("balanced_data.csv")

# 2. Features used by the model
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

# 3. Load scaler
scaler = joblib.load(
    "preprocessed_objects/robust_scaler.pkl"
)

# 4. Load XGBoost model
model = xgb.XGBClassifier()
model.load_model(
    "models/xgboost_smote_tomek.json"
)

# 5. Correct class mapping for this 3-class model
class_names = {
    0: "DDoS",
    1: "DoS",
    2: "Spoofing"
}

print("\n========================================")
print("       IDS MULTI-CLASS TEST")
print("========================================")

# Test one sample from each attack category
for attack in ["DDoS", "DoS", "Spoofing"]:

    sample = df[df["AttackCategory"] == attack].iloc[[0]]

    X = sample[features]

    # Scale features
    X_scaled = scaler.transform(X)

    # Prediction
    prediction = model.predict(X_scaled)

    predicted_class = int(prediction[0])

    predicted_name = class_names[predicted_class]

    print("\nActual Attack    :", attack)
    print("Predicted Attack :", predicted_name)
    print("Class Number     :", predicted_class)

    if attack == predicted_name:
        print("Result           : CORRECT")
    else:
        print("Result           : INCORRECT")

print("\n========================================")
print("              TEST COMPLETE")
print("========================================")