import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("STEP 1: Loading dataset")
df = pd.read_csv("balanced_data.csv")

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

print("Dataset shape:", df.shape)

# Take 100 samples
sample = df.sample(n=100, random_state=42)

X = sample[features]
actual = sample["AttackCategory"]

print("\nSTEP 2: Loading scaler")
scaler = joblib.load("preprocessed_objects/robust_scaler.pkl")

X_scaled = scaler.transform(X)

print("Scaler loaded")

print("\nSTEP 3: Loading model")
model = xgb.XGBClassifier()
model.load_model("models/xgboost_smote_tomek.json")

print("Model loaded")

print("\nSTEP 4: Making predictions")
predicted_classes = model.predict(X_scaled).astype(int)

# Model class mapping:
# 0 = DDoS
# 1 = DoS
# 2 = Spoofing
class_names = {
    0: "DDoS",
    1: "DoS",
    2: "Spoofing"
}

predicted_names = [class_names[int(x)] for x in predicted_classes]

print("\n==============================")
print("     100 SAMPLE RESULTS")
print("==============================")

correct = 0

for i, (actual_name, predicted_name) in enumerate(
    zip(actual.tolist(), predicted_names), start=1
):
    result = "CORRECT" if actual_name == predicted_name else "WRONG"

    if result == "CORRECT":
        correct += 1

    print(
        f"{i:3d}. Actual: {actual_name:10s} | "
        f"Predicted: {predicted_name:10s} | {result}"
    )

accuracy = accuracy_score(actual, predicted_names)

print("\n==============================")
print("       FINAL RESULTS")
print("==============================")

print("Total Samples :", len(actual))
print("Correct       :", correct)
print("Incorrect     :", len(actual) - correct)
print(f"Accuracy      : {accuracy:.4f}")
print(f"Accuracy (%)  : {accuracy * 100:.2f}%")

print("\n==============================")
print(" CLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    actual,
    predicted_names,
    labels=["DDoS", "DoS", "Spoofing"]
))

print("==============================")
print(" CONFUSION MATRIX")
print("==============================")

print(confusion_matrix(
    actual,
    predicted_names,
    labels=["DDoS", "DoS", "Spoofing"]
))
