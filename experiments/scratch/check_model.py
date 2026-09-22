import joblib

model = joblib.load("cic_iot_xgboost_model.pkl")

print("Model loaded successfully!")
print("Model type:", type(model))
print("Number of classes:", model.n_classes_)
print("Number of features:", model.n_features_in_)