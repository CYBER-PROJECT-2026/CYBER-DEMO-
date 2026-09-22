import joblib

encoder = joblib.load("label_encoder.pkl")

print("Encoder classes:")
print(encoder.classes_)

print("\nNumber of classes:")
print(len(encoder.classes_))