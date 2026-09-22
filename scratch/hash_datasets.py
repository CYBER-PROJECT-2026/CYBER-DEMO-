import hashlib
import os

DATA_DIR = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\data\aegis_live_v2_quarantine"

for fname in ["train.csv", "validation.csv", "test.csv"]:
    p = os.path.join(DATA_DIR, fname)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    sz = os.path.getsize(p)
    print(f"{fname:15s} | Size: {sz:,} bytes | SHA256: {h}")
