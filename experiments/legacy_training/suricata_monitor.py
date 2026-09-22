import json
import time

LOG_FILE = r"D:\-AI-Driven-Intrusion-Detection-System-for-IoT-Networks\suricata_logs\eve.json"

print("========================================")
print("       SURICATA IDS MONITOR")
print("========================================")
print("Waiting for Suricata alerts...\n")

with open(LOG_FILE, "r", encoding="utf-8") as f:
    f.seek(0, 2)

    while True:
        line = f.readline()

        if not line:
            time.sleep(0.2)
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("event_type") != "alert":
            continue

        alert = event.get("alert", {})

        print("\n========================================")
        print("        SURICATA IDS ALERT")
        print("========================================")
        print("Source IP       :", event.get("src_ip"))
        print("Source Port     :", event.get("src_port"))
        print("Destination IP  :", event.get("dest_ip"))
        print("Destination Port:", event.get("dest_port"))
        print("Protocol        :", event.get("proto"))
        print("Signature ID    :", alert.get("signature_id"))
        print("Alert           :", alert.get("signature"))
        print("Severity        :", alert.get("severity"))
        print("Action          :", alert.get("action"))
        print("========================================")