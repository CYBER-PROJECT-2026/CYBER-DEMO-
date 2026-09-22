import json

with open('reports/aegis_live_v2_multiclass_validation.json') as f:
    d = json.load(f)

for cls in ['DDoS', 'DoS', 'Spoofing', 'Benign']:
    print(f"### Class: {cls}")
    print("| Feature | Count | Min | Max | Mean | Median | Std | NaN | Inf |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for feat, s in d['feature_stats'][cls].items():
        print(f"| `{feat}` | {int(s['Count']):,} | {s['Min']:.4f} | {s['Max']:.4f} | {s['Mean']:.4f} | {s['Median']:.4f} | {s['Std']:.4f} | {int(s['NaN_Count'])} | {int(s['Inf_Count'])} |")
    print()
