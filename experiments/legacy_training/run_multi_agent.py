"""
run_multi_agent.py
==================
Offline Single-Sample and Multi-Class Testing CLI for the Multi-Agent IDS.

Usage:
    python run_multi_agent.py --sample-idx 100
    python run_multi_agent.py --test-all-classes
    python run_multi_agent.py --sample-idx 0 --json-out alert_0.json
"""

import os
import sys
import json
import argparse
import pandas as pd
from backend.engine.pipeline import CompleteIDSPipeline

# UTF-8 Output Configuration
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV_PATH = os.path.join(BASE_DIR, "Testing_data", "testing_data.csv")


def print_formatted_result(sample_idx: int, ground_truth: str, result: dict):
    print("\n" + "=" * 80)
    print(f"AEGIS-IoT MULTI-AGENT IDS DOSSIER — SAMPLE INDEX #{sample_idx}")
    print("=" * 80)
    print(f"Ground Truth (Post-Hoc Eval Only): {ground_truth}")
    print(f"Detected Attack Type:             {result['attack_type']}")
    print(f"Calibrated Confidence:            {result['confidence']:.6f} ({result['confidence']*100:.2f}%)")
    print(f"Threat Severity:                  {result['threat_assessment']['severity']}")
    print("-" * 80)

    print("\n[1. DETECTION AGENT ANALYSIS]")
    print(f"  Summary: {result['detection']['summary']}")
    print("  Posterior Probabilities:")
    for cls, prob in result['detection']['probabilities'].items():
        print(f"    - {cls:<10s}: {prob:.6f} ({prob*100:.2f}%)")

    print("\n[2. EXPLAINABILITY AGENT (TreeSHAP Evidence)]")
    print(f"  Top Positive Features (Pushing toward {result['attack_type']}):")
    for feat in result['explainability']['supporting_evidence'][:4]:
        print(f"    * {feat['feature']:<18s} | Val: {feat['feature_value']:>10.2f} | SHAP: +{feat['shap_value']:.4f}")
    if result['explainability']['opposing_evidence']:
        print(f"  Top Opposing Features (Pushing away from {result['attack_type']}):")
        for feat in result['explainability']['opposing_evidence'][:2]:
            print(f"    * {feat['feature']:<18s} | Val: {feat['feature_value']:>10.2f} | SHAP: {feat['shap_value']:.4f}")

    print("\n[3. THREAT INTELLIGENCE AGENT — ADAPTIVE HYBRID RAG]")
    ti = result.get('threat_intelligence', {})
    if ti.get('attack_descriptions'):
        print("  Attack Description:")
        for d in ti['attack_descriptions']:
            print(f"    • {d}")
    if ti.get('mechanisms'):
        print("\n  How It Works (Technical Mechanism):")
        for m in ti['mechanisms']:
            print(f"    • {m}")
    if ti.get('iot_impacts'):
        print("\n  IoT/OT Impact:")
        for imp in ti['iot_impacts']:
            print(f"    • {imp}")
    if ti.get('indicators'):
        print("\n  Relevant Indicators & Signatures:")
        for ind in ti['indicators']:
            print(f"    • {ind}")

    print("\n  Retrieved Authoritative Sources:")
    for i, doc in enumerate(ti.get('retrieved_sources', []), 1):
        print(f"    Source {i}: [{doc['source']}] {doc['title']} (Score: {doc['relevance_score']:.4f})")
        print(f"      URL:      {doc['url']}")
        print(f"      Chunk ID: {doc.get('chunk_id', 'N/A')}")
        summary_text = doc.get('content', '')[:300].strip()
        print(f"      Summary:  {summary_text}...")

    print("\n[4. THREAT ASSESSMENT AGENT]")
    print(f"  Severity & Risk:  {result['threat_assessment']['summary']}")
    print(f"  Potential Impact: {result['threat_assessment']['potential_impact']}")
    print(f"  Uncertainty:      {result['threat_assessment'].get('uncertainty', 'Minimal')}")
    ep = result['threat_assessment'].get('epistemic_breakdown', {})
    if ep:
        print("  Epistemic Segregation:")
        print(f"    - FACT:           {'; '.join(ep.get('FACT', []))}")
        print(f"    - INFERENCE:      {'; '.join(ep.get('INFERENCE', []))}")
        print(f"    - RECOMMENDATION: {'; '.join(ep.get('RECOMMENDATION', []))}")

    print("\n[5. RESPONSE RECOMMENDATION AGENT]")
    for r in result['recommendations']:
        print(f"  [{r['priority']}] {r['title']}")
        print(f"    Action Type: {r['action_type']}")
        print(f"    Rationale:   {r['rationale']}")
        print(f"    Guidance:    {r['guidance']}")
        print(f"    Precautions: {r.get('precautions', 'None')}")
        print(f"    Supporting:  {'; '.join(r.get('supporting_sources', ['Guidelines']))}")

    print("\n[6. GROUNDED LLM SYNTHESIS DOSSIER]")
    print(result['final_explanation'])

    print("\n[LATENCY & TELEMETRY BREAKDOWN]")
    for k, v in result.get('_latency_breakdown_ms', {}).items():
        print(f"  - {k:<25s}: {v:.2f} ms")
    print("=" * 80 + "\n")


def run_single(pipeline: CompleteIDSPipeline, df_test: pd.DataFrame, sample_idx: int, json_out: str = None):
    if sample_idx < 0 or sample_idx >= len(df_test):
        raise IndexError(f"sample_idx {sample_idx} is out of bounds (0 to {len(df_test)-1}).")

    row = df_test.iloc[sample_idx]
    # Ground truth isolated
    ground_truth = row.get("attack_class", "UNKNOWN")

    # Run pipeline with NO ground truth passed
    result = pipeline.run_inference_and_reasoning(row)
    print_formatted_result(sample_idx, ground_truth, result)

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[Export] Saved structured output to: {json_out}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent IDS Inference CLI")
    parser.add_argument("--sample-idx", type=int, default=None, help="Row index in Testing_data/testing_data.csv (0 to 4999)")
    parser.add_argument("--test-all-classes", action="store_true", help="Run tests for DDoS, DoS, and Spoofing samples")
    parser.add_argument("--json-out", type=str, default=None, help="Path to output JSON file")
    args = parser.parse_args()

    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Testing dataset not found: {TEST_CSV_PATH}")

    print("Loading testing dataset...")
    df_test = pd.read_csv(TEST_CSV_PATH)
    pipeline = CompleteIDSPipeline()

    if args.test_all_classes:
        print("\n--- RUNNING MULTI-CLASS VERIFICATION (DDoS, DoS, Spoofing) ---")
        # Find first index for each class
        ddos_idx = df_test[df_test['attack_class'] == 'DDoS'].index[0]
        dos_idx = df_test[df_test['attack_class'] == 'DoS'].index[0]
        spoof_idx = df_test[df_test['attack_class'] == 'Spoofing'].index[0]

        print(f"\n[Test 1/3] Running DDoS Sample (Row #{ddos_idx})...")
        run_single(pipeline, df_test, ddos_idx)

        print(f"\n[Test 2/3] Running DoS Sample (Row #{dos_idx})...")
        run_single(pipeline, df_test, dos_idx)

        print(f"\n[Test 3/3] Running Spoofing Sample (Row #{spoof_idx})...")
        run_single(pipeline, df_test, spoof_idx)

    else:
        sample_idx = args.sample_idx if args.sample_idx is not None else 100
        run_single(pipeline, df_test, sample_idx, args.json_out)


if __name__ == "__main__":
    main()
