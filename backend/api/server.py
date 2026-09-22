import os
import sys
import time
import uuid
import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.database.models_db import (
    init_db, insert_alert, get_alerts, get_incidents,
    get_audit_logs, update_incident_decision
)
from backend.detection.detector import IoTIntrusionDetector
from backend.detection.explainer import TreeSHAPExplainer
from backend.detection.confidence_gate import ConfidenceGate
from backend.rag.kb_store import CYBERSECURITY_KNOWLEDGE_BASE
from backend.rag.hybrid_retriever import AdaptiveHybridRetriever
from backend.agents.orchestrator import MultiAgentOrchestrator
from backend.engine.flow_extractor import RealTimeFlowExtractor
from backend.engine.live_sniffer import LivePacketSniffer
from backend.engine.traffic_generator import IoTTrafficGenerator
from backend.enforcement.enforcement import SecurityEnforcementEngine

# Initialize Core Services
app = FastAPI(title="AEGIS-IoT Autonomous SOC Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Singletons
detector = IoTIntrusionDetector()
explainer = TreeSHAPExplainer(detector)
confidence_gate = ConfidenceGate()
retriever = AdaptiveHybridRetriever()
orchestrator = MultiAgentOrchestrator()
enforcement = SecurityEnforcementEngine()

# Connected WebSockets
active_websockets: List[WebSocket] = []

async def broadcast_ws(event_type: str, data: Any):
    payload = json.dumps({"type": event_type, "data": data})
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

def handle_flow_ready(flow_data: Dict[str, Any]):
    """Invoked when a flow is completed/flushed from the live stream"""
    raw_features = flow_data["features"]
    
    # 1. XGBoost Detection (< 0.14 ms) on 46 raw features
    pred = detector.predict(raw_features)
    attack_type = pred["attack_type"]
    confidence = pred["confidence"]

    # 2. TreeSHAP Explainability on validated raw DataFrame
    shap_res = explainer.explain(pred["formatted_df"], pred["predicted_class"], raw_features)

    # 3. Confidence Gate Validation (OOD & Confidence Aware)
    gate_eval = confidence_gate.evaluate(pred, raw_features)
    final_attack_type = "UNKNOWN_OR_REVIEW" if gate_eval["is_abstained"] else attack_type

    alert_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
    alert_record = {
        "id": alert_id,
        "timestamp": time.time(),
        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "src_ip": flow_data.get("src_ip", "0.0.0.0"),
        "src_port": flow_data.get("src_port", 0),
        "dst_ip": flow_data.get("dst_ip", "0.0.0.0"),
        "dst_port": flow_data.get("dst_port", 0),
        "protocol": flow_data.get("proto", "TCP"),
        "attack_type": final_attack_type,
        "advisory_attack_type": attack_type,
        "decision_state": gate_eval.get("decision_state", f"KNOWN_{attack_type}"),
        "is_abstained": gate_eval.get("is_abstained", False),
        "abstention_reasons": gate_eval.get("abstention_reasons", []),
        "confidence": round(confidence, 4),
        "severity": gate_eval["severity"],
        "routing": gate_eval["routing"],
        "raw_features": raw_features,
        "shap_values": shap_res,
        "status": "NEW"
    }

    # Save Alert to DB
    insert_alert(alert_record)

    # 4. If High Confidence -> Trigger Multi-Agent Orchestrator
    incident_record = None
    if gate_eval["escalate_to_rag"]:
        incident_record = orchestrator.process_alert(alert_record)

    # Broadcast to Connected SOC Dashboards via Event Loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_ws("NEW_ALERT", alert_record), loop)
            if incident_record:
                asyncio.run_coroutine_threadsafe(broadcast_ws("NEW_INCIDENT", incident_record), loop)
    except Exception as e:
        pass

# Initialize Flow Extractor & Engine
flow_extractor = RealTimeFlowExtractor(on_flow_ready=handle_flow_ready)
sniffer = LivePacketSniffer(flow_extractor)
generator = IoTTrafficGenerator(flow_extractor)

# Pydantic Request Models
class DecisionRequest(BaseModel):
    decision: str  # "APPROVED" | "REJECTED" | "MODIFIED"
    actions: List[Dict[str, Any]]
    notes: Optional[str] = ""

class SimulateRequest(BaseModel):
    attack_type: str  # "BENIGN" | "DDOS" | "DOS" | "SPOOF"

# --- REST ENDPOINTS ---

@app.get("/api/status")
def get_system_status():
    alerts = get_alerts(limit=100)
    incidents = get_incidents(limit=100)
    audit = get_audit_logs(limit=100)
    
    return {
        "status": "ONLINE",
        "system_name": "AEGIS-IoT Autonomous SOC",
        "sniffer_running": sniffer.is_running,
        "simulation_running": generator.is_running,
        "total_alerts": len(alerts),
        "total_incidents": len(incidents),
        "total_audits": len(audit),
        "active_flows": len(flow_extractor.active_flows),
        "model": "XGBoost Multiclass (46 Features) + TreeSHAP",
        "dataset_compatibility": "CICIoT2023 46-Features (Frozen Production Model)"
    }

@app.get("/api/alerts")
def list_alerts(limit: int = 50):
    return get_alerts(limit=limit)

@app.get("/api/incidents")
def list_incidents(limit: int = 50):
    return get_incidents(limit=limit)

@app.post("/api/incidents/{incident_id}/decision")
def submit_decision(incident_id: str, req: DecisionRequest):
    result = enforcement.handle_analyst_decision(
        incident_id=incident_id,
        decision=req.decision,
        actions=req.actions,
        notes=req.notes or ""
    )
    return result

@app.get("/api/audit")
def list_audit_logs(limit: int = 50):
    return get_audit_logs(limit=limit)

@app.get("/api/mitre")
def list_mitre_knowledge():
    return CYBERSECURITY_KNOWLEDGE_BASE

@app.post("/api/sniffer/start")
def start_sniffer():
    sniffer.start()
    return {"message": "Live packet capture started", "running": True}

@app.post("/api/sniffer/stop")
def stop_sniffer():
    sniffer.stop()
    return {"message": "Live packet capture stopped", "running": False}

@app.post("/api/simulation/start")
def start_simulation():
    generator.start_background_simulation(interval_sec=2.5)
    return {"message": "Background IoT traffic simulation started", "running": True}

@app.post("/api/simulation/stop")
def stop_simulation():
    generator.stop_background_simulation()
    return {"message": "Simulation stopped", "running": False}

@app.post("/api/simulation/inject")
def inject_attack(req: SimulateRequest):
    att = req.attack_type.upper()
    if att == "DDOS":
        generator.inject_ddos_syn_flood()
    elif att == "DOS":
        generator.inject_dos_flood()
    elif att == "SPOOF":
        generator.inject_spoofing_attack()
    else:
        generator.inject_benign_telemetry(20)
    return {"message": f"Injected {att} packet flow successfully", "attack_type": att}

# --- WEBSOCKET LIVE STREAM ---

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        # Send initial state
        initial_payload = {
            "type": "INITIAL_STATE",
            "data": {
                "alerts": get_alerts(limit=20),
                "incidents": get_incidents(limit=10),
                "stats": {
                    "active_flows": len(flow_extractor.active_flows),
                    "sniffer_running": sniffer.is_running,
                    "simulation_running": generator.is_running
                }
            }
        }
        await websocket.send_text(json.dumps(initial_payload))
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    # Start auto background simulation on launch
    generator.start_background_simulation(interval_sec=2.5)
    uvicorn.run(app, host="127.0.0.1", port=8000)
