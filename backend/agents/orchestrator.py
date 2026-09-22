import time
import uuid
from typing import Dict, Any, List
from .agents import MultiAgentSystem
from backend.database.models_db import insert_incident

class MultiAgentOrchestrator:
    def __init__(self, agent_system: MultiAgentSystem = None):
        self.agent_system = agent_system or MultiAgentSystem()

    def process_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the LangGraph Supervisor Multi-Agent workflow:
        Alert -> Threat Intel -> Investigation -> Risk -> Response -> QA -> Compliance -> Report -> DB
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        start_time = time.time()
        agent_trace: List[Dict[str, Any]] = []

        # 1. Supervisor: Step 1 - Threat Intel
        t0 = time.time()
        threat_intel = self.agent_system.threat_intel_agent(alert_data)
        agent_trace.append({
            "step": 1,
            "agent": "Threat Intel Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Mapped to {', '.join(threat_intel['mitre_techniques'])} and CVEs {threat_intel['cves']}"
        })

        # 2. Supervisor: Step 2 - Investigation
        t0 = time.time()
        investigation = self.agent_system.investigation_agent(alert_data, threat_intel)
        agent_trace.append({
            "step": 2,
            "agent": "Incident Investigation Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Target profiled as {investigation['target_asset_profile']} with anomalous flow indicators."
        })

        # 3. Supervisor: Step 3 - Risk Assessment
        t0 = time.time()
        risk = self.agent_system.risk_assessment_agent(alert_data, investigation)
        agent_trace.append({
            "step": 3,
            "agent": "Risk Assessment Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Calculated Composite Risk Score: {risk['composite_risk_score']}/10 ({risk['severity']})"
        })

        # 4. Supervisor: Step 4 - Response Generation
        t0 = time.time()
        response = self.agent_system.response_generation_agent(alert_data, threat_intel, risk)
        agent_trace.append({
            "step": 4,
            "agent": "Response Generation Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Formulated {len(response['recommended_actions'])} containment actions via {response['playbook_applied']}"
        })

        # 5. Supervisor: Step 5 - Reflection & QA
        t0 = time.time()
        qa = self.agent_system.reflection_qa_agent(response, investigation)
        agent_trace.append({
            "step": 5,
            "agent": "Reflection / QA Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Safety Verification: {qa['safety_verification']}. Hallucination Check: {qa['hallucination_check']}"
        })

        # 6. Supervisor: Step 6 - Policy & Compliance
        t0 = time.time()
        compliance = self.agent_system.compliance_agent(risk, response)
        agent_trace.append({
            "step": 6,
            "agent": "Policy & Compliance Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"{compliance['nist_compliance']}. Mandatory HITL gate enforced."
        })

        # 7. Supervisor: Step 7 - Report Generation
        t0 = time.time()
        report = self.agent_system.report_generation_agent(alert_data, threat_intel, investigation, risk, response)
        agent_trace.append({
            "step": 7,
            "agent": "Report Generation Agent",
            "duration_ms": round((time.time() - t0) * 1000, 2),
            "output_summary": f"Generated final incident dossier: '{report['title']}'"
        })

        # Assemble Final Incident Record
        incident_record = {
            "id": incident_id,
            "alert_id": alert_data.get("id"),
            "timestamp": alert_data.get("timestamp", time.time()),
            "title": report["title"],
            "severity": report["severity"],
            "risk_score": risk["composite_risk_score"],
            "mitre_mapping": threat_intel["mitre_techniques"],
            "cves": threat_intel["cves"],
            "ai_explanation": report["ai_explanation"],
            "recommended_actions": response["recommended_actions"],
            "analyst_decision": "PENDING",
            "analyst_notes": "",
            "status": "OPEN",
            "agent_trace": agent_trace
        }

        # Save to SQLite Incidents Table
        insert_incident(incident_record)

        return incident_record
