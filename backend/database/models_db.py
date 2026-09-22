import sqlite3
import json
import os
import time
from typing import Dict, List, Optional, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aegis_soc.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Alerts Table (Raw Detection Events)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        timestamp REAL,
        timestamp_str TEXT,
        src_ip TEXT,
        src_port INTEGER,
        dst_ip TEXT,
        dst_port INTEGER,
        protocol TEXT,
        attack_type TEXT,
        confidence REAL,
        raw_features TEXT,
        shap_values TEXT,
        status TEXT DEFAULT 'NEW'
    )
    """)
    
    # 2. Incidents Table (Triaged Multi-Agent Cases)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        alert_id TEXT,
        timestamp REAL,
        title TEXT,
        severity TEXT,
        risk_score REAL,
        mitre_mapping TEXT,
        cves TEXT,
        ai_explanation TEXT,
        recommended_actions TEXT,
        analyst_decision TEXT DEFAULT 'PENDING',
        analyst_notes TEXT,
        status TEXT DEFAULT 'OPEN',
        agent_trace TEXT,
        FOREIGN KEY (alert_id) REFERENCES alerts(id)
    )
    """)
    
    # 3. Audit Logs Table (Enforcement History)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        incident_id TEXT,
        timestamp REAL,
        timestamp_str TEXT,
        action_type TEXT,
        target TEXT,
        command_executed TEXT,
        status TEXT,
        executed_by TEXT DEFAULT 'HUMAN_APPROVED'
    )
    """)
    
    # 4. Network Telemetry Stats
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS network_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        pps REAL,
        fps REAL,
        byte_rate REAL,
        active_attacks INTEGER
    )
    """)
    
    conn.commit()
    conn.close()

# Database Helper Functions
def insert_alert(alert: Dict[str, Any]) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO alerts (
        id, timestamp, timestamp_str, src_ip, src_port, dst_ip, dst_port,
        protocol, attack_type, confidence, raw_features, shap_values, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert["id"],
        alert.get("timestamp", time.time()),
        alert.get("timestamp_str", time.strftime("%Y-%m-%d %H:%M:%S")),
        alert.get("src_ip", "0.0.0.0"),
        alert.get("src_port", 0),
        alert.get("dst_ip", "0.0.0.0"),
        alert.get("dst_port", 0),
        alert.get("protocol", "TCP"),
        alert["attack_type"],
        alert["confidence"],
        json.dumps(alert.get("raw_features", {})),
        json.dumps(alert.get("shap_values", {})),
        alert.get("status", "NEW")
    ))
    conn.commit()
    conn.close()
    return alert["id"]

def get_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        item = dict(row)
        item["raw_features"] = json.loads(item["raw_features"]) if item["raw_features"] else {}
        item["shap_values"] = json.loads(item["shap_values"]) if item["shap_values"] else {}
        results.append(item)
    return results

def insert_incident(incident: Dict[str, Any]) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO incidents (
        id, alert_id, timestamp, title, severity, risk_score, mitre_mapping,
        cves, ai_explanation, recommended_actions, analyst_decision, analyst_notes, status, agent_trace
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        incident["id"],
        incident.get("alert_id"),
        incident.get("timestamp", time.time()),
        incident.get("title", "Detected Threat Incident"),
        incident.get("severity", "MEDIUM"),
        incident.get("risk_score", 5.0),
        json.dumps(incident.get("mitre_mapping", [])),
        json.dumps(incident.get("cves", [])),
        incident.get("ai_explanation", ""),
        json.dumps(incident.get("recommended_actions", [])),
        incident.get("analyst_decision", "PENDING"),
        incident.get("analyst_notes", ""),
        incident.get("status", "OPEN"),
        json.dumps(incident.get("agent_trace", []))
    ))
    conn.commit()
    conn.close()
    return incident["id"]

def get_incidents(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for row in rows:
        item = dict(row)
        item["mitre_mapping"] = json.loads(item["mitre_mapping"]) if item["mitre_mapping"] else []
        item["cves"] = json.loads(item["cves"]) if item["cves"] else []
        item["recommended_actions"] = json.loads(item["recommended_actions"]) if item["recommended_actions"] else []
        item["agent_trace"] = json.loads(item["agent_trace"]) if item["agent_trace"] else []
        results.append(item)
    return results

def update_incident_decision(incident_id: str, decision: str, notes: str = ""):
    conn = get_db_connection()
    cursor = conn.cursor()
    new_status = "CLOSED" if decision in ["APPROVED", "REJECTED"] else "IN_PROGRESS"
    cursor.execute("""
    UPDATE incidents SET analyst_decision = ?, analyst_notes = ?, status = ? WHERE id = ?
    """, (decision, notes, new_status, incident_id))
    conn.commit()
    conn.close()

def insert_audit_log(audit: Dict[str, Any]):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO audit_logs (id, incident_id, timestamp, timestamp_str, action_type, target, command_executed, status, executed_by)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        audit["id"],
        audit.get("incident_id"),
        audit.get("timestamp", time.time()),
        audit.get("timestamp_str", time.strftime("%Y-%m-%d %H:%M:%S")),
        audit.get("action_type"),
        audit.get("target"),
        audit.get("command_executed"),
        audit.get("status", "SUCCESS"),
        audit.get("executed_by", "HUMAN_APPROVED")
    ))
    conn.commit()
    conn.close()

def get_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize DB on load
init_db()
