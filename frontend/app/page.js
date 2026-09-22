'use client';
import React, { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import AttackSimulatorBar from '../components/AttackSimulatorBar';
import LiveAlertFeed from '../components/LiveAlertFeed';
import ShapWaterfall from '../components/ShapWaterfall';
import MultiAgentTrace from '../components/MultiAgentTrace';
import MitreMatrix from '../components/MitreMatrix';
import AuditLogViewer from '../components/AuditLogViewer';
import HitlModal from '../components/HitlModal';

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/live';

export default function SOCDashboard() {
  const [status, setStatus] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [incidents, setIncidents] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [hitlIncident, setHitlIncident] = useState(null);
  const [activeTab, setActiveTab] = useState('MONITOR'); // 'MONITOR' | 'KNOWLEDGE_AUDIT'

  const wsRef = useRef(null);

  // Fetch initial REST data
  const fetchData = async () => {
    try {
      const [resStatus, resAlerts, resIncidents, resAudit] = await Promise.all([
        fetch(`${API_BASE}/api/status`).then(r => r.json()),
        fetch(`${API_BASE}/api/alerts?limit=50`).then(r => r.json()),
        fetch(`${API_BASE}/api/incidents?limit=20`).then(r => r.json()),
        fetch(`${API_BASE}/api/audit?limit=20`).then(r => r.json()),
      ]);
      setStatus(resStatus);
      setAlerts(resAlerts);
      setIncidents(resIncidents);
      setAuditLogs(resAudit);

      if (!selectedAlert && resAlerts.length > 0) {
        setSelectedAlert(resAlerts[0]);
      }
      if (!selectedIncident && resIncidents.length > 0) {
        setSelectedIncident(resIncidents[0]);
      }
    } catch (err) {
      console.log('Error fetching REST data:', err);
    }
  };

  // Connect WebSocket for live sub-millisecond stream
  useEffect(() => {
    fetchData();

    const connectWs = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'NEW_ALERT') {
            setAlerts(prev => [msg.data, ...prev.slice(0, 49)]);
            setSelectedAlert(msg.data);
          } else if (msg.type === 'NEW_INCIDENT') {
            setIncidents(prev => [msg.data, ...prev.slice(0, 19)]);
            setSelectedIncident(msg.data);
          }
        } catch (e) {
          console.error(e);
        }
      };

      ws.onclose = () => {
        setTimeout(connectWs, 3000);
      };
    };

    connectWs();

    const interval = setInterval(fetchData, 4000);
    return () => {
      clearInterval(interval);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Handlers
  const handleToggleSniffer = async () => {
    const endpoint = status?.sniffer_running ? '/api/sniffer/stop' : '/api/sniffer/start';
    await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
    fetchData();
  };

  const handleToggleSimulation = async () => {
    const endpoint = status?.simulation_running ? '/api/simulation/stop' : '/api/simulation/start';
    await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
    fetchData();
  };

  const handleInjectAttack = async (type) => {
    await fetch(`${API_BASE}/api/simulation/inject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attack_type: type })
    });
    setTimeout(fetchData, 400);
  };

  const handleSelectAlert = (alert) => {
    setSelectedAlert(alert);
    // Find matching incident if any
    const match = incidents.find(i => i.alert_id === alert.id);
    if (match) setSelectedIncident(match);
  };

  const handleDecisionSubmit = async (incidentId, decision, actions, notes) => {
    await fetch(`${API_BASE}/api/incidents/${incidentId}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision, actions, notes })
    });
    fetchData();
  };

  return (
    <div className="min-h-screen pb-12">
      
      {/* Top Navbar */}
      <Navbar 
        status={status} 
        onToggleSniffer={handleToggleSniffer} 
        onToggleSimulation={handleToggleSimulation} 
      />

      <main className="max-w-7xl mx-auto px-4">
        
        {/* Attack Simulator Bar */}
        <AttackSimulatorBar 
          onInjectAttack={handleInjectAttack} 
          simulationRunning={status?.simulation_running}
          onToggleSimulation={handleToggleSimulation}
        />

        {/* Tab Navigation */}
        <div className="flex items-center gap-3 mb-5">
          <button
            onClick={() => setActiveTab('MONITOR')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'MONITOR'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(0,240,255,0.2)]'
                : 'bg-slate-900/60 text-slate-400 border border-slate-800 hover:text-white'
            }`}
          >
            Live SOC Monitor & Multi-Agent RAG
          </button>

          <button
            onClick={() => setActiveTab('KNOWLEDGE_AUDIT')}
            className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'KNOWLEDGE_AUDIT'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-[0_0_12px_rgba(157,78,221,0.2)]'
                : 'bg-slate-900/60 text-slate-400 border border-slate-800 hover:text-white'
            }`}
          >
            MITRE ATT&CK Matrix & Audit Logs
          </button>
        </div>

        {activeTab === 'MONITOR' ? (
          <div className="space-y-6">
            
            {/* Top Grid: Live Alert Feed & TreeSHAP Waterfall */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <LiveAlertFeed 
                alerts={alerts} 
                selectedAlert={selectedAlert} 
                onSelectAlert={handleSelectAlert} 
              />

              <ShapWaterfall 
                alert={selectedAlert} 
              />
            </div>

            {/* Bottom Panel: Multi-Agent Trace & AI Incident Dossier */}
            <MultiAgentTrace 
              incident={selectedIncident || incidents[0]} 
              onOpenHitl={(inc) => setHitlIncident(inc)}
            />

          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <MitreMatrix />
            <AuditLogViewer auditLogs={auditLogs} />
          </div>
        )}

      </main>

      {/* HITL Modal */}
      {hitlIncident && (
        <HitlModal 
          incident={hitlIncident} 
          onClose={() => setHitlIncident(null)}
          onDecisionSubmit={handleDecisionSubmit}
        />
      )}

    </div>
  );
}
