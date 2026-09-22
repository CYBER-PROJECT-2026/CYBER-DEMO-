'use client';
import React, { useState } from 'react';
import { Flame, Zap, ShieldAlert, Cpu, CheckCircle2, Play, Square } from 'lucide-react';

export default function AttackSimulatorBar({ onInjectAttack, simulationRunning, onToggleSimulation }) {
  const [loadingType, setLoadingType] = useState(null);

  const handleInject = async (type) => {
    setLoadingType(type);
    await onInjectAttack(type);
    setTimeout(() => setLoadingType(null), 600);
  };

  return (
    <div className="glass-panel p-4 mb-6 border-slate-800 bg-slate-900/60">
      <div className="flex flex-wrap items-center justify-between gap-4">
        
        {/* Left Label */}
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-400" />
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">IoT Traffic & Attack Injection Engine</h3>
            <p className="text-xs text-slate-400">Generate real-time live packet streams (No external dataset needed)</p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-2">
          
          <button
            onClick={() => handleInject('DDOS')}
            disabled={loadingType === 'DDOS'}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-500/15 hover:bg-red-500/25 text-red-400 border border-red-500/30 text-xs font-semibold transition-all hover:scale-105 active:scale-95 shadow-[0_0_10px_rgba(255,0,85,0.15)]"
          >
            <Flame className="w-4 h-4 text-red-400" />
            <span>Simulate DDoS SYN Flood</span>
          </button>

          <button
            onClick={() => handleInject('DOS')}
            disabled={loadingType === 'DOS'}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-orange-500/15 hover:bg-orange-500/25 text-orange-400 border border-orange-500/30 text-xs font-semibold transition-all hover:scale-105 active:scale-95 shadow-[0_0_10px_rgba(255,158,0,0.15)]"
          >
            <ShieldAlert className="w-4 h-4 text-orange-400" />
            <span>Simulate DoS HTTP Flood</span>
          </button>

          <button
            onClick={() => handleInject('SPOOF')}
            disabled={loadingType === 'SPOOF'}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-500/15 hover:bg-purple-500/25 text-purple-400 border border-purple-500/30 text-xs font-semibold transition-all hover:scale-105 active:scale-95"
          >
            <Cpu className="w-4 h-4 text-purple-400" />
            <span>Simulate ARP/IP Spoofing</span>
          </button>

          <button
            onClick={() => handleInject('BENIGN')}
            disabled={loadingType === 'BENIGN'}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 border border-emerald-500/30 text-xs font-semibold transition-all hover:scale-105 active:scale-95"
          >
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>Inject Benign MQTT Telemetry</span>
          </button>

          <div className="h-6 w-px bg-slate-700 mx-1" />

          <button
            onClick={onToggleSimulation}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
              simulationRunning 
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' 
                : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700'
            }`}
          >
            {simulationRunning ? <Square className="w-3.5 h-3.5 text-amber-400 fill-amber-400" /> : <Play className="w-3.5 h-3.5 text-emerald-400 fill-emerald-400" />}
            <span>{simulationRunning ? 'Stop Auto-Traffic' : 'Start Auto-Traffic'}</span>
          </button>

        </div>

      </div>
    </div>
  );
}
