'use client';
import React from 'react';
import { Shield, Radio, Activity, Cpu, Zap, Database, Terminal } from 'lucide-react';

export default function Navbar({ status, onToggleSniffer, onToggleSimulation }) {
  return (
    <header className="glass-panel p-4 mb-6 border-b border-cyan-500/20 bg-slate-950/80 sticky top-0 z-40">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
        
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-400/40 flex items-center justify-center text-cyan-400 shadow-[0_0_15px_rgba(0,240,255,0.3)]">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-extrabold tracking-wider text-white">AEGIS-IoT</h1>
              <span className="badge badge-medium text-[10px]">SOC v2.0</span>
            </div>
            <p className="text-xs text-slate-400">Autonomous AI Cyber Defense & Multi-Agent RAG</p>
          </div>
        </div>

        {/* Live System Metrics Strip */}
        <div className="flex flex-wrap items-center gap-4 text-xs">
          
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            <span className="text-slate-400">Backend:</span>
            <span className="font-semibold text-emerald-400 mono">ONLINE (8000)</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400">Model Latency:</span>
            <span className="font-semibold text-cyan-400 mono">0.1386 ms</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <Activity className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">Active Flows:</span>
            <span className="font-semibold text-purple-300 mono">{status?.active_flows || 0}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800">
            <Radio className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Live Sniffer:</span>
            <button 
              onClick={onToggleSniffer}
              className={`font-semibold mono px-2 py-0.5 rounded text-[11px] transition-colors ${
                status?.sniffer_running 
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' 
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {status?.sniffer_running ? 'CAPTURING' : 'STANDBY'}
            </button>
          </div>

        </div>

      </div>
    </header>
  );
}
