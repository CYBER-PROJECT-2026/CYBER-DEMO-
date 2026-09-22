'use client';
import React from 'react';
import { AlertTriangle, ShieldCheck, Activity, ChevronRight, Eye } from 'lucide-react';

export default function LiveAlertFeed({ alerts, selectedAlert, onSelectAlert }) {
  return (
    <div className="glass-panel p-4 h-[440px] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Live Detection Feed</h2>
        </div>
        <span className="text-xs text-slate-400 mono">
          {alerts?.length || 0} Total Events
        </span>
      </div>

      {/* Scrolling Alerts Table */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {(!alerts || alerts.length === 0) ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs gap-2">
            <Activity className="w-8 h-8 text-slate-700 animate-pulse" />
            <p>Awaiting live network flows...</p>
          </div>
        ) : (
          alerts.map((alt) => {
            const isSelected = selectedAlert?.id === alt.id;
            const isCritical = alt.attack_type === 'DDoS' || alt.severity === 'CRITICAL';
            const isHigh = alt.attack_type === 'DoS' || alt.severity === 'HIGH';
            const isBenign = alt.attack_type === 'Benign';

            let badgeClass = 'badge-medium';
            if (isCritical) badgeClass = 'badge-critical';
            else if (isHigh) badgeClass = 'badge-high';
            else if (isBenign) badgeClass = 'badge-benign';

            return (
              <div
                key={alt.id}
                onClick={() => onSelectAlert(alt)}
                className={`p-3 rounded-lg border transition-all cursor-pointer flex items-center justify-between gap-3 ${
                  isSelected 
                    ? 'bg-cyan-950/40 border-cyan-500/50 shadow-[0_0_12px_rgba(0,240,255,0.15)]' 
                    : 'bg-slate-900/40 hover:bg-slate-900/80 border-slate-800/80'
                }`}
              >
                
                {/* Left: Icon & IP details */}
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`p-2 rounded-md ${
                    isCritical ? 'bg-red-500/20 text-red-400' : isHigh ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
                  }`}>
                    {isBenign ? <ShieldCheck className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
                  </div>

                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-xs text-slate-200">{alt.attack_type}</span>
                      <span className={`badge ${badgeClass}`}>{alt.severity || 'ALERT'}</span>
                    </div>
                    <div className="text-[11px] text-slate-400 mono truncate mt-0.5">
                      {alt.src_ip}:{alt.src_port} &rarr; {alt.dst_ip}:{alt.dst_port}
                    </div>
                  </div>
                </div>

                {/* Right: Confidence & Inspect */}
                <div className="text-right flex items-center gap-3 shrink-0">
                  <div>
                    <div className="text-xs font-bold text-white mono">
                      {(alt.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-slate-500 mono">
                      {alt.timestamp_str ? alt.timestamp_str.split(' ')[1] : 'NOW'}
                    </div>
                  </div>

                  <button className="p-1 rounded bg-slate-800 text-slate-400 hover:text-cyan-400">
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>

              </div>
            );
          })
        )}
      </div>

    </div>
  );
}
