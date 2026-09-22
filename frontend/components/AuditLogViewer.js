'use client';
import React from 'react';
import { ShieldCheck, Terminal, Clock, CheckCircle2, XCircle } from 'lucide-react';

export default function AuditLogViewer({ auditLogs }) {
  return (
    <div className="glass-panel p-4 h-[380px] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Immutable Security Audit Trail</h2>
        </div>
        <span className="text-xs text-slate-400 mono">
          {auditLogs?.length || 0} Enforcement Logs
        </span>
      </div>

      {/* Audit Logs List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {(!auditLogs || auditLogs.length === 0) ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs gap-2">
            <Terminal className="w-8 h-8 text-slate-700" />
            <p>No enforcement actions logged yet. Approve an incident to view audit records.</p>
          </div>
        ) : (
          auditLogs.map((log) => (
            <div key={log.id} className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-xs">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-200 mono">{log.id}</span>
                  <span className="badge badge-medium text-[9px]">{log.action_type}</span>
                </div>
                <div className="flex items-center gap-1 text-[10px] text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3 h-3" />
                  <span>{log.status}</span>
                </div>
              </div>

              <div className="text-[11px] text-cyan-300 mono bg-slate-950 p-1.5 rounded border border-slate-800 my-1 truncate">
                {log.command_executed}
              </div>

              <div className="flex items-center justify-between text-[10px] text-slate-500 mono pt-0.5">
                <span>Target: <strong className="text-slate-300">{log.target}</strong></span>
                <span>By: {log.executed_by}</span>
                <span>{log.timestamp_str}</span>
              </div>
            </div>
          ))
        )}
      </div>

    </div>
  );
}
