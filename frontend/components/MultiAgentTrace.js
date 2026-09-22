'use client';
import React from 'react';
import { Bot, Network, ShieldCheck, ShieldAlert, CheckCircle2, ChevronRight, Play } from 'lucide-react';

export default function MultiAgentTrace({ incident, onOpenHitl }) {
  if (!incident) {
    return (
      <div className="glass-panel p-6 min-h-[380px] flex flex-col items-center justify-center text-slate-500 text-xs gap-3">
        <Bot className="w-10 h-10 text-slate-700" />
        <p>No active escalated incident selected. Trigger an attack simulation to observe the Multi-Agent Orchestrator.</p>
      </div>
    );
  }

  const agentTrace = incident.agent_trace || [];
  const recommendedActions = incident.recommended_actions || [];
  const isPending = incident.analyst_decision === 'PENDING';

  return (
    <div className="glass-panel p-5 min-h-[380px] flex flex-col">
      
      {/* Header with Incident Info & Review Action */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Bot className="w-5 h-5 text-cyan-400" />
            <h2 className="text-base font-bold text-white tracking-wide">{incident.title}</h2>
            <span className={`badge ${incident.severity === 'CRITICAL' ? 'badge-critical' : 'badge-high'}`}>
              {incident.severity}
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span>ID: <strong className="text-slate-200 mono">{incident.id}</strong></span>
            <span>Risk Score: <strong className="text-red-400 mono">{incident.risk_score}/10</strong></span>
            <span>Decision: <strong className="text-amber-400 mono">{incident.analyst_decision}</strong></span>
          </div>
        </div>

        {/* HITL Review Button */}
        {isPending ? (
          <button
            onClick={() => onOpenHitl(incident)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-[0_0_15px_rgba(0,240,255,0.4)] transition-all hover:scale-105 active:scale-95"
          >
            <ShieldCheck className="w-4 h-4" />
            <span>HITL Review & Enforcement</span>
          </button>
        ) : (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold">
            <CheckCircle2 className="w-4 h-4" />
            <span>Status: {incident.analyst_decision}</span>
          </div>
        )}
      </div>

      {/* AI Explanation Summary */}
      <div className="p-3 mb-4 rounded-lg bg-cyan-950/20 border border-cyan-500/20 text-xs text-slate-300">
        <p className="leading-relaxed text-slate-300">{incident.ai_explanation}</p>
      </div>

      {/* Multi-Agent LangGraph Supervisor Execution Timeline */}
      <div className="mb-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
          <Network className="w-3.5 h-3.5 text-cyan-400" />
          <span>Multi-Agent Reasoning & QA Pipeline (LangGraph Supervisor)</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {agentTrace.map((step, idx) => (
            <div key={idx} className="p-3 rounded-lg bg-slate-900/60 border border-slate-800/80 text-xs flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-bold text-cyan-300">{step.agent}</span>
                  <span className="text-[10px] text-slate-500 mono bg-slate-800 px-1.5 py-0.5 rounded">
                    {step.duration_ms} ms
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-normal">{step.output_summary}</p>
              </div>
              <div className="mt-2 flex items-center gap-1 text-[10px] text-emerald-400 font-semibold">
                <CheckCircle2 className="w-3 h-3" />
                <span>Verified by Reflection Agent</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Playbook Actions */}
      <div>
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
          <span>Recommended Automated Mitigation Playbook</span>
        </h3>

        <div className="space-y-2">
          {recommendedActions.map((act, idx) => (
            <div key={idx} className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800 flex items-center justify-between gap-3 text-xs">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-slate-200">{act.action_id}</span>
                  <span className="badge badge-medium text-[10px]">{act.type}</span>
                </div>
                <p className="text-[11px] text-slate-400 mt-0.5">{act.description}</p>
              </div>
              <code className="text-[10px] text-cyan-300 bg-slate-950 px-2 py-1 rounded border border-slate-800 max-w-xs truncate mono">
                {act.command}
              </code>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
