'use client';
import React, { useState } from 'react';
import { ShieldCheck, X, Check, Ban, Edit3, Terminal, AlertTriangle } from 'lucide-react';

export default function HitlModal({ incident, onClose, onDecisionSubmit }) {
  const [actions, setActions] = useState(incident?.recommended_actions || []);
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!incident) return null;

  const toggleAction = (idx) => {
    const updated = [...actions];
    updated[idx].selected = updated[idx].selected === false ? true : false;
    setActions(updated);
  };

  const handleDecision = async (decision) => {
    setSubmitting(true);
    const selectedActions = actions.filter(a => a.selected !== false);
    await onDecisionSubmit(incident.id, decision, selectedActions, notes);
    setSubmitting(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="glass-panel w-full max-w-2xl bg-slate-900 border-cyan-500/30 p-6 rounded-xl shadow-[0_0_50px_rgba(0,0,0,0.8)]">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-cyan-400" />
            <div>
              <h2 className="text-base font-bold text-white">Human-in-the-Loop (HITL) Enforcement Review</h2>
              <p className="text-xs text-slate-400">Incident: <span className="text-cyan-300 mono">{incident.id}</span> | Risk Score: <span className="text-red-400 mono">{incident.risk_score}/10</span></p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg bg-slate-800 text-slate-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Safety Warning */}
        <div className="p-3 mb-4 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <p>Enforcement actions will modify firewall rules and IoT VLAN routing. Review commands before signing off.</p>
        </div>

        {/* Action Selection List */}
        <div className="mb-4">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2 block">
            Select Mitigation Actions to Execute
          </label>
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {actions.map((act, idx) => {
              const isChecked = act.selected !== false;
              return (
                <div 
                  key={idx} 
                  onClick={() => toggleAction(idx)}
                  className={`p-3 rounded-lg border text-xs cursor-pointer flex items-center justify-between gap-3 transition-all ${
                    isChecked ? 'bg-cyan-950/30 border-cyan-500/40' : 'bg-slate-900/40 border-slate-800 opacity-60'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <input 
                      type="checkbox" 
                      checked={isChecked} 
                      onChange={() => {}}
                      className="rounded accent-cyan-400 w-4 h-4"
                    />
                    <div>
                      <div className="font-bold text-slate-200">{act.description}</div>
                      <div className="text-[11px] text-slate-400 mono mt-0.5">{act.command}</div>
                    </div>
                  </div>
                  <span className="badge badge-medium text-[10px]">{act.type}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Analyst Notes Input */}
        <div className="mb-5">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5 block">
            Analyst Sign-Off Notes (Audit Trail)
          </label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="e.g. Approved containment protocol after validating telemetry."
            className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Decision Action Buttons */}
        <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-800">
          <button
            onClick={() => handleDecision('REJECTED')}
            disabled={submitting}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-red-500/15 hover:bg-red-500/25 text-red-400 border border-red-500/30 text-xs font-bold transition-all"
          >
            <Ban className="w-4 h-4" />
            <span>Reject Mitigation</span>
          </button>

          <button
            onClick={() => handleDecision('APPROVED')}
            disabled={submitting}
            className="flex items-center gap-1.5 px-5 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 text-xs font-extrabold shadow-[0_0_15px_rgba(0,255,136,0.3)] transition-all hover:scale-105 active:scale-95"
          >
            <Check className="w-4 h-4" />
            <span>Approve & Enforce Actions</span>
          </button>
        </div>

      </div>
    </div>
  );
}
