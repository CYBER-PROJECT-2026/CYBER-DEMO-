'use client';
import React, { useState, useEffect } from 'react';
import { Database, ExternalLink, ShieldCheck, Layers } from 'lucide-react';

export default function MitreMatrix({ mitreData }) {
  const [items, setItems] = useState(mitreData || []);

  useEffect(() => {
    if (!mitreData || mitreData.length === 0) {
      fetch('http://127.0.0.1:8000/api/mitre')
        .then(res => res.json())
        .then(data => setItems(data))
        .catch(() => {});
    } else {
      setItems(mitreData);
    }
  }, [mitreData]);

  return (
    <div className="glass-panel p-4 h-[380px] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Trusted Knowledge Base (MITRE & CVEs)</h2>
        </div>
        <span className="badge badge-medium text-[10px]">
          Hybrid RAG Indexed
        </span>
      </div>

      {/* Grid of Knowledge Cards */}
      <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
        {items.map((doc, idx) => (
          <div key={idx} className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span className="font-bold text-cyan-400 mono">{doc.id}</span>
                <span className="badge badge-medium text-[9px]">{doc.category}</span>
              </div>
              {doc.cvss && (
                <span className="font-mono text-red-400 font-bold text-[10px] bg-red-950/40 px-1.5 py-0.5 rounded border border-red-500/20">
                  CVSS {doc.cvss}
                </span>
              )}
            </div>

            <div className="font-semibold text-slate-200 text-[11px] mb-1">{doc.title}</div>
            <p className="text-[11px] text-slate-400 leading-normal mb-2">{doc.description}</p>

            {doc.mitigation && (
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800 text-[10px] text-slate-300">
                <strong className="text-emerald-400">Mitigation: </strong>{doc.mitigation}
              </div>
            )}
          </div>
        ))}
      </div>

    </div>
  );
}
