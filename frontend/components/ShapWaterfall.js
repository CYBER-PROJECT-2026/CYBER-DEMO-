'use client';
import React from 'react';
import { BarChart3, Info, Sparkles, TrendingUp, TrendingDown } from 'lucide-react';

export default function ShapWaterfall({ alert }) {
  if (!alert) {
    return (
      <div className="glass-panel p-4 h-[440px] flex flex-col items-center justify-center text-slate-500 text-xs gap-2">
        <BarChart3 className="w-8 h-8 text-slate-700" />
        <p>Select an alert from the feed to view TreeSHAP feature attribution</p>
      </div>
    );
  }

  const shapData = alert.shap_values || {};
  const topAttributions = shapData.top_attributions || [];
  const narrative = shapData.narrative_explanation || 'Standard feature distribution.';

  return (
    <div className="glass-panel p-4 h-[440px] flex flex-col">
      
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-purple-400" />
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">TreeSHAP Explainability</h2>
        </div>
        <span className="text-xs text-purple-300 mono bg-purple-950/50 px-2 py-0.5 rounded border border-purple-500/30">
          Base Val: {shapData.base_value || 0.0}
        </span>
      </div>

      {/* Narrative Card */}
      <div className="p-3 mb-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
        <div className="flex items-center gap-1.5 font-semibold text-cyan-400 mb-1">
          <Info className="w-3.5 h-3.5" />
          <span>AI Decision Rationale:</span>
        </div>
        <p className="leading-relaxed text-slate-300 text-[11px]">{narrative}</p>
      </div>

      {/* Feature Attribution Bar Chart List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-1">
          Top Driving Features & Contribution
        </div>

        {topAttributions.map((attr, idx) => {
          const isPositive = attr.shap_value > 0;
          const absVal = Math.min(Math.abs(attr.shap_value), 10);
          const barWidth = `${Math.max((absVal / 10) * 100, 8)}%`;

          return (
            <div key={idx} className="p-2 rounded bg-slate-900/40 border border-slate-800/60 text-xs">
              <div className="flex items-center justify-between mb-1 text-[11px]">
                <span className="font-mono font-medium text-slate-200">{attr.feature}</span>
                <div className="flex items-center gap-2">
                  <span className="text-slate-400 mono">val: {attr.raw_value}</span>
                  <span className={`font-mono font-bold ${isPositive ? 'text-red-400' : 'text-emerald-400'}`}>
                    {isPositive ? `+${attr.shap_value}` : attr.shap_value}
                  </span>
                </div>
              </div>

              {/* Visual Bar */}
              <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                <div 
                  className={`h-full rounded-full transition-all ${
                    isPositive ? 'bg-gradient-to-r from-orange-500 to-red-500' : 'bg-gradient-to-r from-teal-500 to-emerald-500'
                  }`}
                  style={{ width: barWidth }}
                />
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}
