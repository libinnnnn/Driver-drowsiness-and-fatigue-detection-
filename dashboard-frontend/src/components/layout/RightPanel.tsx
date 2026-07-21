import React, { useState } from 'react';
import { AlertEvent } from '../../hooks/useMetricsStream';
import { AlertList } from '../ui/AlertList';
import { Bell, Clock, History, Filter } from 'lucide-react';

interface RightPanelProps {
  alerts: AlertEvent[];
  predictionHistory: { timestamp: number; prediction_class: string; risk_level: string }[];
}

export const RightPanel: React.FC<RightPanelProps> = ({ alerts, predictionHistory }) => {
  const [filter, setFilter] = useState<'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM'>('ALL');

  const filteredAlerts = alerts.filter((item) => {
    if (filter === 'ALL') return true;
    return item.risk_level === filter;
  });

  return (
    <div className="flex flex-col space-y-4">
      {/* Recent Alerts Feed Panel */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Bell className="w-4 h-4 text-cyan-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
              Live Alert History Log
            </h2>
          </div>

          {/* Severity Filter Dropdown */}
          <div className="flex items-center space-x-1 text-[11px]">
            <Filter className="w-3 h-3 text-slate-500 mr-1" />
            {(['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'] as const).map((lvl) => (
              <button
                key={lvl}
                onClick={() => setFilter(lvl)}
                className={`px-2 py-0.5 rounded font-semibold transition-colors ${
                  filter === lvl
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>
        </div>

        <AlertList alerts={filteredAlerts} />
      </div>

      {/* ML Prediction Class History Timeline */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md">
        <div className="flex items-center space-x-2 mb-3">
          <History className="w-4 h-4 text-cyan-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            State Change Timeline
          </h2>
        </div>

        {predictionHistory.length === 0 ? (
          <p className="text-xs text-slate-500 text-center py-4">
            No state transitions logged yet
          </p>
        ) : (
          <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
            {predictionHistory.slice().reverse().map((item, idx) => {
              const timeStr = item.timestamp
                ? new Date(item.timestamp * 1000).toLocaleTimeString()
                : 'Just now';

              return (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-xs"
                >
                  <div className="flex items-center space-x-2">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span className="font-semibold text-slate-200">{item.prediction_class}</span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] text-slate-400 font-mono">{timeStr}</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-700 text-slate-300">
                      {item.risk_level}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
