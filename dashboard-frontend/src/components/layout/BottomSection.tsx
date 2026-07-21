import React from 'react';
import { MetricData, SessionSummary } from '../../hooks/useMetricsStream';
import { TimeSeriesCharts } from '../charts/TimeSeriesCharts';
import { Download, RefreshCw, RotateCcw, BarChart3 } from 'lucide-react';

interface BottomSectionProps {
  metricsHistory: MetricData[];
  summary: SessionSummary | null;
  onExportCSV: () => void;
  onNewSession: () => void;
  onResetView: () => void;
}

export const BottomSection: React.FC<BottomSectionProps> = ({
  metricsHistory,
  summary,
  onExportCSV,
  onNewSession,
  onResetView
}) => {
  const avgEar = summary?.avg_ear ?? 0.0;
  const avgMar = summary?.avg_mar ?? 0.0;
  const maxRisk = summary?.max_risk_score ?? 0.0;
  const totalAlerts = summary?.total_alerts ?? 0;
  const totalBlinks = summary?.total_blinks ?? 0;
  const totalYawns = summary?.total_yawns ?? 0;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Time-Series Charts Panel (Spans 2 columns on lg) */}
      <div className="lg:col-span-2 rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md">
        <div className="flex items-center space-x-2 mb-2">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Real-Time Telemetry Time-Series Analytics
          </h2>
        </div>
        <TimeSeriesCharts history={metricsHistory} />
      </div>

      {/* Session Summary KPI Grid & Action Controls */}
      <div className="flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md space-y-4">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
            Session Summary KPIs
          </h2>

          <div className="grid grid-cols-2 gap-2.5">
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Average EAR</span>
              <p className="text-lg font-bold font-['Outfit'] text-slate-100 mt-0.5">{avgEar.toFixed(3)}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Average MAR</span>
              <p className="text-lg font-bold font-['Outfit'] text-slate-100 mt-0.5">{avgMar.toFixed(3)}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Peak Risk Score</span>
              <p className="text-lg font-bold font-['Outfit'] text-amber-400 mt-0.5">{maxRisk.toFixed(0)}/100</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Total Alerts</span>
              <p className="text-lg font-bold font-['Outfit'] text-red-400 mt-0.5">{totalAlerts}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Total Blinks</span>
              <p className="text-lg font-bold font-['Outfit'] text-emerald-400 mt-0.5">{totalBlinks}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Total Yawns</span>
              <p className="text-lg font-bold font-['Outfit'] text-amber-400 mt-0.5">{totalYawns}</p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="pt-2 border-t border-slate-800 flex flex-col space-y-2">
          <button
            onClick={onExportCSV}
            className="w-full flex items-center justify-center space-x-2 py-2 px-4 rounded-lg bg-gradient-to-r from-cyan-600 to-sky-500 hover:from-cyan-500 hover:to-sky-400 text-slate-950 font-bold text-xs shadow-md transition-all"
          >
            <Download className="w-4 h-4" />
            <span>Export Session CSV Log</span>
          </button>

          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={onNewSession}
              className="flex items-center justify-center space-x-1.5 py-1.5 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-semibold text-xs transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-cyan-400" />
              <span>New Session</span>
            </button>
            <button
              onClick={onResetView}
              className="flex items-center justify-center space-x-1.5 py-1.5 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 font-semibold text-xs transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              <span>Reset View</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
