import React, { useState } from 'react';
import { MetricData } from '../../hooks/useMetricsStream';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';

interface TimeSeriesChartsProps {
  history: MetricData[];
}

export const TimeSeriesCharts: React.FC<TimeSeriesChartsProps> = ({ history }) => {
  const [activeTab, setActiveTab] = useState<'ear_mar' | 'perclos' | 'blink' | 'fatigue'>('ear_mar');

  // Format data for charts
  const chartData = history.map((item, idx) => {
    const timeStr = item.timestamp
      ? new Date(item.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : `#${idx}`;

    return {
      time: timeStr,
      ear: Number(item.ear?.toFixed(3) || 0),
      mar: Number(item.mar?.toFixed(3) || 0),
      perclos: Number(((item.perclos || 0) * 100).toFixed(1)),
      blink_rate: Number(item.blink_rate?.toFixed(1) || 0),
      fatigue_score: Number(item.fatigue_score?.toFixed(1) || 0),
      eye_duration: Number(item.eye_closure_duration?.toFixed(1) || 0),
    };
  });

  return (
    <div className="flex flex-col h-full">
      {/* Chart Tab Selectors */}
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-2 mb-3">
        <button
          onClick={() => setActiveTab('ear_mar')}
          className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
            activeTab === 'ear_mar'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          EAR & MAR Ratios
        </button>
        <button
          onClick={() => setActiveTab('perclos')}
          className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
            activeTab === 'perclos'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          PERCLOS (% Eye Closure)
        </button>
        <button
          onClick={() => setActiveTab('blink')}
          className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
            activeTab === 'blink'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Blink Rate & Duration
        </button>
        <button
          onClick={() => setActiveTab('fatigue')}
          className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
            activeTab === 'fatigue'
              ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Fatigue & Risk Score
        </button>
      </div>

      {/* Chart Canvas */}
      <div className="w-full h-64">
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full text-slate-500 text-xs font-medium">
            Waiting for live telemetry metrics update stream...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                itemStyle={{ color: '#f8fafc' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />

              {activeTab === 'ear_mar' && (
                <>
                  <Line type="monotone" dataKey="ear" name="EAR (Eye Ratio)" stroke="#38bdf8" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="mar" name="MAR (Mouth Ratio)" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </>
              )}

              {activeTab === 'perclos' && (
                <Line type="monotone" dataKey="perclos" name="PERCLOS (%)" stroke="#a855f7" strokeWidth={2.5} dot={false} />
              )}

              {activeTab === 'blink' && (
                <>
                  <Line type="monotone" dataKey="blink_rate" name="Blink Rate (blinks/min)" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="eye_duration" name="Closure Duration (s)" stroke="#ef4444" strokeWidth={2} dot={false} />
                </>
              )}

              {activeTab === 'fatigue' && (
                <Line type="monotone" dataKey="fatigue_score" name="Fatigue Score (0-100)" stroke="#f97316" strokeWidth={2.5} dot={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
