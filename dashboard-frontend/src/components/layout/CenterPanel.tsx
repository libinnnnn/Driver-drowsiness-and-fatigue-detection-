import React from 'react';
import { MetricData } from '../../hooks/useMetricsStream';
import { Wifi, WifiOff, ShieldAlert, CheckCircle, AlertCircle, Info, Zap } from 'lucide-react';

interface CenterPanelProps {
  metrics: MetricData | null;
  isConnected: boolean;
  latency: number;
}

export const CenterPanel: React.FC<CenterPanelProps> = ({ metrics, isConnected, latency }) => {
  const predClass = metrics?.prediction_class ?? 'Alert';
  const riskLevel = metrics?.risk_level ?? 'LOW';
  const ear = metrics?.ear ?? 0.3;
  const mar = metrics?.mar ?? 0.2;
  const eyeDuration = metrics?.eye_closure_duration ?? 0.0;
  const alertMsg = metrics?.alert_message ?? 'System monitoring active.';

  // Generate contextual real-time recommendation
  const getRecommendation = () => {
    if (riskLevel === 'CRITICAL' || predClass === 'Fatigued' || eyeDuration >= 1.5) {
      return {
        title: 'CRITICAL ALERT — PULL OVER SAFELY',
        desc: 'Microsleeps / prolonged eye closure detected. Engage hazard lights and pull off the roadway immediately for rest.',
        color: 'bg-red-500/15 border-red-500/40 text-red-300'
      };
    }
    if (riskLevel === 'HIGH' || predClass === 'Drowsy') {
      return {
        title: 'DROWSINESS WARNING — TAKE A BREAK',
        desc: 'Substantial blink degradation and eye closure observed. Open windows for fresh air or plan a coffee stop at the next service area.',
        color: 'bg-orange-500/15 border-orange-500/40 text-orange-300'
      };
    }
    if (riskLevel === 'MEDIUM' || predClass === 'Yawning') {
      return {
        title: 'EARLY FATIGUE NOTICED — STAY VIGILANT',
        desc: 'Yawning or mouth expansion detected. Stay hydrated and ensure proper cabin temperature.',
        color: 'bg-amber-500/15 border-amber-500/40 text-amber-300'
      };
    }
    return {
      title: 'DRIVER STATE OPTIMAL',
      desc: 'All eye aspect ratios and head pose angles within baseline safe operating limits.',
      color: 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
    };
  };

  const rec = getRecommendation();

  return (
    <div className="flex flex-col space-y-4">
      {/* System Connection & Latency Banner */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-3.5 backdrop-blur-md flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${isConnected ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
            {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          </div>
          <div>
            <span className="text-xs font-bold text-slate-200">
              System Engine Pipeline
            </span>
            <p className="text-[11px] text-slate-400 font-medium">
              {isConnected ? `Connected (Latency: ${latency}ms, SocketIO 200ms Cadence)` : 'Disconnected — Waiting for OpenCV detection engine'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-semibold px-3 py-1 rounded-lg bg-slate-800 border border-slate-700 text-slate-300">
          <Zap className="w-3.5 h-3.5 text-cyan-400 mr-1" />
          5 Hz (200ms)
        </div>
      </div>

      {/* Driver State Description Card */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
            Current Driver State Summary
          </span>
          <span className="text-xs font-mono text-cyan-400 font-semibold">
            {metrics?.eye_state ?? 'OPEN'} EYES
          </span>
        </div>

        <p className="text-sm font-semibold text-slate-100">
          {alertMsg}
        </p>
        <p className="text-xs text-slate-400 mt-1">
          Eye Closure Duration: <span className="font-mono text-slate-200">{eyeDuration.toFixed(1)}s</span> | Gaze Fixation: <span className="font-mono text-slate-200">{(metrics?.gaze_fixation_time ?? 0).toFixed(1)}s</span>
        </p>
      </div>

      {/* Dynamic Safety Recommendation Banner */}
      <div className={`rounded-xl border p-4 backdrop-blur-md ${rec.color}`}>
        <div className="flex items-start space-x-3">
          <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-bold tracking-tight">
              {rec.title}
            </h3>
            <p className="text-xs mt-1 leading-relaxed opacity-90">
              {rec.desc}
            </p>
          </div>
        </div>
      </div>

      {/* Live Measurements Feed Ticker / Table */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-md">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
          Live Telemetry Readouts
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono text-slate-300">
            <thead>
              <tr className="border-b border-slate-800 text-slate-500 uppercase text-[10px] tracking-wider">
                <th className="pb-2">Metric</th>
                <th className="pb-2">Current Value</th>
                <th className="pb-2">Threshold</th>
                <th className="pb-2">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              <tr>
                <td className="py-2 text-slate-400">Eye Aspect Ratio (EAR)</td>
                <td className="py-2 font-bold text-slate-100">{ear.toFixed(3)}</td>
                <td className="py-2 text-slate-500">&gt; 0.210</td>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${ear >= 0.21 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                    {ear >= 0.21 ? 'PASS' : 'WARN'}
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-2 text-slate-400">Mouth Aspect Ratio (MAR)</td>
                <td className="py-2 font-bold text-slate-100">{mar.toFixed(3)}</td>
                <td className="py-2 text-slate-500">&lt; 0.550</td>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${mar <= 0.55 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                    {mar <= 0.55 ? 'NORMAL' : 'YAWNING'}
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-2 text-slate-400">PERCLOS (% Eye Closed)</td>
                <td className="py-2 font-bold text-slate-100">{((metrics?.perclos ?? 0) * 100).toFixed(1)}%</td>
                <td className="py-2 text-slate-500">&lt; 25.0%</td>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${(metrics?.perclos ?? 0) < 0.25 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-orange-500/20 text-orange-400'}`}>
                    {(metrics?.perclos ?? 0) < 0.25 ? 'LOW' : 'HIGH'}
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-2 text-slate-400">Head Pose Pitch</td>
                <td className="py-2 font-bold text-slate-100">{(metrics?.head_pitch ?? 0).toFixed(1)}°</td>
                <td className="py-2 text-slate-500">±15.0°</td>
                <td className="py-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-400">
                    ALIGNED
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
