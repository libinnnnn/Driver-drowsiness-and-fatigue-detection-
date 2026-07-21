import React from 'react';
import { MetricData } from '../../hooks/useMetricsStream';
import { RiskGauge } from '../ui/RiskGauge';
import { MetricCard } from '../ui/MetricCard';
import { Eye, Smile, Activity, Compass, AlertOctagon, Timer, EyeOff } from 'lucide-react';

interface LeftPanelProps {
  metrics: MetricData | null;
}

export const LeftPanel: React.FC<LeftPanelProps> = ({ metrics }) => {
  const ear = metrics?.ear ?? 0.30;
  const mar = metrics?.mar ?? 0.20;
  const blinks = metrics?.total_blinks ?? 0;
  const blinkRate = metrics?.blink_rate ?? 0.0;
  const perclos = metrics?.perclos ?? 0.0;
  const pitch = metrics?.head_pitch ?? 0.0;
  const yaw = metrics?.head_yaw ?? 0.0;
  const roll = metrics?.head_roll ?? 0.0;
  const yawns = metrics?.total_yawns ?? 0;
  const eyeDuration = metrics?.eye_closure_duration ?? 0.0;
  const gazeTime = metrics?.gaze_fixation_time ?? 0.0;

  const score = metrics?.fatigue_score ?? 0.0;
  const riskLevel = metrics?.risk_level ?? 'LOW';

  // Status heuristics for cards
  const earStatus = ear < 0.21 ? 'critical' : ear < 0.25 ? 'warning' : 'normal';
  const marStatus = mar > 0.55 ? 'warning' : 'normal';
  const perclosStatus = perclos > 0.4 ? 'critical' : perclos > 0.2 ? 'warning' : 'normal';
  const durationStatus = eyeDuration > 1.5 ? 'critical' : eyeDuration > 0.6 ? 'warning' : 'normal';

  return (
    <div className="flex flex-col space-y-4">
      {/* Risk Gauge Panel */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-4 backdrop-blur-xl flex flex-col items-center justify-center shadow-xl">
        <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-1">
          Driver Risk Gauge
        </h2>
        <RiskGauge score={score} riskLevel={riskLevel} />
      </div>

      {/* 11 Live Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
        <MetricCard
          title="EAR Ratio"
          value={ear.toFixed(3)}
          subtitle="Eye Aspect Ratio"
          status={earStatus}
          icon={<Eye className="w-4 h-4" />}
        />
        <MetricCard
          title="MAR Ratio"
          value={mar.toFixed(3)}
          subtitle="Mouth Aspect Ratio"
          status={marStatus}
          icon={<Smile className="w-4 h-4" />}
        />
        <MetricCard
          title="PERCLOS"
          value={`${(perclos * 100).toFixed(1)}%`}
          subtitle="Eye Closure Time %"
          status={perclosStatus}
          icon={<EyeOff className="w-4 h-4" />}
        />
        <MetricCard
          title="Blink Rate"
          value={blinkRate.toFixed(1)}
          unit="/min"
          subtitle="Blinks per minute"
          icon={<Activity className="w-4 h-4" />}
        />
        <MetricCard
          title="Blinks"
          value={blinks}
          subtitle="Total Session Blinks"
          icon={<Eye className="w-4 h-4" />}
        />
        <MetricCard
          title="Yawns"
          value={yawns}
          subtitle="Total Session Yawns"
          status={yawns > 2 ? 'warning' : 'normal'}
          icon={<AlertOctagon className="w-4 h-4" />}
        />
        <MetricCard
          title="Eye Closed"
          value={`${eyeDuration.toFixed(1)}s`}
          subtitle="Continuous Closure"
          status={durationStatus}
          icon={<Timer className="w-4 h-4" />}
        />
        <MetricCard
          title="Head Pitch"
          value={`${pitch > 0 ? '+' : ''}${pitch.toFixed(1)}°`}
          subtitle="Nodding / Tilt"
          icon={<Compass className="w-4 h-4" />}
        />
        <MetricCard
          title="Head Yaw"
          value={`${yaw > 0 ? '+' : ''}${yaw.toFixed(1)}°`}
          subtitle="Left / Right Turn"
          icon={<Compass className="w-4 h-4" />}
        />
        <MetricCard
          title="Head Roll"
          value={`${roll > 0 ? '+' : ''}${roll.toFixed(1)}°`}
          subtitle="Side Tilt Angle"
          icon={<Compass className="w-4 h-4" />}
        />
        <MetricCard
          title="Gaze Fixation"
          value={`${gazeTime.toFixed(1)}s`}
          subtitle="Continuous Gaze"
          icon={<Timer className="w-4 h-4" />}
        />
      </div>
    </div>
  );
};
