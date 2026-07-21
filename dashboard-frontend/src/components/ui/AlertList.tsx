import React from 'react';
import { AlertEvent } from '../../hooks/useMetricsStream';
import { AlertTriangle, AlertCircle, Info, Bell } from 'lucide-react';

interface AlertListProps {
  alerts: AlertEvent[];
}

export const AlertList: React.FC<AlertListProps> = ({ alerts }) => {
  if (alerts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center text-slate-500">
        <Bell className="w-8 h-8 mb-2 stroke-1 opacity-40 text-slate-400" />
        <p className="text-xs font-medium">No alerts recorded in this session</p>
        <p className="text-[10px] text-slate-400 mt-0.5">Alerts trigger automatically when risk levels escalate</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
      {alerts.map((alert, idx) => {
        let badgeColor = 'bg-amber-500/10 text-amber-400 border-amber-500/30';
        let icon = <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />;

        if (alert.risk_level === 'CRITICAL') {
          badgeColor = 'bg-red-500/15 text-red-400 border-red-500/40 animate-pulse';
          icon = <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />;
        } else if (alert.risk_level === 'HIGH') {
          badgeColor = 'bg-orange-500/15 text-orange-400 border-orange-500/40';
          icon = <AlertTriangle className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />;
        }

        const dateStr = alert.timestamp ? new Date(alert.timestamp * 1000).toLocaleTimeString() : 'Just now';

        return (
          <div
            key={idx}
            className={`flex items-start p-3 rounded-lg border backdrop-blur-md transition-all ${badgeColor}`}
          >
            {icon}
            <div className="ml-2.5 flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wide">
                  {alert.risk_level} — {alert.prediction_class}
                </span>
                <span className="text-[10px] font-mono opacity-75">{dateStr}</span>
              </div>
              <p className="text-xs mt-0.5 opacity-90 leading-tight truncate">
                {alert.message}
              </p>
              <div className="flex items-center space-x-3 text-[10px] mt-1 opacity-75 font-mono">
                <span>Score: {alert.fatigue_score?.toFixed(0)}</span>
                <span>EAR: {alert.ear?.toFixed(3)}</span>
                <span>MAR: {alert.mar?.toFixed(3)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
