import React, { useState, useEffect } from 'react';
import { PredictionBadge } from '../ui/PredictionBadge';
import { Activity, User, Clock, ShieldAlert, Cpu } from 'lucide-react';

interface HeaderProps {
  predictionClass?: string;
  riskLevel?: string;
  isConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  predictionClass = 'Alert',
  riskLevel = 'LOW',
  isConnected
}) => {
  const [driverName, setDriverName] = useState<string>('Driver #01');
  const [isEditingName, setIsEditingName] = useState<boolean>(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTimer = (totalSec: number) => {
    const hrs = Math.floor(totalSec / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);
    const secs = totalSec % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  let riskBadgeColor = 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  if (riskLevel === 'CRITICAL') {
    riskBadgeColor = 'bg-red-500/20 text-red-400 border-red-500/40 animate-pulse';
  } else if (riskLevel === 'HIGH') {
    riskBadgeColor = 'bg-orange-500/20 text-orange-400 border-orange-500/40';
  } else if (riskLevel === 'MEDIUM') {
    riskBadgeColor = 'bg-amber-500/20 text-amber-400 border-amber-500/30';
  }

  return (
    <header className="relative z-20 border-b border-slate-800 bg-[#0a0f1d]/90 backdrop-blur-xl px-6 py-3.5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Title & Brand */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-sky-400 text-slate-950 font-black shadow-glow">
            <Cpu className="w-6 h-6 text-slate-950" />
          </div>
          <div>
            <h1 className="text-lg font-bold font-['Outfit'] tracking-tight text-slate-100 flex items-center">
              Driver Fatigue & Drowsiness Detection System
            </h1>
            <p className="text-xs text-slate-400 font-medium flex items-center">
              <span className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
              Real-Time MediaPipe + Random Forest Socket IO Telemetry
            </p>
          </div>
        </div>

        {/* Center / Right Metadata & Controls */}
        <div className="flex items-center space-x-4 flex-wrap">
          {/* Driver Name Input */}
          <div className="flex items-center px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <User className="w-3.5 h-3.5 mr-2 text-slate-400" />
            {isEditingName ? (
              <input
                type="text"
                value={driverName}
                onChange={(e) => setDriverName(e.target.value)}
                onBlur={() => setIsEditingName(false)}
                onKeyDown={(e) => e.key === 'Enter' && setIsEditingName(false)}
                autoFocus
                className="bg-slate-800 text-slate-100 px-1.5 py-0.5 rounded outline-none w-28 text-xs font-semibold"
              />
            ) : (
              <span
                onClick={() => setIsEditingName(true)}
                className="cursor-pointer font-semibold text-slate-200 hover:text-cyan-400 transition-colors"
                title="Click to edit driver name"
              >
                {driverName}
              </span>
            )}
          </div>

          {/* Session Timer */}
          <div className="flex items-center px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono text-slate-300">
            <Clock className="w-3.5 h-3.5 mr-2 text-cyan-400" />
            {formatTimer(elapsedSeconds)}
          </div>

          {/* ML Prediction Pill */}
          <div className="flex items-center space-x-2">
            <PredictionBadge predictionClass={predictionClass} size="md" />

            <span className={`px-3 py-1 text-xs font-bold uppercase rounded-full border ${riskBadgeColor}`}>
              {riskLevel} RISK
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
