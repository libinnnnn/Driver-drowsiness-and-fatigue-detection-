import React from 'react';
import { motion } from 'framer-motion';

interface RiskGaugeProps {
  score: number; // 0 to 100
  riskLevel: string;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ score = 0, riskLevel = 'LOW' }) => {
  const clampedScore = Math.min(Math.max(score, 0), 100);
  
  // Angle range for gauge: -120 deg to +120 deg (240 deg total arc)
  const angle = -120 + (clampedScore / 100) * 240;

  let scoreColor = 'text-emerald-400';
  let glowColor = 'rgba(16, 185, 129, 0.25)';
  let gradientId = 'gaugeGradientGreen';

  if (riskLevel === 'CRITICAL' || clampedScore >= 80) {
    scoreColor = 'text-red-500';
    glowColor = 'rgba(239, 68, 68, 0.4)';
    gradientId = 'gaugeGradientRed';
  } else if (riskLevel === 'HIGH' || clampedScore >= 55) {
    scoreColor = 'text-orange-400';
    glowColor = 'rgba(249, 115, 22, 0.35)';
    gradientId = 'gaugeGradientOrange';
  } else if (riskLevel === 'MEDIUM' || clampedScore >= 35) {
    scoreColor = 'text-amber-400';
    glowColor = 'rgba(245, 158, 11, 0.3)';
    gradientId = 'gaugeGradientYellow';
  }

  // Calculate arc path for SVG (radius 90, center 120, 120)
  const radius = 80;
  const strokeWidth = 14;
  const circumference = Math.PI * radius * (240 / 180);
  const strokeDashoffset = circumference - (clampedScore / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center p-4">
      {/* Background Glow */}
      <div
        className="absolute w-48 h-48 rounded-full blur-2xl transition-all duration-700 pointer-events-none"
        style={{ backgroundColor: glowColor }}
      />

      <div className="relative w-56 h-56 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 240 240">
          <defs>
            <linearGradient id="gaugeGradientGreen" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#34d399" />
            </linearGradient>
            <linearGradient id="gaugeGradientYellow" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#fbbf24" />
            </linearGradient>
            <linearGradient id="gaugeGradientOrange" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f97316" />
              <stop offset="100%" stopColor="#fb923c" />
            </linearGradient>
            <linearGradient id="gaugeGradientRed" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#f87171" />
            </linearGradient>
          </defs>

          {/* Background Arc */}
          <circle
            cx="120"
            cy="120"
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset="0"
            strokeLinecap="round"
            transform="rotate(-30 120 120)"
          />

          {/* Active Value Arc */}
          <motion.circle
            cx="120"
            cy="120"
            r={radius}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            strokeLinecap="round"
            transform="rotate(-30 120 120)"
          />
        </svg>

        {/* Needle Indicator */}
        <motion.div
          className="absolute w-1 h-24 origin-bottom bottom-1/2 left-1/2 -ml-0.5 rounded-full bg-slate-200 shadow-md"
          style={{ transformOrigin: 'bottom center' }}
          animate={{ rotate: angle }}
          transition={{ type: 'spring', stiffness: 80, damping: 15 }}
        >
          <div className="w-3 h-3 -ml-1 -mt-1 rounded-full bg-cyan-400 border-2 border-slate-900 shadow-glow" />
        </motion.div>

        {/* Center Text Hub */}
        <div className="absolute flex flex-col items-center justify-center text-center mt-6">
          <span className="text-xs uppercase tracking-widest font-semibold text-slate-400 mb-0.5">
            Fatigue Score
          </span>
          <motion.span
            className={`text-4xl font-extrabold font-['Outfit'] tracking-tight ${scoreColor}`}
            key={clampedScore}
            initial={{ scale: 0.9, opacity: 0.8 }}
            animate={{ scale: 1, opacity: 1 }}
          >
            {clampedScore.toFixed(0)}
          </motion.span>
          <span className="text-xs font-semibold px-2.5 py-0.5 mt-1 rounded-md bg-slate-800/80 border border-slate-700 text-slate-300">
            {riskLevel} RISK
          </span>
        </div>
      </div>
    </div>
  );
};
