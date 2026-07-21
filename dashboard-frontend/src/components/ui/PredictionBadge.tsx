import React from 'react';
import { ShieldCheck, AlertTriangle, Moon, Frown } from 'lucide-react';

interface PredictionBadgeProps {
  predictionClass?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const PredictionBadge: React.FC<PredictionBadgeProps> = ({ predictionClass = 'Alert', size = 'md' }) => {
  let bgColor = 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
  let icon = <ShieldCheck className="w-4 h-4 mr-1.5" />;

  switch (predictionClass) {
    case 'Alert':
      bgColor = 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-emerald-950/20';
      icon = <ShieldCheck className="w-4 h-4 mr-1.5 text-emerald-400" />;
      break;
    case 'Yawning':
      bgColor = 'bg-amber-500/10 border-amber-500/30 text-amber-400 shadow-amber-950/20';
      icon = <Frown className="w-4 h-4 mr-1.5 text-amber-400" />;
      break;
    case 'Drowsy':
      bgColor = 'bg-orange-500/10 border-orange-500/30 text-orange-400 shadow-orange-950/20';
      icon = <AlertTriangle className="w-4 h-4 mr-1.5 text-orange-400" />;
      break;
    case 'Fatigued':
      bgColor = 'bg-red-500/15 border-red-500/40 text-red-400 animate-pulse shadow-red-950/30';
      icon = <Moon className="w-4 h-4 mr-1.5 text-red-400" />;
      break;
  }

  const sizeClasses = {
    sm: 'text-xs px-2.5 py-0.5 font-medium',
    md: 'text-sm px-3.5 py-1 font-semibold',
    lg: 'text-base px-4 py-1.5 font-bold',
  }[size];

  return (
    <span className={`inline-flex items-center rounded-full border shadow-sm backdrop-blur-md transition-all duration-300 ${bgColor} ${sizeClasses}`}>
      {icon}
      {predictionClass}
    </span>
  );
};
