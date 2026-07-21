import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  status?: 'normal' | 'warning' | 'critical';
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  status = 'normal',
  trend = 'neutral',
  icon
}) => {
  let borderColor = 'border-slate-800 hover:border-slate-700';
  let valueColor = 'text-slate-100';
  let glowColor = 'from-slate-800/20 to-transparent';

  if (status === 'critical') {
    borderColor = 'border-red-500/40 bg-red-950/20';
    valueColor = 'text-red-400';
    glowColor = 'from-red-500/10 to-transparent';
  } else if (status === 'warning') {
    borderColor = 'border-amber-500/40 bg-amber-950/20';
    valueColor = 'text-amber-400';
    glowColor = 'from-amber-500/10 to-transparent';
  }

  return (
    <motion.div
      className={`relative overflow-hidden rounded-xl border p-3.5 backdrop-blur-md transition-all duration-300 bg-slate-900/80 ${borderColor}`}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
    >
      {/* Background Gradient Glow */}
      <div className={`absolute inset-0 bg-gradient-to-br ${glowColor} pointer-events-none opacity-50`} />

      <div className="relative z-10 flex flex-col justify-between h-full">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            {title}
          </span>
          {icon && <div className="text-cyan-400/80">{icon}</div>}
        </div>

        <div className="flex items-baseline justify-between">
          <div className="flex items-baseline space-x-1">
            <span className={`text-2xl font-bold font-['Outfit'] ${valueColor}`}>
              {value}
            </span>
            {unit && <span className="text-xs text-slate-400 font-medium">{unit}</span>}
          </div>

          {trend !== 'neutral' && (
            <div className={`flex items-center text-xs font-semibold ${trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
              {trend === 'up' ? <TrendingUp className="w-3.5 h-3.5 mr-0.5" /> : <TrendingDown className="w-3.5 h-3.5 mr-0.5" />}
            </div>
          )}
        </div>

        {subtitle && (
          <p className="text-[10px] text-slate-400 font-medium mt-1 truncate">
            {subtitle}
          </p>
        )}
      </div>
    </motion.div>
  );
};
