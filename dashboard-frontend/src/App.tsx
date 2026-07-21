import React from 'react';
import { useMetricsStream } from './hooks/useMetricsStream';
import { Header } from './components/layout/Header';
import { LeftPanel } from './components/layout/LeftPanel';
import { CenterPanel } from './components/layout/CenterPanel';
import { RightPanel } from './components/layout/RightPanel';
import { BottomSection } from './components/layout/BottomSection';

export const App: React.FC = () => {
  const {
    metrics,
    metricsHistory,
    alerts,
    predictionHistory,
    isConnected,
    latency,
    sessionSummary,
    resetSession,
    exportCSV,
  } = useMetricsStream();

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 flex flex-col font-['Inter',sans-serif]">
      {/* Top Navigation Header */}
      <Header
        predictionClass={metrics?.prediction_class}
        riskLevel={metrics?.risk_level}
        isConnected={isConnected}
      />

      {/* Main Grid Workspace */}
      <main className="flex-1 p-4 md:p-6 space-y-6 max-w-[1800px] w-full mx-auto">
        {/* Top 3 Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column — Risk Gauge & 11 Telemetry Cards (Spans 4 cols on lg) */}
          <div className="lg:col-span-4">
            <LeftPanel metrics={metrics} />
          </div>

          {/* Center Column — System Pipeline Status, State & Recommendations (Spans 4 cols on lg) */}
          <div className="lg:col-span-4">
            <CenterPanel
              metrics={metrics}
              isConnected={isConnected}
              latency={latency}
            />
          </div>

          {/* Right Column — Live Alert Log Feed & Timeline (Spans 4 cols on lg) */}
          <div className="lg:col-span-4">
            <RightPanel
              alerts={alerts}
              predictionHistory={predictionHistory}
            />
          </div>
        </div>

        {/* Bottom Section — Time-Series Analytics & Session Controls */}
        <BottomSection
          metricsHistory={metricsHistory}
          summary={sessionSummary}
          onExportCSV={exportCSV}
          onNewSession={resetSession}
          onResetView={() => window.location.reload()}
        />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#090d18] px-6 py-3 text-center text-xs text-slate-500 font-medium">
        Driver Fatigue & Drowsiness Detection System — OpenCV + MediaPipe + RandomForest + Flask-SocketIO + Aceternity UI Dashboard
      </footer>
    </div>
  );
};

export default App;
