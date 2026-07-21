import { useState, useEffect, useCallback } from 'react';
import { socket } from '../lib/socket';

export interface MetricData {
  timestamp: number;
  ear: number;
  mar: number;
  eye_state: string;
  is_yawning: boolean;
  head_pitch: number;
  head_yaw: number;
  head_roll: number;
  total_blinks: number;
  total_yawns: number;
  gaze_fixation_time: number;
  perclos: number;
  blink_rate: number;
  eye_closure_duration: number;
  head_pitch_var: number;
  head_yaw_var: number;
  head_roll_var: number;
  prediction_class: string;
  probabilities: Record<string, number>;
  fatigue_score: number;
  risk_level: string;
  alert_message: string;
}

export interface AlertEvent {
  timestamp: number;
  risk_level: string;
  prediction_class: string;
  message: string;
  fatigue_score: number;
  ear: number;
  mar: number;
}

export interface SessionSummary {
  avg_ear: number;
  avg_mar: number;
  max_risk_score: number;
  total_alerts: number;
  total_blinks: number;
  total_yawns: number;
  session_duration_seconds: number;
}

export function useMetricsStream() {
  const [metrics, setMetrics] = useState<MetricData | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricData[]>([]);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [predictionHistory, setPredictionHistory] = useState<{ timestamp: number; prediction_class: string; risk_level: string }[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(socket.connected);
  const [latency, setLatency] = useState<number>(0);
  const [sessionSummary, setSessionSummary] = useState<SessionSummary | null>(null);

  // Rehydrate initial state from REST API on mount
  const fetchSessionState = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:5000/api/session-state');
      if (res.ok) {
        const data = await res.json();
        if (data.latest_metrics && Object.keys(data.latest_metrics).length > 0) {
          setMetrics(data.latest_metrics);
        }
        if (data.metrics_history) {
          setMetricsHistory(data.metrics_history.slice(-100));
        }
        if (data.alert_history) {
          setAlerts(data.alert_history);
        }
        if (data.summary) {
          setSessionSummary(data.summary);
        }
      }
    } catch (e) {
      console.warn('[METRICS STREAM] Backend initial state fetch failed:', e);
    }
  }, []);

  useEffect(() => {
    fetchSessionState();

    function onConnect() {
      setIsConnected(true);
    }

    function onDisconnect() {
      setIsConnected(false);
    }

    function onMetricsUpdate(data: MetricData) {
      const now = Date.now();
      const payloadTime = data.timestamp ? data.timestamp * 1000 : now;
      const calcLatency = Math.max(0, Math.round(now - payloadTime));
      setLatency(calcLatency);

      setMetrics(data);
      setMetricsHistory((prev) => {
        const updated = [...prev, data];
        return updated.slice(-100); // Keep last 100 points for charts
      });

      // Update prediction history timeline if prediction changed
      setPredictionHistory((prev) => {
        if (prev.length === 0 || prev[prev.length - 1].prediction_class !== data.prediction_class) {
          return [...prev.slice(-30), {
            timestamp: data.timestamp,
            prediction_class: data.prediction_class,
            risk_level: data.risk_level
          }];
        }
        return prev;
      });
    }

    function onNewAlert(data: AlertEvent) {
      setAlerts((prev) => [data, ...prev.slice(0, 100)]);
    }

    function onSessionReset() {
      setMetrics(null);
      setMetricsHistory([]);
      setAlerts([]);
      setPredictionHistory([]);
      fetchSessionState();
    }

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('metrics_update', onMetricsUpdate);
    socket.on('new_alert', onNewAlert);
    socket.on('session_reset', onSessionReset);

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('metrics_update', onMetricsUpdate);
      socket.off('new_alert', onNewAlert);
      socket.off('session_reset', onSessionReset);
    };
  }, [fetchSessionState]);

  const resetSession = async () => {
    try {
      await fetch('http://localhost:5000/api/reset', { method: 'POST' });
    } catch (e) {
      console.error('Reset session error:', e);
    }
  };

  const exportCSV = () => {
    window.open('http://localhost:5000/api/export', '_blank');
  };

  return {
    metrics,
    metricsHistory,
    alerts,
    predictionHistory,
    isConnected,
    latency,
    sessionSummary,
    resetSession,
    exportCSV,
  };
}
