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
  microsleep_count: number;
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
  frame_preview?: string;
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
  total_microsleep_events: number;
  session_duration_seconds: number;
  frame_count?: number;
}

function createEmptySessionSummary(): SessionSummary {
  return {
    avg_ear: 0.0,
    avg_mar: 0.0,
    max_risk_score: 0.0,
    total_alerts: 0,
    total_blinks: 0,
    total_yawns: 0,
    total_microsleep_events: 0,
    session_duration_seconds: 0,
    frame_count: 0,
  };
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
          setSessionSummary({ ...createEmptySessionSummary(), ...data.summary });
        }
      }
    } catch (e) {
      console.warn('[METRICS STREAM] Backend initial state fetch failed:', e);
    }
  }, []);

  useEffect(() => {
    fetchSessionState();

    if (socket.connected) {
      setIsConnected(true);
    }

    function onConnect() {
      setIsConnected(true);
    }

    function onDisconnect() {
      setIsConnected(false);
    }

    function onMetricsUpdate(data: MetricData) {
      setIsConnected(true);

      const now = Date.now();
      const payloadTime = data.timestamp ? data.timestamp * 1000 : now;
      const calcLatency = Math.max(0, Math.round(now - payloadTime));
      setLatency(calcLatency);

      setMetrics(data);
      setMetricsHistory((prev) => {
        const updated = [...prev, data];
        return updated.slice(-100); // Keep last 100 points for charts
      });

      setSessionSummary((prev) => {
        const previous = prev ?? createEmptySessionSummary();
        const frameCount = (previous.frame_count ?? 0) + 1;
        const avgEar = frameCount > 1
          ? ((previous.avg_ear * (frameCount - 1)) + (data.ear ?? 0)) / frameCount
          : (data.ear ?? 0);
        const avgMar = frameCount > 1
          ? ((previous.avg_mar * (frameCount - 1)) + (data.mar ?? 0)) / frameCount
          : (data.mar ?? 0);

        return {
          ...previous,
          avg_ear: avgEar,
          avg_mar: avgMar,
          max_risk_score: Math.max(previous.max_risk_score, data.fatigue_score ?? 0),
          total_blinks: data.total_blinks ?? previous.total_blinks,
          total_yawns: data.total_yawns ?? previous.total_yawns,
          total_microsleep_events: data.microsleep_count ?? previous.total_microsleep_events,
          frame_count: frameCount,
        };
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
      setIsConnected(true);
      setAlerts((prev) => [data, ...prev.slice(0, 100)]);
      setSessionSummary((prev) => {
        const previous = prev ?? createEmptySessionSummary();
        return {
          ...previous,
          total_alerts: previous.total_alerts + 1,
        };
      });
    }

    function onSessionReset() {
      setMetrics(null);
      setMetricsHistory([]);
      setAlerts([]);
      setPredictionHistory([]);
      setSessionSummary(createEmptySessionSummary());
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
