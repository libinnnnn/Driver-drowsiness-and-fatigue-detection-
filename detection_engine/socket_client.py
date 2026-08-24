"""
SocketIO Client for Detection Engine.

Emits live metrics updates throttled to a fixed 200ms cadence to Flask-SocketIO backend.
Fires instantaneous 'new_alert' events whenever risk level escalates.
"""

import time
import socketio
from typing import Dict, Any, Optional


class DetectionSocketClient:
    EMIT_INTERVAL_SECONDS = 0.18  # Align 5 Hz emission cadence with 30 FPS video loops

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.sio = socketio.Client(reconnection=True, reconnection_delay=1)
        self.is_connected = False
        self.last_emit_time = 0.0
        self.last_risk_level = "LOW"
        self.reset_handler = None

        # Register SocketIO callbacks
        @self.sio.event
        def connect():
            self.is_connected = True
            print(f"[SOCKET CLIENT] Connected to Flask-SocketIO backend via WebSocket at {self.server_url}")

        @self.sio.event
        def disconnect():
            self.is_connected = False
            print("[SOCKET CLIENT] Disconnected from Flask-SocketIO backend")

        @self.sio.event
        def connect_error(data):
            self.is_connected = False
            # Silent fallback retry log

        @self.sio.event
        def session_reset(data):
            if self.reset_handler is not None:
                self.reset_handler()

    def set_reset_handler(self, handler):
        self.reset_handler = handler

    def connect(self):
        try:
            self.sio.connect(self.server_url, transports=['websocket'], wait=False)
        except Exception as e:
            print(f"[SOCKET CLIENT NOTICE] Flask backend not reachable at startup ({e}). Will keep retrying...")

    def emit_metrics(self, metrics: Dict[str, Any], prediction: Dict[str, Any], frame_preview: Optional[str] = None):
        """
        Emits 'metrics_update' every 200ms.
        Emits 'new_alert' immediately if risk level escalates.
        """
        now = time.time()
        risk_level = prediction.get("risk_level", "LOW")
        pred_class = prediction.get("prediction_class", "Alert")
        alert_msg = prediction.get("alert_message", "")

        # Instant emission on risk escalation or microsleep trigger
        microsleep_event = bool(prediction.get("microsleep_event", False))
        if microsleep_event:
            microsleep_payload = {
                "timestamp": now,
                "risk_level": "CRITICAL",
                "prediction_class": pred_class,
                "message": "CRITICAL — MICROSLEEP",
                "fatigue_score": prediction.get("fatigue_score", 0.0),
                "ear": metrics.get("ear", 0.0),
                "mar": metrics.get("mar", 0.0),
                "alarm_active": True,
                "microsleep_count": int(prediction.get("microsleep_count", metrics.get("microsleep_count", 0)))
            }
            if self.is_connected:
                try:
                    self.sio.emit("new_alert", microsleep_payload)
                except Exception:
                    pass

        if risk_level in ["MEDIUM", "HIGH", "CRITICAL"] and risk_level != self.last_risk_level:
            self.last_risk_level = risk_level
            alert_payload = {
                "timestamp": now,
                "risk_level": risk_level,
                "prediction_class": pred_class,
                "message": alert_msg,
                "fatigue_score": prediction.get("fatigue_score", 0.0),
                "ear": metrics.get("ear", 0.0),
                "mar": metrics.get("mar", 0.0),
                "alarm_active": bool(prediction.get("alarm_active", False)),
                "microsleep_count": int(prediction.get("microsleep_count", metrics.get("microsleep_count", 0)))
            }
            if self.is_connected:
                try:
                    self.sio.emit("new_alert", alert_payload)
                except Exception:
                    pass

        # Throttled 200ms metrics tick
        if (now - self.last_emit_time) >= self.EMIT_INTERVAL_SECONDS:
            self.last_emit_time = now

            ml_feat = metrics.get("ml_features", {})
            preview_value = frame_preview if frame_preview is not None else metrics.get("frame_preview")
            payload = {
                "timestamp": now,
                # Instantaneous readings
                "ear": metrics.get("ear", 0.0),
                "mar": metrics.get("mar", 0.0),
                "eye_state": metrics.get("eye_state", "OPEN"),
                "is_yawning": metrics.get("is_yawning", False),
                "head_pitch": metrics.get("head_pitch", 0.0),
                "head_yaw": metrics.get("head_yaw", 0.0),
                "head_roll": metrics.get("head_roll", 0.0),
                "total_blinks": metrics.get("total_blinks", 0),
                "total_yawns": metrics.get("total_yawns", 0),
                "gaze_fixation_time": metrics.get("gaze_fixation_time", 0.0),
                # Aggregate features
                "perclos": ml_feat.get("perclos", 0.0),
                "blink_rate": ml_feat.get("blink_rate", 0.0),
                "eye_closure_duration": ml_feat.get("eye_closure_duration", 0.0),
                "head_pitch_var": ml_feat.get("head_pitch_var", 0.0),
                "head_yaw_var": ml_feat.get("head_yaw_var", 0.0),
                "head_roll_var": ml_feat.get("head_roll_var", 0.0),
                # ML Predictions
                "prediction_class": pred_class,
                "probabilities": prediction.get("probabilities", {}),
                "fatigue_score": prediction.get("fatigue_score", 0.0),
                "risk_level": risk_level,
                "alert_message": alert_msg,
                "alarm_active": bool(prediction.get("alarm_active", False)),
                "microsleep_count": int(prediction.get("microsleep_count", metrics.get("microsleep_count", 0)))
            }
            if preview_value is not None:
                payload["frame_preview"] = preview_value

            if self.is_connected:
                try:
                    self.sio.emit("metrics_update", payload)
                except Exception:
                    pass

    def disconnect(self):
        if self.is_connected:
            try:
                self.sio.disconnect()
            except Exception:
                pass
