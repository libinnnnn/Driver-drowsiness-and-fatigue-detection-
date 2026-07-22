"""
Flask-SocketIO Dashboard Backend Server.

Acts as pure WebSocket and REST API server (port 5000).
- Listens to WebSocket events emitted by detection_engine ('metrics_update', 'new_alert').
- Maintains in-memory session metrics history and alert log.
- Re-broadcasts live events to connected React dashboard clients.
- Serves REST APIs: /api/session-state, /api/export (CSV download), /api/reset.
"""

import time
import csv
import io
import sqlite3
from pathlib import Path
from collections import deque
from flask import Flask, jsonify, Response, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fatigue_detection_secret'
CORS(app, resources={r"/*": {"origins": "*"}})

try:
    import gevent
    async_mode = 'gevent'
except ImportError:
    async_mode = 'threading'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# In-memory session state storage
SESSION_MAX_HISTORY = 1000
session_start_time = time.time()
latest_metrics = {}
metrics_history = deque(maxlen=SESSION_MAX_HISTORY)
alert_history = deque(maxlen=200)

project_root = Path(__file__).resolve().parent.parent
DB_PATH = project_root / "outputs" / "session_metrics.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("""
CREATE TABLE IF NOT EXISTS session_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_start_time REAL NOT NULL,
    updated_at REAL NOT NULL,
    microsleep_count INTEGER NOT NULL DEFAULT 0,
    total_blinks INTEGER NOT NULL DEFAULT 0,
    total_yawns INTEGER NOT NULL DEFAULT 0,
    peak_risk_score REAL NOT NULL DEFAULT 0.0,
    total_alerts INTEGER NOT NULL DEFAULT 0
)
""")
conn.commit()
active_session_id = None


def _session_snapshot():
    last_record = metrics_history[-1] if metrics_history else {}
    risk_scores = [m.get("fatigue_score", 0.0) for m in metrics_history]
    return {
        "total_blinks": int(last_record.get("total_blinks", 0)),
        "total_yawns": int(last_record.get("total_yawns", 0)),
        "microsleep_count": int(last_record.get("microsleep_count", 0)),
        "peak_risk_score": round(float(max(risk_scores)) if risk_scores else 0.0, 1),
        "total_alerts": int(len(alert_history)),
    }


def persist_session_metrics():
    global active_session_id
    snapshot = _session_snapshot()
    now = time.time()
    if active_session_id is None:
        cursor = conn.execute(
            """
            INSERT INTO session_log (session_start_time, updated_at, microsleep_count, total_blinks, total_yawns, peak_risk_score, total_alerts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_start_time, now, snapshot["microsleep_count"], snapshot["total_blinks"], snapshot["total_yawns"], snapshot["peak_risk_score"], snapshot["total_alerts"])
        )
        active_session_id = cursor.lastrowid
    else:
        conn.execute(
            """
            UPDATE session_log
            SET updated_at=?, microsleep_count=?, total_blinks=?, total_yawns=?, peak_risk_score=?, total_alerts=?
            WHERE id=?
            """,
            (now, snapshot["microsleep_count"], snapshot["total_blinks"], snapshot["total_yawns"], snapshot["peak_risk_score"], snapshot["total_alerts"], active_session_id)
        )
    conn.commit()


def calculate_session_summary():
    """Computes aggregate summary metrics for the current session."""
    duration = time.time() - session_start_time
    if not metrics_history:
        latest_row = conn.execute(
            "SELECT microsleep_count, total_blinks, total_yawns, peak_risk_score, total_alerts FROM session_log WHERE id=?",
            (active_session_id,)
        ).fetchone() if active_session_id is not None else None
        if latest_row:
            return {
                "avg_ear": 0.0,
                "avg_mar": 0.0,
                "max_risk_score": float(latest_row[3]),
                "total_alerts": int(latest_row[4]),
                "total_blinks": int(latest_row[1]),
                "total_yawns": int(latest_row[2]),
                "total_microsleep_events": int(latest_row[0]),
                "session_duration_seconds": round(duration, 1)
            }
        return {
            "avg_ear": 0.0,
            "avg_mar": 0.0,
            "max_risk_score": 0.0,
            "total_alerts": len(alert_history),
            "total_blinks": 0,
            "total_yawns": 0,
            "total_microsleep_events": 0,
            "session_duration_seconds": round(duration, 1)
        }

    ears = [m.get("ear", 0.0) for m in metrics_history]
    mars = [m.get("mar", 0.0) for m in metrics_history]
    fatigue_scores = [m.get("fatigue_score", 0.0) for m in metrics_history]

    last_record = metrics_history[-1]
    latest_row = conn.execute(
        "SELECT microsleep_count, total_blinks, total_yawns, peak_risk_score, total_alerts FROM session_log WHERE id=?",
        (active_session_id,)
    ).fetchone() if active_session_id is not None else None

    return {
        "avg_ear": round(float(sum(ears) / len(ears)), 4),
        "avg_mar": round(float(sum(mars) / len(mars)), 4),
        "max_risk_score": round(float(max(fatigue_scores)), 1) if fatigue_scores else 0.0,
        "total_alerts": len(alert_history),
        "total_blinks": int(last_record.get("total_blinks", 0)) if latest_row is None else int(latest_row[1]),
        "total_yawns": int(last_record.get("total_yawns", 0)) if latest_row is None else int(latest_row[2]),
        "total_microsleep_events": int(last_record.get("microsleep_count", 0)) if latest_row is None else int(latest_row[0]),
        "session_duration_seconds": round(duration, 1)
    }


# --- SOCKET.IO EVENT HANDLERS ---

@socketio.on('connect')
def handle_connect():
    print(f"[BACKEND SOCKET] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[BACKEND SOCKET] Client disconnected: {request.sid}")

@socketio.on('metrics_update')
def handle_metrics_update(data):
    """Receives metrics tick from detection engine (every 200ms) and relays to dashboard."""
    global latest_metrics
    global latest_metrics
    latest_metrics = data
    metrics_history.append(data)
    persist_session_metrics()
    # Re-broadcast to all connected web browser clients
    socketio.emit('metrics_update', data, skip_sid=request.sid)

@socketio.on('new_alert')
def handle_new_alert(data):
    """Receives immediate alert trigger and relays to dashboard."""
    alert_history.appendleft(data)  # Most recent first
    persist_session_metrics()
    socketio.emit('new_alert', data, skip_sid=request.sid)


# --- REST API ENDPOINTS ---

@app.route('/api/session-state', methods=['GET'])
def get_session_state():
    """Returns current state, metric history for charts, alert log, and summary KPIs."""
    return jsonify({
        "status": "active",
        "session_start_time": session_start_time,
        "session_duration_seconds": round(time.time() - session_start_time, 1),
        "latest_metrics": latest_metrics,
        "metrics_history": list(metrics_history),
        "alert_history": list(alert_history),
        "summary": calculate_session_summary()
    })

@app.route('/api/export', methods=['GET'])
def export_csv():
    """Exports session metrics history as a downloadable CSV file."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Timestamp", "Time_Offset_Sec", "Prediction_Class", "Risk_Level", "Fatigue_Score",
        "EAR", "MAR", "Eye_State", "Is_Yawning", "PERCLOS", "Blink_Rate",
        "Eye_Closure_Duration", "Head_Pitch", "Head_Yaw", "Head_Roll",
        "Total_Blinks", "Total_Yawns", "Microsleep_Count", "Gaze_Fixation_Time"
    ])

    for entry in metrics_history:
        ts = entry.get("timestamp", 0)
        offset = round(ts - session_start_time, 2)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
            offset,
            entry.get("prediction_class", ""),
            entry.get("risk_level", ""),
            entry.get("fatigue_score", 0.0),
            entry.get("ear", 0.0),
            entry.get("mar", 0.0),
            entry.get("eye_state", ""),
            entry.get("is_yawning", False),
            entry.get("perclos", 0.0),
            entry.get("blink_rate", 0.0),
            entry.get("eye_closure_duration", 0.0),
            entry.get("head_pitch", 0.0),
            entry.get("head_yaw", 0.0),
            entry.get("head_roll", 0.0),
            entry.get("total_blinks", 0),
            entry.get("total_yawns", 0),
            entry.get("microsleep_count", 0),
            entry.get("gaze_fixation_time", 0.0)
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=driver_fatigue_session_{int(time.time())}.csv"}
    )

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Resets session metrics and alert history."""
    global session_start_time, latest_metrics, active_session_id
    session_start_time = time.time()
    latest_metrics = {}
    metrics_history.clear()
    alert_history.clear()
    active_session_id = None
    persist_session_metrics()
    socketio.emit('session_reset', {"timestamp": session_start_time})
    return jsonify({"status": "success", "message": "Session reset successfully."})


if __name__ == '__main__':
    print("==========================================================")
    print("   FLASK-SOCKETIO DASHBOARD BACKEND LISTENING ON PORT 5000")
    print("==========================================================")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
