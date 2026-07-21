# Real-Time Driver Fatigue & Drowsiness Detection System

Production-quality, synchronized **Driver Fatigue & Drowsiness Detection System** in Python and React. 
The system features a OpenCV + MediaPipe face & iris tracking pipeline, a Scikit-Learn `RandomForestClassifier` ML fatigue model, a Flask-SocketIO event streaming server, and an Aceternity UI + Tailwind + Framer Motion + Recharts web dashboard.

---

## Architecture & Communication Protocol

```
Webcam / Synthetic Video Stream
  ↓
MediaPipe Face Mesh (468 landmarks + 10 Iris landmarks with refine_landmarks=True)
  ↓
Feature Extractor (EAR, MAR, Head Pose solvePnP, PERCLOS, Blink Rate, Eye Closure Duration)
  ↓
Random Forest Classifier (Predicts class: Alert, Drowsy, Fatigued, Yawning + Probabilities)
  ↓
SocketIO Client (detection_engine) → emits "metrics_update" every 200ms
  ↓
Flask-SocketIO Server (localhost:5000) — In-memory session state & REST API
  ↓
React Dashboard (Vite dev server, localhost:5173) — Aceternity UI + socket.io-client
```

---

## Key Technical Specifications

1. **Detection Engine (`detection_engine/`)**:
   - **Computer Vision:** OpenCV + MediaPipe Face Mesh (`refine_landmarks=True`).
   - **Metrics Computed:**
     - **EAR (Eye Aspect Ratio):** 6-point Euclidean distance ratio. Threshold `EAR < 0.21` signifies closed eye.
     - **MAR (Mouth Aspect Ratio):** Inner mouth height/width ratio. Threshold `MAR > 0.55` signifies mouth opening/yawn.
     - **3D Head Pose Estimation:** `cv2.solvePnP` mapping 3D generic facial keypoints to 2D image points. Extracts `pitch`, `yaw`, and `roll` angles in degrees. Projected 3D Cartesian axes (Red=Pitch, Green=Yaw, Blue=Roll) drawn on nose tip.
     - **PERCLOS:** Percentage of eye closure frames over a rolling 60-second window.
     - **Blink Counter & Rate:** Tracks sluggish/frequent blinks over rolling windows.
     - **Gaze Fixation Duration:** Tracks continuous iris gaze direction stability.
   - **Socket Emission Cadence:** Fixed **200ms** (5 Hz), throttled before emission. `new_alert` events fired instantly on risk level escalation.
   - **Synthetic Fallback Mode:** If no webcam hardware is available, the engine automatically generates a realistic animated face video stream so the entire pipeline is testable in any environment.

2. **ML Prediction Model (`predictor.py` & `training/train_model.py`)**:
   - **Classifier:** `sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10)`.
   - **Target Classes:** `Alert`, `Drowsy`, `Fatigued`, `Yawning`.
   - **Input Features (13 aggregated metrics):** `ear_current`, `ear_mean`, `ear_min`, `mar_current`, `mar_mean`, `mar_max`, `perclos`, `blink_rate`, `eye_closure_duration`, `head_pitch_var`, `head_yaw_var`, `head_roll_var`, `yawn_count`.
   - **Model File:** Serialized to `models/fatigue_rf_model.pkl` with `joblib`.
   - **Rule-Based Fallback:** If the pickle file is missing, the engine falls back to heuristic rule classification and logs a clear console warning.

3. **Dashboard Backend (`dashboard-backend/app.py`)**:
   - Flask + Flask-SocketIO server running on `http://localhost:5000`.
   - CORS enabled for React Vite origin (`http://localhost:5173`).
   - REST Endpoints:
     - `GET /api/session-state`: Session state, telemetry history, alert log, summary KPIs.
     - `GET /api/export`: CSV download of session telemetry log.
     - `POST /api/reset`: Resets session state.

4. **Dashboard Frontend (`dashboard-frontend/`)**:
   - Built with React + Vite + TypeScript + Tailwind CSS + Framer Motion.
   - **Aceternity UI Components & Styling:**
     - Dark-mode background gradients & subtle glow effects (`BentoGrid`, `GlowingCard`).
     - Animated circular `RiskGauge` (0-100 score, dynamic colors, spring needle).
     - Live 11-metric card grid with status highlighting.
     - Recharts time-series line charts with tabs (EAR/MAR, PERCLOS, Blink Rate, Fatigue Score).
     - Filterable Alert History Log & ML Prediction Class Timeline.
     - Action Controls: CSV Export button, New Session button, Reset View button.

---

## Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** & `npm`

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```bash
cd dashboard-frontend
npm install
cd ..
```

---

## Running the Complete System

Launch all modules simultaneously using `run.py`:

```bash
python run.py
```

This single command:
1. Verifies/trains the Random Forest model (`models/fatigue_rf_model.pkl`).
2. Starts the Flask-SocketIO backend on `http://localhost:5000`.
3. Starts the React Vite dashboard on `http://localhost:5173`.
4. Starts the OpenCV detection engine with the HUD window.

Open your browser to: **`http://localhost:5173`**

---

## Retraining the ML Model with Real Data

If you have a custom dataset (e.g. NTHU-DDD or YawDD CSV containing the 13 feature columns and target labels):

```bash
python training/train_model.py --dataset path/to/your_dataset.csv
```

The script fits a new `RandomForestClassifier`, prints classification accuracy & feature importances to the console, and overwrites `models/fatigue_rf_model.pkl`.

---

## Project Structure

```
driver-fatigue-system/
├── detection_engine/
│   ├── main.py                 # Entry point: camera loop, synthetic fallback & orchestration
│   ├── face_mesh_detector.py   # MediaPipe wrapper, 468 points + 10 iris points
│   ├── feature_extractor.py    # EAR, MAR, PERCLOS, solvePnP head pose, blinks, yawns
│   ├── predictor.py            # Loads RF model (or rule-based fallback), runs inference
│   ├── visualizer.py           # OpenCV HUD: mesh, iris highlights, 3D pose axes, alerts
│   └── socket_client.py        # SocketIO client emitting 200ms metrics_update & alerts
├── training/
│   ├── train_model.py          # Trains/retrains RandomForestClassifier
│   └── dataset.csv             # Bootstrapped feature dataset CSV
├── dashboard-backend/
│   └── app.py                  # Flask + SocketIO server (WebSocket/API only)
├── dashboard-frontend/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── lib/socket.ts               # socket.io-client connection
│   │   ├── hooks/useMetricsStream.ts   # WebSocket metric subscription hook
│   │   ├── components/
│   │   │   ├── layout/Header.tsx
│   │   │   ├── layout/LeftPanel.tsx
│   │   │   ├── layout/CenterPanel.tsx
│   │   │   ├── layout/RightPanel.tsx
│   │   │   ├── layout/BottomSection.tsx
│   │   │   ├── ui/RiskGauge.tsx
│   │   │   ├── ui/MetricCard.tsx
│   │   │   ├── ui/AlertList.tsx
│   │   │   ├── ui/PredictionBadge.tsx
│   │   │   └── charts/TimeSeriesCharts.tsx
├── models/
│   └── fatigue_rf_model.pkl    # Serialized RandomForest model (joblib)
├── requirements.txt
├── run.py                      # Multi-process orchestrator
└── README.md
```
