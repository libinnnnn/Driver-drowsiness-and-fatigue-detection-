"""
System Launcher & Process Orchestrator for Driver Fatigue & Drowsiness Detection System.

Orchestrates simultaneous execution of:
1. ML Model training check (trains models/fatigue_rf_model.pkl if missing)
2. Dashboard Backend (Flask + Flask-SocketIO on port 5000)
3. Dashboard Frontend (React + Vite on port 5173)
4. Detection Engine (OpenCV + MediaPipe + SocketIO Client)

Handles graceful termination of all child processes on exit (Ctrl+C).
"""

import sys
import os
import time
import subprocess
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_and_train_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "fatigue_rf_model.pkl")
    if not os.path.exists(model_path):
        print("[LAUNCHER] ML model file missing. Running training/train_model.py...")
        train_script = os.path.join(PROJECT_ROOT, "training", "train_model.py")
        subprocess.run([sys.executable, train_script], check=True)
    else:
        print("[LAUNCHER] Found existing trained Random Forest model at models/fatigue_rf_model.pkl")


def check_frontend_deps():
    frontend_dir = os.path.join(PROJECT_ROOT, "dashboard-frontend")
    node_modules = os.path.join(frontend_dir, "node_modules")
    if not os.path.exists(node_modules):
        print("[LAUNCHER] Frontend node_modules missing. Running 'npm install' in dashboard-frontend...")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True)


def main():
    print("==================================================================")
    print("   LAUNCHING DRIVER FATIGUE & DROWSINESS DETECTION SYSTEM        ")
    print("==================================================================")

    check_and_train_model()
    check_frontend_deps()

    processes = []

    try:
        # 1. Start Flask Backend
        print("\n[1/3] Starting Flask-SocketIO Dashboard Backend (port 5000)...")
        backend_script = os.path.join(PROJECT_ROOT, "dashboard-backend", "app.py")
        backend_proc = subprocess.Popen([sys.executable, backend_script], cwd=PROJECT_ROOT)
        processes.append(("Backend Server", backend_proc))
        time.sleep(2.5)  # Wait for backend server startup

        # 2. Start React Vite Frontend
        print("\n[2/3] Starting React Vite Dashboard Frontend (http://localhost:5173)...")
        frontend_dir = os.path.join(PROJECT_ROOT, "dashboard-frontend")
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_proc = subprocess.Popen([npm_cmd, "run", "dev"], cwd=frontend_dir)
        processes.append(("Frontend Dashboard", frontend_proc))
        time.sleep(2.0)

        # 3. Start Detection Engine
        print("\n[3/3] Starting OpenCV + MediaPipe Detection Engine...")
        engine_script = os.path.join(PROJECT_ROOT, "detection_engine", "main.py")
        engine_proc = subprocess.Popen([sys.executable, engine_script], cwd=PROJECT_ROOT)
        processes.append(("Detection Engine", engine_proc))

        print("\n==================================================================")
        print("   ALL MODULES ACTIVE!                                            ")
        print("   - Dashboard Frontend: http://localhost:5173                    ")
        print("   - Backend WebSocket:  http://localhost:5000                    ")
        print("   - Detection Window:   Native OpenCV HUD Window                 ")
        print("   Press Ctrl+C to terminate all modules cleanly.                 ")
        print("==================================================================\n")

        # Keep launcher thread alive
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"[LAUNCHER NOTICE] Process '{name}' terminated with code {proc.returncode}")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[LAUNCHER] Shutdown signal received (Ctrl+C). Terminating child processes...")
    finally:
        for name, proc in processes:
            print(f"[LAUNCHER] Terminating {name} (PID: {proc.pid})...")
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("[LAUNCHER] All processes shut down cleanly.")


if __name__ == "__main__":
    main()
