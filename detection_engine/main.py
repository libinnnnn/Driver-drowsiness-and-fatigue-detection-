"""
Main Entry Point for Detection Engine.

Orchestrates live video capture, MediaPipe Face Mesh landmark extraction,
feature computation, Random Forest fatigue prediction, OpenCV HUD visualizer,
and 200ms SocketIO streaming to the Flask dashboard backend.

Supports synthetic fallback mode if no camera hardware is available.
"""

import sys
import os
import time
import math
import base64
import warnings
import logging
import cv2
import numpy as np

# Suppress noisy deprecation warnings from protobuf, absl, and mediapipe internals
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, message=".*SymbolDatabase.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*feature names.*")
logging.getLogger("absl").setLevel(logging.ERROR)
os.environ["GLOG_minloglevel"] = "2"  # Suppress TF/mediapipe STDERR C++ logs


# Ensure detection_engine directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_engine.face_mesh_detector import FaceMeshDetector
from detection_engine.feature_extractor import FeatureExtractor
from detection_engine.predictor import FatiguePredictor
from detection_engine.visualizer import Visualizer
from detection_engine.socket_client import DetectionSocketClient
from detection_engine.audio import AlarmController
from detection_engine.config import EAR_THRESHOLD


def create_synthetic_face_frame(t: float, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates a synthetic animated face image with realistic head/eyes/mouth motion
    for test environments without an active camera.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)

    center_x, center_y = width // 2, height // 2

    # Animate subtle head sway
    sway = int(20 * math.sin(t * 0.8))
    cx = center_x + sway
    cy = center_y

    # Face Oval
    cv2.ellipse(frame, (cx, cy), (110, 150), 0, 0, 360, (200, 200, 200), 2)

    # Eyes: Simulate periodic blinking every 4 seconds or continuous closing during cycle
    blink_cycle = t % 5.0
    if 4.0 <= blink_cycle <= 4.2 or 12.0 <= (t % 15.0) <= 14.5:
        # Closed eyes
        cv2.line(frame, (cx - 50, cy - 30), (cx - 20, cy - 30), (0, 255, 255), 3)
        cv2.line(frame, (cx + 20, cy - 30), (cx + 50, cy - 30), (0, 255, 255), 3)
    else:
        # Open eyes
        cv2.ellipse(frame, (cx - 35, cy - 30), (15, 10), 0, 0, 360, (0, 255, 255), 2)
        cv2.ellipse(frame, (cx + 35, cy - 30), (15, 10), 0, 0, 360, (0, 255, 255), 2)
        cv2.circle(frame, (cx - 35, cy - 30), 4, (255, 0, 255), -1)
        cv2.circle(frame, (cx + 35, cy - 30), 4, (255, 0, 255), -1)

    # Nose
    cv2.line(frame, (cx, cy - 10), (cx, cy + 20), (250, 250, 250), 2)

    # Mouth: Simulate periodic yawn every 10 seconds
    yawn_cycle = t % 12.0
    if 9.0 <= yawn_cycle <= 11.5:
        # Yawning mouth
        cv2.ellipse(frame, (cx, cy + 60), (20, 28), 0, 0, 360, (0, 165, 255), -1)
    else:
        # Normal mouth
        cv2.ellipse(frame, (cx, cy + 60), (25, 8), 0, 0, 360, (255, 255, 0), 2)

    # HUD Status notice for synthetic mode
    cv2.putText(frame, "SYNTHETIC DEMO STREAM (NO WEBCAM)", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def encode_preview_frame(frame: np.ndarray, pixel_landmarks=None, target_size=(160, 120), quality: int = 80) -> str:
    """Build a small JPEG preview with minimal eye-outline overlays for the dashboard."""
    preview = cv2.resize(frame, target_size)
    if pixel_landmarks:
        frame_h, frame_w = frame.shape[:2]
        preview_h, preview_w = preview.shape[:2]
        for eye_indices, color in ((FaceMeshDetector.LEFT_EYE, (0, 255, 255)), (FaceMeshDetector.RIGHT_EYE, (255, 0, 255))):
            eye_points = [pixel_landmarks[idx] for idx in eye_indices if idx < len(pixel_landmarks)]
            if len(eye_points) < 3:
                continue
            xs = [int(point[0] * preview_w / max(frame_w, 1)) for point in eye_points]
            ys = [int(point[1] * preview_h / max(frame_h, 1)) for point in eye_points]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)
            pad = 4
            cv2.rectangle(preview, (x_min - pad, y_min - pad), (x_max + pad, y_max + pad), color, 1)

    cv2.putText(preview, "LIVE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    _, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('ascii')


def main():
    print("==========================================================")
    print("   DRIVER FATIGUE & DROWSINESS DETECTION ENGINE STARTING  ")
    print("==========================================================")

    # Initialize modules
    detector = FaceMeshDetector()
    extractor = FeatureExtractor()
    predictor = FatiguePredictor()
    visualizer = Visualizer()
    socket_client = DetectionSocketClient(server_url="http://localhost:5000")
    alarm_controller = AlarmController()
    socket_client.connect()

    # Try opening physical camera
    cap = cv2.VideoCapture(0)
    synthetic_mode = False

    if not cap.isOpened():
        print("[CAMERA NOTICE] Physical webcam unavailable or occupied.")
        print("[CAMERA NOTICE] Activating Synthetic Face Video Mode for seamless testing.")
        synthetic_mode = True
    else:
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("[CAMERA NOTICE] Camera read test failed. Switching to Synthetic Mode.")
            synthetic_mode = True
        else:
            print("[CAMERA SUCCESS] Connected to local webcam.")

    start_time = time.time()
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 30.0
    eye_closure_duration_seconds = 0.0
    consecutive_open_frames = 0
    last_frame_time = time.time()
    microsleep_count = 0
    last_preview_emit_time = time.time()
    last_preview_frame_b64 = None
    preview_emit_interval_seconds = 0.24

    def reset_microsleep_counter():
        nonlocal microsleep_count, eye_closure_duration_seconds, consecutive_open_frames
        microsleep_count = 0
        eye_closure_duration_seconds = 0.0
        consecutive_open_frames = 0
        alarm_controller.stop()

    socket_client.set_reset_handler(reset_microsleep_counter)

    window_name = "Driver Fatigue & Drowsiness Detection System (OpenCV HUD)"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 600)
    except Exception as e:
        print(f"[DISPLAY NOTICE] Could not open GUI window ({e}). Running headless engine.")

    try:
        while True:
            loop_time = time.time() - start_time

            if synthetic_mode:
                frame = create_synthetic_face_frame(loop_time)
                time.sleep(0.033)  # ~30 FPS synthetic rate
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[CAMERA ERROR] Frame read failed. Switching to synthetic mode.")
                    synthetic_mode = True
                    continue
                frame = cv2.flip(frame, 1)  # Mirror frame for intuitive view

            frame_dt = max(time.time() - last_frame_time, 1.0 / max(current_fps, 1.0))
            last_frame_time = time.time()

            # 1. Process frame with MediaPipe Face Mesh
            pixel_landmarks, raw_landmarks = detector.process_frame(frame)

            if pixel_landmarks:
                # 2. Extract metrics and features
                metrics = extractor.process_frame(pixel_landmarks, frame.shape)
                metrics["fps"] = current_fps
                ml_features = metrics.get("ml_features", {})

                # 3. Predict Fatigue & Risk Level with Random Forest Model
                prediction = predictor.predict(ml_features)

                # 4. Eye-closure duration alarm (independent of classifier risk state)
                ear = metrics.get("ear", 0.0)
                eyes_closed = ear < EAR_THRESHOLD

                if eyes_closed:
                    consecutive_open_frames = 0
                    eye_closure_duration_seconds += frame_dt
                else:
                    consecutive_open_frames += 1
                    if consecutive_open_frames >= 2:
                        eye_closure_duration_seconds = 0.0

                alarm_active = alarm_controller.update_from_eye_state(
                    eyes_closed=eyes_closed,
                    closure_duration_seconds=eye_closure_duration_seconds,
                )
                microsleep_event = bool(alarm_controller.last_transitioned_to_active)
                if microsleep_event:
                    microsleep_count += 1
                prediction["alarm_active"] = alarm_active
                prediction["microsleep_event"] = microsleep_event
                prediction["microsleep_count"] = microsleep_count
                metrics["microsleep_count"] = microsleep_count

                now = time.time()
                preview_frame_b64 = None
                if (now - last_preview_emit_time) >= preview_emit_interval_seconds:
                    preview_frame_b64 = encode_preview_frame(frame, pixel_landmarks)
                    last_preview_frame_b64 = preview_frame_b64
                    last_preview_emit_time = now
                else:
                    preview_frame_b64 = last_preview_frame_b64

                if preview_frame_b64:
                    metrics["frame_preview"] = preview_frame_b64

                # 5. Emit socket metrics immediately (before rendering HUD)
                socket_client.emit_metrics(metrics, prediction, frame_preview=preview_frame_b64)

                # 6. Draw telemetry HUD and alerts on OpenCV frame
                output_frame = visualizer.draw_hud(frame, metrics, prediction, pixel_landmarks)
            else:
                # Emit empty/searching metrics immediately
                empty_metrics = {
                    "ear": 0.0, "mar": 0.0, "eye_state": "UNKNOWN",
                    "total_blinks": extractor.total_blinks, "total_yawns": extractor.total_yawns,
                    "fps": current_fps, "ml_features": {}
                }
                empty_prediction = {
                    "prediction_class": "Searching", "risk_level": "LOW",
                    "fatigue_score": 0.0, "alert_message": "Position face in camera view",
                    "probabilities": {}
                }
                empty_prediction["alarm_active"] = alarm_controller.is_alarm_playing
                empty_prediction["microsleep_count"] = microsleep_count
                empty_metrics["microsleep_count"] = microsleep_count

                now = time.time()
                preview_frame_b64 = None
                if (now - last_preview_emit_time) >= preview_emit_interval_seconds:
                    preview_frame_b64 = encode_preview_frame(frame, None)
                    last_preview_frame_b64 = preview_frame_b64
                    last_preview_emit_time = now
                else:
                    preview_frame_b64 = last_preview_frame_b64

                if preview_frame_b64:
                    empty_metrics["frame_preview"] = preview_frame_b64

                socket_client.emit_metrics(empty_metrics, empty_prediction, frame_preview=preview_frame_b64)

                # Render composite HUD sidebar canvas for searching state
                output_frame = visualizer.draw_hud(frame, empty_metrics, empty_prediction, None)

            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            # Display frame in OpenCV window
            try:
                cv2.imshow(window_name, output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Esc or 'q'
                    print("[DETECTION ENGINE] Quit requested by user.")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("[DETECTION ENGINE] KeyboardInterrupt received. Shutting down...")
    finally:
        if not synthetic_mode and cap:
            cap.release()
        detector.close()
        alarm_controller.close()
        socket_client.disconnect()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[DETECTION ENGINE] Engine stopped cleanly.")


if __name__ == "__main__":
    main()
