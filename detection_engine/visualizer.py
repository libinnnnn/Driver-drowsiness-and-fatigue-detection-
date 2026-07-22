"""
OpenCV HUD Visualizer for Driver Fatigue & Drowsiness Detection System.

Renders modern, non-obstructive composite sidebar telemetry layout:
- Live camera feed in left region with subtle face mesh & 3D pose axes
- Dedicated right sidebar panel (340px fixed width, solid dark background)
- Structured vertical sections: Vitals, Activity, Pose, System Status
- Slim top alert banner over camera feed region
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from detection_engine.face_mesh_detector import FaceMeshDetector


class Visualizer:
    # Color Palette (BGR)
    COLOR_CYAN = (255, 215, 0)
    COLOR_GREEN = (120, 230, 0)
    COLOR_YELLOW = (0, 215, 255)
    COLOR_RED = (50, 50, 240)
    COLOR_ORANGE = (0, 140, 255)
    COLOR_MAGENTA = (255, 80, 255)
    COLOR_WHITE = (240, 240, 240)
    COLOR_MUTED = (165, 170, 180)
    COLOR_SIDEBAR_BG = (24, 20, 16)       # Dark BGR
    COLOR_CARD_BG = (34, 28, 22)          # Slightly lighter section card BGR
    COLOR_DIVIDER = (55, 48, 38)          # 20% opacity look

    def draw_rounded_rect(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], radius: int = 8, thickness: int = -1):
        """Draws a rectangle with smooth rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        w, h = x2 - x1, y2 - y1
        r = max(1, min(radius, w // 2, h // 2))

        if thickness < 0:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    def draw_head_pose_axes(self, img: np.ndarray, pitch: float, yaw: float, roll: float,
                            landmarks: List[Tuple[int, int, float]], rvec: np.ndarray, tvec: np.ndarray):
        """Projects 3D axes (X=Red, Y=Green, Z=Blue) onto nose tip."""
        h, w = img.shape[:2]
        nose_tip = (landmarks[1][0], landmarks[1][1])

        axis_length = 50.0
        axis_3d = np.array([
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length]
        ], dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if rvec is not None and tvec is not None and len(rvec) > 0:
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int)

            p_x = (imgpts[0][0][0], imgpts[0][0][1])
            p_y = (imgpts[1][0][0], imgpts[1][0][1])
            p_z = (imgpts[2][0][0], imgpts[2][0][1])

            cv2.line(img, nose_tip, p_x, (0, 0, 255), 2, cv2.LINE_AA)   # Red - Pitch
            cv2.line(img, nose_tip, p_y, (0, 255, 0), 2, cv2.LINE_AA)   # Green - Yaw
            cv2.line(img, nose_tip, p_z, (255, 0, 0), 2, cv2.LINE_AA)   # Blue - Roll
            cv2.circle(img, nose_tip, 3, self.COLOR_WHITE, -1)

    def draw_hud(self, frame: np.ndarray, metrics: Dict[str, Any], prediction: Dict[str, Any],
                 landmarks: Optional[List[Tuple[int, int, float]]]) -> np.ndarray:
        """
        Creates a wide composite canvas: camera feed on left, dedicated sidebar on right.
        """
        cam_h, cam_w, _ = frame.shape
        sidebar_w = 340  # Fixed sidebar width independent of camera resolution
        canvas_w = cam_w + sidebar_w
        canvas_h = cam_h

        # 1. Prepare Camera Frame (Left Region)
        camera_frame = frame.copy()

        risk_level = prediction.get("risk_level", "LOW")
        pred_class = prediction.get("prediction_class", "Alert")
        fatigue_score = prediction.get("fatigue_score", 0.0)
        alert_msg = prediction.get("alert_message", "")
        fps = metrics.get("fps", 30.0)

        # Draw Landmarks on Camera Frame
        if landmarks:
            left_eye_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.LEFT_EYE], dtype=np.int32)
            right_eye_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.RIGHT_EYE], dtype=np.int32)
            cv2.polylines(camera_frame, [left_eye_pts], True, self.COLOR_YELLOW, 1, cv2.LINE_AA)
            cv2.polylines(camera_frame, [right_eye_pts], True, self.COLOR_YELLOW, 1, cv2.LINE_AA)

            if len(landmarks) > 477:
                left_iris = (landmarks[468][0], landmarks[468][1])
                right_iris = (landmarks[473][0], landmarks[473][1])
                cv2.circle(camera_frame, left_iris, 2, self.COLOR_MAGENTA, -1)
                cv2.circle(camera_frame, right_iris, 2, self.COLOR_MAGENTA, -1)

            mouth_color = self.COLOR_ORANGE if metrics.get("is_yawning") else self.COLOR_CYAN
            mouth_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.MOUTH_INNER], dtype=np.int32)
            cv2.polylines(camera_frame, [mouth_pts], True, mouth_color, 1, cv2.LINE_AA)

            rvec, tvec = metrics.get("rvec"), metrics.get("tvec")
            pitch, yaw, roll = metrics.get("head_pitch", 0.0), metrics.get("head_yaw", 0.0), metrics.get("head_roll", 0.0)
            self.draw_head_pose_axes(camera_frame, pitch, yaw, roll, landmarks, rvec, tvec)

        # Calibration Banner — shown during the startup calibration phase
        calibrating = metrics.get("calibrating", False)
        if calibrating:
            cal_banner_h = 40
            overlay = camera_frame.copy()
            cv2.rectangle(overlay, (0, 0), (cam_w, cal_banner_h), (0, 180, 220), -1)
            cv2.addWeighted(overlay, 0.82, camera_frame, 0.18, 0, camera_frame)
            cv2.putText(camera_frame, "CALIBRATING — Look straight ahead, keep eyes open",
                        (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 2, cv2.LINE_AA)

        # Top Alert Banner — Spans ONLY camera feed width (0 to cam_w)
        elif risk_level in ["MEDIUM", "HIGH", "CRITICAL"]:
            banner_h = 36
            bg_color = self.COLOR_RED if risk_level == "CRITICAL" else (self.COLOR_ORANGE if risk_level == "HIGH" else self.COLOR_YELLOW)
            text_color = self.COLOR_WHITE if risk_level != "MEDIUM" else (15, 15, 15)

            overlay = camera_frame.copy()
            cv2.rectangle(overlay, (0, 0), (cam_w, banner_h), bg_color, -1)
            cv2.addWeighted(overlay, 0.85, camera_frame, 0.15, 0, camera_frame)

            banner_text = f"ATTENTION: {alert_msg.upper()} [{pred_class.upper()}]"
            cv2.putText(camera_frame, banner_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)

        # 2. Assemble Composite Canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :] = self.COLOR_SIDEBAR_BG

        # Paste camera feed in left region
        canvas[0:cam_h, 0:cam_w] = camera_frame

        # Vertical Divider line separating camera feed from sidebar
        cv2.line(canvas, (cam_w, 0), (cam_w, canvas_h), self.COLOR_DIVIDER, 1)

        # 3. Render Dedicated Telemetry Sidebar (Right Region)
        sx = cam_w  # Sidebar start X
        ear = metrics.get("ear", 0.0)
        mar = metrics.get("mar", 0.0)
        eye_state = metrics.get("eye_state", "OPEN")
        blinks = metrics.get("total_blinks", 0)
        yawns = metrics.get("total_yawns", 0)
        pitch = metrics.get("head_pitch", 0.0)
        yaw = metrics.get("head_yaw", 0.0)
        roll = metrics.get("head_roll", 0.0)

        state_color = self.COLOR_RED if risk_level in ["HIGH", "CRITICAL"] else (self.COLOR_ORANGE if risk_level == "MEDIUM" else self.COLOR_GREEN)

        # --- Sidebar Header ---
        cv2.circle(canvas, (sx + 22, 28), 5, self.COLOR_GREEN, -1)
        cv2.putText(canvas, "SYSTEM TELEMETRY", (sx + 36, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"5Hz WS | FPS: {fps:.1f}", (sx + 200, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        cv2.line(canvas, (sx + 15, 48), (canvas_w - 15, 48), self.COLOR_DIVIDER, 1)

        # --- SECTION 1: DRIVER STATE & FATIGUE ---
        y_sec1 = 66
        cv2.putText(canvas, "DRIVER STATE", (sx + 18, y_sec1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # State & Risk Class
        cv2.putText(canvas, f"{pred_class.upper()}", (sx + 18, y_sec1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, state_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"[{risk_level} RISK]", (sx + 190, y_sec1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)

        # Fatigue Score & Progress Bar
        cv2.putText(canvas, f"Fatigue Score: {fatigue_score:.1f} / 100", (sx + 18, y_sec1 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_WHITE, 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w, bar_h = sx + 18, y_sec1 + 60, sidebar_w - 36, 8
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 35, 28), -1)
        fill_w = int((min(100.0, max(0.0, fatigue_score)) / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_color, -1)

        cv2.line(canvas, (sx + 15, y_sec1 + 82), (canvas_w - 15, y_sec1 + 82), self.COLOR_DIVIDER, 1)

        # --- SECTION 2: VITALS (EAR & MAR) ---
        y_sec2 = y_sec1 + 100
        cv2.putText(canvas, "VITALS & ASPECT RATIOS", (sx + 18, y_sec2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        # Eye State & EAR
        ear_color = self.COLOR_RED if eye_state == "CLOSED" else self.COLOR_GREEN
        cv2.putText(canvas, f"Eyes: {eye_state}", (sx + 18, y_sec2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"EAR: {ear:.3f}", (sx + 190, y_sec2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Mouth State & MAR
        mar_color = self.COLOR_ORANGE if metrics.get("is_yawning") else self.COLOR_WHITE
        cv2.putText(canvas, f"Mouth: {'YAWNING' if metrics.get('is_yawning') else 'NORMAL'}", (sx + 18, y_sec2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mar_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"MAR: {mar:.3f}", (sx + 190, y_sec2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.line(canvas, (sx + 15, y_sec2 + 66), (canvas_w - 15, y_sec2 + 66), self.COLOR_DIVIDER, 1)

        # --- SECTION 3: ACTIVITY COUNTERS ---
        y_sec3 = y_sec2 + 84
        cv2.putText(canvas, "ACTIVITY COUNTERS", (sx + 18, y_sec3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Total Blinks: {blinks}", (sx + 18, y_sec3 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Total Yawns: {yawns}", (sx + 180, y_sec3 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_ORANGE, 2, cv2.LINE_AA)

        cv2.line(canvas, (sx + 15, y_sec3 + 42), (canvas_w - 15, y_sec3 + 42), self.COLOR_DIVIDER, 1)

        # --- SECTION 4: HEAD POSE ---
        y_sec4 = y_sec3 + 60
        cv2.putText(canvas, "HEAD POSE ANGLES", (sx + 18, y_sec4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Pitch: {pitch:+.1f}°", (sx + 18, y_sec4 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Yaw: {yaw:+.1f}°", (sx + 130, y_sec4 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Roll: {roll:+.1f}°", (sx + 230, y_sec4 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.COLOR_WHITE, 1, cv2.LINE_AA)

        cv2.line(canvas, (sx + 15, y_sec4 + 40), (canvas_w - 15, y_sec4 + 40), self.COLOR_DIVIDER, 1)

        # --- SECTION 5: ACTIVE THRESHOLDS ---
        y_sec5 = y_sec4 + 58
        cv2.putText(canvas, "ACTIVE THRESHOLDS", (sx + 18, y_sec5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        ear_thr = metrics.get("ear_threshold", 0.21)
        mar_thr = metrics.get("yawn_mar_threshold", 0.45)
        thr_color = self.COLOR_YELLOW if metrics.get("calibrating", False) else self.COLOR_CYAN
        cal_label = "(calibrating)" if metrics.get("calibrating", False) else "(calibrated)"
        cv2.putText(canvas, f"EAR<{ear_thr:.3f}  MAR>{mar_thr:.3f}  {cal_label}",
                    (sx + 18, y_sec5 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, thr_color, 1, cv2.LINE_AA)

        cv2.line(canvas, (sx + 15, y_sec5 + 36), (canvas_w - 15, y_sec5 + 36), self.COLOR_DIVIDER, 1)

        # --- SECTION 6: SYSTEM PIPELINE STATUS ---
        y_sec6 = y_sec5 + 54
        cv2.putText(canvas, "PIPELINE STATUS", (sx + 18, y_sec6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_MUTED, 1, cv2.LINE_AA)

        cv2.putText(canvas, "Flask-SocketIO Active", (sx + 18, y_sec6 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "MediaPipe + RF Eng", (sx + 180, y_sec6 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_CYAN, 1, cv2.LINE_AA)

        return canvas


