"""
OpenCV HUD Visualizer for Driver Fatigue & Drowsiness Detection System.

Renders high-contrast, professional telemetry overlays on native camera feed:
- Face mesh wireframe (soft cyan)
- Eyes & Iris separate landmark highlights (bright yellow/magenta)
- Mouth inner/outer contour
- 3D Head Pose Projection Axes (Red=Pitch, Green=Yaw, Blue=Roll)
- Instantaneous metrics HUD card (EAR, MAR, Blinks, Pose, Eye State)
- Dynamic Alert Banner at top (escalating red/yellow banner)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from detection_engine.face_mesh_detector import FaceMeshDetector


class Visualizer:
    # Colors (BGR)
    COLOR_CYAN = (255, 255, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_MAGENTA = (255, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    COLOR_DARK_BG = (15, 15, 15)

    def draw_head_pose_axes(self, img: np.ndarray, pitch: float, yaw: float, roll: float, landmarks: List[Tuple[int, int, float]], rvec: np.ndarray, tvec: np.ndarray):
        """Projects 3D Cartesian axes (X=Red, Y=Green, Z=Blue) onto the nose tip."""
        h, w = img.shape[:2]
        nose_tip = (landmarks[1][0], landmarks[1][1])

        # 3D points for axes lines (length 80mm)
        axis_length = 80.0
        axis_3d = np.array([
            [axis_length, 0.0, 0.0],   # X axis (Pitch - Red)
            [0.0, axis_length, 0.0],   # Y axis (Yaw - Green)
            [0.0, 0.0, axis_length]    # Z axis (Roll - Blue)
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

            cv2.line(img, nose_tip, p_x, (0, 0, 255), 3)  # Red - Pitch
            cv2.line(img, nose_tip, p_y, (0, 255, 0), 3)  # Green - Yaw
            cv2.line(img, nose_tip, p_z, (255, 0, 0), 3)  # Blue - Roll
            cv2.circle(img, nose_tip, 4, self.COLOR_WHITE, -1)

    def draw_hud(self, frame: np.ndarray, metrics: Dict[str, Any], prediction: Dict[str, Any], landmarks: Optional[List[Tuple[int, int, float]]]):
        """Renders complete telemetry HUD and alert overlays onto frame."""
        h, w, _ = frame.shape
        annotated = frame.copy()

        risk_level = prediction.get("risk_level", "LOW")
        pred_class = prediction.get("prediction_class", "Alert")
        fatigue_score = prediction.get("fatigue_score", 0.0)
        alert_msg = prediction.get("alert_message", "")

        # 1. Draw Facial Mesh & Highlights if landmarks present
        if landmarks:
            # Draw eye contours (yellow)
            left_eye_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.LEFT_EYE], dtype=np.int32)
            right_eye_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.RIGHT_EYE], dtype=np.int32)
            cv2.polylines(annotated, [left_eye_pts], True, self.COLOR_YELLOW, 2)
            cv2.polylines(annotated, [right_eye_pts], True, self.COLOR_YELLOW, 2)

            # Draw iris highlights (magenta)
            if len(landmarks) > 477:
                left_iris = (landmarks[468][0], landmarks[468][1])
                right_iris = (landmarks[473][0], landmarks[473][1])
                cv2.circle(annotated, left_iris, 3, self.COLOR_MAGENTA, -1)
                cv2.circle(annotated, right_iris, 3, self.COLOR_MAGENTA, -1)

            # Draw mouth inner contour (cyan / orange if yawning)
            mouth_color = self.COLOR_ORANGE if metrics.get("is_yawning") else self.COLOR_CYAN
            mouth_pts = np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.MOUTH_INNER], dtype=np.int32)
            cv2.polylines(annotated, [mouth_pts], True, mouth_color, 2)

            # Draw 3D Head Pose projection axes
            rvec = metrics.get("rvec")
            tvec = metrics.get("tvec")
            pitch = metrics.get("head_pitch", 0.0)
            yaw = metrics.get("head_yaw", 0.0)
            roll = metrics.get("head_roll", 0.0)
            self.draw_head_pose_axes(annotated, pitch, yaw, roll, landmarks, rvec, tvec)

        # 2. Draw Top Alert Banner if Risk Level is elevated
        if risk_level in ["HIGH", "CRITICAL"]:
            banner_color = self.COLOR_RED if risk_level == "CRITICAL" else self.COLOR_ORANGE
            cv2.rectangle(annotated, (0, 0), (w, 50), banner_color, -1)
            banner_text = f"WARNING: {alert_msg.upper()} [{pred_class.upper()}]"
            cv2.putText(annotated, banner_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_WHITE, 2, cv2.LINE_AA)
        elif risk_level == "MEDIUM":
            cv2.rectangle(annotated, (0, 0), (w, 40), self.COLOR_YELLOW, -1)
            banner_text = f"ATTENTION: {alert_msg.upper()}"
            cv2.putText(annotated, banner_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # 3. Draw Telemetry HUD Panel on Top-Left
        hud_w, hud_h = 320, 240
        hud_x, hud_y = 15, 60 if risk_level != "LOW" else 15

        # Semi-transparent dark overlay background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 60, 60), 1)

        # Telemetry Text Lines
        ear = metrics.get("ear", 0.0)
        mar = metrics.get("mar", 0.0)
        eye_state = metrics.get("eye_state", "OPEN")
        blinks = metrics.get("total_blinks", 0)
        yawns = metrics.get("total_yawns", 0)
        pitch = metrics.get("head_pitch", 0.0)
        yaw = metrics.get("head_yaw", 0.0)
        roll = metrics.get("head_roll", 0.0)

        # Color choices based on thresholds
        ear_color = self.COLOR_RED if ear < 0.21 else self.COLOR_GREEN
        mar_color = self.COLOR_ORANGE if mar > 0.55 else self.COLOR_GREEN
        state_color = self.COLOR_RED if eye_state == "CLOSED" else self.COLOR_GREEN

        lines = [
            ("DRIVER MONITORING HUD", self.COLOR_CYAN, 0.5, 2),
            (f"State: {pred_class} (Risk: {risk_level})", self.COLOR_WHITE, 0.5, 1),
            (f"Fatigue Score: {fatigue_score}/100", self.COLOR_YELLOW, 0.5, 1),
            (f"Eye Status: {eye_state} (EAR: {ear:.3f})", state_color, 0.5, 1),
            (f"Mouth Ratio (MAR): {mar:.3f}", mar_color, 0.5, 1),
            (f"Blinks: {blinks} | Yawns: {yawns}", self.COLOR_WHITE, 0.5, 1),
            (f"Pose: P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}", self.COLOR_WHITE, 0.45, 1),
            ("WebSockets: Emitting 200ms", self.COLOR_CYAN, 0.4, 1)
        ]

        ty = hud_y + 25
        for text, color, scale, thickness in lines:
            cv2.putText(annotated, text, (hud_x + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            ty += 26

        return annotated
