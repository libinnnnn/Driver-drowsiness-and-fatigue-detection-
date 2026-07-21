"""
Feature Extractor for Driver Fatigue & Drowsiness Detection System.

Calculates:
- Eye Aspect Ratio (EAR) & Eye Closure Durations
- Mouth Aspect Ratio (MAR) & Yawn Detection
- Head Pose Estimation via cv2.solvePnP (Pitch, Yaw, Roll)
- PERCLOS (Percentage of Eye Closure) over rolling window
- Blink Rates & Counts
- Gaze Fixation Time
- Rolling Window aggregations for Random Forest ML input
"""

import time
import math
import numpy as np
import cv2
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from detection_engine.face_mesh_detector import FaceMeshDetector


def euclidean_dist(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


class FeatureExtractor:
    # Threshold Constants
    EAR_THRESHOLD = 0.21         # EAR below this is considered eye closed
    MAR_THRESHOLD = 0.55         # MAR above this is considered open mouth / yawn
    BLINK_MIN_FRAMES = 2         # Min frames for a valid blink
    YAWN_MIN_FRAMES = 12         # Min frames for a valid yawn (~0.4s at 30fps)
    ROLLING_WINDOW_SECONDS = 60  # Time window for PERCLOS & Blink Rate

    # Generic 3D facial model coordinates (in mm) for PnP Head Pose Estimation
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -330.0, -65.0),     # Chin (152)
        (-225.0, 170.0, -135.0),  # Left eye corner (33)
        (225.0, 170.0, -135.0),   # Right eye corner (263)
        (-150.0, -150.0, -125.0), # Left mouth corner (61)
        (150.0, -150.0, -125.0)   # Right mouth corner (291)
    ], dtype=np.float64)

    def __init__(self):
        # State tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.yawn_counter = 0
        self.total_yawns = 0

        self.eye_closed_start_time: Optional[float] = None
        self.current_eye_closure_duration: float = 0.0

        # Rolling buffers (timestamps & values)
        self.eye_state_history = deque()  # (timestamp, is_closed: bool)
        self.blink_timestamps = deque()    # timestamps of completed blinks
        self.feature_history = deque(maxlen=60)  # rolling history of frame features for aggregation

        # Gaze tracking state
        self.last_gaze_direction = "center"
        self.gaze_fixation_start_time = time.time()
        self.gaze_fixation_time = 0.0

    def compute_ear(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[float, float, float]:
        """
        Computes Left EAR, Right EAR, and Average EAR.
        EAR formula: (||P2-P6|| + ||P3-P5||) / (2.0 * ||P1-P4||)
        """
        def single_eye_ear(indices: List[int]) -> float:
            p1 = (landmarks[indices[0]][0], landmarks[indices[0]][1]) # outer corner
            p2 = (landmarks[indices[1]][0], landmarks[indices[1]][1]) # top-left
            p3 = (landmarks[indices[2]][0], landmarks[indices[2]][1]) # top-right
            p4 = (landmarks[indices[3]][0], landmarks[indices[3]][1]) # inner corner
            p5 = (landmarks[indices[4]][0], landmarks[indices[4]][1]) # bottom-right
            p6 = (landmarks[indices[5]][0], landmarks[indices[5]][1]) # bottom-left

            v_dist1 = euclidean_dist(p2, p6)
            v_dist2 = euclidean_dist(p3, p5)
            h_dist = euclidean_dist(p1, p4)

            if h_dist == 0:
                return 0.0
            return (v_dist1 + v_dist2) / (2.0 * h_dist)

        left_ear = single_eye_ear(FaceMeshDetector.LEFT_EYE)
        right_ear = single_eye_ear(FaceMeshDetector.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        return left_ear, right_ear, avg_ear

    def compute_mar(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Computes Mouth Aspect Ratio (MAR).
        MAR = vertical inner mouth height / horizontal mouth width
        """
        top = (landmarks[13][0], landmarks[13][1])
        bottom = (landmarks[14][0], landmarks[14][1])
        left = (landmarks[78][0], landmarks[78][1])
        right = (landmarks[308][0], landmarks[308][1])

        v_dist = euclidean_dist(top, bottom)
        h_dist = euclidean_dist(left, right)

        if h_dist == 0:
            return 0.0
        return v_dist / h_dist

    def estimate_head_pose(self, landmarks: List[Tuple[int, int, float]], frame_shape: Tuple[int, int]) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Estimates Head Pose (Pitch, Yaw, Roll) in degrees via cv2.solvePnP.
        """
        h, w = frame_shape[:2]
        
        # 2D image points matching POSE_LANDMARKS [1, 152, 33, 263, 61, 291]
        image_points_2d = np.array([
            (landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.POSE_LANDMARKS
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64) # Assume zero lens distortion

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, np.zeros((3, 1)), np.zeros((3, 1))

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Extract Euler angles (Pitch, Yaw, Roll)
        # Using decomposition of rotation matrix
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        roll_deg = math.degrees(roll)

        return pitch_deg, yaw_deg, roll_deg, rvec, tvec

    def estimate_gaze_fixation(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """Estimates continuous gaze fixation duration (in seconds)."""
        now = time.time()
        # Compute iris center relative to eye bounding box
        left_iris_center = landmarks[468] if len(landmarks) > 468 else landmarks[33]
        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]

        eye_width = euclidean_dist((left_eye_outer[0], left_eye_outer[1]), (left_eye_inner[0], left_eye_inner[1]))
        if eye_width > 0:
            rel_pos = (left_iris_center[0] - left_eye_outer[0]) / eye_width
            if rel_pos < 0.35:
                direction = "left"
            elif rel_pos > 0.65:
                direction = "right"
            else:
                direction = "center"
        else:
            direction = "center"

        if direction == self.last_gaze_direction:
            self.gaze_fixation_time = now - self.gaze_fixation_start_time
        else:
            self.last_gaze_direction = direction
            self.gaze_fixation_start_time = now
            self.gaze_fixation_time = 0.0

        return self.gaze_fixation_time

    def process_frame(self, landmarks: List[Tuple[int, int, float]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Main extraction function per frame.
        Computes instantaneous and aggregate metrics.
        Returns dictionary containing all raw and aggregated features.
        """
        now = time.time()

        # 1. Compute EAR and MAR
        _, _, ear = self.compute_ear(landmarks)
        mar = self.compute_mar(landmarks)

        # 2. Compute Head Pose
        pitch, yaw, roll, rvec, tvec = self.estimate_head_pose(landmarks, frame_shape)

        # 3. Eye Closure & Blink Logic
        is_eye_closed = ear < self.EAR_THRESHOLD

        if is_eye_closed:
            self.blink_counter += 1
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
            self.current_eye_closure_duration = round(now - self.eye_closed_start_time, 2)
        else:
            if self.blink_counter >= self.BLINK_MIN_FRAMES:
                self.total_blinks += 1
                self.blink_timestamps.append(now)
            self.blink_counter = 0
            self.eye_closed_start_time = None
            self.current_eye_closure_duration = 0.0

        # 4. Yawn Logic
        is_yawning = mar > self.MAR_THRESHOLD
        if is_yawning:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.YAWN_MIN_FRAMES:
                self.total_yawns += 1
            self.yawn_counter = 0

        # 5. Maintain rolling buffers for PERCLOS & Blink Rate
        self.eye_state_history.append((now, is_eye_closed))
        while self.eye_state_history and (now - self.eye_state_history[0][0]) > self.ROLLING_WINDOW_SECONDS:
            self.eye_state_history.popleft()

        while self.blink_timestamps and (now - self.blink_timestamps[0]) > self.ROLLING_WINDOW_SECONDS:
            self.blink_timestamps.popleft()

        # Calculate PERCLOS (% of closed eye frames in window)
        if self.eye_state_history:
            closed_frames = sum(1 for _, closed in self.eye_state_history if closed)
            perclos = closed_frames / len(self.eye_state_history)
        else:
            perclos = 0.0

        # Calculate Blink Rate (blinks/minute)
        window_duration_min = min((now - self.eye_state_history[0][0]) / 60.0, 1.0) if self.eye_state_history else 1.0
        blink_rate = len(self.blink_timestamps) / max(window_duration_min, 0.1)

        # 6. Gaze Fixation
        gaze_time = self.estimate_gaze_fixation(landmarks)

        # 7. Store frame metrics into rolling feature buffer for aggregate calculations
        frame_dict = {
            "ear": ear,
            "mar": mar,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "timestamp": now
        }
        self.feature_history.append(frame_dict)

        # Compute rolling window feature aggregations (last 30 frames ~ 1 sec)
        ears = [f["ear"] for f in self.feature_history]
        mars = [f["mar"] for f in self.feature_history]
        pitches = [f["pitch"] for f in self.feature_history]
        yaws = [f["yaw"] for f in self.feature_history]
        rolls = [f["roll"] for f in self.feature_history]

        ear_current = float(ear)
        ear_mean = float(np.mean(ears))
        ear_min = float(np.min(ears))
        mar_current = float(mar)
        mar_mean = float(np.mean(mars))
        mar_max = float(np.max(mars))
        head_pitch_var = float(np.var(pitches)) if len(pitches) > 1 else 0.0
        head_yaw_var = float(np.var(yaws)) if len(yaws) > 1 else 0.0
        head_roll_var = float(np.var(rolls)) if len(rolls) > 1 else 0.0

        # Package feature vector formatted for RandomForest input schema
        ml_features = {
            "ear_current": round(ear_current, 4),
            "ear_mean": round(ear_mean, 4),
            "ear_min": round(ear_min, 4),
            "mar_current": round(mar_current, 4),
            "mar_mean": round(mar_mean, 4),
            "mar_max": round(mar_max, 4),
            "perclos": round(perclos, 4),
            "blink_rate": round(blink_rate, 2),
            "eye_closure_duration": round(self.current_eye_closure_duration, 2),
            "head_pitch_var": round(head_pitch_var, 4),
            "head_yaw_var": round(head_yaw_var, 4),
            "head_roll_var": round(head_roll_var, 4),
            "yawn_count": self.total_yawns
        }

        # Complete telemetry dict for visualizer and socket client
        return {
            # Instantaneous & Status values
            "ear": round(ear_current, 4),
            "mar": round(mar_current, 4),
            "eye_state": "CLOSED" if is_eye_closed else "OPEN",
            "is_yawning": is_yawning,
            "head_pitch": round(pitch, 1),
            "head_yaw": round(yaw, 1),
            "head_roll": round(roll, 1),
            "rvec": rvec,
            "tvec": tvec,
            "total_blinks": self.total_blinks,
            "total_yawns": self.total_yawns,
            "gaze_fixation_time": round(gaze_time, 2),
            # Aggregated ML Feature Dictionary
            "ml_features": ml_features
        }
