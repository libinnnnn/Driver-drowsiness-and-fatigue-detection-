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

Head Pose Conventions (after fix):
  Pitch: ~0° = looking straight at camera, negative = looking down, positive = looking up
  Yaw:   ~0° = facing forward, negative = turned left, positive = turned right
  Roll:  ~0° = head level, negative = tilted left, positive = tilted right
"""

import time
import math
import numpy as np
import cv2
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from detection_engine.face_mesh_detector import FaceMeshDetector
import detection_engine.config as cfg


def euclidean_dist(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


class FeatureExtractor:
    """
    Extracts fatigue-relevant features from MediaPipe Face Mesh landmarks.

    3D Model Coordinate Convention (OpenCV camera space):
        +X = right, +Y = down (screen), +Z = toward camera
    MODEL_POINTS_3D is defined in this convention so cv2.solvePnP returns
    a correct, physically-meaningful rotation vector.
    """

    # Generic 3D facial model coordinates (mm) for PnP Head Pose Estimation.
    # NOTE: These use a head-centric coordinate system (not strict OpenCV convention).
    # cv2.RQDecomp3x3 correctly handles the resulting rotation matrix.
    #   Nose tip is the origin; chin is below (-Y in head space), eyes are above (+Y in head space).
    MODEL_POINTS_3D = np.array([
        (0.0,     0.0,    0.0),    # 0: Nose tip     (landmark 1)
        (0.0,  -330.0,  -65.0),   # 1: Chin          (landmark 152)
        (-225.0, 170.0, -135.0),  # 2: Left eye outer corner  (landmark 33)
        (225.0,  170.0, -135.0),  # 3: Right eye outer corner (landmark 263)
        (-150.0,-150.0, -125.0),  # 4: Left mouth corner  (landmark 61)
        (150.0, -150.0, -125.0),  # 5: Right mouth corner (landmark 291)
    ], dtype=np.float64)

    def __init__(self):
        # ------------------------------------------------------------------ #
        #  Calibration state                                                   #
        # ------------------------------------------------------------------ #
        self.calibration_active = True
        self.calibration_start_time = time.time()
        self._cal_ear_samples: List[float] = []
        self._cal_mar_samples: List[float] = []

        # Effective thresholds (overwritten after calibration completes)
        self.ear_threshold = cfg.EAR_THRESHOLD
        self.yawn_mar_threshold = cfg.YAWN_MAR_THRESHOLD

        # ------------------------------------------------------------------ #
        #  Blink / yawn state tracking                                         #
        # ------------------------------------------------------------------ #
        self.blink_counter = 0          # consecutive frames with eyes closed
        self.total_blinks = 0
        self.yawn_counter = 0           # consecutive frames with mouth open
        self.total_yawns = 0

        self.eye_closed_start_time: Optional[float] = None
        self.current_eye_closure_duration: float = 0.0

        # ------------------------------------------------------------------ #
        #  Rolling buffers                                                      #
        # ------------------------------------------------------------------ #
        self.ear_smoothing_buffer = deque(maxlen=3)
        self.eye_state_history: deque = deque()   # (timestamp, is_closed: bool)
        self.blink_timestamps: deque = deque()    # timestamps of completed blinks
        self.feature_history: deque = deque(maxlen=60)  # last ~2s frame features

        # ------------------------------------------------------------------ #
        #  Gaze tracking state                                                 #
        # ------------------------------------------------------------------ #
        self.last_gaze_direction = "center"
        self.gaze_fixation_start_time = time.time()
        self.gaze_fixation_time = 0.0

        # ------------------------------------------------------------------ #
        #  Verification logging (print raw values for first 15 seconds)        #
        # ------------------------------------------------------------------ #
        self._log_start = time.time()
        self._log_active = True
        print("[TELEMETRY] Verification logging active for first 15 seconds.")
        print("[TELEMETRY]  t(s) | EAR    | MAR    | Pitch  | Blinks | Yawns")
        print("[TELEMETRY] " + "-" * 58)

    # ---------------------------------------------------------------------- #
    #  Calibration                                                             #
    # ---------------------------------------------------------------------- #
    def _update_calibration(self, ear: float, mar: float) -> bool:
        """
        Accumulates baseline samples during the startup calibration phase.
        Returns True while calibration is active, False once it completes.
        """
        if not self.calibration_active:
            return False

        elapsed = time.time() - self.calibration_start_time
        if elapsed < cfg.CALIBRATION_DURATION_SECONDS:
            # Collect only plausible open-eye / closed-mouth values
            if ear > 0.10:
                self._cal_ear_samples.append(ear)
            if mar < 0.35:
                self._cal_mar_samples.append(mar)
            return True
        else:
            # Calibration complete — compute personalised thresholds
            if len(self._cal_ear_samples) >= 10:
                baseline_ear = float(np.median(self._cal_ear_samples))
                self.ear_threshold = round(baseline_ear * cfg.EAR_CLOSED_FACTOR, 4)
                print(f"[CALIBRATION] Baseline EAR = {baseline_ear:.4f} → "
                      f"EAR threshold set to {self.ear_threshold:.4f}")
            else:
                print(f"[CALIBRATION] Insufficient EAR samples; "
                      f"using default threshold {self.ear_threshold:.4f}")

            if len(self._cal_mar_samples) >= 10:
                baseline_mar = float(np.median(self._cal_mar_samples))
                self.yawn_mar_threshold = round(baseline_mar + cfg.YAWN_MAR_FACTOR, 4)
                print(f"[CALIBRATION] Baseline MAR = {baseline_mar:.4f} → "
                      f"Yawn MAR threshold set to {self.yawn_mar_threshold:.4f}")
            else:
                print(f"[CALIBRATION] Insufficient MAR samples; "
                      f"using default threshold {self.yawn_mar_threshold:.4f}")

            self.calibration_active = False
            print("[CALIBRATION] Calibration complete. Detection now active.")
            return False

    # ---------------------------------------------------------------------- #
    #  EAR                                                                     #
    # ---------------------------------------------------------------------- #
    def compute_ear(self, landmarks: List[Tuple[int, int, float]]) -> Tuple[float, float, float]:
        """
        Computes Left EAR, Right EAR, and Average EAR.

        6-point EAR formula (Soukupová & Čech, 2016):
            EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

        MediaPipe landmark indices:
            LEFT_EYE  = [362, 385, 387, 263, 373, 380]
                         P1    P2    P3    P4    P5    P6
            RIGHT_EYE = [33,  160,  158,  133,  153,  144]
                         P1    P2    P3    P4    P5    P6
        """
        def single_eye_ear(indices: List[int]) -> float:
            p1 = (landmarks[indices[0]][0], landmarks[indices[0]][1])  # outer corner
            p2 = (landmarks[indices[1]][0], landmarks[indices[1]][1])  # upper-outer
            p3 = (landmarks[indices[2]][0], landmarks[indices[2]][1])  # upper-inner
            p4 = (landmarks[indices[3]][0], landmarks[indices[3]][1])  # inner corner
            p5 = (landmarks[indices[4]][0], landmarks[indices[4]][1])  # lower-inner
            p6 = (landmarks[indices[5]][0], landmarks[indices[5]][1])  # lower-outer

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

    # ---------------------------------------------------------------------- #
    #  MAR                                                                     #
    # ---------------------------------------------------------------------- #
    def compute_mar(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Computes Mouth Aspect Ratio (MAR).

        MAR = vertical inner-lip opening / horizontal mouth width

        Landmark indices:
            13  = upper inner lip (top of mouth opening)
            14  = lower inner lip (bottom of mouth opening)
            78  = left mouth corner (inner)
            308 = right mouth corner (inner)

        The ratio is distance-invariant: camera distance and slight head
        rotation do not affect the normalized value.
        """
        top    = (landmarks[13][0],  landmarks[13][1])
        bottom = (landmarks[14][0],  landmarks[14][1])
        left   = (landmarks[78][0],  landmarks[78][1])
        right  = (landmarks[308][0], landmarks[308][1])

        v_dist = euclidean_dist(top, bottom)
        h_dist = euclidean_dist(left, right)

        if h_dist == 0:
            return 0.0
        return v_dist / h_dist

    # ---------------------------------------------------------------------- #
    #  Head Pose                                                               #
    # ---------------------------------------------------------------------- #
    def estimate_head_pose(
        self,
        landmarks: List[Tuple[int, int, float]],
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Estimates Head Pose (Pitch, Yaw, Roll) in degrees via cv2.solvePnP +
        cv2.RQDecomp3x3.

        Sign conventions after correction:
            Pitch: 0° = looking straight; negative = looking down; positive = looking up
            Yaw:   0° = facing forward; negative = turned left; positive = turned right
            Roll:  0° = level; negative = tilted left; positive = tilted right
        """
        h, w = frame_shape[:2]

        # 2D image points matching POSE_LANDMARKS [1, 152, 33, 263, 61, 291]
        image_points_2d = np.array([
            (landmarks[idx][0], landmarks[idx][1]) for idx in FaceMeshDetector.POSE_LANDMARKS
        ], dtype=np.float64)

        # Camera intrinsic matrix approximation (no calibration file available)
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1         ]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume zero lens distortion

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

        # Extract Euler angles using cv2.RQDecomp3x3 (numerically stable)
        # Returns (rx, ry, rz) in degrees
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch_deg = angles[0]
        yaw_deg   = angles[1]
        roll_deg  = angles[2]

        # RQDecomp3x3 can return angles in the range [-180, 180].
        # Normalise to the intuitive range [-90, 90] for pitch/yaw.
        def _norm(a: float) -> float:
            """Bring angle into (-90, 90] by flipping ±180° if needed."""
            if a > 90:
                return a - 180.0
            if a < -90:
                return a + 180.0
            return a

        pitch_deg = _norm(pitch_deg)
        yaw_deg   = _norm(yaw_deg)
        # Roll is allowed its full ±180° range

        return pitch_deg, yaw_deg, roll_deg, rvec, tvec

    # ---------------------------------------------------------------------- #
    #  Gaze Fixation                                                           #
    # ---------------------------------------------------------------------- #
    def estimate_gaze_fixation(self, landmarks: List[Tuple[int, int, float]]) -> float:
        """Estimates continuous gaze fixation duration (in seconds)."""
        now = time.time()
        # Compute iris center relative to eye bounding box
        left_iris_center = landmarks[468] if len(landmarks) > 468 else landmarks[33]
        left_eye_outer   = landmarks[33]
        left_eye_inner   = landmarks[133]

        eye_width = euclidean_dist(
            (left_eye_outer[0], left_eye_outer[1]),
            (left_eye_inner[0], left_eye_inner[1])
        )
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

    # ---------------------------------------------------------------------- #
    #  Main per-frame processing                                               #
    # ---------------------------------------------------------------------- #
    def process_frame(
        self,
        landmarks: List[Tuple[int, int, float]],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Main extraction function per frame.
        Computes instantaneous and aggregate metrics.
        Returns dictionary containing all raw and aggregated features.
        """
        now = time.time()

        # ---- 1. Compute EAR and MAR -------------------------------------- #
        _, _, ear = self.compute_ear(landmarks)
        mar = self.compute_mar(landmarks)
        self.ear_smoothing_buffer.append(ear)
        smoothed_ear = sum(self.ear_smoothing_buffer) / len(self.ear_smoothing_buffer)

        # ---- 2. Startup calibration -------------------------------------- #
        calibrating = self._update_calibration(ear, mar)

        # ---- 3. Head Pose ------------------------------------------------ #
        pitch, yaw, roll, rvec, tvec = self.estimate_head_pose(landmarks, frame_shape)

        # ---- 4. Eye Closure & Blink Logic -------------------------------- #
        is_eye_closed = smoothed_ear < self.ear_threshold

        if is_eye_closed:
            self.blink_counter += 1
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
            self.current_eye_closure_duration = round(now - self.eye_closed_start_time, 2)
        else:
            # Eyes just reopened — check if it was a valid blink (not a microsleep)
            if cfg.BLINK_MIN_FRAMES <= self.blink_counter <= cfg.BLINK_MAX_FRAMES:
                self.total_blinks += 1
                self.blink_timestamps.append(now)
            self.blink_counter = 0
            self.eye_closed_start_time = None
            self.current_eye_closure_duration = 0.0

        # ---- 5. Yawn Logic ----------------------------------------------- #
        mouth_vertical_opening = euclidean_dist(
            (landmarks[13][0], landmarks[13][1]),
            (landmarks[14][0], landmarks[14][1])
        )
        mouth_width = euclidean_dist(
            (landmarks[78][0], landmarks[78][1]),
            (landmarks[308][0], landmarks[308][1])
        )
        is_yawning = (
            mar > self.yawn_mar_threshold
            and mouth_width > 0
            and mouth_vertical_opening > mouth_width * 0.5
        )
        if is_yawning:
            self.yawn_counter += 1
        else:
            # Mouth just closed — check if it was open long enough to be a yawn
            if self.yawn_counter >= cfg.YAWN_MIN_FRAMES:
                self.total_yawns += 1
            self.yawn_counter = 0

        # ---- 6. Rolling buffers for PERCLOS & Blink Rate ----------------- #
        self.eye_state_history.append((now, is_eye_closed))
        while self.eye_state_history and (now - self.eye_state_history[0][0]) > cfg.PERCLOS_WINDOW_SECONDS:
            self.eye_state_history.popleft()

        while self.blink_timestamps and (now - self.blink_timestamps[0]) > cfg.PERCLOS_WINDOW_SECONDS:
            self.blink_timestamps.popleft()

        # PERCLOS: % of frames with closed eyes in the rolling window
        if self.eye_state_history:
            closed_frames = sum(1 for _, closed in self.eye_state_history if closed)
            perclos = closed_frames / len(self.eye_state_history)
        else:
            perclos = 0.0

        # Blink Rate (blinks per minute, from rolling window)
        window_duration_min = (
            min((now - self.eye_state_history[0][0]) / 60.0, 1.0)
            if self.eye_state_history else 1.0
        )
        blink_rate = len(self.blink_timestamps) / max(window_duration_min, 0.1)

        # ---- 7. Gaze Fixation -------------------------------------------- #
        gaze_time = self.estimate_gaze_fixation(landmarks)

        # ---- 8. Rolling feature history ---------------------------------- #
        frame_dict = {
            "ear": ear, "mar": mar,
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "timestamp": now
        }
        self.feature_history.append(frame_dict)

        ears    = [f["ear"]   for f in self.feature_history]
        mars    = [f["mar"]   for f in self.feature_history]
        pitches = [f["pitch"] for f in self.feature_history]
        yaws    = [f["yaw"]   for f in self.feature_history]
        rolls   = [f["roll"]  for f in self.feature_history]

        ear_current      = float(ear)
        ear_mean         = float(np.mean(ears))
        ear_min          = float(np.min(ears))
        mar_current      = float(mar)
        mar_mean         = float(np.mean(mars))
        mar_max          = float(np.max(mars))
        head_pitch_var   = float(np.var(pitches)) if len(pitches) > 1 else 0.0
        head_yaw_var     = float(np.var(yaws))    if len(yaws)    > 1 else 0.0
        head_roll_var    = float(np.var(rolls))   if len(rolls)   > 1 else 0.0

        # ---- 9. Verification logging (first 15 seconds) ------------------ #
        if self._log_active:
            t_elapsed = now - self._log_start
            if t_elapsed <= 15.0:
                print(
                    f"[TELEMETRY] {t_elapsed:5.1f}s | "
                    f"EAR={ear:.4f} | MAR={mar:.4f} | "
                    f"Pitch={pitch:+.1f}° | "
                    f"Blinks={self.total_blinks} | Yawns={self.total_yawns}"
                )
            else:
                print("[TELEMETRY] Logging window ended.")
                self._log_active = False

        # ---- 10. Package feature vector for RandomForest input ----------- #
        ml_features = {
            "ear_current":          round(ear_current,    4),
            "ear_mean":             round(ear_mean,        4),
            "ear_min":              round(ear_min,         4),
            "mar_current":          round(mar_current,    4),
            "mar_mean":             round(mar_mean,        4),
            "mar_max":              round(mar_max,         4),
            "perclos":              round(perclos,         4),
            "blink_rate":           round(blink_rate,      2),
            "eye_closure_duration": round(self.current_eye_closure_duration, 2),
            "head_pitch_var":       round(head_pitch_var,  4),
            "head_yaw_var":         round(head_yaw_var,    4),
            "head_roll_var":        round(head_roll_var,   4),
            "yawn_count":           self.total_yawns,
        }

        # Complete telemetry dict for visualizer and socket client
        return {
            # Instantaneous & status values
            "ear":                  round(ear_current, 4),
            "mar":                  round(mar_current, 4),
            "eye_state":            "CLOSED" if is_eye_closed else "OPEN",
            "is_yawning":           is_yawning,
            "head_pitch":           round(pitch, 1),
            "head_yaw":             round(yaw,   1),
            "head_roll":            round(roll,  1),
            "rvec":                 rvec,
            "tvec":                 tvec,
            "total_blinks":         self.total_blinks,
            "total_yawns":          self.total_yawns,
            "gaze_fixation_time":   round(gaze_time, 2),
            "calibrating":          calibrating,
            "ear_threshold":        round(self.ear_threshold, 4),
            "yawn_mar_threshold":   round(self.yawn_mar_threshold, 4),
            # Aggregated ML Feature Dictionary
            "ml_features":          ml_features,
        }
