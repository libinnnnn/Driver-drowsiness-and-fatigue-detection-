"""
MediaPipe Face Mesh detector wrapper with Iris Landmark Refinement.
Supports both legacy MediaPipe Solutions API and modern MediaPipe Tasks Vision API (FaceLandmarker).
Provides landmark extraction for eyes, irises, mouth, and key head pose points.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional

MODEL_TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceMeshDetector:
    # MediaPipe 468 landmark index mappings
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    MOUTH_INNER = [13, 14, 78, 308, 82, 87]
    MOUTH_OUTER = [61, 291, 0, 17, 37, 84]

    # Facial landmarks for 3D Head Pose Estimation (PnP)
    # 1: Nose tip, 152: Chin, 33: Left eye outer corner, 263: Right eye outer corner,
    # 61: Left mouth corner, 291: Right mouth corner
    POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.use_tasks_api = False
        self.face_mesh = None
        self.landmarker = None

        # Check if legacy mp.solutions.face_mesh is available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("[FACE DETECTOR] Initialized using legacy MediaPipe Solutions FaceMesh.")
                return
            except Exception as e:
                print(f"[FACE DETECTOR NOTICE] Could not initialize mp.solutions ({e}). Switching to Tasks API.")

        # Fall back to modern MediaPipe Tasks Vision API (FaceLandmarker)
        self.use_tasks_api = True
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task_path = os.path.join(project_root, "models", "face_landmarker.task")
        os.makedirs(os.path.dirname(task_path), exist_ok=True)

        if not os.path.exists(task_path):
            print(f"[FACE DETECTOR] Downloading face_landmarker.task model to {task_path}...")
            try:
                urllib.request.urlretrieve(MODEL_TASK_URL, task_path)
                print("[FACE DETECTOR] Model download complete.")
            except Exception as e:
                print(f"[FACE DETECTOR ERROR] Failed to download model task: {e}")

        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python import BaseOptions

            base_options = BaseOptions(model_asset_path=task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_tracking_confidence
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
            print("[FACE DETECTOR] Initialized using MediaPipe Vision Tasks FaceLandmarker.")
        except Exception as e:
            print(f"[FACE DETECTOR ERROR] Failed to initialize FaceLandmarker Tasks API: {e}")

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[List[Tuple[int, int, float]]], Optional[np.ndarray]]:
        """
        Processes a BGR camera frame and returns:
        - landmarks: List of (x_px, y_px, z_rel) for facial points (or None if no face detected)
        - raw_landmarks_array: (N, 3) numpy array normalized coordinates
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.use_tasks_api and self.face_mesh is not None:
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None, None

            face_landmarks = results.multi_face_landmarks[0]
            pixel_landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_landmarks.landmark]
            raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            return pixel_landmarks, raw_landmarks

        elif self.use_tasks_api and self.landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)

            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return None, None

            face_landmarks = result.face_landmarks[0]
            pixel_landmarks = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_landmarks]
            raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])
            return pixel_landmarks, raw_landmarks

        return None, None

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()
        if self.landmarker:
            self.landmarker.close()
