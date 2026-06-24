"""
pose_utils.py — Shared MediaPipe pose utilities for extract.py and pole_infer.py.
"""

import os
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Standard MediaPipe pose landmark connections (index pairs)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Shoulders
    (11, 12),
    # Right arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Right leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


def make_landmarker() -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=ensure_model())
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_skeleton(landmarks, image_size: tuple = (224, 224)) -> np.ndarray:
    """Draw pose skeleton on a black background.

    Args:
        landmarks: list of NormalizedLandmark from MediaPipe Tasks API
        image_size: (width, height)

    Returns:
        BGR uint8 array of shape (height, width, 3)
    """
    w, h = image_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if (landmarks[start_idx].visibility > 0.5 and
                landmarks[end_idx].visibility > 0.5):
            cv2.line(canvas, points[start_idx], points[end_idx],
                     (200, 200, 200), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(points):
        if landmarks[i].visibility > 0.5:
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)

    return canvas


def detect_and_draw(frame_bgr: np.ndarray, landmarker: mp_vision.PoseLandmarker,
                    image_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """Run pose detection on a BGR frame and return skeleton on black background.

    Returns None if no person is detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return draw_skeleton(result.pose_landmarks[0], image_size)


def extract_landmarks(frame_bgr: np.ndarray, landmarker: mp_vision.PoseLandmarker) -> Optional[np.ndarray]:
    """Run pose detection and return a normalized landmark vector of shape (99,).

    Each of the 33 MediaPipe landmarks contributes (x, y, visibility).
    x/y are centered on the hip midpoint and scaled by torso length so the
    vector is position- and scale-invariant.

    Returns None if no person is detected or the torso length is degenerate.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return None

    lms = result.pose_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.visibility] for lm in lms], dtype=np.float32)  # (33, 3)

    hip_mid = (coords[23, :2] + coords[24, :2]) / 2        # midpoint of hips
    shoulder_mid = (coords[11, :2] + coords[12, :2]) / 2   # midpoint of shoulders
    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_len < 1e-6:
        return None

    coords[:, :2] = (coords[:, :2] - hip_mid) / torso_len
    return coords.flatten()  # (99,)
