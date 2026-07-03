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

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


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
