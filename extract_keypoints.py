"""
extract_keypoints.py — Extract MediaPipe Pose keypoints from pole shape videos.

Mirrors extract.py but instead of saving JPEG frames it saves numpy arrays of
pose landmarks. Each video produces one .npy file containing all successfully
detected frames, shaped (N, 99) where 99 = 33 landmarks × (x, y, visibility).

Output structure (parallel to data/images/):
    data/keypoints/train/<shape>/<video_name>.npy
    data/keypoints/val/<shape>/<video_name>.npy
    data/keypoints/test/<shape>/<video_name>.npy

Usage:
    python extract_keypoints.py          # all shapes
    python extract_keypoints.py --shapes invert pencil
"""

import argparse
import os
import random
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DATA_ROOT = "data"
SHAPES = ["inside_leg_hang", "outside_leg_hang", "airwalk", "invert", "climb", "pencil", "unknown"]

# 33 landmarks × 3 values (x, y, visibility)
N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 3

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose landmarker model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


def _make_landmarker() -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=_ensure_model())
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def video_folder(shape: str) -> str:
    return os.path.join(DATA_ROOT, "clips", shape)


def keypoints_folder(split: str, shape: str) -> str:
    return os.path.join(DATA_ROOT, "keypoints", split, shape)


def _output_path(split: str, shape: str, video_name: str) -> str:
    return os.path.join(keypoints_folder(split, shape), f"{video_name}.npy")


def extract_keypoints_from_video(
    video_path: str,
    frame_interval: int = 2,
    skip_existing: bool = True,
) -> "np.ndarray | None":
    """
    Run MediaPipe Pose on every Nth frame of a video.

    Returns a float32 array of shape (N_detected, 99), or None if no poses
    were detected. Frames where MediaPipe finds no person are silently skipped.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return None

    landmarker = _make_landmarker()
    rows = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]  # first detected person
                row = np.array(
                    [[l.x, l.y, l.visibility] for l in lm], dtype=np.float32
                ).flatten()  # shape (99,)
                rows.append(row)

        frame_count += 1

    cap.release()
    landmarker.close()

    if not rows:
        print(f"  No poses detected: {video_name}")
        return None

    return np.stack(rows)  # (N, 99)


def extract_split(
    shape: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    frame_interval: int = 2,
) -> None:
    """
    Extract keypoints from all videos for a shape, splitting at the video level
    into train/val/test (same split logic as extract.py).

    Saves one .npy per video to data/keypoints/<split>/<shape>/<video_name>.npy.
    Already-extracted videos are skipped.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    video_dir = video_folder(shape)
    videos = sorted(
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not videos:
        raise ValueError(f"No video files found in: {video_dir}")

    random.seed(seed)
    shuffled = videos[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    for i, video_file in enumerate(shuffled):
        split_name = (
            "train" if i < n_train
            else "val" if i < n_train + n_val
            else "test"
        )

        video_name = os.path.splitext(video_file)[0]
        out_path = _output_path(split_name, shape, video_name)

        if os.path.exists(out_path):
            print(f"  Skipping (exists): {out_path}")
            continue

        print(f"  [{split_name}] {video_file} ...", end=" ", flush=True)
        keypoints = extract_keypoints_from_video(
            os.path.join(video_dir, video_file),
            frame_interval=frame_interval,
        )

        if keypoints is not None:
            os.makedirs(keypoints_folder(split_name, shape), exist_ok=True)
            np.save(out_path, keypoints)
            print(f"saved {len(keypoints)} frames → {out_path}")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all keypoint arrays for a given split into X, y arrays.

    Returns:
        X : float32 array of shape (N_total_frames, 99)
        y : int array of shape (N_total_frames,)  — class indices
        class_names : sorted list of shape names
    """
    split_dir = os.path.join(DATA_ROOT, "keypoints", split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Keypoints split directory not found: {split_dir}")

    class_names = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )

    X_parts, y_parts = [], []
    for class_idx, shape in enumerate(class_names):
        shape_dir = os.path.join(split_dir, shape)
        for fname in sorted(os.listdir(shape_dir)):
            if not fname.endswith(".npy"):
                continue
            arr = np.load(os.path.join(shape_dir, fname))  # (N, 99)
            X_parts.append(arr)
            y_parts.append(np.full(len(arr), class_idx, dtype=np.int32))

    if not X_parts:
        raise ValueError(f"No keypoint files found in: {split_dir}")

    return np.concatenate(X_parts), np.concatenate(y_parts), class_names


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from pole shape videos.")
    parser.add_argument(
        "--shapes", nargs="+", default=SHAPES, choices=SHAPES,
        help="Which shape classes to process (default: all).",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=2, dest="frame_interval",
        help="Extract keypoints every Nth frame (default: 2).",
    )
    args = parser.parse_args()

    for shape in args.shapes:
        vdir = video_folder(shape)
        if not os.path.isdir(vdir):
            print(f"[{shape}] Video directory not found, skipping: {vdir}")
            continue
        print(f"\n[{shape}]")
        extract_split(shape, train_ratio=0.7, val_ratio=0.15, frame_interval=args.frame_interval)

    print("\nDone. Load data in your notebook with:")
    print("  from extract_keypoints import load_split")
    print("  X_train, y_train, class_names = load_split('train')")
    print("  X_val,   y_val,   _           = load_split('val')")


if __name__ == "__main__":
    main()
