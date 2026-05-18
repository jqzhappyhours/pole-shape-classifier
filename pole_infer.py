from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import tensorflow as tf

from extract import center_crop


DEFAULT_CLASS_NAMES = ["inside_leg_hang", "outside_leg_hang"]


@dataclass(frozen=True)
class VideoInferenceConfig:
    image_size: tuple[int, int] = (224, 224)  # (width, height)
    frame_interval: int = 30
    max_frames: int = 64


def _resolve_model_path(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p

    for name in ("efficientnetb0_v2.keras", "efficientnetb0_v1.keras"):
        p = Path(name)
        if p.is_file():
            return p

    raise FileNotFoundError(
        "No model found. Put your exported model file in the repo root as "
        "`efficientnetb0_v2.keras` (or `efficientnetb0_v1.keras`), "
        "or pass an explicit model path."
    )


def load_model(model_path: Optional[str] = None) -> tf.keras.Model:
    path = _resolve_model_path(model_path)
    return tf.keras.models.load_model(path)


def iter_video_frames(
    video_path: str,
    *,
    image_size: tuple[int, int],
    frame_interval: int,
    max_frames: int,
) -> Iterable[np.ndarray]:
    """
    Yield RGB float32 frames shaped (H, W, 3), resized to image_size.

    Notes:
    - Training used `keras.utils.image_dataset_from_directory(image_size=(224,224))`
      without explicit rescaling/preprocessing, so we keep pixel values in [0,255].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    yielded = 0
    target_w, target_h = image_size
    try:
        while yielded < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_bgr = center_crop(frame_bgr, (target_w, target_h))
                frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb.astype(np.float32)
                yielded += 1

            frame_idx += 1
    finally:
        cap.release()


def predict_video(
    model: tf.keras.Model,
    video_path: str,
    *,
    class_names: Optional[list[str]] = None,
    config: VideoInferenceConfig = VideoInferenceConfig(),
) -> dict:
    class_names = class_names or DEFAULT_CLASS_NAMES

    frames = list(
        iter_video_frames(
            video_path,
            image_size=config.image_size,
            frame_interval=config.frame_interval,
            max_frames=config.max_frames,
        )
    )
    if not frames:
        raise ValueError("No frames were extracted from the video.")

    x = np.stack(frames, axis=0)  # (N, H, W, 3)
    probs = model.predict(x, verbose=0)  # (N, num_classes)
    mean_probs = probs.mean(axis=0)

    pred_idx = int(np.argmax(mean_probs))
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    confidence = float(mean_probs[pred_idx])

    return {
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "confidence": confidence,
        "mean_probs": mean_probs.tolist(),
        "n_frames": int(x.shape[0]),
        "image_size": list(config.image_size),
        "frame_interval": int(config.frame_interval),
        "max_frames": int(config.max_frames),
    }

