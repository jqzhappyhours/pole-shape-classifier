from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import tensorflow as tf

from extract import center_crop
from pose_utils import detect_and_draw, extract_landmarks, make_landmarker


DEFAULT_CLASS_NAMES = ['airwalk',
 'climb',
 'inside_leg_hang',
 'invert',
 'outside_leg_hang',
 'pencil',
 'unknown']

@dataclass(frozen=True)
class VideoInferenceConfig:
    image_size: tuple[int, int] = (224, 224)  # (width, height)
    frame_interval: int = 30
    max_frames: int = 64
    smooth_window: int = 5  # majority-vote window size for sequence smoothing
    use_pose_skeleton: bool = False  # must match training preprocessing


def _smooth_predictions(pred_indices: np.ndarray, window: int) -> np.ndarray:
    """Majority vote over a sliding window to remove single-frame noise."""
    if window <= 1:
        return pred_indices
    n = len(pred_indices)
    smoothed = np.empty(n, dtype=pred_indices.dtype)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        counts = np.bincount(pred_indices[start:end])
        smoothed[i] = np.argmax(counts)
    return smoothed


def _resolve_model_path(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Model file not found: {p}")
        return p

    for name in ("efficientnetb0_fine_tuned_v1.keras", "efficientnetb0_v2.keras", "efficientnetb0_v1.keras"):
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
    use_pose_skeleton: bool = False,
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

    landmarker = make_landmarker() if use_pose_skeleton else None
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

                if use_pose_skeleton:
                    skeleton = detect_and_draw(frame_bgr, landmarker, image_size)
                    if skeleton is None:
                        frame_idx += 1
                        continue  # skip frames where no person detected
                    frame_rgb = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                yield frame_rgb.astype(np.float32)
                yielded += 1

            frame_idx += 1
    finally:
        if landmarker is not None:
            landmarker.close()
        cap.release()


def predict_video_sequence(
    model: tf.keras.Model,
    video_path: str,
    *,
    class_names: Optional[list[str]] = None,
    config: VideoInferenceConfig = VideoInferenceConfig(),
) -> list[dict]:
    """Return deduplicated shape segments: [{start, end, label, confidence}, ...]"""
    class_names = class_names or DEFAULT_CLASS_NAMES

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    frames = list(
        iter_video_frames(
            video_path,
            image_size=config.image_size,
            frame_interval=config.frame_interval,
            max_frames=config.max_frames,
            use_pose_skeleton=config.use_pose_skeleton,
        )
    )
    if not frames:
        raise ValueError("No frames were extracted from the video.")

    x = np.stack(frames, axis=0)
    probs = model.predict(x, verbose=0)  # (N, num_classes)

    pred_indices = np.argmax(probs, axis=1)
    pred_indices = _smooth_predictions(pred_indices, config.smooth_window)
    confidences = probs[np.arange(len(probs)), pred_indices]
    seconds_per_sample = config.frame_interval / fps

    segments: list[dict] = []
    for i, (pred_idx, conf) in enumerate(zip(pred_indices, confidences)):
        label = class_names[int(pred_idx)] if int(pred_idx) < len(class_names) else str(pred_idx)
        t_start = i * seconds_per_sample
        t_end = (i + 1) * seconds_per_sample

        if segments and segments[-1]["label"] == label:
            seg = segments[-1]
            seg["end"] = t_end
            seg["_n"] += 1
            seg["confidence"] += (float(conf) - seg["confidence"]) / seg["_n"]
        else:
            segments.append({"start": t_start, "end": t_end, "label": label, "confidence": float(conf), "_n": 1})

    min_duration = config.frame_interval / fps  # at least 2 frames worth
    segments = [s for s in segments if (s["end"] - s["start"]) >= min_duration * 2]

    for seg in segments:
        seg.pop("_n")

    return segments


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
            use_pose_skeleton=config.use_pose_skeleton,
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


def annotate_video(
    video_path: str,
    segments: list[dict],
    output_path: str,
) -> str:
    """Write a copy of the video with the predicted shape label overlaid on each frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        label = ""
        confidence = 0.0
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                label = seg["label"]
                confidence = seg["confidence"]
                break

        if label:
            text = f"{label}  {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.6, w / 1000)
            thickness = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = (w - tw) // 2
            cv2.rectangle(frame, (x - 5, 10), (x + tw + 5, 20 + th + 8), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, 15 + th), font, scale, (0, 255, 0), thickness)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path


def predict_class(
    model: tf.keras.Model,
    video_path: str,
    *,
    class_names: Optional[list[str]] = None,
    config: VideoInferenceConfig = VideoInferenceConfig(),
) -> str:
    result = predict_video(model, video_path, class_names=class_names, config=config)
    return result["pred_name"]


# ---------------------------------------------------------------------------
# Coordinate-based inference (coord_mlp_v1.keras)
# ---------------------------------------------------------------------------

def iter_video_coords(
    video_path: str,
    *,
    frame_interval: int,
    max_frames: int,
) -> Iterable[np.ndarray]:
    """Yield normalized landmark vectors (99,) for each frame with a detected pose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    landmarker = make_landmarker()
    frame_idx = 0
    yielded = 0
    try:
        while yielded < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                coords = extract_landmarks(frame_bgr, landmarker)
                if coords is not None:
                    yield coords
                    yielded += 1
            frame_idx += 1
    finally:
        landmarker.close()
        cap.release()


def predict_video_coords(
    model: tf.keras.Model,
    video_path: str,
    *,
    class_names: Optional[list[str]] = None,
    config: VideoInferenceConfig = VideoInferenceConfig(),
) -> dict:
    """Predict shape from pose coordinates extracted from a video.

    Use this with a model trained on coordinate features (e.g. coord_mlp_v1.keras)
    instead of the image-based predict_video().
    """
    class_names = class_names or DEFAULT_CLASS_NAMES

    coords_list = list(
        iter_video_coords(
            video_path,
            frame_interval=config.frame_interval,
            max_frames=config.max_frames,
        )
    )
    if not coords_list:
        raise ValueError("No frames with a detected pose were extracted from the video.")

    x = np.stack(coords_list, axis=0)   # (N, 99)
    probs = model.predict(x, verbose=0)  # (N, num_classes)
    mean_probs = probs.mean(axis=0)

    pred_idx = int(np.argmax(mean_probs))
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    return {
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "confidence": float(mean_probs[pred_idx]),
        "mean_probs": mean_probs.tolist(),
        "n_frames": int(x.shape[0]),
        "frame_interval": int(config.frame_interval),
        "max_frames": int(config.max_frames),
    }

