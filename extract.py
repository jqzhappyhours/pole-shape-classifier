import cv2
import os
import random

def center_crop(frame, target_size: tuple):
    """
    Center-crop an OpenCV BGR frame to match the target aspect ratio.

    Parameters
    ----------
    frame : np.ndarray
        OpenCV image (H, W, C) in BGR.
    target_size : tuple
        (target_width, target_height)
    """
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        return frame

    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h if h else 0

    # Crop to match target aspect ratio while keeping the crop centered.
    if current_ratio > target_ratio:
        # Frame is too wide -> crop width.
        new_w = int(round(h * target_ratio))
        x1 = max((w - new_w) // 2, 0)
        x2 = min(x1 + new_w, w)
        return frame[:, x1:x2]
    else:
        # Frame is too tall -> crop height.
        new_h = int(round(w / target_ratio)) if target_ratio != 0 else h
        y1 = max((h - new_h) // 2, 0)
        y2 = min(y1 + new_h, h)
        return frame[y1:y2, :]


def _has_extracted_frames(output_dir: str, video_name: str, shape: str) -> bool:
    """True if any frame files for this video+shape already exist in output_dir."""
    prefix = f"{video_name}_{shape}_frame_"
    try:
        names = os.listdir(output_dir)
    except FileNotFoundError:
        return False
    return any(
        n.startswith(prefix) and n.lower().endswith((".jpg", ".jpeg", ".png"))
        for n in names
    )


def extract_frames_from_video(
    shape: str,
    video_path: str,
    output_dir: str,
    frame_interval: int = 200,
    resize: tuple = None,
    skip_existing: bool = True,
):
    """
    Extract frames from a video file at a fixed interval.

    Parameters:
    -----------
    shape : str
        Shape of the video. Example: "inside_leg_hang"
    video_path : str
        Path to input video file.
    output_dir : str
        Directory to save extracted frames.
    frame_interval : int
        Save every Nth frame (default=200).
    resize : tuple
        Optional output size (width, height). If provided, frames will be
        center-cropped to the target aspect ratio and then resized.
    skip_existing : bool
        If True, do not extract when frames for this video already exist in
        output_dir (same naming pattern as this function uses).

    Returns:
    --------
    saved_count : int
        Number of frames saved.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if skip_existing and _has_extracted_frames(output_dir, video_name, shape):
        print(f"Skipping (already extracted): {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:

            if resize is not None:
                frame = center_crop(frame, resize)
                frame = cv2.resize(frame, resize)

            frame_filename = os.path.join(
            output_dir,
            f"{video_name}_{shape}_frame_{saved_count:05d}.jpg"
            )

            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Finished. Saved {saved_count} frames to '{output_dir}'")
    return saved_count

DATA_ROOT = "data"

def video_folder(shape: str):
    return os.path.join(DATA_ROOT, "clips", shape)

def output_folder(shape: str):
    return os.path.join(DATA_ROOT, "images", shape)


def extract_split(
    shape: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    frame_interval: int = 5,
):
    """
    Extract frames from all videos for a shape, splitting at the video level
    into train/val/test to prevent data leakage.

    Frames are saved to:
        data/images/train/<shape>/
        data/images/val/<shape>/
        data/images/test/<shape>/
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
        if i < n_train:
            split_name = "train"
        elif i < n_train + n_val:
            split_name = "val"
        else:
            split_name = "test"
        out_dir = os.path.join(DATA_ROOT, "images", split_name, shape)
        extract_frames_from_video(
            shape,
            os.path.join(video_dir, video_file),
            out_dir,
            frame_interval=frame_interval,
            resize=(224, 224),
        )


'''
Extract frames from the videos in the inside_leg_hang and outside_leg_hang folders and
save them to the images/inside_leg_hang and images/outside_leg_hang folders. The frames
are resized to 224x224.
'''

if __name__ == "__main__":
    shapes = ["inside_leg_hang", "outside_leg_hang", "airwalk", "invert", "climb", "pencil", "unknown"]
    for shape in shapes:
        if not os.path.isdir(video_folder(shape)):
            raise ValueError(f"Video directory does not exist: {video_folder(shape)}")
        extract_split(shape, train_ratio=0.7, val_ratio=0.15, frame_interval=2)