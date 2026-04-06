import cv2
import os

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


def extract_frames_from_video(
    shape: str,
    video_path: str,
    output_dir: str,
    frame_interval: int = 15,
    resize: tuple = None
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
        Save every Nth frame (default=15).
    resize : tuple
        Optional output size (width, height). If provided, frames will be
        center-cropped to the target aspect ratio and then resized.

    Returns:
    --------
    saved_count : int
        Number of frames saved.
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

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


'''
Extract frames from the videos in the inside_leg_hang and outside_leg_hang folders and
save them to the images/inside_leg_hang and images/outside_leg_hang folders. The frames
are resized to 224x224.
'''

if __name__ == "__main__":
    shapes = ["inside_leg_hang", "outside_leg_hang"]
    for shape in shapes:
        video_dir = video_folder(shape)
        out_dir = output_folder(shape)

        if not os.path.isdir(video_dir):
            raise ValueError(f"Video directory does not exist: {video_dir}")

        video_files = sorted(
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        )
        if not video_files:
            raise ValueError(f"No video files found in: {video_dir}")

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            extract_frames_from_video(shape, video_path, out_dir, resize=(224, 224))