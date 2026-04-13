import os
from extract import extract_frames_from_video

shape = ['inside_leg_hang', 'outside_leg_hang']
DATA_ROOT = "data/stress_test"
for s in shape:
    video_dir = os.path.join(DATA_ROOT, "clips", s)
    if not os.path.isdir(video_dir):
            raise ValueError(f"Video directory does not exist: {video_dir}")
    
    video_files = sorted(
            f for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        )
    if not video_files:
        raise ValueError(f"No video files found in: {video_dir}")

    for video in os.listdir(video_dir):
        extract_frames_from_video(s, os.path.join(video_dir, video), os.path.join(DATA_ROOT, "images",s),resize=(224, 224))