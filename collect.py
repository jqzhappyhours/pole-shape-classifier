"""
collect.py — Download pole shape video clips from YouTube into data/clips/<shape>/

Usage:
    python collect.py                        # all classes, 20 videos each
    python collect.py --shapes invert pencil # specific classes only
    python collect.py --per-class 40         # more videos per class

Requirements:
    pip install yt-dlp

Downloads go to data/clips/<shape>/ so extract.py works unchanged.
Already-downloaded videos are skipped automatically.
"""

import argparse
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Search queries per class. Multiple queries are tried in order; duplicates
# (same YouTube video ID) are skipped automatically.
# ---------------------------------------------------------------------------
QUERIES: dict[str, list[str]] = {
    "invert": [
        "pole dance invert tutorial",
        "pole invert beginner",
        "pole inside leg hang tutorial",
        "aerial invert pole dance",
    ],
    "pencil": [
        "pencil spin pole dance tutorial",
        "pole pencil spin",
        "pole dance pencil",
        "pencil pole trick tutorial",
    ],
    "inside_leg_hang": [
        "inside leg hang pole dance tutorial",
        "pole inside leg hang",
        "inside leg hang pole trick",
    ],
    "outside_leg_hang": [
        "outside leg hang pole dance tutorial",
        "pole outside leg hang",
        "outside leg hang pole trick",
    ],
    "airwalk": [
        "airwalk pole dance tutorial",
        "pole airwalk trick",
        "pole dance airwalk beginner",
    ],
    "climb": [
        "pole climb tutorial",
        "how to climb pole dance",
        "pole climbing technique",
        "pole climb beginner",
    ],
    "unknown": [
        "pole dance combo tutorial",
        "pole dance choreography",
        "pole tricks compilation",
    ],
}

# Keep clips short (tutorials) — skip anything over this many seconds.
MAX_DURATION = 600  # 10 minutes


def clips_dir(shape: str) -> str:
    return os.path.join("data", "clips", shape)


def already_downloaded(shape: str) -> set[str]:
    """Return the set of video IDs already present in data/clips/<shape>/."""
    d = clips_dir(shape)
    if not os.path.isdir(d):
        return set()
    ids = set()
    for fname in os.listdir(d):
        # yt-dlp default output is <title>_<id>.mp4 or just <id>.mp4
        # We name files as %(id)s.mp4 so the stem IS the video ID.
        stem = os.path.splitext(fname)[0]
        if stem:
            ids.add(stem)
    return ids


def search_video_ids(query: str, max_results: int) -> list[str]:
    """
    Use yt-dlp's built-in ytsearch to get video IDs without downloading.
    Returns a list of YouTube video IDs.
    """
    search_url = f"ytsearch{max_results}:{query}"
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "id",
        "--match-filter", f"duration < {MAX_DURATION}",
        "--no-warnings",
        search_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ids


def download_video(video_id: str, shape: str) -> bool:
    """
    Download a single video into data/clips/<shape>/<video_id>.mp4.
    Returns True on success.
    """
    out_dir = clips_dir(shape)
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",
        "--match-filter", f"duration < {MAX_DURATION}",
        "--no-warnings",
        "--quiet",
        "--progress",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def collect(shape: str, target: int) -> None:
    queries = QUERIES.get(shape, [])
    if not queries:
        print(f"[{shape}] No queries defined — skipping.")
        return

    existing = already_downloaded(shape)
    print(f"\n[{shape}] {len(existing)} videos already downloaded, targeting {target} total.")

    seen: set[str] = set(existing)
    downloaded = 0
    needed = max(0, target - len(existing))

    for query in queries:
        if downloaded >= needed:
            break
        print(f"  Searching: {query!r}")
        ids = search_video_ids(query, max_results=50)
        for vid_id in ids:
            if downloaded >= needed:
                break
            if vid_id in seen:
                continue
            seen.add(vid_id)
            print(f"  Downloading {vid_id} ...", end=" ", flush=True)
            ok = download_video(vid_id, shape)
            if ok:
                downloaded += 1
                print(f"ok ({downloaded}/{needed})")
            else:
                print("failed (skipped)")

    print(f"[{shape}] Done. Downloaded {downloaded} new videos.")


def main():
    parser = argparse.ArgumentParser(description="Collect pole shape clips from YouTube.")
    parser.add_argument(
        "--shapes", nargs="+", default=list(QUERIES.keys()),
        choices=list(QUERIES.keys()),
        help="Which shape classes to collect (default: all).",
    )
    parser.add_argument(
        "--per-class", type=int, default=20, dest="per_class",
        help="Target total number of videos per class (default: 20). Already-downloaded videos count toward this.",
    )
    args = parser.parse_args()

    # Check yt-dlp is available
    if subprocess.run(["which", "yt-dlp"], capture_output=True).returncode != 0:
        print("Error: yt-dlp is not installed. Run:  pip install yt-dlp")
        sys.exit(1)

    for shape in args.shapes:
        collect(shape, args.per_class)

    print("\nAll done. Run `python extract.py` to extract frames from the new clips.")


if __name__ == "__main__":
    main()
