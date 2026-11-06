"""Extract frames from videos so you can label them in Roboflow.

    python scripts/extract_frames.py video1.mp4 video2.mp4 --every 10 --out frames
"""
import argparse
from pathlib import Path

import cv2


def extract(video: Path, out_dir: Path, every: int, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"Could not open {video}")
        return 0
    saved = idx = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            cv2.imwrite(str(out_dir / f"{video.stem}_{saved:04d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("frames"))
    ap.add_argument("--every", type=int, default=10, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=500, help="limit per video")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        n = extract(video, args.out, args.every, args.max_frames)
        print(f"{video.name}: saved {n} frames")


if __name__ == "__main__":
    main()
