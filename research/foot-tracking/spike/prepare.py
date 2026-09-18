import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .common import PREPARED, RAW, VIEWS

MAX_SIDE = 1600
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v"}


def read_oriented(path: Path):
    if path.suffix.lower() != ".heic":
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{path.stem}.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)], check=True, capture_output=True)
        return cv2.imread(str(out), cv2.IMREAD_COLOR)


def downscale(image: np.ndarray) -> np.ndarray:
    scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
    return image if scale == 1.0 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(cv2.resize(image, (540, 960)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def video_frames(path: Path, every_seconds: float, drop_blurriest: float) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_seconds))
    frames = []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                frames.append((round(index / fps * 1000), downscale(frame)))
        index += 1
    capture.release()
    if len(frames) < 4:
        return frames
    scores = np.array([sharpness(frame) for _, frame in frames])
    cutoff = np.quantile(scores, drop_blurriest)
    return [item for item, score in zip(frames, scores) if score >= cutoff]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize photos and video frames into data/prepared")
    parser.add_argument("--every", type=float, default=0.5, help="seconds between sampled video frames")
    parser.add_argument("--drop-blurriest", type=float, default=0.25, help="share of the blurriest sampled frames to drop per video")
    args = parser.parse_args()

    PREPARED.mkdir(parents=True, exist_ok=True)
    written = 0
    for view in VIEWS:
        sources = sorted((RAW / view).glob("*"))
        photos = [p for p in sources if p.suffix.lower() in PHOTO_EXTENSIONS]
        videos = [p for p in sources if p.suffix.lower() in VIDEO_EXTENSIONS]
        for index, source in enumerate(photos, start=1):
            image = read_oriented(source)
            if image is None:
                print(f"skip unreadable {source}", file=sys.stderr)
                continue
            cv2.imwrite(str(PREPARED / f"{view}_{index:03d}.jpg"), downscale(image), [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
        for video in videos:
            frames = video_frames(video, args.every, args.drop_blurriest)
            for millis, frame in frames:
                cv2.imwrite(str(PREPARED / f"{view}_{video.stem}_{millis:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += len(frames)
            print(f"{video.name}: {len(frames)} frames")
    print(f"prepared {written} images into {PREPARED}")


if __name__ == "__main__":
    main()
