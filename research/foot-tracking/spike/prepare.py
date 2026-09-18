import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from .common import PREPARED, RAW, VIEWS

MAX_SIDE = 1600
EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def read_oriented(path: Path):
    if path.suffix.lower() != ".heic":
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{path.stem}.jpg"
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)], check=True, capture_output=True)
        return cv2.imread(str(out), cv2.IMREAD_COLOR)


def main() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    written = 0
    for view in VIEWS:
        sources = sorted(p for p in (RAW / view).glob("*") if p.suffix.lower() in EXTENSIONS)
        for index, source in enumerate(sources, start=1):
            image = read_oriented(source)
            if image is None:
                print(f"skip unreadable {source}", file=sys.stderr)
                continue
            scale = min(1.0, MAX_SIDE / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(PREPARED / f"{view}_{index:03d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            written += 1
    print(f"prepared {written} images into {PREPARED}")


if __name__ == "__main__":
    main()
