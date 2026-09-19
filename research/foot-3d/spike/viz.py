import sys
from pathlib import Path

import cv2
import numpy as np

from .toc import letterbox, load_model, predict

OUT = Path(__file__).resolve().parents[1] / "results/viz"


def overlay(image: np.ndarray, prediction) -> np.ndarray:
    padded, _ = letterbox(image)
    colored = (prediction.toc[..., ::-1] * 255).astype(np.uint8)
    blended = padded.copy()
    blended[prediction.mask] = (0.25 * padded[prediction.mask] + 0.75 * colored[prediction.mask]).astype(np.uint8)
    cv2.putText(blended, prediction.footedness, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return np.hstack([padded, blended])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model("mps")
    for path in sys.argv[1:]:
        image = cv2.imread(path)
        cv2.imwrite(str(OUT / Path(path).name), overlay(image, predict(model, image, "mps")))
    print(f"wrote {len(sys.argv) - 1} overlays to {OUT}")
