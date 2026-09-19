import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor/FOCUS"), str(ROOT / "vendor/FOCUS/FOCUS/toc_prediction/DSINE")]

from FOCUS.toc_prediction.model import FootPredictorModel  # noqa: E402

MODEL = ROOT / "data/toc_model/densedepth_toc_predictor.pth"
INPUT_W, INPUT_H = 480, 640
MASK_THRESHOLD = 0.5


@dataclass
class TocPrediction:
    mask: np.ndarray
    toc: np.ndarray
    toc_std: np.ndarray
    footedness: str
    to_crop: np.ndarray


def load_model(device: str) -> FootPredictorModel:
    model = FootPredictorModel.load(MODEL, device=device)
    model.eval()
    return model


def letterbox(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(INPUT_W / width, INPUT_H / height)
    new_w, new_h = int(width * scale), int(height * scale)
    pad_x, pad_y = (INPUT_W - new_w) // 2, (INPUT_H - new_h) // 2
    out = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = cv2.resize(image, (new_w, new_h))
    # Maps network pixels back to input pixels.
    inverse = np.array([[1 / scale, 0, -pad_x / scale], [0, 1 / scale, -pad_y / scale]])
    return out, inverse


def predict(model: FootPredictorModel, crop_bgr: np.ndarray, device: str) -> TocPrediction:
    padded, inverse = letterbox(crop_bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        out = model(tensor)
    mask = out["hm"][0, 0].cpu().numpy() > MASK_THRESHOLD
    toc = out["TOC"][0].permute(1, 2, 0).cpu().numpy()
    toc_std = np.exp(out["TOC_unc_log_var"][0].permute(1, 2, 0).cpu().numpy()) ** 0.5
    footedness = "LR"[int(out["footedness"][0].argmax())]
    return TocPrediction(mask, toc, toc_std, footedness, inverse)
