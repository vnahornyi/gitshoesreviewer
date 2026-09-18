import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .common import MASKS, MODELS, Foot

Predictor = Callable[[np.ndarray, str], list[Foot]]

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
RTMW_POSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip"
)

MEDIAPIPE_FEET = (
    {"ankle": 27, "heel": 29, "big_toe": 31},
    {"ankle": 28, "heel": 30, "big_toe": 32},
)
COCO_WHOLEBODY_FEET = (
    {"ankle": 15, "big_toe": 17, "small_toe": 18, "heel": 19},
    {"ankle": 16, "big_toe": 20, "small_toe": 21, "heel": 22},
)


def _download(url: str, name: str) -> Path:
    target = MODELS / name
    if not target.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
    return target


def _feet_from(keypoints: np.ndarray, scores: np.ndarray, mapping) -> list[Foot]:
    feet = []
    for side in mapping:
        foot = Foot()
        for name, index in side.items():
            foot.points[name] = (float(keypoints[index][0]), float(keypoints[index][1]))
            foot.scores[name] = float(scores[index])
        feet.append(foot)
    return feet


def mediapipe_pose() -> Predictor:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_download(MEDIAPIPE_MODEL_URL, "pose_landmarker_full.task"))),
            num_poses=1,
        )
    )

    def predict(image: np.ndarray, _stem: str) -> list[Foot]:
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(rgb)
        if not result.pose_landmarks:
            return []
        height, width = image.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints = np.array([[lm.x * width, lm.y * height] for lm in landmarks])
        scores = np.array([lm.visibility for lm in landmarks])
        return _feet_from(keypoints, scores, MEDIAPIPE_FEET)

    return predict


def rtmw_with_detector() -> Predictor:
    from rtmlib import Wholebody

    model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")

    def predict(image: np.ndarray, _stem: str) -> list[Foot]:
        keypoints, scores = model(image)
        if len(keypoints) == 0:
            return []
        best = int(np.argmax(scores[:, 15:23].mean(axis=1)))
        return _feet_from(keypoints[best], scores[best], COCO_WHOLEBODY_FEET)

    return predict


def rtmw_full_frame() -> Predictor:
    from rtmlib import RTMPose

    model = RTMPose(RTMW_POSE_URL, model_input_size=(192, 256), backend="onnxruntime", device="cpu")

    def predict(image: np.ndarray, _stem: str) -> list[Foot]:
        height, width = image.shape[:2]
        keypoints, scores = model(image, bboxes=[[0, 0, width, height]])
        return _feet_from(keypoints[0], scores[0], COCO_WHOLEBODY_FEET)

    return predict


def _component_foot(pixels: np.ndarray) -> Foot | None:
    if len(pixels) < 200:
        return None
    centered = pixels - pixels.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    axis = vectors[:, -1]
    if axis[1] > 0:
        axis = -axis

    projection = centered @ axis
    toe_t = projection.max()
    length = toe_t - projection.min()
    along = toe_t - projection
    lateral = centered @ np.array([-axis[1], axis[0]])

    bins = 40
    edges = np.linspace(0, length, bins + 1)
    widths = np.zeros(bins)
    for i in range(bins):
        in_bin = (along >= edges[i]) & (along < edges[i + 1])
        if in_bin.any():
            widths[i] = lateral[in_bin].max() - lateral[in_bin].min()

    ball = int(np.argmax(widths[: int(bins * 0.4)]))
    search_from = min(bins - 1, ball + int(bins * 0.15))
    search_to = max(search_from + 1, int(bins * 0.9))
    heel_bin = search_from + int(np.argmin(widths[search_from:search_to]))
    heel_along = (edges[heel_bin] + edges[heel_bin + 1]) / 2

    foot_part = pixels[along <= heel_along]
    if len(foot_part) > 50:
        foot_centered = foot_part - foot_part.mean(axis=0)
        _, foot_vectors = np.linalg.eigh(np.cov(foot_centered.T))
        foot_axis = foot_vectors[:, -1]
        if foot_axis @ axis < 0:
            foot_axis = -foot_axis
        axis = foot_axis
        origin = foot_part.mean(axis=0)
        projection = (foot_part - origin) @ axis
        toe = origin + axis * projection.max()
        heel = origin + axis * projection.min()
    else:
        toe = pixels.mean(axis=0) + axis * toe_t
        heel = toe - axis * heel_along

    foot = Foot()
    foot.points = {"big_toe": tuple(toe), "heel": tuple(heel), "small_toe": None, "ankle": None}
    foot.scores = {"big_toe": 1.0, "heel": 1.0}
    return foot


def geometric_from_mask(mask_dir: Path = MASKS, min_area_fraction: float = 0.01) -> Predictor:
    def predict(image: np.ndarray, stem: str) -> list[Foot]:
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            return []
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        count, components, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
        min_area = min_area_fraction * mask.size
        feet = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(components == label)
            foot = _component_foot(np.stack([xs, ys], axis=1).astype(float))
            if foot:
                feet.append(foot)
        return feet

    return predict


CANDIDATES: dict[str, Callable[[], Predictor]] = {
    "mediapipe": mediapipe_pose,
    "rtmw-det": rtmw_with_detector,
    "rtmw-full": rtmw_full_frame,
    "geometric": geometric_from_mask,
}
