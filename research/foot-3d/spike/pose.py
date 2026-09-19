import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .toc import TocPrediction

FIND = Path(__file__).resolve().parents[1] / "vendor/FOCUS/data/find"
SAMPLES = 600
MAX_TOC_STD = 0.05
RANSAC_PX = 0.01


def _template():
    verts = np.array([list(map(float, l.split()[1:4])) for l in open(FIND / "template.obj") if l.startswith("v ")])
    rows = {r[0]: r[1:] for r in csv.reader(open(FIND / "keypoints.csv"))}
    keypoints = {name: verts[int(i)] for name, i in zip(rows["model"], rows["FIND"])}
    return verts, verts.min(0), verts.max(0), keypoints


TEMPLATE, TEMPLATE_MIN, TEMPLATE_MAX, TEMPLATE_KEYPOINTS = _template()


@dataclass
class FootPose:
    rvec: np.ndarray
    tvec: np.ndarray
    mirrored: bool
    inliers: int
    samples: int
    error_px: float


def intrinsics(width: int, height: int, long_side_fov_deg: float = 65.0) -> np.ndarray:
    focal = max(width, height) / 2 / np.tan(np.radians(long_side_fov_deg) / 2)
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])


def mirror(points: np.ndarray, mirrored: bool) -> np.ndarray:
    return points * np.array([1, -1, 1]) if mirrored else points


def correspondences(prediction: TocPrediction, crop_to_frame: np.ndarray, rng: np.random.Generator):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(prediction.mask.astype(np.uint8))
    if count < 2:
        return None, None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    confident = (labels == largest) & (prediction.toc_std.max(axis=-1) < MAX_TOC_STD)
    ys, xs = np.nonzero(confident)
    if len(xs) < 20:
        return None, None
    pick = rng.choice(len(xs), size=min(SAMPLES, len(xs)), replace=False)
    xs, ys = xs[pick], ys[pick]
    net = np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))], axis=1)
    crop = net @ prediction.to_crop.T
    frame = np.concatenate([crop, np.ones((len(crop), 1))], axis=1) @ crop_to_frame.T
    points3d = prediction.toc[ys, xs] * (TEMPLATE_MAX - TEMPLATE_MIN) + TEMPLATE_MIN
    return frame.astype(np.float64), points3d.astype(np.float64)


def solve(points2d: np.ndarray, points3d: np.ndarray, camera: np.ndarray, frame_size: tuple[int, int]) -> FootPose | None:
    best = None
    threshold = RANSAC_PX * max(frame_size)
    for mirrored in (False, True):
        object_points = mirror(points3d, mirrored)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, points2d, camera, None,
            reprojectionError=threshold, iterationsCount=300, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inliers is None or len(inliers) < 12:
            continue
        idx = inliers[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(object_points[idx], points2d[idx], camera, None, rvec, tvec)
        projected, _ = cv2.projectPoints(object_points[idx], rvec, tvec, camera, None)
        error = float(np.median(np.linalg.norm(projected[:, 0] - points2d[idx], axis=1)))
        pose = FootPose(rvec, tvec, mirrored, len(idx), len(points2d), error)
        if tvec[2, 0] > 0 and (best is None or (pose.inliers, -pose.error_px) > (best.inliers, -best.error_px)):
            best = pose
    return best


def draw(frame: np.ndarray, pose: FootPose, camera: np.ndarray, color=(0, 255, 255)) -> None:
    rng = np.random.default_rng(0)
    verts = mirror(TEMPLATE[rng.choice(len(TEMPLATE), 1500, replace=False)], pose.mirrored)
    projected, _ = cv2.projectPoints(verts, pose.rvec, pose.tvec, camera, None)
    for x, y in projected[:, 0]:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    for name, point in TEMPLATE_KEYPOINTS.items():
        if name in ("big toe", "little toe", "heel", "lower heel"):
            projected_point, _ = cv2.projectPoints(mirror(point[None], pose.mirrored), pose.rvec, pose.tvec, camera, None)
            x, y = projected_point[0, 0]
            cv2.circle(frame, (int(x), int(y)), 9, (0, 0, 255), -1)
            cv2.putText(frame, name, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
