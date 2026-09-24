import json
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from spike.pose import TEMPLATE_KEYPOINTS

from .data import KEYPOINTS, ROOT, Sample, ids

CACHE = ROOT.parent / "foot_keypoints.json"
VIEWS_PER_FOOT = 400
FIND_KEYPOINTS = np.array([TEMPLATE_KEYPOINTS[name] for name in KEYPOINTS])


@dataclass(frozen=True)
class Foot:
    """A scanned foot's 8 keypoints in the scene frame, and the similarity that maps the FIND template onto them."""

    keypoints: np.ndarray
    scale: float
    R: np.ndarray
    t: np.ndarray

    def from_find(self, points: np.ndarray) -> np.ndarray:
        return self.scale * points @ self.R.T + self.t


@dataclass(frozen=True)
class Pose:
    """Camera-frame pose of the scanned foot (scene frame → camera), recovered from the sample's keypoints."""

    R: np.ndarray
    t: np.ndarray
    error_px: float


def _triangulate(samples: list[Sample]) -> np.ndarray:
    points = []
    for k in range(len(KEYPOINTS)):
        rows = []
        for sample in samples:
            P = sample.camera.K @ np.hstack([sample.camera.R, sample.camera.t[:, None]])
            u, v = sample.keypoints[k]
            rows += [u * P[2] - P[0], v * P[2] - P[1]]
        homogeneous = np.linalg.svd(np.array(rows))[2][-1]
        points.append(homogeneous[:3] / homogeneous[3])
    return np.array(points)


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: target ≈ s · R · source + t."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    U, S, Vt = np.linalg.svd(b.T @ a / len(source))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_t - s * R @ mu_s


def build_feet(stride: int = 5) -> dict[str, Foot]:
    by_foot = defaultdict(list)
    for id in ids()[::stride]:
        sample = Sample.load(id)
        if len(by_foot[sample.foot]) < VIEWS_PER_FOOT:
            by_foot[sample.foot].append(sample)
    feet = {}
    for name, samples in sorted(by_foot.items()):
        keypoints = _triangulate(samples)
        feet[name] = Foot(keypoints, *_similarity(FIND_KEYPOINTS, keypoints))
    CACHE.write_text(json.dumps({
        name: {"keypoints": foot.keypoints.tolist(), "scale": foot.scale, "R": foot.R.tolist(), "t": foot.t.tolist()}
        for name, foot in feet.items()
    }, indent=1))
    return feet


def load_feet() -> dict[str, Foot]:
    if not CACHE.exists():
        return build_feet()
    raw = json.loads(CACHE.read_text())
    return {name: Foot(np.array(f["keypoints"]), f["scale"], np.array(f["R"]), np.array(f["t"])) for name, f in raw.items()}


def solve(sample: Sample, foot: Foot) -> Pose:
    # The label camera is off by a small per-sample foot offset (~1.5 cm); the keypoints are exact, so refit from them.
    camera = sample.camera
    _, rvec, tvec = cv2.solvePnP(
        foot.keypoints, sample.keypoints, camera.K, None,
        cv2.Rodrigues(camera.R)[0], camera.t.reshape(3, 1).copy(), useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    projected = cv2.projectPoints(foot.keypoints, rvec, tvec, camera.K, None)[0][:, 0]
    error = float(np.median(np.linalg.norm(projected - sample.keypoints, axis=1)))
    return Pose(cv2.Rodrigues(rvec)[0], tvec[:, 0], error)


def project(points: np.ndarray, pose: Pose, K: np.ndarray) -> np.ndarray:
    return cv2.projectPoints(np.asarray(points, float), cv2.Rodrigues(pose.R)[0], pose.t, K, None)[0][:, 0]
