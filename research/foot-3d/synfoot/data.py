import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1] / "data/synfoot/V1"
WIDTH, HEIGHT = 480, 640
KEYPOINTS = ("big toe", "2nd toe", "3rd toe", "4th toe", "little toe", "heel", "outer extrema", "inner extrema")
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class Camera:
    """OpenCV pinhole camera, world → camera: x_cam = R @ x_world + t."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @classmethod
    def from_label(cls, label: dict) -> "Camera":
        # Blender camera: XYZ Euler, looks down −Z with +Y up, FOV spans the long (640 px) side.
        world_from_camera = Rotation.from_euler("xyz", label["euler"]).as_matrix()
        R = BLENDER_TO_OPENCV @ world_from_camera.T
        focal = max(WIDTH, HEIGHT) / 2 / np.tan(np.radians(label["fov"]) / 2)
        K = np.array([[focal, 0, WIDTH / 2], [0, focal, HEIGHT / 2], [0, 0, 1]])
        return cls(K, R, -R @ np.array(label["pos"]))

    def project(self, points: np.ndarray) -> np.ndarray:
        rvec = cv2.Rodrigues(self.R)[0]
        return cv2.projectPoints(np.asarray(points, float), rvec, self.t, self.K, None)[0][:, 0]


@dataclass(frozen=True)
class Sample:
    """One SynFoot V1 render. All feet are left feet; keypoints are projected mesh vertices, occluded or not."""

    id: str
    foot: str
    keypoints: np.ndarray
    camera: Camera

    @classmethod
    def load(cls, id: str) -> "Sample":
        label = json.loads((ROOT / "labels" / f"{id}.json").read_text())
        keypoints = np.array([label["keypoints"][name] for name in KEYPOINTS], float)
        return cls(id, label["foot"], keypoints, Camera.from_label(label["camera"]))

    @cached_property
    def rgb(self) -> np.ndarray:
        return cv2.imread(str(ROOT / "rgb" / f"{self.id}.png"), cv2.IMREAD_COLOR)

    @cached_property
    def mask(self) -> np.ndarray:
        return cv2.imread(str(ROOT / "mask" / f"{self.id}.png"), cv2.IMREAD_GRAYSCALE) > 127

    @cached_property
    def normals(self) -> np.ndarray:
        """Unit normals decoded from the PNG, zero off the foot."""
        bgr = cv2.imread(str(ROOT / "normals" / f"{self.id}.png"), cv2.IMREAD_COLOR)
        normals = bgr[..., ::-1].astype(np.float32) / 127.5 - 1
        return np.where(self.mask[..., None], normals, 0)


def ids() -> list[str]:
    return sorted(path.stem for path in (ROOT / "labels").glob("*.json"))
