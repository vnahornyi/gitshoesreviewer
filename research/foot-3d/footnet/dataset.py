import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from synfoot.data import Sample

SIZE = 256
SIGMA = 3.0
RENDER_WIDTH, RENDER_HEIGHT = 720, 960
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = np.mgrid[0:SIZE, 0:SIZE][::-1].astype(np.float32)


def crop_transform(keypoints: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """Square crop around the foot's keypoints, rotated and jittered when training. Returns a 2×3 affine."""
    low, high = keypoints.min(0), keypoints.max(0)
    center, side = (low + high) / 2, float((high - low).max())
    if rng is None:
        return square_affine(center, side * 1.3, 0.0)
    center = center + rng.uniform(-0.15, 0.15, 2) * side
    return square_affine(center, side * rng.uniform(1.15, 1.7), rng.uniform(-180, 180))


def square_affine(center: np.ndarray, side: float, angle: float) -> np.ndarray:
    scale = SIZE / side
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    matrix[:, 2] += SIZE / 2 - center
    return matrix


def heatmaps(points: np.ndarray) -> np.ndarray:
    d2 = ((GRID[None] - points[:, :, None, None]) ** 2).sum(1)
    return np.exp(-d2 / (2 * SIGMA**2)).astype(np.float32)


def sock(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Repaint the foot as a sock: a random colour, optional knit stripes, the original shading kept."""
    shade = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    shade = cv2.GaussianBlur(shade, (0, 0), rng.uniform(0.8, 2.5))
    shade = (shade / max(float(np.percentile(shade[mask], 90)), 1e-3)).clip(0, 1.3)
    color = rng.uniform(0, 255, 3).astype(np.float32)
    if rng.random() < 0.4:
        second = rng.uniform(0, 255, 3).astype(np.float32)
        ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]].astype(np.float32)
        angle = rng.uniform(0, np.pi)
        stripes = np.sin(2 * np.pi * (xs * np.cos(angle) + ys * np.sin(angle)) / rng.uniform(6, 30)) > 0
        cloth = np.where(stripes[..., None], color, second)
    else:
        cloth = np.broadcast_to(color, image.shape)
    knit = rng.normal(1, 0.06, image.shape[:2]).astype(np.float32)[..., None]
    painted = (cloth * shade[..., None] * knit).clip(0, 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
    return (soft * painted + (1 - soft) * image).astype(np.uint8)


def photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.5, 1.4) + rng.uniform(-20, 20)
    image = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        length = int(rng.integers(3, 15))
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2] = 1 / length
        rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1)
        image = cv2.filter2D(image, -1, cv2.warpAffine(kernel, rotation, (length, length)))
    elif rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 2))
    noise = rng.normal(0, rng.uniform(0, 8), image.shape)
    return (image + noise).clip(0, 255).astype(np.uint8)


def make_item(image, mask, keypoints, right: bool, rng: np.random.Generator | None, sock_rate: float) -> dict:
    """Mirror to swap sides when training, repaint as a sock, crop around the keypoints, augment, build targets."""
    if rng is not None and rng.random() < 0.5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        keypoints = keypoints * [-1, 1] + [image.shape[1] - 1, 0]
        right = not right
    if rng is not None and mask.any() and rng.random() < sock_rate:
        image = sock(image, mask, rng)

    affine = crop_transform(keypoints, rng)
    image = cv2.warpAffine(image, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask.astype(np.uint8), affine, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE) > 0
    points = keypoints @ affine[:, :2].T + affine[:, 2]
    if rng is not None:
        image = photometric(image, rng)

    tensor = ((image[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return {
        "image": torch.from_numpy(tensor.copy()),
        "mask": torch.from_numpy(mask[None].astype(np.float32)),
        "heatmaps": torch.from_numpy(heatmaps(points)),
        "points": torch.from_numpy(points.astype(np.float32)),
        "right": torch.tensor([float(right)]),
    }


class SynFootCrops(Dataset):
    """SynFoot V1 foot crops (all left feet, mirrored into right ones half the time)."""

    def __init__(self, ids: list[str], train: bool, sock_rate: float = 0.5):
        self.ids = ids
        self.train = train
        self.sock_rate = sock_rate

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sample = Sample.load(self.ids[index])
        image, mask, keypoints = sample.rgb, sample.mask, sample.keypoints.copy()
        if not self.train and index % 2 == 1:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            keypoints = keypoints * [-1, 1] + [image.shape[1] - 1, 0]
            return make_item(image, mask, keypoints, True, None, 0)
        return make_item(image, mask, keypoints, False, np.random.default_rng() if self.train else None, self.sock_rate)


class RenderCrops(Dataset):
    """Crops of our own renders (render/render.py): one item per foot with enough of it in frame and visible."""

    def __init__(self, root: Path, train: bool, names: list[str] | None = None):
        self.root = root
        self.train = train
        self.items = []
        for path in sorted(root.glob("*.json")) if names is None else [root / f"{n}.json" for n in names]:
            label = json.loads(path.read_text())
            for side, foot in label["feet"].items():
                points = np.array(foot["keypoints"])
                inside = ((points >= 0) & (points < [RENDER_WIDTH, RENDER_HEIGHT])).all(1)
                if inside.sum() >= 6 and sum(foot["visible"]) >= 2 and np.ptp(points, 0).max() >= 24:
                    self.items.append((path.stem, side))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, side = self.items[index]
        label = json.loads((self.root / f"{name}.json").read_text())
        image = cv2.imread(str(self.root / f"{name}.jpg"), cv2.IMREAD_COLOR)
        colours = cv2.imread(str(self.root / f"{name}_mask.png"), cv2.IMREAD_COLOR)
        mask = colours[..., 2 if side == "left" else 1] > 127
        keypoints = np.array(label["feet"][side]["keypoints"], float)
        # The renderer already dresses the feet, so no painted socks on top.
        return make_item(image, mask, keypoints, side == "right", np.random.default_rng() if self.train else None, 0)
