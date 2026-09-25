import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from assets import ASSETS

from .dataset import MEAN, RENDER_HEIGHT, RENDER_WIDTH, SIZE, STD, heatmaps, photometric, square_affine

RENDERS = ASSETS / "synth/renders"
KEYPOINT_COUNT = 8


def render_names(root: Path = RENDERS) -> tuple[list[str], list[str]]:
    names = sorted(path.stem for path in root.glob("*.json"))
    return ([name for name in names if int(name) % 20],
            [name for name in names if int(name) % 20 == 0])


@lru_cache(maxsize=12)
def read_render(root: str, name: str):
    path = Path(root)
    label = json.loads((path / f"{name}.json").read_text())
    image = cv2.imread(str(path / f"{name}.jpg"), cv2.IMREAD_COLOR)
    colours = cv2.imread(str(path / f"{name}_mask.png"), cv2.IMREAD_COLOR)
    if image is None or colours is None:
        raise FileNotFoundError(f"missing render image or mask for {name} under {path}")
    left = colours[..., 2] > 127
    right = colours[..., 1] > 127
    return image, {"left": left, "right": right}, left | right, label


def _bounds(mask: np.ndarray) -> tuple[np.ndarray, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    low = np.array([xs.min(), ys.min()], np.float32)
    high = np.array([xs.max(), ys.max()], np.float32)
    center = (low + high) / 2
    side = max(float((high - low).max()), 24.0)
    return center, side


def _positive_crop(image, mask, points, visible, right, train, rng):
    box = _bounds(mask)
    if box is None:
        return None
    center, side = box

    if train:
        center = center + rng.uniform(-0.15, 0.15, 2) * side
        scale = rng.uniform(1.15, 1.8)
        angle = rng.uniform(-180, 180)
    else:
        scale, angle = 1.8, 0.0

    affine = square_affine(center, side * scale, angle)
    return _make_item(image, mask, points, visible, right, affine, train, rng,
                      mode="positive")


def _empty_crop(image, all_masks, side_masks, right, train, rng):
    boxes = [box for box in (_bounds(side_masks["left"]), _bounds(side_masks["right"])) if box]
    if not boxes:
        return None

    width, height = image.shape[1], image.shape[0]
    for _ in range(40):
        center, foot_side = boxes[int(rng.integers(0, len(boxes)))]
        side = min(foot_side * rng.uniform(1.15, 1.8), min(width, height) * 0.9)
        center = np.array([rng.uniform(side / 2, width - side / 2),
                           rng.uniform(side / 2, height - side / 2)], np.float32)
        angle = rng.uniform(-180, 180) if train else 0.0
        affine = square_affine(center, side, angle)
        crop_mask = cv2.warpAffine(all_masks.astype(np.uint8), affine, (SIZE, SIZE),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if not crop_mask.any():
            return _make_item(image, np.zeros_like(all_masks), np.zeros((KEYPOINT_COUNT, 2), np.float32),
                              np.zeros(KEYPOINT_COUNT, bool), right, affine, train, rng,
                              mode="empty")
    return None


def _make_item(image, mask, points, visible, right, affine, train, rng, mode):
    crop_image = cv2.warpAffine(image, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    crop_mask = cv2.warpAffine(mask.astype(np.uint8), affine, (SIZE, SIZE), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT) > 0
    crop_points = points @ affine[:, :2].T + affine[:, 2]
    in_crop = ((crop_points >= 0) & (crop_points < SIZE)).all(1)
    crop_visible = visible & in_crop

    if train and rng.random() < 0.5:
        crop_image = crop_image[:, ::-1].copy()
        crop_mask = crop_mask[:, ::-1].copy()
        crop_points[:, 0] = SIZE - 1 - crop_points[:, 0]
        right = not right
    if train:
        crop_image = photometric(crop_image, rng)

    tensor = ((crop_image[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    target_heatmaps = heatmaps(crop_points)
    target_heatmaps[~crop_visible] = 0
    return {
        "image": torch.from_numpy(tensor.copy()),
        "mask": torch.from_numpy(crop_mask[None].astype(np.float32)),
        "heatmaps": torch.from_numpy(target_heatmaps),
        "points": torch.from_numpy(crop_points.astype(np.float32)),
        "visible": torch.from_numpy(crop_visible.astype(np.float32)),
        "right": torch.tensor([float(right)]),
        "side_valid": torch.tensor([float(mode == "positive")]),
        "is_positive": torch.tensor([float(mode == "positive")]),
    }


class RenderFootMasks(Dataset):
    """Per-foot crops from the owned renders; hidden landmarks never define the crop."""

    def __init__(self, root: Path, names: list[str], train: bool, negatives: bool):
        self.root = root
        self.names = names
        self.train = train
        self.items = [(name, side) for name in names for side in ("left", "right")]
        self.negative_count = len(self.items) // 5 if negatives else 0

    def __len__(self) -> int:
        return len(self.items) + self.negative_count

    def __getitem__(self, index: int):
        if index < len(self.items):
            name, side = self.items[index]
            image, masks, _, label = read_render(str(self.root), name)
            foot = label["feet"][side]
            points = np.asarray(foot["keypoints"], np.float32)
            visible = np.asarray(foot["visible"], bool)
            visible &= ((points >= 0) & (points < [RENDER_WIDTH, RENDER_HEIGHT])).all(1)
            rng = np.random.default_rng() if self.train else np.random.default_rng(int(name))
            item = _positive_crop(image, masks[side], points, visible, side == "right", self.train, rng)
            if item is None:
                item = _empty_crop(image, masks["left"] | masks["right"], masks,
                                   side == "right", self.train, rng)
            if item is None:
                affine = square_affine(np.array([RENDER_WIDTH / 2, RENDER_HEIGHT / 2], np.float32),
                                       min(RENDER_WIDTH, RENDER_HEIGHT) / 2, 0)
                item = _make_item(image, np.zeros_like(masks[side]), np.zeros_like(points),
                                  np.zeros_like(visible), side == "right", affine, self.train, rng,
                                  mode="empty")
            metadata = (label["mode"], "sock" if label["sock"] else "bare",
                        "trousers" if label["trousers"] else "no_trousers")
            return {**item, "mode": metadata[0], "material": metadata[1],
                    "clothing": metadata[2]}

        name = self.names[(index - len(self.items)) % len(self.names)]
        image, masks, all_masks, label = read_render(str(self.root), name)
        side = bool((index - len(self.items)) % 2)
        seed = int(name) * 1009 + index
        rng = np.random.default_rng(seed)
        item = _empty_crop(image, all_masks, masks, side, self.train, rng)
        if item is None:
            fallback = (index - len(self.items)) % len(self.items)
            return self[fallback]
        return {**item, "mode": label["mode"],
                "material": "sock" if label["sock"] else "bare",
                "clothing": "trousers" if label["trousers"] else "no_trousers"}
