from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KEYPOINTS = ("big_toe", "small_toe", "heel", "ankle")
VIEWS = ("mirror-full", "mirror-lower", "third", "top", "close")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PREPARED = DATA / "prepared"
MASKS = DATA / "masks"
LABELS = DATA / "labels.json"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

Point = tuple[float, float]


@dataclass
class Foot:
    points: dict[str, Point | None] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        point = self.points.get(name)
        return None if point is None else np.asarray(point, dtype=float)


def view_of(stem: str) -> str:
    prefix = stem.split("_", 1)[0]
    return prefix if prefix in VIEWS else "unknown"


def prepared_images() -> list[Path]:
    return sorted(p for p in PREPARED.glob("*.jpg"))
