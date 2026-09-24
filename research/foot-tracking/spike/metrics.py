from dataclasses import dataclass, field

import numpy as np

from .common import KEYPOINTS, Foot

PCK_THRESHOLDS = (0.05, 0.10)
MATCH_RADIUS = 0.5


@dataclass
class FootResult:
    view: str
    detected: bool
    foot_length_px: float
    errors: dict[str, float] = field(default_factory=dict)
    axis_error_deg: float | None = None


def labeled_foot(raw: dict) -> Foot:
    return Foot(points={name: (tuple(raw[name]) if raw.get(name) is not None else None) for name in KEYPOINTS})


def foot_length(foot: Foot) -> float | None:
    toe, heel = foot.get("big_toe"), foot.get("heel")
    if toe is None or heel is None:
        return None
    return float(np.linalg.norm(toe - heel))


def axis_angle_deg(a: Foot, b: Foot) -> float | None:
    points = [a.get("big_toe"), a.get("heel"), b.get("big_toe"), b.get("heel")]
    if any(p is None for p in points):
        return None
    va, vb = points[0] - points[1], points[2] - points[3]
    cosine = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def match(truth: list[Foot], predicted: list[Foot], view: str) -> list[FootResult]:
    remaining = [p for p in predicted if p.get("big_toe") is not None]
    results = []
    for gt in truth:
        length = foot_length(gt)
        toe = gt.get("big_toe")
        if length is None or toe is None:
            continue
        best = min(remaining, key=lambda p: np.linalg.norm(p.get("big_toe") - toe), default=None)
        if best is None or np.linalg.norm(best.get("big_toe") - toe) > MATCH_RADIUS * length:
            results.append(FootResult(view=view, detected=False, foot_length_px=length))
            continue
        remaining.remove(best)
        errors = {
            name: float(np.linalg.norm(best.get(name) - gt.get(name))) / length
            for name in KEYPOINTS
            if gt.get(name) is not None and best.get(name) is not None
        }
        results.append(
            FootResult(view=view, detected=True, foot_length_px=length, errors=errors, axis_error_deg=axis_angle_deg(gt, best))
        )
    return results


def summarize(results: list[FootResult]) -> dict:
    total = len(results)
    detected = [r for r in results if r.detected]
    summary = {"feet": total, "detection_rate": len(detected) / total if total else 0.0}
    for threshold in PCK_THRESHOLDS:
        for name in KEYPOINTS:
            values = [r.errors[name] for r in detected if name in r.errors]
            if not total:
                summary[f"pck@{threshold:.2f}_{name}"] = None
            elif detected and not values:
                summary[f"pck@{threshold:.2f}_{name}"] = None
            else:
                summary[f"pck@{threshold:.2f}_{name}"] = sum(v <= threshold for v in values) / total
    angles = [r.axis_error_deg for r in detected if r.axis_error_deg is not None]
    summary["axis_error_median_deg"] = float(np.median(angles)) if angles else None
    return summary
