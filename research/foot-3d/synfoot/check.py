"""Check SynFoot V1: camera convention, per-sample foot pose, FIND template fit, keypoint visibility. Writes a contact sheet."""

from pathlib import Path

import cv2
import numpy as np

from spike.pose import TEMPLATE

from .data import KEYPOINTS, Sample, ids
from .foot_pose import FIND_KEYPOINTS, load_feet, project, solve

OUT = Path(__file__).resolve().parents[1] / "results/synfoot"
COLORS = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 0, 0), (255, 255, 255)]
STATS_SAMPLES = 2000
SHEET_SAMPLES = 16


def tilt_deg(sample: Sample) -> float:
    down = np.array([0.0, 0.0, -1.0])
    view = sample.camera.R.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(np.arccos(np.clip(view @ down, -1, 1))))


def template_error_px(sample: Sample) -> float:
    """PnP on the generic FIND keypoints, as the app will do: how much the per-person foot shape costs."""
    K = sample.camera.K
    _, rvec, tvec = cv2.solvePnP(FIND_KEYPOINTS, sample.keypoints, K, None, flags=cv2.SOLVEPNP_SQPNP)
    projected = cv2.projectPoints(FIND_KEYPOINTS, rvec, tvec, K, None)[0][:, 0]
    return float(np.median(np.linalg.norm(projected - sample.keypoints, axis=1)))


def off_mask(sample: Sample) -> np.ndarray:
    """Keypoints that land off the silhouette: certainly hidden. On-silhouette ones may still be hidden."""
    mask = cv2.dilate(sample.mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    xy = np.round(sample.keypoints).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    hit = np.zeros(len(xy), bool)
    hit[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return ~hit


def draw(sample: Sample, feet) -> np.ndarray:
    foot = feet[sample.foot]
    pose = solve(sample, foot)
    image = sample.rgb.copy()
    template = project(foot.from_find(TEMPLATE[::40]), pose, sample.camera.K)
    for x, y in template.astype(int):
        cv2.circle(image, (x, y), 1, (200, 200, 200), -1)
    contours, _ = cv2.findContours(sample.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    for (x, y), color, hidden in zip(sample.keypoints.astype(int), COLORS, off_mask(sample)):
        cv2.circle(image, (x, y), 6, color, 1 if hidden else -1)
    cv2.putText(image, f"{sample.id} {sample.foot} tilt {tilt_deg(sample):.0f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    all_ids = ids()
    feet = load_feet()
    rng = np.random.default_rng(0)

    picked = [Sample.load(id) for id in rng.choice(all_ids, STATS_SAMPLES, replace=False)]
    exact = np.array([solve(s, feet[s.foot]).error_px for s in picked])
    generic = np.array([template_error_px(s) for s in picked])
    tilts = np.array([tilt_deg(s) for s in picked])
    hidden = np.array([off_mask(s) for s in picked])
    coverage = np.array([s.mask.mean() for s in picked])

    print(f"{len(all_ids)} samples, {len(feet)} feet (all left): {', '.join(feet)}")
    print("foot length (heel → big toe), cm:", {n: round(100 * float(np.linalg.norm(f.keypoints[0] - f.keypoints[5])), 1) for n, f in feet.items()})
    print("FIND → foot scale:", {n: round(float(f.scale), 3) for n, f in feet.items()})
    print(f"camera tilt from straight down, deg (p10/50/90/max): {np.percentile(tilts, [10, 50, 90, 100]).round(0)}")
    print(f"mask coverage of frame (p10/50/90): {np.percentile(coverage, [10, 50, 90]).round(2)}")
    print(f"PnP on the scanned foot's keypoints, median px (p50/p99): {np.percentile(exact, [50, 99]).round(2)}")
    print(f"PnP on generic FIND keypoints, median px (p50/p90): {np.percentile(generic, [50, 90]).round(1)}")
    print("keypoints off the silhouette (certainly hidden), %:",
          {name: round(100 * float(h), 1) for name, h in zip(KEYPOINTS, hidden.mean(0))})

    sheet = [draw(Sample.load(id), feet) for id in rng.choice(all_ids, SHEET_SAMPLES, replace=False)]
    rows = [np.hstack(sheet[i:i + 4]) for i in range(0, SHEET_SAMPLES, 4)]
    cv2.imwrite(str(OUT / "sheet.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))
    print(f"wrote {OUT / 'sheet.jpg'}")
