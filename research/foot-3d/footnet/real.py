"""Run FootNet on real frames: RTMPose finds each foot, the crop goes to FootNet, PnP puts the FIND template on it.

    uv run python -m footnet.real [--checkpoint data/footnet/best.pt] [--per-video 2] [images…]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from rtmlib import RTMPose

from spike.fit_scene import FEET, MIN_SCORE, RTMPOSE_M
from spike.pose import TEMPLATE, intrinsics
from synfoot.foot_pose import FIND_KEYPOINTS

from .dataset import MEAN, SIZE, STD, square_affine
from .model import FootNet, decode_points

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT.parent / "foot-tracking/data/prepared"
OUT = ROOT / "results/footnet/real"
CROP_CONTEXT = 1.8
# About the 5th percentile of peak scores on held-out SynFoot after training: below it a point is a guess.
MIN_POINT_SCORE = 0.3
COLORS = {"left": (0, 255, 255), "right": (255, 0, 255)}


def crop_from_body(points: np.ndarray) -> np.ndarray:
    low, high = points.min(0), points.max(0)
    return square_affine((low + high) / 2, max(float((high - low).max()), 30.0) * CROP_CONTEXT, 0.0)


def run_foot(model, frame: np.ndarray, affine: np.ndarray):
    crop = cv2.warpAffine(frame, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    tensor = torch.from_numpy(((crop[..., ::-1].astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
    with torch.no_grad():
        mask, heatmaps, right = model(tensor[None].to("mps"))
    points, scores = decode_points(heatmaps)
    inverse = cv2.invertAffineTransform(affine)
    points = points[0].cpu().numpy() @ inverse[:, :2].T + inverse[:, 2]
    mask = cv2.warpAffine((torch.sigmoid(mask[0, 0]).cpu().numpy() > 0.5).astype(np.uint8), inverse,
                          (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST)
    return points, scores[0].cpu().numpy(), mask > 0, torch.sigmoid(right).item()


def fit_template(points: np.ndarray, scores: np.ndarray, right: bool, camera: np.ndarray):
    keep = scores >= MIN_POINT_SCORE
    if keep.sum() < 5:
        return None
    template = FIND_KEYPOINTS * ([1, -1, 1] if right else 1)
    ok, rvec, tvec = cv2.solvePnP(template[keep], points[keep], camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or tvec[2, 0] <= 0:
        return None
    projected = cv2.projectPoints(template[keep], rvec, tvec, camera, None)[0][:, 0]
    return rvec, tvec, float(np.median(np.linalg.norm(projected - points[keep], axis=1)))


def draw(canvas, points, scores, mask, right_probability, pose, camera, color):
    tint = np.zeros_like(canvas)
    tint[mask] = color
    canvas[:] = cv2.addWeighted(canvas, 1, tint, 0.35, 0)
    right = right_probability > 0.5
    if pose is not None:
        rvec, tvec, _ = pose
        template = TEMPLATE[::60] * ([1, -1, 1] if right else 1)
        for x, y in cv2.projectPoints(template, rvec, tvec, camera, None)[0][:, 0].astype(int):
            cv2.circle(canvas, (x, y), 1, (255, 255, 255), -1)
    for (x, y), score in zip(points.astype(int), scores):
        cv2.circle(canvas, (x, y), 5, color, -1 if score >= MIN_POINT_SCORE else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "data/footnet/best.pt"))
    parser.add_argument("--per-video", type=int, default=2)
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    images = [Path(p) for p in args.images]
    if not images:
        by_video = {}
        for path in sorted(FRAMES.glob("*.jpg")):
            by_video.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
        images = [p for paths in by_video.values() for p in paths[:: max(1, len(paths) // args.per_video)][:args.per_video]]

    model = FootNet(pretrained=False).to("mps").eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="mps"))
    body = RTMPose(RTMPOSE_M, model_input_size=(192, 256), backend="onnxruntime", device="cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in images:
        frame = cv2.imread(str(path))
        height, width = frame.shape[:2]
        camera = intrinsics(width, height)
        keypoints, body_scores = body(frame, bboxes=[[0, 0, width, height]])
        canvas, report = frame.copy(), []
        for side, indices in FEET.items():
            seen = [keypoints[0][i] for i in indices if body_scores[0][i] >= MIN_SCORE]
            if len(seen) < 2:
                report.append(f"{side}: no foot")
                continue
            points, scores, mask, right = run_foot(model, frame, crop_from_body(np.array(seen)))
            pose = fit_template(points, scores, right > 0.5, camera)
            draw(canvas, points, scores, mask, right, pose, camera, COLORS[side])
            report.append(f"{side}: model says {'right' if right > 0.5 else 'left'} ({right:.2f}), "
                          f"{(scores >= MIN_POINT_SCORE).sum()}/8 points, "
                          + (f"PnP {pose[2]:.1f}px" if pose else "no PnP"))
        cv2.imwrite(str(OUT / path.name), canvas)
        print(f"{path.name}: " + "; ".join(report))


if __name__ == "__main__":
    main()
