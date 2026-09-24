import sys
from pathlib import Path

import cv2
import numpy as np
from rtmlib import RTMPose

from .pose import correspondences, draw, intrinsics, solve
from .toc import load_model, predict

OUT = Path(__file__).resolve().parents[1] / "results/fit"
RTMPOSE_M = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
)
# Halpe26: ankle, big toe, small toe, heel for each side.
FEET = {"left": (15, 20, 22, 24), "right": (16, 21, 23, 25)}
MIN_SCORE = 0.2
CONTEXT = 2.2


def foot_crop(points: np.ndarray, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = frame_size
    center = (points.min(0) + points.max(0)) / 2
    side = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 40.0) * CONTEXT
    half_w, half_h = side * 0.75 / 2 * 1.2, side / 2 * 1.2
    x0, y0 = int(max(0, center[0] - half_w)), int(max(0, center[1] - half_h))
    x1, y1 = int(min(width, center[0] + half_w)), int(min(height, center[1] + half_h))
    return x0, y0, x1, y1


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model("mps")
    body = RTMPose(RTMPOSE_M, model_input_size=(192, 256), backend="onnxruntime", device="cpu")
    rng = np.random.default_rng(0)
    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        height, width = frame.shape[:2]
        camera = intrinsics(width, height)
        keypoints, scores = body(frame, bboxes=[[0, 0, width, height]])
        canvas = frame.copy()
        report = []
        for side, indices in FEET.items():
            seen = [keypoints[0][i] for i in indices if scores[0][i] >= MIN_SCORE]
            if len(seen) < 2:
                report.append(f"{side}: not found")
                continue
            x0, y0, x1, y1 = foot_crop(np.array(seen), (width, height))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 128, 0), 2)
            prediction = predict(model, frame[y0:y1, x0:x1], "mps")
            to_frame = np.array([[1.0, 0, x0], [0, 1.0, y0]])
            points2d, points3d = correspondences(prediction, to_frame, rng)
            if points2d is None:
                report.append(f"{side}: no confident pixels")
                continue
            pose = solve(points2d, points3d, camera, (width, height))
            if pose is None:
                report.append(f"{side}: PnP failed")
                continue
            draw(canvas, pose, camera, color=(0, 255, 255) if side == "left" else (255, 0, 255))
            report.append(f"{side}: {prediction.footedness} {pose.inliers}/{pose.samples} err {pose.error_px:.1f}px depth {pose.tvec[2, 0]:.2f}m")
        cv2.imwrite(str(OUT / Path(path).name), canvas)
        print(f"{Path(path).name}: " + "; ".join(report))
