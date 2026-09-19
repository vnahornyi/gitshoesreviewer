import sys
from pathlib import Path

import cv2
import numpy as np

from .pose import correspondences, draw, intrinsics, solve
from .toc import load_model, predict

OUT = Path(__file__).resolve().parents[1] / "results/fit"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model("mps")
    rng = np.random.default_rng(0)
    for path in sys.argv[1:]:
        frame = cv2.imread(path)
        height, width = frame.shape[:2]
        prediction = predict(model, frame, "mps")
        identity = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        points2d, points3d = correspondences(prediction, identity, rng)
        name = Path(path).name
        if points2d is None:
            print(f"{name}: no confident foot pixels")
            continue
        camera = intrinsics(width, height)
        pose = solve(points2d, points3d, camera, (width, height))
        if pose is None:
            print(f"{name}: PnP failed")
            continue
        draw(frame, pose, camera)
        cv2.imwrite(str(OUT / name), frame)
        print(f"{name}: foot {prediction.footedness}, mirrored {pose.mirrored}, inliers {pose.inliers}/{pose.samples}, "
              f"median error {pose.error_px:.1f} px, depth {pose.tvec[2, 0]:.2f} m")
