import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_W, INPUT_H = 192, 256
SIMCC_SPLIT = 2.0
FEET = {"left_big_toe": 17, "left_heel": 19, "right_big_toe": 20, "right_heel": 22}


def full_frame_input(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = max(width / INPUT_W, height / INPUT_H)
    matrix = np.array(
        [[1 / scale, 0, (INPUT_W - width / scale) / 2], [0, 1 / scale, (INPUT_H - height / scale) / 2]],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(image, matrix, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None], matrix


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = simcc_x.argmax(axis=-1) / SIMCC_SPLIT
    y = simcc_y.argmax(axis=-1) / SIMCC_SPLIT
    scores = np.minimum(simcc_x.max(axis=-1), simcc_y.max(axis=-1))
    inverse = cv2.invertAffineTransform(matrix)
    return np.stack([x, y, np.ones_like(x)], axis=-1) @ inverse.T, scores


def session(path: str, provider: str) -> ort.InferenceSession:
    if provider == "coreml":
        options = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
        return ort.InferenceSession(path, providers=[("CoreMLExecutionProvider", options), "CPUExecutionProvider"])
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def timed(run, tensor, repeats: int = 10) -> tuple[list[np.ndarray], float]:
    outputs = run(tensor)
    started = time.perf_counter()
    for _ in range(repeats):
        run(tensor)
    return outputs, (time.perf_counter() - started) / repeats * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare foot keypoints of a candidate ONNX model/provider against the fp32 CPU reference")
    parser.add_argument("reference", help="fp32 ONNX model, run on CPU")
    parser.add_argument("candidate", help="ONNX model to check, e.g. the fp16 one")
    parser.add_argument("--provider", choices=["cpu", "coreml"], default="coreml")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    reference = session(args.reference, "cpu")
    candidate = session(args.candidate, args.provider)
    print(f"candidate providers: {candidate.get_providers()}")

    worst = 0.0
    reference_ms, candidate_ms = [], []
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"cannot read image: {path}")
        tensor, matrix = full_frame_input(image)
        (ref_x, ref_y), ref_time = timed(lambda t: reference.run(None, {"input": t}), tensor)
        (can_x, can_y), can_time = timed(lambda t: candidate.run(None, {"input": t}), tensor)
        reference_ms.append(ref_time)
        candidate_ms.append(can_time)
        ref_points, ref_scores = decode(ref_x[0], ref_y[0], matrix)
        can_points, can_scores = decode(can_x[0], can_y[0], matrix)
        indices = list(FEET.values())
        point_diff = float(np.linalg.norm(ref_points[indices] - can_points[indices], axis=-1).max())
        score_diff = float(np.abs(ref_scores[indices] - can_scores[indices]).max())
        worst = max(worst, point_diff)
        print(f"{Path(path).name}: max foot point diff {point_diff:.1f} px, max score diff {score_diff:.3f}")
    print(f"worst foot point difference: {worst:.1f} px")
    print(f"median latency: reference cpu {np.median(reference_ms):.0f} ms, candidate {args.provider} {np.median(candidate_ms):.0f} ms")


if __name__ == "__main__":
    main()
