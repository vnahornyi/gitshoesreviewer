"""Compare FootNet checkpoints on hand-labelled real frames.

Run from research/foot-3d with:
    .venv/bin/python -m footnet.audit_checkpoints > results/footnet/audit-checkpoints.csv

The oracle mode isolates FootNet using a crop centred on the labelled toe/heel. The seed mode
uses an offline RTMPose crop proxy with the same joints, threshold and crop geometry as the iOS
detector. It does not emulate the native tracker's motion or crop history.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from assets import ASSETS
from footnet.dataset import MEAN, SIZE, STD, square_affine
from footnet.model import FootNet, decode_points

LABELS = ASSETS / "capture/labels.json"
FRAMES = ASSETS / "capture/prepared"
CHECKPOINTS = ASSETS / "checkpoints"
RTMPOSE_MODEL = ASSETS / "onnx/rtmpose-m.onnx"
CONTEXT = 1.8
MIN_BODY_SCORE = 0.2
MIN_FOOTNET_SCORE = 0.3
FOOT_JOINTS = {"left": (0, (15, 20, 22, 24)), "right": (1, (16, 21, 23, 25))}
EXCLUDED_CLIPS = {"IMG_3244", "IMG_3246", "IMG_3256"}
KEY_RE = re.compile(r"^(?P<view>.+?)_(?P<clip>IMG_\d+)_(?P<frame>\d+)$")


@dataclass
class Example:
    view: str
    crop: np.ndarray
    affine: np.ndarray
    truth: np.ndarray
    foot_length: float
    crop_center_error: float | None = None
    clip: str = ""


def crop_affine(points: np.ndarray) -> np.ndarray:
    low, high = points.min(0), points.max(0)
    side = max(float((high - low).max()), 24.0) * CONTEXT
    return square_affine((low + high) / 2, side, 0.0)


def make_crop(frame: np.ndarray, affine: np.ndarray) -> np.ndarray:
    # The iOS crop leaves off-frame pixels black. OpenCV matches its geometry; its interpolation
    # is a close proxy for vImage scaling, not a bit-identical implementation.
    return cv2.warpAffine(
        frame,
        affine,
        (SIZE, SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def tensor_for(crop: np.ndarray) -> torch.Tensor:
    rgb = crop[..., ::-1].astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def parse_checkpoints(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return [
            ("renders-7k", CHECKPOINTS / "v2-renders-7k.pt"),
            ("best", CHECKPOINTS / "best.pt"),
            ("last", CHECKPOINTS / "last.pt"),
        ]
    parsed = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Checkpoint must be NAME=PATH, got {value!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        parsed.append((name, path))
    return parsed


def load_examples(mode: str, detector=None) -> tuple[list[Example], int, dict[str, int]]:
    labels = __import__("json").loads(LABELS.read_text())
    frame_cache: dict[str, np.ndarray] = {}
    examples: list[Example] = []
    excluded_feet = 0
    missing_seed_by_view: dict[str, int] = {}

    for name, label in sorted(labels.items()):
        match = KEY_RE.match(name)
        if not match:
            raise ValueError(f"Unexpected label key: {name}")
        view, clip = match.group("view"), match.group("clip")
        feet = label.get("feet", [])
        if clip in EXCLUDED_CLIPS:
            excluded_feet += len(feet)
            continue

        if name not in frame_cache:
            image_path = FRAMES / f"{name}.jpg"
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read labelled frame: {image_path}")
            frame_cache[name] = frame
        frame = frame_cache[name]

        body_candidates = []
        if mode == "seed":
            height, width = frame.shape[:2]
            keypoints, scores = detector(frame, bboxes=[[0, 0, width, height]])
            for side, (ankle, indices) in FOOT_JOINTS.items():
                seen = [keypoints[0][index] for index in indices if scores[0][index] >= MIN_BODY_SCORE]
                if len(seen) < 2 and scores[0][ankle] >= MIN_BODY_SCORE:
                    seen.append(keypoints[0][ankle])
                if len(seen) < 2:
                    continue
                points = np.asarray(seen, dtype=np.float32)
                affine = crop_affine(points)
                center = (points.min(0) + points.max(0)) / 2
                body_candidates.append((side, affine, center))

        valid_feet = []
        for foot in feet:
            if not foot.get("big_toe") or not foot.get("heel"):
                excluded_feet += 1
                continue
            truth = np.asarray([foot["big_toe"], foot["heel"]], dtype=np.float32)
            foot_length = float(np.linalg.norm(truth[0] - truth[1]))
            if not np.isfinite(foot_length) or foot_length <= 0:
                excluded_feet += 1
                continue
            valid_feet.append((truth, foot_length))

        if mode == "oracle":
            for truth, foot_length in valid_feet:
                affine = crop_affine(truth)
                crop = make_crop(frame, affine)
                examples.append(Example(view, crop, affine, truth, foot_length, clip=clip))
        else:
            if not valid_feet:
                continue
            if not body_candidates:
                missing_seed_by_view[view] = missing_seed_by_view.get(view, 0) + len(valid_feet)
                continue
            costs = np.asarray(
                [
                    [np.linalg.norm(candidate[2] - truth.mean(0)) / foot_length for candidate in body_candidates]
                    for truth, foot_length in valid_feet
                ],
                dtype=np.float32,
            )
            assigned_feet, assigned_candidates = linear_sum_assignment(costs)
            assignments = dict(zip(assigned_feet.tolist(), assigned_candidates.tolist()))
            for foot_index, (truth, foot_length) in enumerate(valid_feet):
                if foot_index not in assignments:
                    missing_seed_by_view[view] = missing_seed_by_view.get(view, 0) + 1
                    continue
                _, affine, center = body_candidates[assignments[foot_index]]
                target_center = truth.mean(0)
                crop_center_error = float(np.linalg.norm(center - target_center) / foot_length)
                examples.append(
                    Example(view, make_crop(frame, affine), affine, truth, foot_length, crop_center_error, clip)
                )
    return examples, excluded_feet, missing_seed_by_view


def evaluate(model: FootNet, examples: list[Example], device: torch.device, batch_size: int):
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = torch.stack([tensor_for(example.crop) for example in batch_examples]).to(device)
            _, heatmaps, _ = model(batch)
            points, scores = decode_points(heatmaps)
            points = points.cpu().numpy()
            scores = scores.cpu().numpy()

            for example, crop_points, crop_scores in zip(batch_examples, points, scores):
                inverse = cv2.invertAffineTransform(example.affine)
                frame_points = crop_points[[0, 5]] @ inverse[:, :2].T + inverse[:, 2]
                normalized_errors = np.linalg.norm(frame_points - example.truth, axis=1) / example.foot_length
                expected_vector = example.truth[1] - example.truth[0]
                predicted_vector = frame_points[1] - frame_points[0]
                expected_angle = np.degrees(np.arctan2(expected_vector[1], expected_vector[0]))
                predicted_angle = np.degrees(np.arctan2(predicted_vector[1], predicted_vector[0]))
                angle_error = abs((predicted_angle - expected_angle + 180) % 360 - 180)
                predicted_length = float(np.linalg.norm(predicted_vector))
                rows.append(
                    {
                        "view": example.view,
                        "errors": normalized_errors,
                        "scores": crop_scores[[0, 5]],
                        "angle_error": angle_error,
                        "length_error": abs(predicted_length / example.foot_length - 1),
                        "crop_center_error": example.crop_center_error,
                    }
                )
    return rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def emit(output, mode: str, checkpoint: str, view: str, rows: list[dict], labelled: int) -> None:
    errors = [float(value) for row in rows for value in row["errors"]]
    scores = [float(value) for row in rows for value in row["scores"]]
    centers = [row["crop_center_error"] for row in rows if row["crop_center_error"] is not None]
    angle = [float(row["angle_error"]) for row in rows]
    length = [float(row["length_error"]) for row in rows]
    points = len(errors)
    output.writerow(
        {
            "crop": mode,
            "checkpoint": checkpoint,
            "view": view,
            "labelled_feet": labelled,
            "predicted_feet": len(rows),
            "coverage_pct": round(100 * len(rows) / labelled, 1) if labelled else 0,
            "points": points,
            "median_error_foot_lengths": round(percentile(errors, 50), 4),
            "p90_error_foot_lengths": round(percentile(errors, 90), 4),
            "median_error_mm_at_276mm": round(percentile(errors, 50) * 276, 1),
            "score_ge_0_3_pct": round(100 * sum(score >= MIN_FOOTNET_SCORE for score in scores) / len(scores), 1) if scores else 0,
            "median_direction_error_deg": round(percentile(angle, 50), 2),
            "median_length_error_pct": round(100 * percentile(length, 50), 1),
            "median_seed_center_error_foot_lengths": round(percentile(centers, 50), 3) if centers else "",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", help="checkpoint as NAME=PATH; repeat to override defaults")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop", choices=("both", "oracle", "seed"), default="both")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(
        "mps" if args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()) else "cpu"
    )
    checkpoints = parse_checkpoints(args.checkpoint)
    for name, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {name!r} not found: {path}")
    print(f"device={device}; excluded clips={','.join(sorted(EXCLUDED_CLIPS))}", file=sys.stderr)

    modes = ("oracle", "seed") if args.crop == "both" else (args.crop,)
    prepared: dict[str, tuple[list[Example], int, dict[str, int]]] = {}
    for mode in modes:
        detector = None
        if mode == "seed":
            from rtmlib import RTMPose

            if not RTMPOSE_MODEL.is_file():
                raise FileNotFoundError(f"Local RTMPose model not found: {RTMPOSE_MODEL}")
            detector = RTMPose(str(RTMPOSE_MODEL), model_input_size=(192, 256), backend="onnxruntime", device="cpu")
        examples, excluded, missing = load_examples(mode, detector)
        prepared[mode] = (examples, excluded, missing)
        print(
            f"{mode}: {len(examples)} crops; excluded={excluded}; missing RTMPose seed by view={missing}",
            file=sys.stderr,
        )

    columns = (
        "crop", "checkpoint", "view", "labelled_feet", "predicted_feet", "coverage_pct", "points",
        "median_error_foot_lengths", "p90_error_foot_lengths", "median_error_mm_at_276mm",
        "score_ge_0_3_pct", "median_direction_error_deg", "median_length_error_pct",
        "median_seed_center_error_foot_lengths",
    )
    output = csv.DictWriter(sys.stdout, fieldnames=columns)
    output.writeheader()

    for checkpoint_name, checkpoint_path in checkpoints:
        model = FootNet(pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.to(device).eval()
        for mode in modes:
            examples, excluded, missing = prepared[mode]
            evaluated = evaluate(model, examples, device, args.batch_size)
            views = sorted({row["view"] for row in evaluated})
            for view in ["all", *views]:
                subset = evaluated if view == "all" else [row for row in evaluated if row["view"] == view]
                labelled = len(subset)
                if mode == "seed" and view == "all":
                    labelled += sum(missing.values())
                elif mode == "seed":
                    labelled += missing.get(view, 0)
                emit(output, mode, checkpoint_name, view, subset, labelled)


if __name__ == "__main__":
    main()
