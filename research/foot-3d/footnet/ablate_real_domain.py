"""Test one-at-a-time image-domain changes on the labelled real-foot crops.

Run from research/foot-3d:
    uv run python -m footnet.ablate_real_domain > results/footnet/domain-ablation.csv
"""

import argparse
import csv
import sys
from contextlib import redirect_stdout
from collections import defaultdict

import cv2
import numpy as np
import torch
from rtmlib import RTMPose

from assets import ASSETS
from footnet.audit_checkpoints import (
    RTMPOSE_MODEL,
    Example,
    load_examples,
    tensor_for,
)
from footnet.dataset import SIZE
from footnet.model import FootNet, decode_points

SYNTHETIC_SOURCE_PIXELS_PER_FOOT = 243.0
REAL_SOURCE_PIXELS_PER_FOOT = 848.0
SYNTHETIC_SATURATION = 18.7
REAL_SATURATION = 97.5
SYNTHETIC_LAPLACIAN = 9.9
CHECKPOINT = ASSETS / "checkpoints/v2-renders-7k.pt"


def laplacian_median(crops: list[np.ndarray], transform) -> float:
    values = []
    for crop in crops:
        image = transform(crop)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        values.append(float(cv2.Laplacian(gray, cv2.CV_32F).std()))
    return float(np.median(values))


def denoise(crop: np.ndarray, strength: float) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(crop, None, strength, strength)


def calibrate_strength(crops: list[np.ndarray], target: float, transform_factory) -> float:
    low, high = 0.0, 1.0
    while laplacian_median(crops, transform_factory(high)) > target and high < 255:
        low, high = high, min(high * 2, 255.0)
    if laplacian_median(crops, transform_factory(high)) > target:
        raise RuntimeError("could not reach the measured synthetic sharpness target")
    while high - low > 0.05:
        middle = (low + high) / 2
        if laplacian_median(crops, transform_factory(middle)) > target:
            low = middle
        else:
            high = middle
    return high


def desaturate(crop: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= factor
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).std())


def saturation(crop: np.ndarray) -> float:
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1]))


def infer(model: FootNet, examples: list[Example], transform, batch_size: int) -> list[dict]:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            crops = [transform(example.crop) for example in batch_examples]
            batch = torch.stack([tensor_for(crop) for crop in crops])
            _, heatmaps, _ = model(batch)
            points, scores = decode_points(heatmaps)
            points = points.numpy()
            scores = scores.numpy()

            for example, crop_points, crop_scores, crop in zip(batch_examples, points, scores, crops):
                inverse = cv2.invertAffineTransform(example.affine)
                frame_points = crop_points[[0, 5]] @ inverse[:, :2].T + inverse[:, 2]
                errors = np.linalg.norm(frame_points - example.truth, axis=1) / example.foot_length
                output.append(
                    {
                        "view": example.view,
                        "errors": errors,
                        "scores": crop_scores[[0, 5]],
                        "sharpness": sharpness(crop),
                        "saturation": saturation(crop),
                    }
                )
    return output


def summarise(rows: list[dict], baseline: list[dict], indices: list[int], mode: str, variant: str) -> dict:
    errors = np.asarray([rows[i]["errors"] for i in indices]).reshape(-1)
    scores = np.asarray([rows[i]["scores"] for i in indices]).reshape(-1)
    per_foot = np.asarray([rows[i]["errors"].mean() for i in indices])
    baseline_per_foot = np.asarray([baseline[i]["errors"].mean() for i in indices])
    return {
        "crop": mode,
        "variant": variant,
        "view": rows[indices[0]]["view"] if len({rows[i]["view"] for i in indices}) == 1 else "all",
        "feet": len(indices),
        "points": len(errors),
        "median_error_foot_lengths": np.percentile(errors, 50),
        "p90_error_foot_lengths": np.percentile(errors, 90),
        "score_ge_0_3_pct": 100 * np.mean(scores >= 0.3),
        "median_paired_error_delta": np.median(per_foot - baseline_per_foot),
        "feet_improved_pct": 100 * np.mean(per_foot < baseline_per_foot),
        "median_laplacian_std": np.median([rows[i]["sharpness"] for i in indices]),
        "median_hsv_saturation": np.median([rows[i]["saturation"] for i in indices]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--crop", choices=("both", "oracle", "seed"), default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    oracle, excluded, _ = load_examples("oracle")
    print(f"oracle crops={len(oracle)}; excluded={excluded}", file=sys.stderr)
    crops = [example.crop for example in oracle]
    calibration_indices = np.linspace(0, len(crops) - 1, min(16, len(crops)), dtype=int)
    calibration_crops = [crops[index] for index in calibration_indices]
    blur_sigma = calibrate_strength(
        crops,
        SYNTHETIC_LAPLACIAN,
        lambda sigma: lambda crop: cv2.GaussianBlur(crop, (0, 0), sigma),
    )
    denoise_h = calibrate_strength(
        calibration_crops,
        SYNTHETIC_LAPLACIAN,
        lambda strength: lambda crop: denoise(crop, strength),
    )
    scale = SYNTHETIC_SOURCE_PIXELS_PER_FOOT / REAL_SOURCE_PIXELS_PER_FOOT
    saturation_factor = SYNTHETIC_SATURATION / REAL_SATURATION
    print(
        f"target factors: resolution={scale:.4f}; saturation={saturation_factor:.4f}; "
        f"blur_sigma={blur_sigma:.3f}; denoise_h={denoise_h:.3f}; "
        f"sharpness target={SYNTHETIC_LAPLACIAN:.1f}",
        file=sys.stderr,
    )

    transforms = {
        "baseline": lambda crop: crop,
        "downsample": lambda crop: cv2.resize(
            cv2.resize(
                crop,
                (max(1, round(SIZE * scale)), max(1, round(SIZE * scale))),
                interpolation=cv2.INTER_AREA,
            ),
            (SIZE, SIZE),
            interpolation=cv2.INTER_LINEAR,
        ),
        "desaturate": lambda crop: desaturate(crop, saturation_factor),
        "blur": lambda crop: cv2.GaussianBlur(crop, (0, 0), blur_sigma),
        "denoise": lambda crop: denoise(crop, denoise_h),
    }
    modes = ("oracle", "seed") if args.crop == "both" else (args.crop,)
    prepared = {"oracle": (oracle, excluded, {})}
    if "seed" in modes:
        with redirect_stdout(sys.stderr):
            detector = RTMPose(
                str(RTMPOSE_MODEL), model_input_size=(192, 256), backend="onnxruntime", device="cpu"
            )
        prepared["seed"] = load_examples("seed", detector)
        print(f"seed crops={len(prepared['seed'][0])}; excluded={prepared['seed'][1]}", file=sys.stderr)

    model = FootNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    output_rows = []
    for mode in modes:
        examples = prepared[mode][0]
        baseline_rows = None
        for variant, transform in transforms.items():
            rows = infer(model, examples, transform, args.batch_size)
            if variant == "baseline":
                baseline_rows = rows
            assert baseline_rows is not None
            groups: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                groups[row["view"]].append(index)
                groups["all"].append(index)
            output_rows.extend(
                summarise(rows, baseline_rows, indices, mode, variant)
                for _, indices in sorted(groups.items())
            )

    fieldnames = (
        "crop", "variant", "view", "feet", "points", "median_error_foot_lengths",
        "p90_error_foot_lengths", "score_ge_0_3_pct", "median_paired_error_delta",
        "feet_improved_pct", "median_laplacian_std", "median_hsv_saturation",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in output_rows:
        writer.writerow(
            {
                key: value if isinstance(value, (str, int)) else f"{value:.5f}"
                for key, value in row.items()
            }
        )


if __name__ == "__main__":
    main()
