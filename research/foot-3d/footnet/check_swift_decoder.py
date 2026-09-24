"""Compare the app's actual Swift decoder with the training decoder on real labelled crops.

Run from research/foot-3d with:
    uv run python -m footnet.check_swift_decoder
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from assets import ASSETS
from footnet.audit_checkpoints import load_examples, parse_checkpoints, tensor_for
from footnet.model import FootNet, decode_points

ROOT = Path(__file__).resolve().parents[3]
SWIFT_DECODER = ROOT / "ios/FootNetDecoder.swift"
SWIFT_PROBE = Path(__file__).with_name("decoder_probe.swift")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", help="checkpoint as NAME=PATH; repeat to override defaults")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    examples, excluded, _ = load_examples("oracle")
    print(f"Using {len(examples)} labelled oracle crops; skipped {excluded} incomplete/private records", file=sys.stderr)
    checkpoints = parse_checkpoints(args.checkpoint)
    for name, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint {name!r} not found: {path}")

    rows = []
    with tempfile.TemporaryDirectory(prefix="footnet-swift-decoder-") as directory:
        binary = Path(directory) / "decoder-probe"
        module_cache = Path(directory) / "module-cache"
        subprocess.run(
            [
                "swiftc", "-module-cache-path", str(module_cache), "-O",
                str(SWIFT_DECODER), str(SWIFT_PROBE), "-o", str(binary),
            ],
            check=True,
        )

        for checkpoint_name, checkpoint_path in checkpoints:
            model = FootNet(pretrained=False).eval()
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            fp32_point_errors = []
            fp32_score_errors = []
            fp16_point_errors = []
            fp16_score_errors = []
            for start in range(0, len(examples), args.batch_size):
                batch_examples = examples[start : start + args.batch_size]
                batch = torch.stack([tensor_for(example.crop) for example in batch_examples])
                with torch.inference_mode():
                    _, logits, _ = model(batch)
                    python_points, python_scores = decode_points(logits)
                    coreml_like = logits.to(torch.float16).to(torch.float32).contiguous()
                    fp16_points, fp16_scores = decode_points(coreml_like)

                swift = subprocess.run(
                    [str(binary)],
                    input=coreml_like.numpy().astype("<f4", copy=False).tobytes(),
                    capture_output=True,
                    check=True,
                )
                decoded_rows = [np.fromstring(line, sep=",") for line in swift.stdout.decode().splitlines()]
                if len(decoded_rows) != len(batch_examples) or any(row.size != 24 for row in decoded_rows):
                    raise RuntimeError("Swift decoder returned an unexpected number of values")
                decoded = np.stack(decoded_rows)
                swift_points = decoded.reshape(len(batch_examples), 8, 3)[..., :2]
                swift_scores = decoded.reshape(len(batch_examples), 8, 3)[..., 2]
                fp32_point_errors.append(
                    np.linalg.norm(swift_points - python_points.numpy(), axis=-1).reshape(-1)
                )
                fp32_score_errors.append(np.abs(swift_scores - python_scores.numpy()).reshape(-1))
                fp16_point_errors.append(
                    np.linalg.norm(swift_points - fp16_points.numpy(), axis=-1).reshape(-1)
                )
                fp16_score_errors.append(np.abs(swift_scores - fp16_scores.numpy()).reshape(-1))

            fp32_points = np.concatenate(fp32_point_errors)
            fp32_scores = np.concatenate(fp32_score_errors)
            fp16_points = np.concatenate(fp16_point_errors)
            fp16_scores = np.concatenate(fp16_score_errors)
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "feet": len(examples),
                    "joints": len(fp32_points),
                    "coordinate_delta_fp32_median_px": np.percentile(fp32_points, 50),
                    "coordinate_delta_fp32_p90_px": np.percentile(fp32_points, 90),
                    "coordinate_delta_fp32_max_px": fp32_points.max(),
                    "score_delta_fp32_median": np.percentile(fp32_scores, 50),
                    "score_delta_fp32_p90": np.percentile(fp32_scores, 90),
                    "score_delta_fp32_max": fp32_scores.max(),
                    "coordinate_delta_fp16_median_px": np.percentile(fp16_points, 50),
                    "coordinate_delta_fp16_p90_px": np.percentile(fp16_points, 90),
                    "coordinate_delta_fp16_max_px": fp16_points.max(),
                    "score_delta_fp16_median": np.percentile(fp16_scores, 50),
                    "score_delta_fp16_p90": np.percentile(fp16_scores, 90),
                    "score_delta_fp16_max": fp16_scores.max(),
                }
            )

    output = csv.DictWriter(
        sys.stdout,
        fieldnames=(
            "checkpoint", "feet", "joints",
            "coordinate_delta_fp32_median_px", "coordinate_delta_fp32_p90_px", "coordinate_delta_fp32_max_px",
            "score_delta_fp32_median", "score_delta_fp32_p90", "score_delta_fp32_max",
            "coordinate_delta_fp16_median_px", "coordinate_delta_fp16_p90_px", "coordinate_delta_fp16_max_px",
            "score_delta_fp16_median", "score_delta_fp16_p90", "score_delta_fp16_max",
        ),
    )
    output.writeheader()
    for row in rows:
        output.writerow({key: value if isinstance(value, str) or isinstance(value, int) else f"{value:.7f}" for key, value in row.items()})


if __name__ == "__main__":
    main()
