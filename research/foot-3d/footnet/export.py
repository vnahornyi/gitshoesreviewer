"""Export FootNet to ONNX fp16 for the app, with the keypoint decoding baked into the graph, and check it.

    uv run python -m footnet.export [--checkpoint data/footnet/best.pt] [--target ../../modules/foot-pose/model/footnet-fp16.onnx]

The app feeds float32 and reads float32, as for RTMPose (see tools/model-convert).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16

from synfoot.data import ids

from .dataset import SIZE, SynFootCrops
from .model import FootNet, decode_points

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT.parents[1] / "modules/foot-pose/model/footnet-fp16.onnx"
SAMPLES = 16


class Exported(torch.nn.Module):
    """Probabilities out, nothing else: decoding the peaks in the graph split Core ML into 22 partitions and broke the
    float16 conversion on its integer maths. The app decodes the 8 peaks itself, as it already does for RTMPose."""

    def __init__(self, model: FootNet):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        mask, heatmaps, right = self.model(image)
        return torch.sigmoid(mask), torch.sigmoid(heatmaps), torch.sigmoid(right)


def to_onnx(model: FootNet, path: Path) -> None:
    exported = Exported(model).eval()
    torch.onnx.export(
        exported,
        (torch.zeros(1, 3, SIZE, SIZE),),
        str(path),
        input_names=["image"],
        output_names=["mask", "heatmaps", "right"],
        dynamo=False,
        opset_version=17,
    )


def check(session: ort.InferenceSession, exported: Exported, batch: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        expected = exported(batch)
        wanted, _ = decode_points(expected[1])
    errors, mask_agreement = [], []
    started = time.perf_counter()
    for index in range(len(batch)):
        mask, heatmaps, right = session.run(None, {"image": batch[index : index + 1].numpy()})
        points, _ = decode_points(torch.from_numpy(heatmaps))
        errors.append(np.linalg.norm(points[0].numpy() - wanted[index].numpy(), axis=-1))
        mask_agreement.append(((mask[0, 0] > 0.5) == (expected[0][index, 0].numpy() > 0.5)).mean())
    elapsed = (time.perf_counter() - started) / len(batch) * 1000
    return {
        "point_median_px": float(np.median(errors)),
        "point_max_px": float(np.max(errors)),
        "mask_agreement": float(np.mean(mask_agreement)),
        "latency_ms": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "data/footnet/best.pt"))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    model = FootNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    work = ROOT / "data/footnet"
    float32 = work / "footnet.onnx"
    to_onnx(model, float32)
    converted = float16.convert_float_to_float16(onnx.load(float32), keep_io_types=True)
    onnx.save(converted, args.target)
    print(f"wrote {args.target} ({Path(args.target).stat().st_size / 1e6:.1f} MB)")

    dataset = SynFootCrops(ids()[::997][:SAMPLES], False)
    batch = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    exported = Exported(model).eval()
    for name, path in (("float32", float32), ("float16", Path(args.target))):
        for provider in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
            session = ort.InferenceSession(str(path), providers=[provider])
            metrics = check(session, exported, batch)
            print(f"{name} on {provider.replace('ExecutionProvider', '')}: "
                  + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
