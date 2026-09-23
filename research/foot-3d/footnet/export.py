"""Export FootNet to Core ML for the app, compile it, and check it against PyTorch.

    uv run python -m footnet.export [--checkpoint ../../assets/checkpoints/best.pt] [--target ../../modules/foot-pose/model]

Core ML rather than ONNX Runtime: ONNX Runtime's Core ML execution provider cut this graph into 22 partitions and
copied the tensors out and back at every one, which made the Neural Engine slower than the CPU (44 ms against 20).
Converted straight from PyTorch the whole network stays on the Neural Engine and takes 1.3 ms.

The app hands over a 256×256 BGRA pixel buffer of the foot crop and reads the 8 keypoint heatmaps; the peaks are
decoded there, as they are for RTMPose. The mask and right-foot heads are research-only and are left out.
"""

import argparse
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import ASSETS
from synfoot.data import ids

from .dataset import MEAN, SIZE, STD, SynFootCrops
from .model import FootNet, decode_points

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT.parents[1] / "modules/foot-pose/model"
NAME = "footnet"
SAMPLES = 16


class Exported(torch.nn.Module):
    """Pixels 0…1 in, keypoint probabilities out. Core ML scales the pixel buffer by 1/255 itself; the ImageNet
    normalisation is part of the graph so the app has no per-pixel work of its own left."""

    def __init__(self, model: FootNet):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, heatmaps, _ = self.model((image - self.mean) / self.std)
        return torch.sigmoid(heatmaps)


def to_coreml(exported: Exported, units: ct.ComputeUnit) -> ct.models.MLModel:
    traced = torch.jit.trace(exported, torch.zeros(1, 3, SIZE, SIZE))
    return ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="heatmaps", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=units,
    )


def latency(model: ct.models.MLModel, image) -> float:
    for _ in range(3):
        model.predict({"image": image})
    started = time.perf_counter()
    for _ in range(15):
        model.predict({"image": image})
    return (time.perf_counter() - started) / 15 * 1000


def check(model: ct.models.MLModel, exported: Exported, batch: torch.Tensor) -> dict[str, float]:
    """Core ML takes an image, so each sample goes back from the normalised tensor to the 0…255 pixels it came from."""
    from PIL import Image

    with torch.no_grad():
        wanted, _ = decode_points(exported(batch * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)))
    errors = []
    for index in range(len(batch)):
        pixels = (batch[index] * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
        image = Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        heatmaps = model.predict({"image": image})["heatmaps"].astype(np.float32)
        points, _ = decode_points(torch.from_numpy(heatmaps))
        errors.append(np.linalg.norm(points[0].numpy() - wanted[index].numpy(), axis=-1))
    return {"point_median_px": float(np.median(errors)), "point_max_px": float(np.max(errors))}


def compile_model(package: Path, target: Path) -> Path:
    """Compile on the Mac, as Xcode would, so the app has no compilation to do on its first launch."""
    compiled = target / f"{NAME}.mlmodelc"
    subprocess.run(["rm", "-rf", str(compiled)], check=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(target)], check=True)
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ASSETS / "checkpoints/best.pt"))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()

    model = FootNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    exported = Exported(model).eval()

    target = Path(args.target)
    package = ASSETS / f"checkpoints/{NAME}.mlpackage"
    to_coreml(exported, ct.ComputeUnit.ALL).save(str(package))
    compiled = compile_model(package, target)
    size = sum(f.stat().st_size for f in compiled.rglob("*") if f.is_file())
    print(f"wrote {compiled} ({size / 1e6:.1f} MB)")

    dataset = SynFootCrops(ids()[::997][:SAMPLES], False)
    batch = torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    loaded = ct.models.MLModel(str(package))
    metrics = check(loaded, exported, batch)
    print("against PyTorch: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    from PIL import Image

    sample = Image.fromarray(np.random.randint(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
    for units in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        timed = ct.models.MLModel(str(package), compute_units=units)
        print(f"{str(units).replace('ComputeUnit.', ''):12s} {latency(timed, sample):6.1f} ms")


if __name__ == "__main__":
    main()
