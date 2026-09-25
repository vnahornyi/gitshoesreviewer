import argparse
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import ASSETS

from .dataset import MEAN, SIZE, STD
from .mask_dataset import RENDERS, RenderFootMasks, render_names
from .mask_model import FootMaskNet
from .model import decode_points

DEFAULT_CHECKPOINT = ASSETS / "checkpoints/footmask-v1/best.pt"
DEFAULT_PACKAGE = ASSETS / "checkpoints/footmask-v1/footmask-v1.mlpackage"


class Exported(torch.nn.Module):
    """RGB pixels in; normalization and all mask/pose heads stay in Core ML."""

    def __init__(self, model: FootMaskNet):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor):
        mask, heatmaps, right = self.model((image - self.mean) / self.std)
        return mask, heatmaps, right


def convert(exported: Exported, compute_units: ct.ComputeUnit) -> ct.models.MLModel:
    traced = torch.jit.trace(exported, torch.zeros(1, 3, SIZE, SIZE))
    return ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIZE, SIZE), scale=1 / 255,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="mask_logits", dtype=np.float16),
                 ct.TensorType(name="heatmaps", dtype=np.float16),
                 ct.TensorType(name="right_logit", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=compute_units,
    )


def compile_package(package: Path) -> Path:
    compiled = package.parent / "footmask-v1.mlmodelc"
    if compiled.exists():
        raise FileExistsError(f"compiled model already exists: {compiled}; export to a fresh package directory")
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(package.parent)], check=True)
    return compiled


def _image_for_coreml(tensor: torch.Tensor):
    from PIL import Image

    pixels = (tensor * torch.tensor(STD).view(3, 1, 1)
              + torch.tensor(MEAN).view(3, 1, 1)).clamp(0, 1)
    return Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def parity(model: ct.models.MLModel, exported: Exported, batch: torch.Tensor,
           visible: torch.Tensor) -> dict[str, float]:
    errors = {"mask_logits": [], "heatmaps": [], "right_logit": []}
    mask_disagreements, point_deltas, confident_point_deltas, side_matches = [], [], [], []
    confidence_disagreements = 0
    visible_point_count = 0
    confident_point_count = 0
    for index in range(len(batch)):
        image = _image_for_coreml(batch[index])
        pixels = np.asarray(image).astype(np.float32) / 255
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)
        wanted = exported(tensor)
        result = model.predict({"image": image})
        references = []
        for name, expected in zip(errors, wanted):
            values = result[name].astype(np.float32)
            reference = expected[0].detach().cpu().numpy().astype(np.float16).astype(np.float32)
            errors[name].append(float(np.max(np.abs(values - reference))))
            references.append(reference)
        mask_disagreements.append(float(np.mean((result["mask_logits"] > 0) != (references[0] > 0))))
        exported_points, exported_scores = decode_points(torch.from_numpy(result["heatmaps"].astype(np.float32)))
        reference_points, reference_scores = decode_points(torch.from_numpy(references[1][None]))
        point_error = (exported_points - reference_points).norm(dim=-1)[0]
        labelled = visible[index] > 0.5
        confident = labelled & (exported_scores[0] >= 0.3) & (reference_scores[0] >= 0.3)
        labelled_scores_differ = (exported_scores[0, labelled] >= 0.3) != (reference_scores[0, labelled] >= 0.3)
        confidence_disagreements += int(labelled_scores_differ.sum())
        visible_point_count += int(labelled.sum())
        confident_point_count += int(confident.sum())
        point_deltas.extend(point_error[labelled].tolist())
        confident_point_deltas.extend(point_error[confident].tolist())
        side_matches.append(float(np.all((result["right_logit"] > 0) == (references[2] > 0))))

    def error_stats(name: str, values: list[float]) -> dict[str, float]:
        return {f"{name}_p99_abs": float(np.quantile(values, 0.99)),
                f"{name}_max_abs": float(np.max(values))}

    return {
        **{key: value for name, values in errors.items() for key, value in error_stats(name, values).items()},
        "mask_threshold_disagreement": float(np.mean(mask_disagreements)),
        "landmark_median_delta_px": float(np.median(point_deltas)),
        "landmark_p90_delta_px": float(np.quantile(point_deltas, 0.9)),
        "landmark_max_delta_px": float(np.max(point_deltas)),
        "visible_landmark_count": float(visible_point_count),
        "confident_landmark_count": float(confident_point_count),
        "confident_landmark_median_delta_px": float(np.median(confident_point_deltas)),
        "confident_landmark_p90_delta_px": float(np.quantile(confident_point_deltas, 0.9)),
        "confident_landmark_max_delta_px": float(np.max(confident_point_deltas)),
        "confidence_disagreement_at_0.3": confidence_disagreements / max(1, visible_point_count),
        "side_agreement": float(np.mean(side_matches)),
    }


def latency(model: ct.models.MLModel, image) -> float:
    for _ in range(3):
        model.predict({"image": image})
    started = time.perf_counter()
    for _ in range(20):
        model.predict({"image": image})
    return (time.perf_counter() - started) / 20 * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    args = parser.parse_args()

    checkpoint_path, package = Path(args.checkpoint), Path(args.package)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = FootMaskNet().eval()
    model.load_state_dict(checkpoint["state_dict"])
    exported = Exported(model).eval()
    package.parent.mkdir(parents=True, exist_ok=True)
    converted = convert(exported, ct.ComputeUnit.ALL)
    converted.save(str(package))
    compiled = compile_package(package)
    package_size = sum(path.stat().st_size for path in compiled.rglob("*") if path.is_file())
    print(f"compiled {compiled} ({package_size / 1e6:.1f} MB)")

    _, validation_names = render_names(RENDERS)
    validation = RenderFootMasks(RENDERS, validation_names[:10], False, negatives=False)
    items = [validation[index] for index in range(min(16, len(validation)))]
    batch = torch.stack([item["image"] for item in items])
    visible = torch.stack([item["visible"] for item in items])
    loaded = ct.models.MLModel(str(package))
    print("float16 parity: " + ", ".join(
        f"{key} {value:.5f}" for key, value in parity(loaded, exported, batch, visible).items()))

    sample = _image_for_coreml(batch[0])
    for units in (ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        timed = ct.models.MLModel(str(package), compute_units=units)
        label = str(units).replace("ComputeUnit.", "")
        print(f"{label:12s} {latency(timed, sample):6.2f} ms/crop")


if __name__ == "__main__":
    main()
