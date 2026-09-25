import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from assets import ASSETS

from .mask_dataset import RENDERS, RenderFootMasks, render_names
from .mask_model import FootMaskNet
from .train_mask import evaluate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ASSETS / "checkpoints/footmask-v1/best.pt"
DEFAULT_OUTPUT = ROOT / "results/footmask-v1/final-validation.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FootMaskNet().to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    _, validation_names = render_names(RENDERS)
    dataset = RenderFootMasks(RENDERS, validation_names, False, negatives=True)
    loader = DataLoader(dataset, args.batch, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    metrics = evaluate(model, loader, device)
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "validation_frames": len(validation_names),
        "validation_crops": len(dataset),
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
