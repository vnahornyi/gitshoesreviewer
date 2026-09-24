"""Fine-tune the deployed FootNet checkpoint with partial real labels and render replay.

Run from research/foot-3d:
    uv run python -m footnet.finetune_real

The split is by source clip: IMG_3248 is validation, IMG_3250 is final test, and the remaining
eligible clips train. Only visible big-toe and heel labels contribute to the real loss; each update
also includes a fully labelled synthetic-render batch to retain the other outputs.
"""

import argparse
import csv
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from assets import ASSETS
from footnet.audit_checkpoints import RTMPOSE_MODEL, Example, load_examples, tensor_for
from footnet.dataset import RenderCrops, SIZE, heatmaps
from footnet.model import FootNet, decode_points
from footnet.train import (
    COORDINATE_WEIGHT,
    HEATMAP_WEIGHT,
    RENDERS,
    SIDE_WEIGHT,
    loss_terms,
    render_split,
    soft_argmax,
)
from rtmlib import RTMPose

INITIAL_CHECKPOINT = ASSETS / "checkpoints/v2-renders-7k.pt"
VALIDATION_CLIP = "IMG_3248"
TEST_CLIP = "IMG_3250"
OUTPUT = ASSETS / "checkpoints/real-finetune-156"


class PartialRealCrops(Dataset):
    def __init__(self, examples: list[Example], augment: bool):
        self.examples = examples
        self.augment = augment
        self.rng = np.random.default_rng(3248)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        crop = example.crop.copy()
        points = example.truth @ example.affine[:, :2].T + example.affine[:, 2]
        if self.augment and self.rng.random() < 0.5:
            crop = cv2.flip(crop, 1)
            points[:, 0] = SIZE - 1 - points[:, 0]

        valid = ((points >= 0) & (points < SIZE)).all(axis=1)
        target_points = np.zeros((2, 2), dtype=np.float32)
        target_points[valid] = points[valid]
        target_maps = heatmaps(target_points)
        return {
            "image": torch.from_numpy(tensor_for(crop).numpy()),
            "heatmaps": torch.from_numpy(target_maps),
            "points": torch.from_numpy(target_points),
            "valid": torch.from_numpy(valid.astype(np.float32)),
        }


def partial_loss(outputs, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    heatmaps = outputs[1][:, [0, 5]]
    weights = batch["valid"]
    denominator = weights.sum().clamp(min=1)
    heatmap_loss = F.binary_cross_entropy_with_logits(
        heatmaps, batch["heatmaps"], reduction="none"
    ).mean(dim=(-1, -2))
    predicted_points, _ = decode_points(heatmaps)
    point_loss = F.smooth_l1_loss(predicted_points, batch["points"], reduction="none").mean(-1)
    return (
        HEATMAP_WEIGHT * (heatmap_loss * weights).sum() / denominator
        + COORDINATE_WEIGHT * (point_loss * weights).sum() / denominator
    )


@torch.no_grad()
def evaluate(model: FootNet, examples: list[Example], batch_size: int) -> dict[str, float]:
    model.eval()
    errors, scores = [], []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = torch.stack([tensor_for(example.crop) for example in batch_examples])
        _, logits, _ = model(batch)
        points, confidence = decode_points(logits)
        points, confidence = points.numpy(), confidence.numpy()
        for example, crop_points, crop_scores in zip(batch_examples, points, confidence):
            truth_crop = example.truth @ example.affine[:, :2].T + example.affine[:, 2]
            valid = ((truth_crop >= 0) & (truth_crop < SIZE)).all(axis=1)
            if not valid.any():
                continue
            inverse = cv2.invertAffineTransform(example.affine)
            frame_points = crop_points[[0, 5]] @ inverse[:, :2].T + inverse[:, 2]
            errors.extend(
                (np.linalg.norm(frame_points[valid] - example.truth[valid], axis=1) / example.foot_length).tolist()
            )
            scores.extend(crop_scores[[0, 5]][valid].tolist())
    model.train()
    return {
        "feet": len(examples),
        "points": len(errors),
        "median_error_foot_lengths": float(np.percentile(errors, 50)) if errors else float("nan"),
        "p90_error_foot_lengths": float(np.percentile(errors, 90)) if errors else float("nan"),
        "score_ge_0_3_pct": 100 * float(np.mean(np.asarray(scores) >= 0.3)) if scores else 0.0,
    }


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(INITIAL_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    device = torch.device("cpu")

    detector = RTMPose(str(RTMPOSE_MODEL), model_input_size=(192, 256), backend="onnxruntime", device="cpu")
    all_examples, excluded, missing = load_examples("seed", detector)
    train_examples = [example for example in all_examples if example.clip not in {VALIDATION_CLIP, TEST_CLIP}]
    validation_examples = [example for example in all_examples if example.clip == VALIDATION_CLIP]
    test_examples = [example for example in all_examples if example.clip == TEST_CLIP]
    if not train_examples or not validation_examples or not test_examples:
        raise RuntimeError("clip split is empty; re-check source labels and excluded clips")
    print(
        f"seed crops={len(all_examples)}; train={len(train_examples)}; validation={len(validation_examples)} "
        f"({VALIDATION_CLIP}); test={len(test_examples)} ({TEST_CLIP}); excluded={excluded}; missing={missing}",
        file=sys.stderr,
    )

    model = FootNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "metrics.csv").open("w", newline="") as log_file:
        columns = (
            "epoch", "minutes", "loss_real", "loss_render", "validation_feet", "validation_points",
            "validation_median", "validation_p90", "validation_score_ge_0_3_pct",
        )
        writer = csv.DictWriter(log_file, fieldnames=columns)
        writer.writeheader()
        train_loader = DataLoader(
            PartialRealCrops(train_examples, augment=True),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        render_train, _ = render_split()
        render_loader = DataLoader(
            RenderCrops(RENDERS, True, render_train),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        render_iterator = iter(render_loader)
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            total_steps=args.epochs * len(train_loader),
            pct_start=0.05,
        )
        best_validation = float("inf")
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            real_losses, render_losses = [], []
            model.train()
            for real_batch in train_loader:
                render_batch, render_iterator = next_batch(render_loader, render_iterator)
                real_batch = {key: value.to(device) for key, value in real_batch.items()}
                render_batch = {key: value.to(device) for key, value in render_batch.items()}
                real_loss = partial_loss(model(real_batch["image"]), real_batch)
                render_terms = loss_terms(model(render_batch["image"]), render_batch)
                render_loss = sum(render_terms.values())
                loss = real_loss + render_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                real_losses.append(float(real_loss.detach()))
                render_losses.append(float(render_loss.detach()))

            validation = evaluate(model, validation_examples, args.batch_size)
            row = {
                "epoch": epoch,
                "minutes": round((time.time() - started) / 60, 2),
                "loss_real": float(np.mean(real_losses)),
                "loss_render": float(np.mean(render_losses)),
                "validation_feet": validation["feet"],
                "validation_points": validation["points"],
                "validation_median": validation["median_error_foot_lengths"],
                "validation_p90": validation["p90_error_foot_lengths"],
                "validation_score_ge_0_3_pct": validation["score_ge_0_3_pct"],
            }
            writer.writerow(row)
            log_file.flush()
            print(row, flush=True)
            torch.save(model.state_dict(), OUTPUT / "last.pt")
            if validation["median_error_foot_lengths"] < best_validation:
                best_validation = validation["median_error_foot_lengths"]
                torch.save(model.state_dict(), OUTPUT / "best.pt")

    best_model = FootNet(pretrained=False).to(device).eval()
    best_model.load_state_dict(torch.load(OUTPUT / "best.pt", map_location=device, weights_only=True))
    baseline = FootNet(pretrained=False).to(device).eval()
    baseline.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    for split_name, examples in (("validation", validation_examples), ("test", test_examples)):
        print(
            f"{split_name} baseline={evaluate(baseline, examples, args.batch_size)} "
            f"fine_tuned={evaluate(best_model, examples, args.batch_size)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
