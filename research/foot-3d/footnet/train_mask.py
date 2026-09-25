import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assets import ASSETS

from .dataset import SIZE
from .mask_dataset import RENDERS, RenderFootMasks, render_names
from .mask_model import FootMaskNet
from .model import decode_points

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ASSETS / "checkpoints/footmask-v1"
LOG = ROOT / "results/footmask-v1/metrics.csv"
HEATMAP_WEIGHT = 20.0
COORDINATE_WEIGHT = 0.05
SIDE_WEIGHT = 0.2
MASK_WEIGHT = 2.0
THRESHOLDS = (0.3, 0.5, 0.7)


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    weights = F.softmax(heatmaps.flatten(2), -1).view(b, k, h, w)
    xs = torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
    return torch.stack([(weights.sum(2) * xs).sum(-1), (weights.sum(3) * ys).sum(-1)], -1)


def loss_terms(outputs, batch) -> dict[str, torch.Tensor]:
    mask_logits, heatmaps_logits, right_logits = outputs
    mask_target = batch["mask"]
    probability = torch.sigmoid(mask_logits)
    dice_loss = 1 - (2 * (probability * mask_target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + mask_target.sum((1, 2, 3)) + 1
    )
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target) + dice_loss.mean()
    heatmap_loss = F.binary_cross_entropy_with_logits(heatmaps_logits, batch["heatmaps"])
    visible = batch["visible"]
    point_error = F.smooth_l1_loss(soft_argmax(heatmaps_logits), batch["points"], reduction="none").sum(-1)
    coordinate_loss = (point_error * visible).sum() / visible.sum().clamp_min(1)

    side_valid = batch["side_valid"].flatten() > 0.5
    if side_valid.any():
        side_loss = F.binary_cross_entropy_with_logits(
            right_logits[side_valid], batch["right"][side_valid]
        )
    else:
        side_loss = right_logits.sum() * 0

    return {
        "mask": MASK_WEIGHT * mask_loss,
        "heatmaps": HEATMAP_WEIGHT * heatmap_loss,
        "points": COORDINATE_WEIGHT * coordinate_loss,
        "side": SIDE_WEIGHT * side_loss,
    }


def _new_accumulator():
    return {
        "crop_iou": [],
        "crop_dice": [],
        "intersection": 0,
        "union": 0,
        "true_positive": 0,
        "predicted_positive": 0,
        "target_positive": 0,
        "visible_point_errors": [],
        "side_correct": 0,
        "side_count": 0,
        "empty_predicted_pixels": 0,
        "empty_pixel_count": 0,
        "samples": 0,
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    aggregates = {key: _new_accumulator() for key in ("all", "bare", "sock", "mirror", "top", "third",
                                                        "trousers", "no_trousers", "empty")}
    for batch in loader:
        image = batch["image"].to(device)
        mask_target = batch["mask"] > 0.5
        mask_logits, heatmaps_logits, right_logits = model(image)
        points, _ = decode_points(heatmaps_logits)
        visible = batch["visible"] > 0.5
        errors = (points - batch["points"].to(device)).norm(dim=-1).cpu()
        probs = torch.sigmoid(mask_logits).cpu().numpy()[:, 0]
        targets = mask_target.cpu().numpy()[:, 0]
        point_errors = errors.numpy()
        visible_cpu = visible.cpu().numpy()
        side_prediction = (right_logits.flatten() > 0).cpu().numpy()
        side_truth = (batch["right"].flatten() > 0.5).numpy()
        side_valid = (batch["side_valid"].flatten() > 0.5).numpy()

        for index, (mode, material, clothing) in enumerate(zip(batch["mode"], batch["material"], batch["clothing"])):
            is_positive = bool(batch["is_positive"][index].item())
            groups = ["all", "positive", material, mode, clothing] if is_positive else ["all", "empty"]
            for threshold in THRESHOLDS:
                predicted = probs[index] >= threshold
                truth = targets[index]
                intersection = int(np.logical_and(predicted, truth).sum())
                predicted_count = int(predicted.sum())
                target_count = int(truth.sum())
                union = predicted_count + target_count - intersection
                iou = intersection / union if union else 1.0
                dice = (2 * intersection) / (predicted_count + target_count) if predicted_count + target_count else 1.0
                for group in groups:
                    key = f"{group}:{threshold:.1f}"
                    if key not in aggregates:
                        aggregates[key] = _new_accumulator()
                    accumulator = aggregates[key]
                    accumulator["crop_iou"].append(iou)
                    accumulator["crop_dice"].append(dice)
                    accumulator["intersection"] += intersection
                    accumulator["union"] += union
                    accumulator["true_positive"] += intersection
                    accumulator["predicted_positive"] += predicted_count
                    accumulator["target_positive"] += target_count
                    accumulator["samples"] += 1
                    if group == "empty":
                        accumulator["empty_predicted_pixels"] += predicted_count
                        accumulator["empty_pixel_count"] += predicted.size
            if is_positive:
                valid_errors = point_errors[index][visible_cpu[index]]
                groups_with_keypoints = ["all", "positive", material, mode, clothing]
                for group in groups_with_keypoints:
                    for threshold in THRESHOLDS:
                        key = f"{group}:{threshold:.1f}"
                        aggregates[key]["visible_point_errors"].extend(valid_errors.tolist())
                        if side_valid[index]:
                            aggregates[key]["side_correct"] += int(side_prediction[index] == side_truth[index])
                            aggregates[key]["side_count"] += 1

    metrics = {}
    for key, accumulator in aggregates.items():
        if ":" not in key:
            continue
        prefix, threshold = key.rsplit(":", 1)
        if not accumulator["samples"]:
            continue
        suffix = f"t{threshold}"
        metrics[f"{prefix}_mask_iou_{suffix}"] = float(np.mean(accumulator["crop_iou"]))
        metrics[f"{prefix}_mask_dice_{suffix}"] = float(np.mean(accumulator["crop_dice"]))
        metrics[f"{prefix}_pixel_precision_{suffix}"] = accumulator["true_positive"] / max(1, accumulator["predicted_positive"])
        metrics[f"{prefix}_pixel_recall_{suffix}"] = accumulator["true_positive"] / max(1, accumulator["target_positive"])
        if accumulator["empty_pixel_count"]:
            metrics[f"{prefix}_crop_fp_area_{suffix}"] = (
                accumulator["empty_predicted_pixels"] / accumulator["empty_pixel_count"]
            )
        if threshold == "0.5" and accumulator["visible_point_errors"]:
            errors = np.asarray(accumulator["visible_point_errors"])
            metrics[f"{prefix}_visible_kp_median_px"] = float(np.median(errors))
            metrics[f"{prefix}_visible_kp_p90_px"] = float(np.quantile(errors, 0.9))
        if threshold == "0.5" and accumulator["side_count"]:
            metrics[f"{prefix}_side_accuracy"] = accumulator["side_correct"] / accumulator["side_count"]
    model.train(was_training)
    return metrics


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="restrict render frame names for a short training run")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINTS))
    parser.add_argument("--log", default=str(LOG))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_names, validation_names = render_names(RENDERS)
    if args.limit:
        train_names = train_names[:args.limit]
        validation_names = validation_names[:max(1, args.limit // 20)]
    train_set = RenderFootMasks(RENDERS, train_names, True, negatives=True)
    validation_set = RenderFootMasks(RENDERS, validation_names, False, negatives=True)
    train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0, drop_last=True)
    validation_loader = DataLoader(validation_set, args.batch, num_workers=args.workers,
                                   persistent_workers=args.workers > 0)

    model = FootMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.05
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    log_path = Path(args.log)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train_set)} ({len(train_names)} frames), validation {len(validation_set)} "
          f"({len(validation_names)} frames), {len(train_loader)} steps/epoch, {device}", flush=True)

    best = -1.0
    with log_path.open("w", newline="") as log_file:
        writer = None
        for epoch in range(1, args.epochs + 1):
            started, totals = time.time(), {}
            model.train()
            for step, batch in enumerate(train_loader, 1):
                batch = _move_batch(batch, device)
                terms = loss_terms(model(batch["image"]), batch)
                loss = sum(terms.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + value.item()
                if step % 200 == 0:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    elapsed = (time.time() - started) / step
                    print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                          + " ".join(f"{name} {value / step:.4f}" for name, value in totals.items())
                          + f" {elapsed:.3f} s/step", flush=True)
            if device.type == "mps":
                torch.mps.synchronize()
            validation = evaluate(model, validation_loader, device)
            metrics = {
                "epoch": epoch,
                "minutes": round((time.time() - started) / 60, 1),
                **{f"loss_{name}": round(value / len(train_loader), 4) for name, value in totals.items()},
                **{name: round(value, 4) for name, value in validation.items()},
            }
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if writer is None:
                writer = csv.DictWriter(log_file, list(metrics))
                writer.writeheader()
            writer.writerow(metrics)
            log_file.flush()

            checkpoint = {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "epoch": epoch,
                "seed": args.seed,
                "input_size": SIZE,
                "encoder": "tu-mobilenetv3_small_100",
                "loss_weights": {"mask": MASK_WEIGHT, "heatmaps": HEATMAP_WEIGHT,
                                 "points": COORDINATE_WEIGHT, "side": SIDE_WEIGHT},
                "thresholds_evaluated": list(THRESHOLDS),
                "validation": metrics,
            }
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            score = metrics.get("all_mask_dice_t0.5", -1.0)
            if score > best:
                best = score
                torch.save(checkpoint, checkpoint_dir / "best.pt")
    os._exit(0)


if __name__ == "__main__":
    main()
