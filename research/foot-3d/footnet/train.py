import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from assets import ASSETS
from synfoot.data import Sample, ids

from .dataset import RenderCrops, SynFootCrops
from .model import FootNet, decode_points

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ASSETS / "checkpoints"
LOG = ROOT / "results/footnet/log.csv"
RENDERS = ASSETS / "synth/renders"
RENDER_VALIDATION_EVERY = 20
VALIDATION_FOOT = "0033-A"
VALIDATION_SAMPLES = 2000
HEATMAP_WEIGHT = 20.0
COORDINATE_WEIGHT = 0.05
SIDE_WEIGHT = 0.2


def split() -> tuple[list[str], list[str]]:
    """Hold out one scanned foot entirely, so validation measures a person the model has not seen."""
    train, validation = [], []
    for id in ids():
        (validation if Sample.load(id).foot == VALIDATION_FOOT else train).append(id)
    return train, validation[:VALIDATION_SAMPLES]


def render_split() -> tuple[list[str], list[str]]:
    names = sorted(path.stem for path in RENDERS.glob("*.json"))
    return ([n for n in names if int(n) % RENDER_VALIDATION_EVERY],
            [n for n in names if int(n) % RENDER_VALIDATION_EVERY == 0])


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    weights = F.softmax(heatmaps.flatten(2), -1).view(b, k, h, w)
    xs = torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
    return torch.stack([(weights.sum(2) * xs).sum(-1), (weights.sum(3) * ys).sum(-1)], -1)


def loss_terms(outputs, batch) -> dict[str, torch.Tensor]:
    mask, heatmaps, right = outputs
    probability = torch.sigmoid(mask)
    dice = 1 - (2 * (probability * batch["mask"]).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + batch["mask"].sum((1, 2, 3)) + 1)
    return {
        "mask": F.binary_cross_entropy_with_logits(mask, batch["mask"]) + dice.mean(),
        "heatmaps": HEATMAP_WEIGHT * F.binary_cross_entropy_with_logits(heatmaps, batch["heatmaps"]),
        "points": COORDINATE_WEIGHT * F.smooth_l1_loss(soft_argmax(heatmaps), batch["points"]),
        "side": SIDE_WEIGHT * F.binary_cross_entropy_with_logits(right, batch["right"]),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    errors, ious, correct, count = [], [], 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mask, heatmaps, right = model(batch["image"])
        points, _ = decode_points(heatmaps)
        errors.append((points - batch["points"]).norm(dim=-1).cpu())
        predicted, truth = mask > 0, batch["mask"] > 0.5
        ious.append(((predicted & truth).sum((1, 2, 3)) / (predicted | truth).sum((1, 2, 3)).clamp(min=1)).cpu())
        correct += ((right > 0) == (batch["right"] > 0.5)).sum().item()
        count += len(right)
    errors = torch.cat(errors)
    model.train()
    return {
        "kp_median_px": errors.median().item(),
        "kp_p90_px": errors.flatten().quantile(0.9).item(),
        "heel_median_px": errors[:, 5].median().item(),
        "mask_iou": torch.cat(ious).mean().item(),
        "side_acc": correct / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="train on the first N samples only (smoke test)")
    parser.add_argument("--renders", action="store_true", help="also train on render/render.py output, validate on it separately")
    parser.add_argument("--init", help="start from this checkpoint")
    parser.add_argument("--render-repeat", type=int, default=3,
                        help="with --renders, repeat the render crops so they weigh more against SynFoot's 44k")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_ids, validation_ids = split()
    if args.limit:
        train_ids, validation_ids = train_ids[:args.limit], validation_ids[:args.limit // 4 or 1]
    train_set = SynFootCrops(train_ids, True)
    validation_sets = {"": SynFootCrops(validation_ids, False)}
    if args.renders:
        render_train, render_validation = render_split()
        renders = RenderCrops(RENDERS, True, render_train)
        train_set = ConcatDataset([train_set] + [renders] * args.render_repeat)
        validation_sets["render_"] = RenderCrops(RENDERS, False, render_validation)
    train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                              persistent_workers=True, drop_last=True)
    validation_loaders = {prefix: DataLoader(dataset, args.batch, num_workers=args.workers, persistent_workers=True)
                          for prefix, dataset in validation_sets.items()}

    model = FootNet(pretrained=args.init is None).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.05)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train_set)}, validation " + ", ".join(f"{p or 'synfoot_'}{len(d)}" for p, d in validation_sets.items())
          + f", {len(train_loader)} steps per epoch, {device}")

    best = float("inf")
    with LOG.open("w", newline="") as log_file:
        log = None
        for epoch in range(1, args.epochs + 1):
            started, totals = time.time(), {}
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                terms = loss_terms(model(batch["image"]), batch)
                loss = sum(terms.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                for name, value in terms.items():
                    totals[name] = totals.get(name, 0.0) + value.item()
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                          + " ".join(f"{k} {v / step:.4f}" for k, v in totals.items())
                          + f" {(time.time() - started) / step:.3f} s/step", flush=True)
            metrics = {"epoch": epoch, "minutes": round((time.time() - started) / 60, 1),
                       **{f"loss_{k}": round(v / len(train_loader), 4) for k, v in totals.items()},
                       **{prefix + k: round(v, 3) for prefix, loader in validation_loaders.items()
                          for k, v in evaluate(model, loader, device).items()}}
            print(metrics, flush=True)
            if log is None:
                log = csv.DictWriter(log_file, list(metrics))
                log.writeheader()
            log.writerow(metrics)
            log_file.flush()
            torch.save(model.state_dict(), CHECKPOINTS / "last.pt")
            if metrics["kp_median_px"] < best:
                best = metrics["kp_median_px"]
                torch.save(model.state_dict(), CHECKPOINTS / "best.pt")


if __name__ == "__main__":
    main()
    # Persistent DataLoader workers can hang the interpreter's exit on macOS; everything is saved by now.
    os._exit(0)
