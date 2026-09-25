import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from assets import ASSETS

from .dataset import MEAN, SIZE, STD
from .mask_model import FootMaskNet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ASSETS / "checkpoints/footmask-v1/best.pt"
DEFAULT_OUTPUT = ROOT / "results/footmask-v1/real-review.jpg"
CAPTURES = ASSETS / "capture/prepared"
CAPTURE_OUTPUT = ROOT / "results/footmask-v1/capture-review"


def input_tensor(image: np.ndarray) -> torch.Tensor:
    rgb = image[..., ::-1].astype(np.float32) / 255
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy()).unsqueeze(0)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    tint = np.zeros_like(image)
    tint[mask] = color
    return cv2.addWeighted(output, 1.0, tint, 0.35, 0)


def square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    top = (SIZE - resized.shape[0]) // 2
    left = (SIZE - resized.shape[1]) // 2
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas, (left, top, resized.shape[1], resized.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--captures", action="store_true", help="also run RTMPose-seeded crops on prepared real frames")
    parser.add_argument("--per-video", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FootMaskNet().eval()
    model.load_state_dict(checkpoint["state_dict"])
    crops = sorted((ASSETS / "real-crops").glob("*.jpg"))
    panels = []
    with torch.no_grad():
        for path in crops:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            canvas, (left, top, width, height) = square(image)
            tensor = input_tensor(canvas)
            mask_logits, _, _ = model(tensor)
            probability = torch.sigmoid(mask_logits[0, 0]).numpy()
            binary = probability >= args.threshold
            overlay = overlay_mask(canvas, binary, (40, 70, 255))
            side_by_side = np.concatenate([canvas, overlay], axis=1)
            crop_mask = binary[top:top + height, left:left + width]
            score = float(probability[top:top + height, left:left + width].mean())
            cv2.putText(side_by_side, path.stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(side_by_side, f"mask {crop_mask.mean():.2f} mean-p {score:.2f}",
                        (6, SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(side_by_side)

    if not panels:
        raise FileNotFoundError(f"no real crops found under {ASSETS / 'real-crops'}")
    while len(panels) < 4:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.concatenate([np.concatenate(panels[:2], axis=1),
                            np.concatenate(panels[2:4], axis=1)], axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"could not write review image to {output}")
    print(output)

    if args.captures:
        from rtmlib import RTMPose

        from spike.fit_scene import FEET, MIN_SCORE, RTMPOSE_M

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device).eval()
        body = RTMPose(RTMPOSE_M, model_input_size=(192, 256), backend="onnxruntime", device="cpu")
        by_video = {}
        for path in sorted(CAPTURES.glob("*.jpg")):
            by_video.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
        selected = [paths[index] for paths in by_video.values()
                    for index in (np.arange(len(paths)) if max(1, args.per_video) >= len(paths) else
                                  np.rint(np.linspace(0, len(paths) - 1, max(1, args.per_video) + 2)[1:-1]).astype(int))]
        CAPTURE_OUTPUT.mkdir(parents=True, exist_ok=True)
        thumbnails = []
        for path in selected:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            height, width = frame.shape[:2]
            keypoints, scores = body(frame, bboxes=[[0, 0, width, height]])
            annotated = frame.copy()
            found = 0
            for side_index, (side, joint_indices) in enumerate(FEET.items()):
                visible_points = [keypoints[0][joint] for joint in joint_indices
                                  if scores[0][joint] >= MIN_SCORE]
                if len(visible_points) < 2:
                    continue
                points = np.asarray(visible_points, np.float32)
                low, high = points.min(0), points.max(0)
                center = (low + high) / 2
                crop_side = max(float((high - low).max()), 30.0) * 1.8
                affine = cv2.getRotationMatrix2D(tuple(center), 0.0, SIZE / crop_side)
                affine[:, 2] += SIZE / 2 - center
                crop = cv2.warpAffine(frame, affine, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                with torch.no_grad():
                    mask_logits, _, _ = model(input_tensor(crop).to(device))
                probability = torch.sigmoid(mask_logits[0, 0]).cpu().numpy()
                inverse = cv2.invertAffineTransform(affine)
                full_probability = cv2.warpAffine(probability, inverse, (width, height),
                                                  flags=cv2.INTER_LINEAR)
                mask = full_probability >= args.threshold
                color = (40, 70, 255) if side_index == 0 else (255, 50, 230)
                annotated = overlay_mask(annotated, mask, color)
                found += 1
            cv2.putText(annotated, path.stem, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            target = CAPTURE_OUTPUT / path.name
            if not cv2.imwrite(str(target), annotated):
                raise OSError(f"could not write capture review to {target}")
            preview_top = int(height * 0.45)
            preview_height = height - preview_top
            preview_width = min(width, round(preview_height * 160 / 284))
            preview_left = max(0, min(width - preview_width, width // 2 - preview_width // 2))
            preview = annotated[preview_top:, preview_left:preview_left + preview_width]
            thumbnail = cv2.resize(preview, (160, 284), interpolation=cv2.INTER_AREA)
            thumbnails.append(thumbnail)
            print(f"{path.name}: {found} seeded masks")

        if thumbnails:
            columns = 5
            rows = (len(thumbnails) + columns - 1) // columns
            sheet = np.zeros((rows * 284, columns * 160, 3), np.uint8)
            for index, thumbnail in enumerate(thumbnails):
                row, column = divmod(index, columns)
                sheet[row * 284:(row + 1) * 284, column * 160:(column + 1) * 160] = thumbnail
            contact_sheet = CAPTURE_OUTPUT / "contact-sheet.jpg"
            if not cv2.imwrite(str(contact_sheet), sheet):
                raise OSError(f"could not write capture contact sheet to {contact_sheet}")
            print(contact_sheet)


if __name__ == "__main__":
    main()
