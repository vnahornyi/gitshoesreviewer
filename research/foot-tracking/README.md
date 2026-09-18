# Foot tracking spike (TRYON-1, step 2)

This spike decides whether the try-on pipeline can find a foot in the demo view: the developer looks down at their own feet with an iPhone 11. The answer is measured on real photos before any app code depends on it.

## Candidates

| Name | What it is | License |
|---|---|---|
| `mediapipe` | MediaPipe Pose Landmarker (full): ankle, heel, foot index per foot | Apache-2.0 |
| `rtmw-det` | RTMW whole-body (COCO-WholeBody feet: big toe, small toe, heel, ankle) behind the YOLOX person detector | Apache-2.0 |
| `rtmw-full` | The same RTMW pose model run on the whole frame, with no detector — for views where no full person is visible | Apache-2.0 |
| `geometric` | No keypoint model. It reads the Apple Vision **person** mask, splits it into per-leg components, and uses PCA to get the heel→toe axis. The toe is the end away from the bottom edge, and the heel is where the width narrows behind the ball of the foot | — |
| `geometric-fg` | The same geometry on Apple Vision's **foreground instance** mask (subject lifting), which may separate shoes from the floor better in a close-up | — |

Apple Vision body pose (2D and 3D) is not a candidate: its skeleton ends at the ankle, with no heel or toe joints, so it cannot give the heel→toe axis.

## Setup

```bash
uv sync
```

MediaPipe is pinned to 0.10.35: 1.0.x aborts on macOS in `TensorsToDetectionsCalculator` ("Service is unavailable"). Models download on first use. MediaPipe's go to `models/`, RTMW's to `~/.cache/rtmlib`.

## 1. Shoot (iPhone 11)

About 60 photos, stood up, phone at chest height, looking down:

| Folder | Count | What |
|---|---|---|
| `data/raw/top/` | 40 | both feet from above, feet turned at different angles, sometimes only one foot in frame |
| `data/raw/step/` | 10 | one foot stepped forward, so the foot is tilted |
| `data/raw/34/` | 10 | three-quarter view from above |

Vary what is on the feet (socks, barefoot, sneakers), use two floor surfaces, and shoot in two kinds of light. HEIC is fine; AirDrop the photos into the folders.

## 2. Prepare

```bash
uv run python -m spike.prepare
```

This applies EXIF orientation, downscales to 1600 px, and writes `data/prepared/<view>_<nnn>.jpg`.

## 3. Masks (for the geometric candidates)

```bash
swift build -c release --package-path vision-cli
vision-cli/.build/release/FootVisionCLI data/prepared data/masks
```

This writes `data/masks/person/<stem>.png` and `data/masks/foreground/<stem>.png` at image size. `--kind person|foreground` limits it to one kind.

## 4. Label

```bash
uv run python -m spike.label
```

Per foot, click in order: **big toe → small toe → heel → ankle**. In the top view the heel is usually hidden behind the shin. Click where it would be anyway; the heel→toe line is what the angle metric uses. Keys: `x` not visible and cannot be estimated, `n` next foot, `u` undo, `s` save the image, `q` quit (progress is kept in `data/labels.json`).

## 5. Run

```bash
uv run python -m spike.run
uv run python -m spike.run --candidates rtmw-full,geometric-fg
uv run python -m spike.run --self-test
```

Output: `results/report.md` (the table), `results/report.json`, and `results/overlays/<candidate>/` (green = label, red = prediction).

Metrics are computed per view and over all photos:
- **detected** — a predicted big toe within half a foot length of the labeled one;
- **PCK@0.05 / 0.10** — share of feet whose point lies within 5% / 10% of the labeled heel→toe length;
- **axis err** — median angle between the labeled and predicted heel→toe direction;
- **ms (Mac)** — median latency on the Mac CPU. The iPhone 11 is measured separately, once the winner is converted.

## Decision

Thresholds come from plan step 2. They are working assumptions, not requirements from a document.

- **GO** — a keypoint candidate reaches PCK@0.05 ≥ 0.8 in the `top` view, **or** `geometric` / `geometric-fg` reaches an axis error ≤ 8° with a toe error ≤ 5% of foot length.
- **PARTIAL** — only feet standing flat pass (`top` passes, `step` fails): photo mode ships with a "stand straight" hint.
- **NO-GO** — nothing passes: stop, and decide separately about fine-tuning on ~500–1000 labeled photos.

## Results

_Not run yet: waiting for the photos._
