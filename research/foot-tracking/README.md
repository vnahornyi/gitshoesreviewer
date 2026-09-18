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

One 30–40 s video per scenario, in socks and in sneakers (not only barefoot). Put each `.MOV` into its scenario folder. Photos (HEIC is fine) work too.

| Folder | Scenario |
|---|---|
| `data/raw/mirror-full/` | the phone's back camera points at a mirror, and the whole body is visible |
| `data/raw/mirror-lower/` | the same, but only the legs and feet are visible |
| `data/raw/third/` | someone else films your feet from the side or front |
| `data/raw/top/` | standing, phone at chest height, looking down at your own feet |
| `data/raw/close/` | handheld 20–40 cm from one foot, any side including the sole |

Frames are sampled every 0.5 s, and the blurriest quarter of them is dropped. Vary the floor and the light.

## 2. Prepare

```bash
uv run python -m spike.prepare
```

This applies EXIF orientation, downscales to 1600 px, and writes `data/prepared/<view>_<nnn>.jpg` for photos and `data/prepared/<view>_<video>_<ms>.jpg` for video frames. `--every 0.25` samples denser, and `--drop-blurriest 0.4` is stricter.

## 3. Masks (for the geometric candidates)

```bash
swift build -c release --package-path vision-cli
vision-cli/.build/release/FootVisionCLI data/prepared data/masks
```

This writes `data/masks/person/<stem>.png` and `data/masks/foreground/<stem>.png` at image size. `--kind person|foreground` limits it to one kind.

## 4. Label

```bash
uv run python -m spike.label --points toe-heel
uv run python -m spike.label
```

`--points toe-heel` asks for two clicks per foot, which is enough for the GO/NO-GO metrics. Without it, per foot, click in order: **big toe → small toe → heel → ankle**. In the top view the heel is usually hidden behind the shin. Click where it would be anyway; the heel→toe line is what the angle metric uses. Keys: `x` not visible and cannot be estimated, `n` next foot, `u` undo, `s` save the image, `d` discard the image (a blurry frame, no foot, or a person's face in it), `q` quit (progress is kept in `data/labels.json`).

## 5. Run

```bash
uv run python -m spike.run
uv run python -m spike.run --candidates rtmw-full,geometric-fg
uv run python -m spike.run --self-test
```

Output: `results/report.md` (the table), `results/report.json`, and `results/overlays/<candidate>/` (green = label, red = prediction).

`--preview` needs no labels: it draws every candidate's heel→toe arrows into `results/preview/<candidate>/`, for a first look (`--limit N` for a quick pass). Keypoint candidates keep a foot only when both its toe and heel scores reach `MIN_AXIS_SCORE` (0.25 in `spike/candidates.py`). That value is a working one from eyeballing the `close` frames (correct feet scored 0.30–0.48, wrong ones 0.12–0.22); recalibrate it on labeled data.

Metrics are computed per view and over all photos:
- **detected** — a predicted big toe within half a foot length of the labeled one;
- **PCK@0.05 / 0.10** — share of feet whose point lies within 5% / 10% of the labeled heel→toe length;
- **axis err** — median angle between the labeled and predicted heel→toe direction;
- **ms (Mac)** — median latency on the Mac CPU. The iPhone 11 is measured separately, once the winner is converted.

## Decision

Thresholds come from plan step 2. They are working assumptions, not requirements from a document.

- **GO** — a keypoint candidate reaches PCK@0.05 ≥ 0.8 in the `top` view, **or** `geometric` / `geometric-fg` reaches an axis error ≤ 8° with a toe error ≤ 5% of foot length.
- The decision is made per scenario (`mirror-full`, `mirror-lower`, `third`, `top`, `close`).
- **PARTIAL** — a scenario passes only under a constraint (for example standing still, or one angle): it ships with a hint for the user.
- **NO-GO** — nothing passes: stop, and decide separately about fine-tuning on ~500–1000 labeled photos.

## Results

**2026-09-18, `close` preview, no labels.** 68 frames from 3 videos (one adult foot, two child feet, barefoot, handheld 20–40 cm, sole views included). These are eyeballed, not measured:

- `rtmw-full` keeps a foot on 40 of 68 frames at score ≥ 0.25 (29 at 0.30) (~70 ms on the Mac). The arrows checked by eye point the right way, including on sole views. Frames with only the heel in view come out wrong and score low.
- `rtmw-det` keeps a foot on 28 of 68, because the person detector often misses a lone foot.
- `mediapipe` finds a foot on 3 of 68 (it needs the whole body).
- `geometric` on the person mask points the wrong way in the close view: its "toe away from the bottom edge" assumption holds only for `top`. `geometric-fg` is right on some frames.

**2026-09-18, preview of all scenarios, no labels.** Videos IMG_3247 (`mirror-full`), IMG_3248 (`mirror-lower`), IMG_3249 and IMG_3250 (`top`, standing and walking), IMG_3251 (`third`: someone else films the feet close to floor level). White socks, one adult, one room. Share of frames with a foot kept, and what the overlays show when checked by eye:

| Scenario | frames | `mediapipe` | `rtmw-det` | `rtmw-full` | `geometric-fg` |
|---|---|---|---|---|---|
| `mirror-full` | 23 | 21 — right when both feet are visible | 23 — right | 23 — right | wrong: the mask is the whole body |
| `mirror-lower` | 17 | 10 — right on frontal frames | 17 — mostly right, side views too | 17 — mostly right | wrong |
| `third` | 23 | 0 | 21 — right on most frames, one foot flipped when the soles face the camera | 21 — same as `rtmw-det` | wrong |
| `top` | 41 | 1 | 35 — right on sharp frames | 35 — right on sharp frames, wrong on some motion-blurred ones | 40 — right when the foot points up the frame, as designed |
| `close` | 68 | 3 | 28 | 40 | partial |

First read, before metrics: **RTMW whole-body covers every scenario**, including the mirror and the top view. `geometric-fg` is a fallback for `top` only, and `mediapipe` for `mirror-full` only. Motion blur while walking (IMG_3250) breaks every candidate.

Open:
- only white socks, no sneakers;
- RTMW here is the `dw-x-l` model (≈ 70 ms on the Mac CPU), and whether a mobile-size RTMW keeps this on an iPhone 11 is the next thing to measure;
- labels for real metrics.
