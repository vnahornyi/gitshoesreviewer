---
name: footnet
description: Training, exporting and shipping FootNet, this project's own foot keypoint model — the SynFoot and Blender render datasets, the train/export/deploy loop, what the validation numbers mean, and which failures are the model's rather than the app's. Use when retraining or fine-tuning FootNet, when adding or rendering training data, when the on-device keypoints are wrong or unsure, or when changing what the model predicts.
---

# FootNet

Our own model: a 256×256 crop of **one** foot in, 8 keypoint heatmaps out. It exists because
RTMPose gives two usable points per foot (big toe, heel) and puts the heel on the ankle when the
heel is hidden, which is most of the time a person looks down at their own feet.

Everything below lives in `research/foot-3d/`. Read its `README.md` first — it holds the dataset
findings and the measured results this skill only points at.

## The loop

```bash
cd research/foot-3d
uv run python -m footnet.train --renders --init assets/checkpoints/best.pt --epochs 8 --lr 3e-4 --render-repeat 1
uv run python -m footnet.real          # RTMPose crop → FootNet → PnP on real frames, results/footnet/real/
uv run python -m footnet.export        # best.pt → model/footnet.mlmodelc, with a parity check
cp -R assets/checkpoints/footnet.mlmodelc ../../model/
cd ../../ios && bundle exec pod install
```

Training writes `results/footnet/log.csv` (one row per epoch) and two checkpoints:
`assets/checkpoints/last.pt` and `assets/checkpoints/best.pt`, where "best" means the lowest **SynFoot**
median — not the render median. Save a named copy of the checkpoint you start from
(`v1-synfoot.pt`, `v2-renders-7k.pt`, …), because `best.pt` is overwritten in place and there is
otherwise nothing to compare against or fall back to.

## Datasets and their balance

| Source | Size | What it covers | What it does not |
|---|---|---|---|
| SynFoot V1 | 43 801 train crops | bare feet, close, from above | mirror views, socks as geometry, shoes, trousers |
| `render/render.py` | ~1.7 crops per frame | mirror / top / third-person, socks, trousers, our own floors and walls | **shoes** |

Neither has footwear. When a real-frame failure is on a shod foot, it is a data gap, not a bug —
say so instead of tuning thresholds around it.

`--render-repeat` weighs the renders against SynFoot's 44k. Check the ratio before trusting the
default: the validated recipe was roughly 45 % renders. Count the crops, do not assume —

```bash
uv run python -c "
from footnet.train import split, render_split, RENDERS
from footnet.dataset import RenderCrops
print(len(split()[0]), len(RenderCrops(RENDERS, True, render_split()[0])))"
```

The render validation split is every 20th frame **by index**, so it is stable across renders of
different sizes: frames held out before stay held out, and starting from an earlier checkpoint
leaks nothing.

## Reading the numbers

`log.csv` carries each metric twice, once per validation set: bare name for SynFoot,
`render_` prefix for the renders. The ones that matter:

- `render_kp_median_px` and `render_kp_p90_px` — the p90 is the honest one. A good median with a
  large p90 means the model is bimodal: sure or blind, nothing between. That is exactly what the
  device shows (scores 0.7–0.9 or 0.0–0.15), and it is what to watch across a run.
- `heel_median_px` — the heel is the hard point, and index 5 in `FOOT_NET_JOINTS`.
- SynFoot metrics barely move during a render fine-tune. That is fine; it means no forgetting.

A validation number is measured on synthetic data. It does **not** predict device behaviour, and
has repeatedly not predicted it. Only `footnet.real` and the on-device log say what the phone sees.

## Export

`footnet/export.py` converts `best.pt` with coremltools and compiles the `.mlpackage` with
`xcrun coremlcompiler`. It is not a thin wrapper — it changes the graph:

- ImageNet mean/std are folded in, so Core ML gets an `ImageType` with `scale=1/255` and does the
  normalisation itself. The app must not normalise.
- Only `sigmoid(heatmaps)` is returned. The mask and side heads are training-time supervision and
  are dropped, which is most of the speed.
- Output is float16; the app converts to float32 with vImage before finding peaks.

The export prints a parity check against PyTorch. Anything above ~1 px means the conversion broke
something — do not ship it. The last good run: median 0.001 px, max 0.974 px, 9.3 MB.

## Things that were tried and did not work

Do not re-derive these; they were measured.

- **Guessing the hidden heel in image space.** Axis extrapolation: 54 mm mean error, 22 % of foot
  length. A 2D similarity fit on 237 views: 56° direction error. Both useless. The 3D fit on the
  floor plane works (25 mm, 4.5°), and that is the open direction if the heel has to be recovered.
- **ONNX Runtime for this model.** Its Core ML execution provider cut the graph into 22 partitions
  and the Neural Engine came out slower than the CPU. See the `apple-neural-engine` skill.
- **fp16/bf16 autocast on MPS** for training: slower than fp32. `channels_last` fails in the decoder.
