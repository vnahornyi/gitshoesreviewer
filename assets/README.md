# Assets

Everything large that is **not in git**: datasets, model checkpoints, captured footage, downloaded
models. Together they are around 40 GB, and every byte of it is either downloadable from its source
or regenerable from code in this repository — but regenerating some of it costs 18 hours of
rendering, so back it up rather than re-deriving it.

One folder per dataset, each self-contained, so any single folder can be zipped on its own:

```bash
tar -C assets -czf synth-2026-09-23.tar.gz synth
```

Code never hard-codes a path into here. It resolves this directory once and joins onto it:
`research/foot-3d/assets.py` and `research/foot-tracking/spike/common.py`. Set **`SHOE_ASSETS`** to
an absolute path to work off an external disk without editing anything.

| Folder | Size | What it is | Where it comes from |
|---|---|---|---|
| `synfoot/` | 31 G | [SynFoot](https://github.com/OllieBoyne/SynFoot) V1 — 50 000 renders of bare feet with masks, keypoints and camera poses | Downloaded. `research/foot-3d/README.md` has the link |
| `synth/` | 3.6 G | **Our own** renders: `renders/` (20 000 frames, feet in socks and trousers, mirror/top/third camera) and `shapes/` (FIND feet sampled for the renderer) | `python -m synth.export_shapes`, then `render/render.py`. ~18 h on one Mac |
| `checkpoints/` | 133 M | FootNet weights: `best.pt`, named copies of earlier runs, the exported `.mlpackage`, and the unpromoted `real-finetune-156/` experiment (`best.pt`, `last.pt`, `metrics.csv`) | `python -m footnet.train`, `python -m footnet.export`, `python -m footnet.finetune_real` |
| `capture/` | 3.9 G | Real footage shot on an iPhone 11: `raw/` videos per scenario, `prepared/` sampled frames, `masks/`, `labels.json` | Filmed. `research/foot-tracking/README.md` says how |
| `real-crops/` | 128 K | Four hand-picked real foot crops, the quick eyeball check for a new checkpoint | Cut by hand from `capture/prepared` |
| `onnx/` | 407 M | RTMPose-m and RTMW-x-l, fp32 as downloaded and fp16 as converted | `~/.cache/rtmlib` → `tools/model-convert` |
| `find-toc/` | 277 M | FOCUS's dense TOC predictor, used only by the research spike | Google Drive link in the FOCUS README |
| `mediapipe/` | 9 M | MediaPipe Pose Landmarker, a rejected candidate kept so the comparison can be re-run | Downloads itself on first use |
| `shoe-photos/` | 1.4 M | Retailer product photos, one folder per shoe, with `SOURCE.txt` | Saved from product pages. **Never commit these** — they are not ours |
| `shoe-builds/` | 12 M | The asset pipeline's intermediate and finished 3D shoes: cutouts, `raw.glb`, normalized `model.glb`, `model.usdz` | `tools/asset-pipeline` |

The finished shoes that the app actually ships are **not** here — they are checked into
`shoes/` next to the library, because they are part of the package.

## Restoring from nothing

In the order that unblocks the most:

1. `onnx/` — download and convert. 20 minutes, and the app needs it to run at all.
2. `checkpoints/` — copy a backup, or retrain (about 4 hours) once `synfoot/` and `synth/` are back.
3. `synfoot/` — download. Large but unattended.
4. `synth/` — re-render. 18 hours, so this is the one worth having a zip of.
5. `capture/` — cannot be restored. It is footage of a specific person in a specific flat. Back it
   up or lose it.
