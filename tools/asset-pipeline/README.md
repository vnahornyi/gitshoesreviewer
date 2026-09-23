# Shoe asset pipeline

Offline, on the developer's Mac: a product photo becomes a normalized GLB that the app loads.
User data never passes through here; only catalog photos do.

```
assets/shoe-photos/<id>/*.png
  ──cutout.swift──▶ assets/shoe-builds/<id>/*-cutout-N.png
  ──generate3d.sh──▶ assets/shoe-builds/<id>/raw.glb
  ──normalize.mjs──▶ assets/shoe-builds/<id>/model.glb
  ──glb2usdz.py──▶ shoes/<id>-{right,left}.usdz
```

Inputs and intermediates live under `assets/` (`shoe-photos/`, `shoe-builds/`), which is git-ignored; `vendor/` is too. Product photos are retailer images, so they stay local and are never committed.

## Requirements

- Apple Silicon, 24 GB+ unified memory (TRELLIS.2 peaks around 18 GB)
- Xcode with the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`
- `uv`, Node ≥ 22.11, Eigen headers (`brew install eigen`)
- A free HuggingFace account. `microsoft/TRELLIS.2-4B` is open, but the port also loads two gated
  models: request access to `facebook/dinov3-vitl16-pretrain-lvd1689m` (Meta approves it by hand)
  and `briaai/RMBG-2.0` (approved automatically, CC BY-NC). Log in once with `hf auth login`.

## Setup

```bash
./setup.sh
```

This pins [trellis-mac](https://github.com/shivampkumar/trellis-mac) to a fixed commit under `vendor/`, runs its setup (Python 3.11 venv, PyTorch, Metal extensions), and installs the Node dependencies. `constraints.txt` pins PyTorch to 2.11: from 2.14 on it requires C++20, while the port's Metal extensions build with `-std=c++17`. The script stops early when the Metal Toolchain or Eigen is missing, because otherwise the port only warns and texture baking quietly degrades.

## Per shoe

```bash
swift cutout.swift ../../assets/shoe-photos/<id>/01.png ../../assets/shoe-builds/<id> --instance all
./generate3d.sh <id> ../../assets/shoe-builds/<id>/01-cutout-0.png --pipeline-type 1024
node normalize.mjs ../../assets/shoe-builds/<id>/raw.glb ../../assets/shoe-builds/<id>/model.glb
uv run glb2usdz.py ../../assets/shoe-builds/<id>/model.glb ../../shoes/<id>-right.usdz
uv run glb2usdz.py ../../assets/shoe-builds/<id>/model.glb ../../shoes/<id>-left.usdz --mirror
```

Swap `-right` and `-left` if the source model is a left shoe. On a normalized shoe, the arch cutout of the sole and the big-toe bulge are on the medial side: on `+x` for a right shoe.

- `cutout.swift` uses Apple Vision's foreground instance mask. A photo of a pair touching each other comes out as one instance. A top view usually splits into two, one per shoe.
- TRELLIS.2 takes a **single** image. Pick the view that shows the most of one shoe (side or 3/4). Anything it cannot see is invented.
- `normalize.mjs` puts the heel at the origin, the sole on `y = 0`, and the toe along `+Z`, with the sole length equal to `1`. The app scales the model to the real size in millimetres. The heel end is detected as the taller end. If a model comes out backwards, pass `--flip`. If it lies on its side (a warning says so), pass `--pre-rotate-x 90` or `-90`.
- Textures are re-encoded as JPEG at `--texture-size` (default 1024). Mesh compression is off, because the USDZ step reads plain glTF buffers.
- `glb2usdz.py` writes a USDZ with `UsdPreviewSurface` materials (base color, metallic-roughness, normal, occlusion) that RealityKit loads natively. Check it with `usdchecker --arkit <file>.usdz`. `--mirror` flips `x` and the triangle winding, which turns a right shoe into a left one.

## Check

```bash
npm run self-test
```
