# Shoe asset pipeline

Offline, on the developer's Mac: a product photo becomes a normalized GLB that the app loads.
User data never passes through here; only catalog photos do.

```
input/<id>/*.png ──cutout.swift──▶ work/<id>/*-cutout-N.png ──generate3d.sh──▶ work/<id>/raw.glb ──normalize.mjs──▶ assets/shoes/<id>/model.glb
```

`input/`, `work/` and `vendor/` are git-ignored. Product photos are retailer images, so they stay local.

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
swift cutout.swift input/<id>/01.png work/<id> --instance all
./generate3d.sh <id> work/<id>/01-cutout-0.png --pipeline-type 1024
node normalize.mjs work/<id>/raw.glb ../../assets/shoes/<id>/model.glb
```

- `cutout.swift` uses Apple Vision's foreground instance mask. A photo of a pair touching each other comes out as one instance. A top view usually splits into two, one per shoe.
- TRELLIS.2 takes a **single** image. Pick the view that shows the most of one shoe (side or 3/4). Anything it cannot see is invented.
- `normalize.mjs` puts the heel at the origin, the sole on `y = 0`, and the toe along `+Z`, with the sole length equal to `1`. The app scales the model to the real size in millimetres. The heel end is detected as the taller end. If a model comes out backwards, pass `--flip`. If it lies on its side (a warning says so), pass `--pre-rotate-x 90` or `-90`.
- Textures are re-encoded as JPEG at `--texture-size` (default 1024). Mesh compression is off until the app side confirms Filament decodes it.

## Check

```bash
npm run self-test
```
