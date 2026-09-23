# Shoe virtual try-on (TRYON-1)

A React Native prototype that puts a 3D sneaker on your real foot, live in the camera, entirely
on-device. iOS first; Android is a later stage.

Two scenarios are equally in scope, and they are harder than they look for opposite reasons:

- **Mirror** — the phone points at a mirror, you see yourself and your feet from 1–2 m away.
- **On your feet** — you stand and look down at your own shoes, 20–40 cm away. The heel is hidden
  behind the foot in this view, which is why half of this repository exists.

User data never leaves the phone. Shoe assets are prepared offline on a developer Mac from catalog
photos and shipped as static files.

> **Status: prototype.** One debug screen, no product UI, no Android. Numbers below are measured on
> an iPhone 11 unless said otherwise. See [docs/state.md](docs/state.md) for what works today and
> what does not.

## Running it

```bash
npm install
npm run skills       # links skills/ into .claude/skills/ for Claude Code; harmless otherwise
```

Models are **not in git** — build or copy them first, or the detector never leaves its error state:

```bash
cp tools/model-convert/work/rtmpose-m-fp16.onnx tools/model-convert/work/rtmw-x-l-fp16.onnx modules/foot-pose/model/
cd research/foot-3d && uv run python -m footnet.export && cd ../..
cp -R research/foot-3d/data/footnet/footnet.mlmodelc modules/foot-pose/model/
```

Then:

```bash
bundle install
cd ios && bundle exec pod install && cd ..
npm start
npm run ios          # a real device: the simulator has no usable camera
```

`npm test` (Jest), `npx tsc --noEmit` and `npm run lint` all have to be green before anything is
staged.

## How it is laid out

```
App.tsx                  one screen, the debug screen
src/foot-debug/          the app-side pipeline: tracking, smoothing, 2D → 3D shoe pose
modules/foot-pose/       Swift: RTMPose (ONNX Runtime) + FootNet (Core ML)
modules/shoe-stage/      Swift: RealityKit shoe rendering, person matte, scene light, gravity
tools/model-convert/     RTMPose ONNX → fp16, verified against fp32
tools/asset-pipeline/    product photo → 3D shoe → USDZ, offline on the Mac
research/foot-tracking/  which body model to use, and why
research/foot-3d/        FootNet: our own foot model, its datasets and its Blender renderer
docs/                    architecture, decisions, measurements, troubleshooting
skills/                  what to know before touching a given area, agent-neutral
```

Every directory above with real complexity has its own `README.md` that documents its contract.
Those are the source of truth for that layer; this file only points at them.

## The short version of how it works

A camera frame arrives every 33 ms. For each frame:

1. **FootNet** runs on a tight crop of each foot, taken from where that foot was on the previous
   frame. 8 keypoints per foot, ~15 ms per foot on the Neural Engine.
2. **RTMPose** — the whole-frame body model — runs only to pick up a foot that is not being
   followed, and at most 4 times a second. It is the detector; FootNet is the tracker.
3. The points are smoothed, carried between runs, and turned into a 6-DoF shoe pose against the
   floor plane using device gravity.
4. **RealityKit** draws the shoe, masked by an Apple Vision person matte so the real leg comes out
   of the collar.

This detector/tracker split is what took the app from 6 fps to 28–30. The full reasoning, with the
alternatives that were measured and rejected, is in [docs/architecture.md](docs/architecture.md)
and [docs/decisions.md](docs/decisions.md).

## Documentation map

| Document | What it answers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | What happens to a frame, in order, with the time budget |
| [docs/decisions.md](docs/decisions.md) | Why each model, runtime and approach — and what was rejected |
| [docs/measurements.md](docs/measurements.md) | Every number in this repo, with its date and method |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Failures that have actually happened here |
| [docs/state.md](docs/state.md) | What works, what does not, what is next |
| [AGENTS.md](AGENTS.md) | Conventions and rules, for an agent or a new developer |
| [skills/](skills/README.md) | What to know before touching a given area |

`AGENTS.md` is the shared instruction file; [CLAUDE.md](CLAUDE.md) adds the Claude Code specifics on
top of it. The skills are agent-neutral on purpose — they say what and when, never which tool to
call.

## Licences

The prototype depends on research code with mixed licences, and some of it is **not usable in a
shipped product**. `research/foot-3d/README.md` names which. Shoe asset licences are in
`modules/shoe-stage/shoes/LICENSES.md`.
