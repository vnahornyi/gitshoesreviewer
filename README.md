# react-native-shoe-tryon

A React Native library that puts a 3D sneaker on a real foot, live in the camera, entirely
on-device. iOS first; Android is a later stage.

Two scenarios are equally in scope, and they are harder than they look for opposite reasons:

- **Mirror** — the phone points at a mirror, you see yourself and your feet from 1–2 m away.
- **On your feet** — you stand and look down at your own shoes, 20–40 cm away. The heel is hidden
  behind the foot in this view, which is why half of this repository exists.

User data never leaves the phone. Shoe assets are prepared offline on a developer Mac from catalog
photos and shipped as static files.

> **Status: prototype.** One debug screen in `example/`, no product UI, no Android, and the models
> are not in the package yet (see below). Numbers here are measured on an iPhone 11 unless said
> otherwise. [docs/state.md](docs/state.md) has what works today and what does not.

## Installing it

```bash
npm install github:vnahornyi/aishoesreviewer
cd ios && pod install
```

Peer dependencies: `react-native-nitro-modules`, `react-native-vision-camera` (v5),
`react-native-worklets`.

> **The keypoint models are not in the package.** They are 150 MB and not in git, so `model/` is
> empty after an install and the detector stays in its error state. Copy the three files into
> `node_modules/react-native-shoe-tryon/model/` **before** `pod install`;
> [ios/README.md](ios/README.md) says where they come from. Shipping them properly — git-lfs, a
> release asset, or a download step — is open work.

```tsx
import { ShoeLayer, useFootPose } from 'react-native-shoe-tryon';

const { feet, frameSize, onFrame } = useFootPose({ legMatte: true, sceneLight: true });
// … feed onFrame from a VisionCamera frame processor, then:
<ShoeLayer feet={feet} frame={frameSize} view={layout} />
```

[src/index.ts](src/index.ts) is the whole public surface, and [src/README.md](src/README.md)
explains the pieces behind it.

## Running the example

The example is a full React Native app with one debug screen, and it is how this is developed.

```bash
npm install                     # the library
npm run skills                  # links skills/ into .claude/skills/ for Claude Code; harmless otherwise
cp assets/onnx/rtmpose-m-fp16.onnx assets/onnx/rtmw-x-l-fp16.onnx model/
cd research/foot-3d && uv run python -m footnet.export && cd ../..

cd example
npm install
bundle install
cd ios && bundle exec pod install && cd ..
npm start
npm run ios                     # a real device: the simulator has no usable camera
```

At the root, `npm test` (Jest), `npx tsc --noEmit` and `npm run lint` all have to be green before
anything is staged; the example has its own `typecheck` and `lint`.

## How it is laid out

```
src/                     the library: tracking, smoothing, 2D → 3D shoe pose, the Nitro specs
ios/                     the library's Swift: one pod, five Nitro HybridObjects
shoes/                   the shipped .usdz shoe models
model/                   the keypoint models — not in git
example/                 a React Native app with one debug screen
assets/                  every large local file, one folder per dataset — not in git
tools/model-convert/     RTMPose ONNX → fp16, verified against fp32
tools/asset-pipeline/    product photo → 3D shoe → USDZ, offline on the Mac
research/foot-tracking/  which body model to use, and why
research/foot-3d/        FootNet: our own foot model, its datasets and its Blender renderer
docs/                    architecture, decisions, measurements, troubleshooting, and a primer
skills/                  what to know before touching a given area, agent-neutral
```

Four layers with four different jobs, and the split is deliberate:

- **the library** (`src/`, `ios/`, `shoes/`) is what an app installs;
- **`example/`** is the only consumer, and proves the library works from the outside;
- **`tools/`** builds the artifacts the library ships — models and shoes — and runs on a Mac, never
  on a phone;
- **`research/`** answers questions with measurements and produces `tools/`' inputs. Its licences
  are mixed and some of it cannot ship.

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
| [docs/primer/](docs/primer/README.md) | **Start here to learn the subject** — the theory, from the camera up, with references |
| [docs/architecture.md](docs/architecture.md) | What happens to a frame, in order, with the time budget |
| [docs/decisions.md](docs/decisions.md) | Why each model, runtime and approach — and what was rejected |
| [docs/measurements.md](docs/measurements.md) | Every number in this repo, with its date and method |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Failures that have actually happened here |
| [docs/state.md](docs/state.md) | What works, what does not, what is next |
| [docs/audit.md](docs/audit.md) | Where the AR quality actually breaks, measured end to end |
| [assets/README.md](assets/README.md) | What each dataset is, what it cost, how to restore it |
| [AGENTS.md](AGENTS.md) | Conventions and rules, for an agent or a new developer |
| [skills/](skills/README.md) | What to know before touching a given area |

`AGENTS.md` is the shared instruction file; [CLAUDE.md](CLAUDE.md) adds the Claude Code specifics on
top of it. The skills are agent-neutral on purpose — they say what and when, never which tool to
call.

## Licences

The prototype depends on research code with mixed licences, and some of it is **not usable in a
shipped product**. `research/foot-3d/README.md` names which. Shoe asset licences are in
[shoes/LICENSES.md](shoes/LICENSES.md).
