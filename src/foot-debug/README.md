# The app-side pipeline

Everything between "the native module returned points" and "RealityKit draws a shoe". The native
side is in `modules/`; the whole path end to end is in [`docs/architecture.md`](../../docs/architecture.md).

| File | What it owns |
|---|---|
| `useFootPose.ts` | The frame processor: runs the detectors, measures fps, logs once a second |
| `footTracker.ts` | Matching feet across frames, One Euro smoothing, carrying FootNet's points |
| `footAxes.ts` | Choosing which points a shoe can be built from |
| `shoePose.ts` | Image points → a 6-DoF shoe transform, via the floor plane |
| `oneEuro.ts`, `vec3.ts` | Small, dependency-free helpers |
| `FootDebugScreen.tsx` | The only screen: camera, stats line, one layer toggle |
| `FootOverlay.tsx` | Draws points and axes |
| `ShoeLayer.tsx` | Hands the transforms to `ShoeView` |

`footAxes`, `footTracker` and `shoePose` are pure and unit-tested (`*.test.ts`) — they are the parts
where a mistake is invisible on screen but wrong in the maths. Keep them pure.

## Contracts worth knowing before editing

**Points are normalized to the frame.** x and y are each in `[0, 1]` relative to width and height,
so they are *not* comparable as distances: a speed of 1 in the smoother means one frame **width**
per second. Convert with `frameToView` only at the drawing edge.

**Zeros mean absent.** A foot the detector did not find, or FootNet did not refine, comes back as
all zeros — not as a missing array. Check before trusting a point.

**Raw scores are published before any filtering.** `refined` carries every point with its own
score, and the decision to use it happens here, not in the native module. This is deliberate: the
diagnostics must be able to distinguish "the model saw nothing" from "we discarded it". Do not move
filtering upstream.

**Flags live on the native objects.** A VisionCamera frame processor keeps the closure it started
with, so `matte.enabled`, `light.enabled` and `detector.refine` are set from an effect, never
captured in the worklet.

## Two models, one set of feet

RTMPose gives 2 usable points per foot plus the ankle, on the frames where it runs at all. FootNet
gives 8, every frame, per foot. `footTracker` bridges them by storing FootNet's points **relative to
RTMPose's big toe** — the one landmark both produce — and moving them by however far that toe moved,
for up to `CARRY_MS` (600 ms). After that the body model's own points take over.

`footAxes` then prefers FootNet's toes whenever it has the big toe, takes FootNet's heel when it is
confident of it, and substitutes RTMPose's ankle when it is not. The heel is missing about half the
time by physics, not by bug: looking down at your own feet hides it behind the foot.

## The debug screen

One toggle — arrows / 3D / both. The stats line reads:

```
fps · стопи <refine ms> <sure>/16 · кроп <bright0>/<bright1> · пошук <ms> · маска <ms> · WxH
```

Other toggles (foot side, mirroring, model choice) existed and were removed once each question had
been settled by measurement. A toggle is a decision not yet made; keeping it after the decision
spreads branches through the code.

For what the per-second console log means and how to read it, see the `device-diagnostics` skill.
