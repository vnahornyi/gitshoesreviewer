# What happens to a camera frame

One frame every 33 ms. This is the whole path, in order, with where the time goes.

```
VisionCamera frame (BGRA, upright)
  │
  ├─ personMatte.update(frame)      background queue, skips frames while busy
  ├─ sceneLight.update(frame)       32×32 average over the lower half
  │
  ├─ HybridFootPoseDetector.detect(frame)
  │    ├─ for each foot being followed: FootNet on its own crop   ~15 ms / foot
  │    └─ only if a foot is not followed, ≤ 4×/s: RTMPose on the whole frame
  │
  └─ scheduleOnRN → JS
       ├─ footTracker: match, smooth (One Euro), carry FootNet's points
       ├─ footAxes:    pick the points a shoe needs
       ├─ shoePose:    back-project onto the floor plane → 4×4 transform
       └─ ShoeView:    RealityKit draws the shoe, matte cuts the leg out
```

## Why it is split this way

The naive pipeline — whole-frame body model every frame, then a refinement — ran at **6 fps**.
The split is the MediaPipe/BlazePose pattern:

- **RTMPose is the detector.** Expensive, sees the whole frame, only needed to *find* a foot.
- **FootNet is the tracker.** Cheap, sees one tight crop, takes that crop from where the foot was
  on the previous frame.

So RTMPose runs only when a foot is not being followed, and at most every 250 ms — otherwise a foot
that is simply out of frame would make it search on every frame and eat the budget for nothing.
Result: **28–30 fps**.

The general form of this pattern is the `realtime-vision-pipeline` skill. What follows is what is
specific to this app.

## Tracking, in the native module

State per foot (`HybridFootPoseDetector.swift`): the crop being followed, when the foot was last
seen, the last crop's brightness, and the points RTMPose last found (the seed for picking it up
again).

The rules that were arrived at by measurement, not by design:

- **A crop may change size by at most 20 % per frame** (`sizeStep = 1.2`). Building the next crop
  only from *confident* points collapses it onto part of the foot, frame after frame, until it is
  following a toe. Allowing 2× per frame made that collapse instant.
- **A lost foot keeps its crop for 300 ms** (`lostAfter`). Without this, losing a track meant
  waiting up to 250 ms for the next search — a dead window where FootNet ran on nothing and the
  log read `feet 0 ms`.
- **A stored seed** lets a foot be picked up from the last RTMPose points without waiting for a new
  search.
- **Two crops that land on the same foot**: the less certain one is released, and RTMPose picks that
  foot up again.

Every point is published with its raw score. Filtering happens downstream, never before the
diagnostics — see the `device-diagnostics` skill for why that rule exists.

## Carrying points between models, in JS

FootNet gives 8 points per foot; RTMPose gives 2 usable ones (big toe, heel) plus the ankle. On a
frame where FootNet was sure and RTMPose did not run, the app still needs points.

`footTracker.ts` stores, for each of FootNet's points, **where it sat relative to RTMPose's big
toe** — the one point both models produce. For up to 600 ms (`CARRY_MS`) after FootNet last ran,
those offsets are moved by however far that big toe moved. After that the body model's own points
take over.

Smoothing is One Euro, on normalized coordinates, so a speed of 1 means one frame **width** per
second — x and y are not comparable in pixels.

## Choosing the points a shoe needs

`footAxes.ts`. A shoe needs a front and a back:

- FootNet's **toes** are always preferred when it has the big toe — they are what the shoe is
  anchored to.
- FootNet's **heel** comes along when it is sure of it, which is about half the time: looking down
  at your own feet hides the heel behind the foot.
- When FootNet's heel is missing, **RTMPose's ankle** stands in. The ankle is a worse landmark but
  a real observation, which a geometric guess at the heel is not — two different guessing schemes
  were measured and both were far worse (54 mm, and 56° of direction error).

## 2D to 3D

`shoePose.ts`, and the domain reasoning is in the `shoe-try-on` skill. In short: gravity plus an
assumed camera height turns each image point into a 3D point on its own horizontal plane, because a
standing foot's landmarks are at known heights above the floor. Depth from apparent size does not
work in the mirror scenario.

## Rendering

`ios` — RealityKit `ARView` in non-AR mode over the camera preview, with the shoe
transform handed in as a column-major 4×4 in camera space. The Apple Vision person matte cuts the
view out where the real leg crosses the shoe collar, so the leg comes out of the shoe. Scene light
matching takes the frame's exposure and part of its colour cast. Its README is the contract.

## The budget

At 30 fps there are 33 ms per frame, shared:

| Stage | Cost |
|---|---|
| FootNet, per foot | ~15 ms |
| RTMPose, whole frame | ~40 ms, at most 4×/s |
| Person matte | background queue, skips frames while busy |
| Everything in JS | small, but it is on the JS thread |

Two feet at 15 ms leave very little. This is why the search is throttled and why cutting FootNet's
decoder (heatmaps at 64×64 instead of 256×256) is still on the list.
