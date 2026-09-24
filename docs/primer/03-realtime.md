# 3. Doing it 30 times a second

A model that takes 200 ms is not a slow version of a real-time system. It is a different system.
This chapter is about the constraint and the architectures it forces.

## The budget is the design

At 30 fps a frame arrives every **33 ms**. Everything for that frame — copying it, preprocessing,
every model, the maths, the drawing — fits in 33 ms, or the queue grows and latency climbs until
frames are dropped.

Write the budget down before optimising anything. Ours:

| Stage | Cost |
|---|---|
| FootNet, per foot | ~15 ms |
| RTMPose, whole frame | ~40 ms |
| Person matte | background, skips frames while busy |
| Everything in JS | small, but on the JS thread |

Two feet at 15 ms is 30 ms of the 33. Adding a 40 ms whole-frame model to every frame is not
"somewhat slower" — it is three times over budget. That arithmetic, not profiling, is what decides
the architecture.

## The detector/tracker split

The pattern comes from [BlazePose](https://arxiv.org/abs/2006.10204) (Bazarevsky et al., 2020), the
model behind MediaPipe Pose, and it is the single most useful idea in on-device vision.

Split the work by **how often it is actually needed**:

- The **detector** searches the whole frame. Expensive, and only needed when you do not already
  know where the thing is.
- The **tracker** takes a tight crop around where the thing was on the *previous* frame, and
  refines it. Cheap, because the crop is small and the object fills it.

Between frames a foot barely moves, so the previous frame's answer is an excellent guess for this
one. The detector then runs only when tracking is lost.

```
frame 1:  detector (40 ms) → found feet → crops
frame 2:  tracker on each crop (15 ms) → new crops
frame 3:  tracker … 
frame N:  lost a foot → detector again
```

This is why our numbers moved from **6 fps to 28–30**. Not a faster model — a model that runs less
often.

There is a second benefit, and it is about quality, not speed. A crop of one foot, scaled to
256×256, gives the model far more pixels on the foot than a whole frame shrunk to 256×192 does.
The tracker is both cheaper *and* more accurate than the detector at the same job.

### The rules that keep it stable

A tracker feeding itself is a feedback loop, and feedback loops need damping. Ours, in
[`HybridFootPoseDetector.swift`](../../ios/HybridFootPoseDetector.swift), each
learned from a failure:

**The crop may change size by at most 20 % per frame.** Building the next crop only from the points
the model is *confident* about shrinks it: fewer points → smaller box → less context → fewer
confident points. Within a few frames the crop is following one toe. Allowing 2× per frame made the
collapse instant. The clamp turns a runaway into, at worst, a slow drift that the detector fixes.

**A lost foot keeps its crop for 300 ms.** Without this, losing a track meant waiting for the next
detector run — up to 250 ms of doing nothing, which the log showed as `feet 0 ms`.

**The detector runs at most 4 times a second.** Otherwise a foot that is simply out of frame makes
it search on *every* frame — the worst case becomes "never found, always searching", which is
exactly when you can least afford it.

**Two crops that land on the same foot**: release the less confident one and let the detector sort
it out. Trackers drift; two drifting towards the same object is normal and must be handled.

> The general form of these rules is in the `realtime-vision-pipeline` skill. The specific numbers
> are in this repository because they were measured here.

## Latency is not throughput

Thirty frames a second with 300 ms of lag feels broken. They are different quantities:

- **Throughput** — frames per second. Fixed by the budget above.
- **Latency** — how old the thing on screen is. Sum of capture, inference, maths and draw.

Overlapping work helps throughput and can hurt latency. Our person matte runs on a background
queue and *skips* frames while busy: it would rather be 100 ms stale than push the pose off
budget. That is a deliberate choice about which quantity matters where — a slightly late leg mask
is invisible, a late shoe is not.

## Smoothing, and why it is not cheating

Per-frame predictions jitter. The foot is still; the points wobble by a pixel or two, and a
rendered shoe shakes. Averaging over the last N frames fixes the shake and adds lag — and lag is
exactly what you cannot afford.

The [**1€ filter**](https://gery.casiez.net/1euro/) (Casiez, Roussel & Vogel, CHI 2012) resolves
the trade-off by making it adaptive. It is a low-pass filter whose cutoff frequency **rises with
speed**:

```
cutoff = minCutoff + β · |estimated speed|
α      = 1 / (1 + τ/Δt),   τ = 1 / (2π · cutoff)
value  = value + α · (measurement − value)
```

Standing still, speed ≈ 0, the cutoff is low, and jitter is smoothed away. Moving fast, the cutoff
is high, α approaches 1, and the filter barely filters — so there is no lag when it would be felt.
It is forty lines ([`oneEuro.ts`](../../src/track/oneEuro.ts)) and has two meaningful knobs:

- `minCutoff` — lower means steadier when still, at the cost of lag;
- `β` — higher means more responsive when moving, at the cost of jitter during motion.

Tune `minCutoff` first with the object still, then `β` with it moving. Ours are `1.5` and `6` on
frame-normalized coordinates — so "speed 1" means one frame **width** per second, and x and y are
not comparable in pixels.

## Filling the gaps between two models

Our two models run at different rates, and the slow one produces points the fast one does not.
When the detector has not run this frame, its ankle is missing — but the shoe still needs a back
point.

[`footTracker.ts`](../../src/track/footTracker.ts) stores FootNet's points **relative to the
one landmark both models produce**, RTMPose's big toe, and moves them by however far that toe
moved. For up to 600 ms this keeps a plausible full set; after that the body model's own points
take over.

This is dead reckoning, and its assumption is explicit: the foot moves rigidly over the short
window. That holds for 600 ms of walking and fails for a sudden turn — hence the expiry. **Every
gap-filling scheme should carry a stated assumption and a timeout.**

## The mistake worth remembering

For a while the pipeline zeroed a foot's points when fewer than three passed the confidence
threshold. It looked tidy. It meant the logs could not distinguish *the model saw nothing* from
*we threw it away* — and we spent real time debugging the model when the bug was in the filter.

**Publish raw values with their scores; decide downstream.** The native module now emits every
point with its own score, and the app filters. That rule is in the `device-diagnostics` skill
because it cost us a day.

## Try it

1. Work out what happens if the detector interval is 1 s instead of 250 ms. Which failures get
   worse? Which get better?
2. Set `β` to 0 in `SMOOTHING` and move your foot quickly. You should be able to predict the
   symptom before you see it.
3. Run the app and watch `search` in the per-second log. It should say `skipped` most of the time.
   When it does not, ask why the tracker is losing feet — that is the question, not the frame rate.

## Further reading

- [BlazePose](https://arxiv.org/abs/2006.10204) — detector/tracker on a phone, the original.
- [MediaPipe's pose landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
  — the same pipeline as a shipped product.
- [The 1€ filter page](https://gery.casiez.net/1euro/) — paper, demo and implementations in a dozen
  languages. The interactive demo teaches the trade-off faster than the paper does.
