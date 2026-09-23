---
name: device-diagnostics
description: Finding out what the app actually does on the phone — the per-second console log and what each field means, the on-screen stats line, and the reasoning discipline that this project learned the hard way. Use when frame rate or keypoint quality is wrong on the device, when a change needs to be verified on real frames, or before proposing any fix for behaviour that has not been measured.
---

# Measuring on the device

The Mac lies. FootNet ran correctly on the Mac while the phone saw nothing, twice. Nothing here is
settled by a simulator run or an offline probe.

## The two instruments

**The per-second log** (`logSample` in `src/foot-debug/useFootPose.ts`), one line a second to Metro:

```
[foot] 28.4 fps, frame 1280x720, feet 31 ms, search skipped
  left:  at 0.41,0.63 side 0.22 bright 0.34 scores 0.91 0.88 0.72 0.55 0.81 0.12 0.77 0.69
  right: none
```

Read it in this order:

1. `search skipped` vs a number — whether RTMPose ran this frame. It should be *skipped* most of
   the time; if it runs every frame, the tracker is losing feet and the frame rate will show it.
2. `at`/`side` — where FootNet looked, normalized to the frame. `none` means no crop at all, which
   is a **tracking** failure, not a model failure.
3. `bright` — the crop's average brightness. Near zero means the crop is off the frame or on
   darkness; the model is then blameless.
4. `scores` — the 8 per-point confidences. The heel is index 5.

**The stats line** on the debug screen: fps, refine time and how many of 16 points are sure, both
crop brightnesses, search time, matte time, frame size.

## What the numbers mean

- A crop with sensible `at`/`side`/`bright` and scores near zero is the **model** failing.
- `none`, or a crop that has drifted off the foot, is the **tracker** failing.
- Scores that are either 0.7–0.9 or 0.0–0.15 with nothing in between are the known bimodality: the
  synthetic/real gap. It is a data problem (see the `footnet` skill), not a threshold problem.
- `feet 0 ms` means FootNet did not run at all — it had no crop. Look upstream.

## The discipline

These are corrections this project actually needed, not general advice:

- **Publish raw before filtering.** For a while the pipeline zeroed a foot's points when fewer than
  three passed threshold, so the log could not tell "the model saw nothing" from "we discarded it".
  Diagnostics must carry the unfiltered value and its score.
- **A warning is not noise until you have checked.** `VerifyEachNodeIsAssignedToAnEp` was dismissed
  twice as ordinary chatter. It was the whole diagnosis: 22 Core ML partitions. Cost: a day.
- **The crash line is where it surfaced.** See the `nitro-native-modules` skill.
- **Layer the probe.** When the Mac and the phone disagree, prove each layer separately — model,
  crop code, readback — until the disagreement has nowhere left to hide.
- **Ask for the log rather than a screenshot** when the question is numeric. The developer runs the
  app and pastes Metro output; that is the fastest loop here and it is the established one.

## Frame geometry

Frames arrive upright (`enablePhysicalBufferRotation`) and BGRA (`pixelFormat: 'rgb'`). Points are
normalized to the frame, so x and y are comparable across resolutions but **not** square: a speed
of 1 is one frame width per second.

Known open issue: a landscape frame (seen at 1472×828) draws points far from the feet. Not
diagnosed. If it reappears, log `frameWidth`/`frameHeight` together with `isMirrored` before
theorising.
