---
name: realtime-vision-pipeline
description: How a camera-rate (30 fps) on-device vision pipeline is built — the detector/tracker split, cropping from the previous frame's result, landmark models that regress coordinates instead of decoding full-resolution heatmaps, per-frame time budgets, and where the time actually goes. Use when a model runs on every camera frame, when frames per second are too low, when choosing between running a model per frame or every Nth frame, when designing or retraining a keypoint/landmark/segmentation model for a phone, or when deciding what to measure before optimising.
---

> Copied from `ai-god/skills/realtime-vision-pipeline`, which is the canonical copy: this skill is general, not about
> this project. Edit it there and copy it back, or the two drift apart.

# Real-time vision on a phone

A camera frame arrives every 33 ms. Everything that happens for that frame — copying it,
preprocessing, every model, the drawing — shares those 33 ms. Design backwards from that number,
not forwards from a model that happens to be accurate.

## The detector/tracker split

This is the single largest win and it costs no retraining. MediaPipe's pose, hand and face
pipelines all use it, and the Snapchat/Instagram-class effects work the same way:

- A **detector** finds the object in the whole frame. It runs on the first frame, and afterwards
  only when tracking is lost.
- A **tracker** (the landmark model) runs every frame on a **tight crop taken from the previous
  frame's landmarks**, and also emits a presence score and a refined region for the next frame.
- When the presence score drops, the detector runs again.

Two consequences people miss:

1. The whole-frame detector disappears from the per-frame budget. Searching a 30× larger area for
   an object that moved 5 px is pure waste.
2. The landmark model gets a **tighter, better-framed crop**, so accuracy goes *up*, not down. A
   foot filling a 192 px crop carries more pixels than a foot filling 60 px of a 256 px crop.

A crop from the previous frame must be padded for motion (10–25 % beyond the landmark box) and
made square and axis-aligned, or rotated to the object's axis when the model was trained that way.

## Landmark models: regress, do not decode

A U-Net-shaped model that outputs one heatmap per keypoint at input resolution spends most of its
compute in the decoder, purely to upsample. Count the multiply-accumulates per stage before
optimising anything — the answer is usually "the last two upsampling stages are half the model".

What the mobile-grade models do instead (BlazePose is the documented example):

- A small encoder, then a **direct coordinate regression head** — `2 × joints` numbers out of a
  dense layer, plus a visibility/presence score per point.
- Heatmap and offset heads are kept **only during training** to supervise the shared features, and
  the layers are **removed before export**. Gradients from the regression head are stopped from
  flowing into the heatmap-trained features; this improves both heads.
- Input 128–256 px, and the model is in the 0.1–0.3 GMac range, not 1–3 GMac.

If full heatmaps are wanted anyway, put them at 1/4 or 1/8 of the input and refine the peak with a
softmax-weighted mean over a small window; the quantisation error is then well under a pixel and
the decoder shrinks by the square of the factor.

## Preprocessing is not free

Copying a 1080p frame into a float array with a scalar loop on the CPU costs as much as a small
model. Rules:

- Never write a per-pixel loop in application code. Scale with vImage, convert with vImage /
  vDSP in one pass, or let the inference framework take the pixel buffer directly (Core ML image
  inputs accept a `CVPixelBuffer` and apply scale and bias themselves).
- Better still, fold the normalisation (transpose, channel order, `/255`, mean/std) into the
  exported graph so it runs on the accelerator.
- Keep the frame in one representation. Every extra copy of a full frame is a millisecond.

## Budget, then measure

Write the budget down before optimising:

| stage | 30 fps budget |
|---|---|
| frame copy + preprocess | 1–3 ms |
| per-frame model(s) | 10–18 ms |
| detector (amortised over ~30 frames) | 1–3 ms |
| drawing / compositing | 3–8 ms |

Then measure each stage separately on the **device**, not the Mac, and check the numbers add up to
the observed frame time. When they do not, the missing time is real work you have not named:
frame delivery, a bridge hop to JS, a synchronous GPU wait, thermal throttling.

Two sanity checks worth doing early:

- **Effective throughput.** `GMac / seconds` for each model. If one model gets far fewer
  operations per second than another of the same size, it is running somewhere slower, not
  "just being slow".
- **The same model on another compute unit.** If CPU-only is nearly as fast as "all", the
  accelerator was never really used.

## Running something less often than every frame

When a model cannot fit the per-frame budget, running it every Nth frame is legitimate, but the
frames in between must not silently fall back to a worse estimate — that shows as a periodic
twitch, which is more objectionable than a constant small error. Carry the good result forward
instead: store it **relative to something the cheap per-frame signal also has** (an anchor point,
an affine fit, optical flow) and move it with that anchor. Bound the carry with a timeout so a
stale result cannot survive the object leaving.

## Temporal stability

- Filter every published point (One Euro is the usual choice: little lag when fast, little jitter
  when still).
- Feed the previous mask back into a segmentation model as an extra input channel — that is how
  selfie-segmentation models stay tiny; most of the work is refining an edge, not finding a person.
- Smooth pose in the pose domain (position, axis, scale), not in the rendered image.

## Sources

- [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204)
- [On-device, Real-time Body Pose Tracking with MediaPipe BlazePose](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/)
- [MediaPipe Pose landmark detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [Core ML Tools: Image Input and Output](https://apple.github.io/coremltools/docs-guides/source/image-inputs.html)
