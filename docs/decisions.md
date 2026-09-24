# Decisions, and what was rejected

Each entry says what was chosen, what it was chosen over, and what evidence decided it. A decision
without evidence is marked as such — those are the ones to revisit first.

## Body model: RTMPose-m (Halpe26), not RTMW x-l

RTMW x-l sees 133 whole-body joints; RTMPose-m sees 26. The foot joints are the same.

On the iPhone 11 RTMW takes ~150 ms a frame (~6 fps) — unusable live. RTMPose-m fp16 takes 6 ms on
the Core ML EP against 18 ms on CPU. `research/foot-tracking/README.md` has the comparison.

## ONNX Runtime, not TFLite, for RTMPose

`onnx2tf` converts RTMW without error and produces a **wrong** model: even float32 puts foot points
up to ~940 px away from the ONNX model on the same input. The NCHW → NHWC rewrite breaks the SimCC
head. ONNX Runtime runs the original graph unchanged, and the same `.onnx` will serve Android.

Full detail in `tools/model-convert/README.md`.

## Core ML directly, not ONNX Runtime, for FootNet

**This was the single biggest win in the project, and it was found late.**

ONNX Runtime's Core ML execution provider cut FootNet's graph into **22 partitions**, copying
tensors out and back at each boundary. The Neural Engine came out *slower than the CPU* (44 ms
against 20). Converted from PyTorch with coremltools and run through Core ML, the whole network
stays on the Neural Engine: **1.9 ms on the Mac, 65 → 15 ms per foot on the iPhone 11.**

The warning that said so (`VerifyEachNodeIsAssignedToAnEp`) was dismissed as noise twice before it
was read properly. Cost: about a day. The general lesson is in the `apple-neural-engine` skill.

RTMPose still runs through ONNX Runtime, because coremltools cannot read its ONNX source. Doing the
same conversion for it is open work.

## Our own foot model (FootNet), not just the body model

RTMPose gives two usable points per foot. From the front the heel is hidden and the model places it
on the ankle, so a shoe anchored to it lands wrong. Dense-correspondence research (FOCUS/TOC)
showed the approach is right but its pretrained model fails on socks and mirror views — and its
licence is non-commercial anyway. Hence a small model of our own.

`research/foot-3d/README.md` has the spike results that led here.

## Detector/tracker split, not "run the refiner every Nth frame"

Both were considered. Running FootNet every Nth frame keeps the whole-frame body model on the
critical path, so it caps out at the body model's rate. The split removes the body model from the
common case entirely: 6 fps → 28–30 fps.

## Floor-plane back-projection, not depth from apparent size

In the mirror scenario, solving depth from heel-to-toe length is ill-conditioned. A standing foot's
landmarks sit at known heights above the floor, so gravity plus a camera height gives a 3D point
per image point from any view. See the `shoe-try-on` skill.

Camera height is **assumed** (1.3 m), not measured. Open.

## Guessing the hidden heel: rejected, twice

| Approach | Error | Verdict |
|---|---|---|
| Axis extrapolation from toes and extrema | 54 mm mean (22 % of foot length) | Rejected |
| 2D similarity fit of the template, 237 views | **56° direction error** | Rejected |
| 3D fit on the floor plane | 25 mm, 4.5° direction | Viable, not implemented |

The app instead uses RTMPose's ankle when FootNet's heel is missing: a worse landmark, but a real
observation rather than a guess.

## Crop context 1.8×, not the 1.3× training validated at

Training saw crops 1.15–1.7× the keypoint box and validated at 1.3. On 94 real feet the wider 1.8
crop measured better: 3.9 of 8 confident points against 3.0, and a fifth of feet with nothing at
all instead of a third. The number traces to that measurement, not to the training config.

## One debug screen, one toggle

Toggles for foot side, mirroring and model choice were removed once each had been decided by
measurement. What remains is the layer toggle (arrows / 3D / both). A toggle is a decision not yet
made; keeping it after the decision spreads the branches through the code.

## Open, with no decision yet

- Retraining FootNet **with footwear** — the largest known quality gap. Needs shoe meshes fitted to
  the FIND template in the renderer; the only shoe asset in the repo today is a placeholder.
- Converting RTMPose to Core ML the way FootNet was.
- Cutting FootNet's decoder to 64×64 heatmaps.
- Android.
- A landscape frame (1472×828) drawing points far from the feet. Not diagnosed.
