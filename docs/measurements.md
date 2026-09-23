# Measurements

Every number this project relies on, with the date and how it was obtained. The rule: a number that
traces only to code is an implementation artifact, not a requirement. If a number is not here, it
has not been measured.

Device is an **iPhone 11 (A13)** unless stated. Mac is an **M1 Pro**.

## Frame rate, end to end

| Date | Pipeline | fps |
|---|---|---|
| 2026-09-22 | RTMPose every frame + FootNet refinement | 6 |
| 2026-09-23 | FootNet every frame per foot, RTMPose only when a foot is lost | 28–30 |

## Model latency

| Model | Where | Latency | Date |
|---|---|---|---|
| RTMW x-l fp16 | iPhone, Core ML EP | ~150 ms/frame | 2026-09-18 |
| RTMW x-l fp32 | Mac CPU | 66–68 ms | 2026-09-18 |
| RTMW x-l fp16 | Mac, Core ML EP | 18 ms | 2026-09-18 |
| RTMPose-m fp16 | Mac, Core ML EP | 6 ms | 2026-09-19 |
| RTMPose-m fp32 | Mac CPU | 18 ms | 2026-09-19 |
| FootNet, ONNX Runtime CoreML EP | iPhone | 65 ms/foot (44 ms ANE vs 20 ms CPU on the Mac) | 2026-09-23 |
| **FootNet, Core ML** | **iPhone** | **15 ms/foot** | 2026-09-23 |
| FootNet, Core ML | Mac, `.all` | 1.9 ms | 2026-09-23 |
| FootNet, Core ML | Mac, `CPU_AND_NE` / `CPU_ONLY` | 4.6 / 7.1 ms | 2026-09-23 |

The ONNX Runtime figure is the one that mattered: its Core ML EP produced **22 graph partitions**.

## FootNet accuracy

Validation, `research/foot-3d/results/footnet/log.csv`. Median and p90 are keypoint error in pixels
at 256×256.

| Date | Training | SynFoot median | Render median | Render p90 | Render IoU |
|---|---|---|---|---|---|
| 2026-09-22 | 20 epochs, SynFoot only | 2.1 px | 11.7 px | 113 px | 0.73 |
| 2026-09-22 | + 8 epochs with 7 254 render frames | 2.3 px | 3.3 px | 16 px | 0.93 |
| 2026-09-23 | + 8 epochs with 20 000 render frames | *in progress* | | | |

On the developer's own frames after the 2026-09-22 run, the FIND template fits to **0.3–2 px in the
mirror** and **1–3 px from above**, bare or in socks. **Shoes and the side view fail** — no dataset
has footwear.

On the device the per-point scores are **bimodal**: 0.6–0.9 or 0.0–0.15, little between.

## Export parity

FootNet PyTorch vs Core ML, 2026-09-23: median **0.001 px**, max **0.974 px**, 9.3 MB. Above ~1 px
means the conversion broke something.

## Crop context

2026-09-23, 94 real feet:

| Context | Confident points of 8 | Feet with nothing |
|---|---|---|
| 1.3× (training validation) | 3.0 | a third |
| **1.8× (shipped)** | **3.9** | a fifth |

## Heel estimation attempts

2026-09-23, on SynFoot views:

| Approach | Error |
|---|---|
| Axis extrapolation | 54 mm mean (22 % of foot length) |
| 2D similarity fit, 237 views | 56° direction error |
| 3D fit on the floor plane | 25 mm, 4.5° direction |

## Dataset sizes

2026-09-23, after the 20 000-frame render:

| Set | Count |
|---|---|
| SynFoot train crops | 43 801 |
| SynFoot validation crops | 2 000 (one held-out scanned foot, `0033-A`) |
| Render frames | 20 000 (19 000 train / 1 000 validation, every 20th held out) |
| Render crops | 33 115 train / 1 750 validation |

Render composition, sampled: `mirror` ~52 %, `top` ~29 %, `third` ~19 %; socks ~58 %; trousers
~65 %; **shoes 0 %**.

## Training throughput

| Setup | Cost |
|---|---|
| SynFoot only, M1 Pro, fp32 | 0.45 s per batch of 32 |
| With renders mixed in | 0.78 s/step — JPEG and PNG decode per item |
| Blender render | 3.3–3.5 s/frame (6–9 s when training shares the GPU) |

## Unmeasured, and known to be

These are assumptions in the code. Any estimate depending on them inherits their uncertainty.

- **Camera height 1.3 m** (`ShoeLayer.tsx`) — assumed, a phone at chest height.
- **Shoe length 0.29 m**, about EU 43 — placeholder until catalog sole lengths land.
- **Landmark heights on the shoe** — ankle ≈ 28 % of length up and a quarter along, little toe
  ≈ 80 % along (`shoePose.ts`) — anatomical estimates, not measurements.
- Scene-light and grain constants in `shoe-stage` — "estimates to tune on device", per its README.
