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
| 2026-09-25 18:01 | Example camera with synchronous mask preview selected, screen debug counter | 4.1–4.4 |
| 2026-09-25 21:25 | Same preview, later screen recording after timing breakdown was added | 10.8–11.5 |
| 2026-09-25 22:10 | Same preview, new screen recording after crop-alignment retention change | 8.3–11.3 |

These figures are the debug UI's delivered-sample rates from iPhone screen recordings, not Instruments
traces. They include the rest of the app's frame path. The camera movement and number of active foot
crops differ, so they are not controlled before/after comparisons and do not establish an FPS gain.

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

## Mask-preview latency

| Path | Where | Latency | Date |
|---|---|---|---|
| footmask-v1 preview path | iPhone, first recording's active crop(s), screen debug counter | 133–142 ms/frame | 2026-09-25 18:01 |
| footmask-v1 preview path | iPhone, later recording's active crops, screen debug counter | 46.9–49.0 ms/frame | 2026-09-25 21:25 |
| footmask-v1 preview path | iPhone, new recording's active crops, screen debug counter | 46.9–52.5 ms/frame | 2026-09-25 21:52 |
| footmask-v1 preview path | iPhone, new recording's active crops, screen debug counter | 45.7–53.2 ms/frame | 2026-09-25 22:10 |

The timer wraps the Core ML prediction, float16 output conversion, threshold/downsample and native
overlay storage for every active crop in that frame; UIKit/Core Animation image creation and drawing
are outside it. In the later recording, the last crop reported 9.4–11.6 ms for the Core ML call and
12.6–12.7 ms for native post-processing. The full-foot refinement was 79–81 ms and includes this
mask work. In the 21:52 recording, sampled active-crop frames delivered 10.1–10.8 fps, with 84–94 ms
for full-foot refinement; the last crop reported 9.9–14.6 ms for Core ML and 11.2–12.7 ms for
post-processing. A sampled frame with no active crops showed 14.4 fps and 0 ms total mask time, while
the status text still displayed the last crop's timing. Scene, crop count, movement and mask threshold
differ between recordings, so these values do not establish a before/after speed change. The latest
recording also shows masks displaced from moving feet in sampled frames. This is qualitative evidence;
without real mask labels it cannot distinguish stale retained overlays from model spill. No
post-processing optimization has been measured on a phone yet. This is a diagnostic path, not the
production shoe path.

In the 22:10 recording, sampled active-crop frames delivered 8.3–11.3 fps and 81–91 ms full-foot
refinement. The last crop reported 9.2–16.6 ms for Core ML and 11.5–13.8 ms for post-processing.
The threshold changes during the recording. Some sampled frames at 0.7 align the overlay with most
of a foot; other moving frames spill onto the shin or nearby objects. This is qualitative review
without real mask labels and does not isolate the effect of the crop-alignment retention change.

## FootNet accuracy

Validation, `research/foot-3d/results/footnet/log.csv`. Median and p90 are keypoint error in pixels
at 256×256.

| Date | Training | SynFoot median | Render median | Render p90 | Render IoU |
|---|---|---|---|---|---|
| 2026-09-22 | 20 epochs, SynFoot only | 2.1 px | 11.7 px | 113 px | 0.73 |
| 2026-09-22 | + 8 epochs with 7 254 render frames | 2.3 px | 3.3 px | 16 px | 0.93 |
| 2026-09-23 | + 8 epochs with 20 000 render frames | 2.330 px | 2.938 px | 11.824 px | 0.942 |

The 20 000-frame run started at 17:33 EEST and completed by 20:42 EEST; all eight epochs are in
`research/foot-3d/results/footnet/log.csv`. `best.pt` is selected by the lowest SynFoot median
(2.304 px at epoch 7), while epoch 8 `last.pt` has the best render p90 (11.824 px). These are
synthetic validation metrics only. The real-frame comparison is recorded below; it does not replace
on-device validation.

### Real-label checkpoint comparison

2026-09-23, PyTorch on Mac CPU (`torch.backends.mps.is_available()` was false). Errors are per point,
normalized by the labelled toe-to-heel distance. The oracle crop uses the labelled points; the seed
crop uses an offline RTMPose-M full-frame pass and the iOS detector's joint threshold/context as a
one-frame proxy. It does not reproduce the native tracker's crop history or run on the phone.

The comparison uses 156 feet with both labels after excluding 19 incomplete label records and the
whole `IMG_3244` / `IMG_3246` clips, which the project handoff flags for child appearances in some
frames. RTMPose produced a seed crop for 153 of 156 feet (98.1 %).

| Checkpoint | Oracle median / p90 | RTMPose seed median / p90 | Oracle score ≥0.3 | Seed score ≥0.3 |
|---|---|---|---|---|
| `v2-renders-7k.pt` | 0.1575 / 0.3821 | 0.1947 / 1.1420 | 53.8 % | 50.7 % |
| 20k `best.pt` | 0.1539 / 0.3837 | 0.2020 / 1.1629 | 57.4 % | 50.3 % |
| 20k `last.pt` | 0.1545 / 0.3904 | 0.1938 / 1.1631 | 57.4 % | 52.0 % |

The 20k checkpoints slightly improve the oracle median and score coverage, but do not improve p90
on either crop protocol. This does not support exporting a new model to the phone yet. Per-view
rows and the full command are in `research/foot-3d/footnet/audit_checkpoints.py` and the ignored
output `research/foot-3d/results/footnet/audit-checkpoints.csv`.

On the developer's own frames after the 2026-09-22 run, the FIND template fits to **0.3–2 px in the
mirror** and **1–3 px from above**, bare or in socks. **Shoes and the side view fail** — no dataset
has footwear.

On the device the per-point scores are **bimodal**: 0.6–0.9 or 0.0–0.15, little between.

## Shoe placement on the device

From a 28 s screen recording on 2026-09-23, mirror and top-down, in socks, with the debugger
attached. The numbers come from the per-second `[shoe]` log (`ShoeLayer.tsx`), which reports the
shoe length the image implies (`impliedShoeLengthM`) and the spread of the shoe's origin.

| Quantity | Value | How |
|---|---|---|
| Implied shoe length, locked on | **290 mm**, range 281–306 | 9 windows with ≥10 frames and spread ≤5 mm |
| Implied shoe length, bad frames | 385–2174 mm | every window where the model was unsure |
| Jitter of the shoe's origin | x ~15, y ~6, z ~15 mm | standard deviation over one second, standing still |
| Developer's foot | 276 mm; 280 mm insole | tape measure |

A sneaker with a 280 mm insole is about 290–300 mm outside, so the measured 290 mm is consistent
with the assumed 1.3 m camera height. It does **not** confirm it: the landmark fractions in
`shoePose.ts` are estimates too, and one reading cannot separate two unknowns. Measuring the phone's
height with a tape and re-reading would.

The plausibility window in `ShoeLayer.tsx` (180–330 mm) comes from this table: it separates every
good window from every bad one in this recording, with a wide margin on both sides.

Quality tracked the on-screen `N/16` exactly: 8/16 put both shoes correctly on the feet, 2/16 drew
nothing, 0/16 drew a shoe at the wrong size and angle. The last case is what the plausibility
window now rejects.

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

## Decoder and coordinate checks (2026-09-24)

| Check | Result |
|---|---|
| Old Swift decoder vs PyTorch, 156 labelled crops / 1,248 joints | 0.095 px median, 0.232 px p90, 3.149 px max coordinate delta; score p90 0.000184 |
| Updated Swift vs PyTorch on identical float16-rounded logits | 0.000009 px median, 0.000020 px p90, 0.000045 px max; score max 0.0000001 |
| Updated Swift vs full-precision PyTorch, including float16 peak ties | 0.000307 px p90; rare argmax switches up to 4.769 px on `last.pt` |
| Core ML CPU-only package vs PyTorch, 16 SynFoot crops, original `v2-renders-7k.pt` | 0.010 px median, 0.985 px max |
| Swift crop → frame pixel → normalized coordinate probe (1472×828) | Pass |
| JS frame → portrait view round trip (1472×828 to 393×852) | Pass |

The exporter now returns logits, the native decoder turns the peak logit into its confidence score,
and the locally compiled `model/footnet.mlmodelc` was regenerated with the original 7k weights.
These are CPU/Mac checks; the new bundle has not been exercised in an iOS build or on a phone.

The 156 complete toe/heel labels are spread across six eligible source clips (16–34 feet per clip).
Five view labels are represented; `top` appears in two clips. The pilot used source-level holdouts,
but the small number of clips is not enough to claim generalization.

## Real-image ablation and partial-label pilot (2026-09-24)

All errors below are measured on the same labelled toe/heel coordinates with the original
`v2-renders-7k.pt` weights. Oracle crops remove RTMPose crop placement; seed crops are a single-frame
CPU proxy and omit three top-view feet with no RTMPose crop.

| Transform | Oracle median / p90 | RTMPose seed median / p90 |
|---|---:|---:|
| Baseline | 0.1575 / 0.3821 | 0.1947 / 1.1420 |
| Downsample at 0.2866× then upsample | 0.1499 / 0.3780 | 0.1895 / 1.0846 |
| Reduce saturation to measured 18.7 median | 0.1628 / 0.3924 | 0.1899 / 1.1851 |
| Gaussian blur to Laplacian std ≈9.9 | 0.1565 / 0.3795 | 0.1952 / 1.1540 |
| Non-local denoise to Laplacian std ≈9.9 | 0.1588 / 0.4035 | 0.2018 / 1.1928 |

The downsample ratio comes from the measured source pixels per foot: 243 px synthetic vs 848 px real.
It overshoots the sharpness target (resulting median Laplacian std 2.60), so the small gain does
not isolate source resolution from loss of detail. Denoise strength was calibrated on 16 evenly
spaced crops, then measured across the full sets. None of the transforms is a deployment fix.

The partial-label pilot started from the 7k checkpoint, trained on 111 seed crops from four clips,
validated on 16 crops from `IMG_3248`, and tested on 26 crops from `IMG_3250`. It used 8 epochs at
the earlier render fine-tune learning rate of 3e-4 and mixed one fully labelled render batch into
each real batch. Metrics count only landmark labels inside each crop.

| Split | Model | Visible points | Median | p90 | Score ≥0.3 |
|---|---|---:|---:|---:|---:|
| Validation | Baseline | 31 | 0.1660 | 0.3504 | 54.8 % |
| Validation | Fine-tuned epoch 7 | 31 | 0.1705 | 0.2970 | 45.2 % |
| Test | Baseline | 48 | 0.1681 | 1.3899 | 43.8 % |
| Test | Fine-tuned epoch 7 | 48 | 0.1380 | 1.5241 | 39.6 % |

The median improvement on one held-out clip did not carry to its tail or confidence recall. Keep
the original checkpoint until there are more source clips and footwear labels.

## From-scratch visible-foot mask experiment (2026-09-25)

`footmask-v1` uses a MobileNetV3-Small U-Net with mask, landmark and side outputs. It was initialized
randomly and trained on 45,600 per-foot crops from 19,000 owned Blender frames; 2,400 crops from
1,000 other frames were held out. MPS was available during training and evaluation. The 20-epoch
run took about 2 h 45 min total (epoch times varied from 7.3 to 11.2 min). `best.pt` is epoch 16,
selected by mean crop Dice at threshold 0.5; epoch 20 is `last.pt`.

| Mask threshold | Dice, all 2,400 crops | Dice, positive only | Pixel precision, all | Pixel recall | Empty-crop false-positive area |
|---|---:|---:|---:|---:|---:|
| 0.3 | 0.8993 | 0.8870 | 0.8371 | 0.9611 | 0.376 % |
| 0.5 | 0.9048 | 0.8936 | 0.8672 | 0.9438 | 0.274 % |
| 0.7 | 0.9057 | 0.8942 | 0.8934 | 0.9206 | 0.195 % |

The all-crop Dice and precision include 400 empty crops. The positive-only Dice excludes them;
empty-crop false-positive area is reported separately. Threshold 0.5 positive-crop groups:

| Holdout group | Dice | Pixel precision | Pixel recall |
|---|---:|---:|---:|
| Mirror | 0.9144 | 0.8899 | 0.9479 |
| Top | 0.8930 | 0.8852 | 0.9502 |
| Third person | 0.8538 | 0.8255 | 0.9301 |
| Bare | 0.8952 | 0.8752 | 0.9460 |
| Sock | 0.8925 | 0.8688 | 0.9423 |
| Trousers | 0.8979 | 0.8787 | 0.9451 |
| No trousers | 0.8849 | 0.8554 | 0.9408 |

Across visible labelled landmarks, median error is 5.66 px and p90 is 14.88 px at 256×256;
side accuracy is 91.0 %. These are synthetic holdout figures. They do not score real mask edges.
`research/foot-3d/results/footmask-v1/final-validation.json` and `last-validation.json` hold the
full threshold sweeps for best and last.

The Core ML package and compiled model are each 2.8 MB. On the M1 Pro, per-crop prediction takes
1.25 ms with `.all`, 1.24 ms with `CPU_AND_NE`, and 3.82 ms with `CPU_ONLY`. This is a Mac
measurement, not iPhone 11 latency or proof of ANE placement. On 16 held-out crops, Core ML and
PyTorch disagree on the mask-logit sign at 0.022 % of pixels. Of 78 visible labelled landmarks,
10 have score ≥0.3 in both frameworks; these coordinates differ by at most 0.029 px. One
low-confidence point (score 0.005) differs by 101.6 px; no labelled point crosses the 0.3 score
threshold between frameworks. Output absolute error p99 / max is 2.19 / 2.26 for mask logits,
6.93 / 7.34 for heatmaps and 0.119 / 0.121 for the side logit, so this is decision-level mask
parity on a small sample, not full-precision logit equality. Core ML conversion warned that
PyTorch 2.11 is newer than the latest version tested by the installed coremltools (2.7);
conversion and prediction completed. The compiled `.mlmodelc` also loaded in a Swift Core ML
smoke check on the Mac and returned the expected `[1,1,256,256]` mask, `[1,8,256,256]` heatmaps
and `[1,1]` side outputs; this is not an iOS build or phone trace.

The real-frame review sampled one middle frame from each of 23 prepared clips. RTMPose returned 42
seeded feet out of 46 possible. Some predicted masks miss the selected sock or spill onto adjacent
regions at thresholds 0.5 and 0.7. This is qualitative only: the real frames have no foot-instance
mask labels, and RTMPose seeding means the experiment does not test full-frame acquisition.
`results/footmask-v1/real-review.jpg` and `capture-review/contact-sheet.jpg` are ignored local
artifacts.

An iPhone 11 latency trace, per-layer execution-provider proof, real dense-mask precision/recall,
and live temporal-stability measurement have not been obtained.

## Unmeasured, and known to be

These are assumptions in the code. Any estimate depending on them inherits their uncertainty.

- **Camera height 1.3 m** (`ShoeLayer.tsx`) — assumed, a phone at chest height. It no longer
  affects what is drawn, because the shoe is drawn at the length the same construction measures and
  the two scale together; it still has to be right to report a real size in millimetres.
- **Shoe length** is no longer assumed: it is measured per foot and smoothed. `SHOE_LENGTH_M = 0.29`
  is only where the measurement starts. A catalog sole length would replace the measurement for a
  real try-on, where seeing a size that does not fit is the point.
- **Landmark heights on the shoe** — ankle ≈ 28 % of length up and a quarter along, little toe
  ≈ 80 % along (`shoePose.ts`) — anatomical estimates, not measurements.
- Scene-light and grain constants in `shoe-stage` — "estimates to tune on device", per its README.
