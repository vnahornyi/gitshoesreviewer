# Audit: what stands between this and an AR shoe that stays on the foot

2026-09-23. An end-to-end review of the pipeline against one criterion — **on a live camera the shoe
looks stuck to the foot, moves and scales with it, and does not flicker** — rather than against a
validation score.

Everything below is labelled. **FACT** is read off the code. **MEASUREMENT** was run for this audit
and can be re-run. **HYPOTHESIS** is untested. **CONCLUSION** is an inference, with its confidence.

The measurements here were made with `assets/checkpoints/v2-renders-7k.pt` — the checkpoint that was
on the phone — on the 189 hand-labelled real feet in `assets/capture/labels.json`. Errors are given
as a fraction of the foot's own length, so they transfer across distances and devices. On this
developer's 276 mm foot, **0.1 ≈ 28 mm**.

## The method, so the numbers can be judged

Each real foot has two hand-clicked points: the big toe and the back of the heel. For every
measurement the crop is built **from the ground truth**, 1.8× the toe-to-heel span, exactly as
`FootNetRunner` builds it in the app. That is deliberately generous: it removes the detector from
the comparison and measures the model's localisation alone. This is an oracle-crop reference, not
an upper bound for every individual frame. A follow-up RTMPose seed-crop proxy below measures how
crop placement changes the aggregate result.

The label file has 208 foot records; 19 lack one or both labelled points and are excluded from the
189-foot measurements below. The 0.1 → 28 mm conversion is a foot-length equivalent for this
developer's foot, not a direct measurement of 3D shoe displacement.

### Follow-up: the 20k-render checkpoint

The original measurements below use `v2-renders-7k.pt`. A further eight-epoch run on 20 000
synthetic renders completed on 2026-09-23. Its held-out-render p90 improved, but the real-label
checkpoint comparison did not improve p90. The comparison used 156 complete feet, excluding
`IMG_3244` and `IMG_3246` because the project handoff flags child appearances in some frames. It
ran on Mac CPU. The RTMPose seed crop is a one-frame proxy; it does not run the iOS crop tracker.

| Checkpoint | Crop | n feet | Median | p90 | Score ≥0.3 |
|---|---|---:|---:|---:|---:|
| `v2-renders-7k.pt` | Oracle | 156 | 0.1575 | 0.3821 | 53.8 % |
| 20k `best.pt` | Oracle | 156 | 0.1539 | 0.3837 | 57.4 % |
| 20k `last.pt` | Oracle | 156 | 0.1545 | 0.3904 | 57.4 % |
| `v2-renders-7k.pt` | RTMPose seed | 153 | 0.1947 | 1.1420 | 50.7 % |
| 20k `best.pt` | RTMPose seed | 153 | 0.2020 | 1.1629 | 50.3 % |
| 20k `last.pt` | RTMPose seed | 153 | 0.1938 | 1.1631 | 52.0 % |

Reproduce with `uv run python -m footnet.audit_checkpoints` from `research/foot-3d/`; the per-view
CSV is written under the ignored `results/footnet/` directory.

### Follow-up: image-domain ablation and partial-label fine-tune

**MEASUREMENT.** One-factor image transforms on the original 7k checkpoint; values are median / p90
landmark error in foot lengths. Oracle uses 156 GT crops; seed uses 153 RTMPose crops. The seed
proxy does not reproduce native crop tracking.

| Input | Oracle median / p90 | Seed median / p90 |
|---|---:|---:|
| Baseline | 0.1575 / 0.3821 | 0.1947 / 1.1420 |
| Downsample then upsample | 0.1499 / 0.3780 | 0.1895 / 1.0846 |
| Match median saturation | 0.1628 / 0.3924 | 0.1899 / 1.1851 |
| Gaussian blur | 0.1565 / 0.3795 | 0.1952 / 1.1540 |
| Non-local denoise | 0.1588 / 0.4035 | 0.2018 / 1.1928 |

The resolution factor `243 / 848 = 0.2866` and saturation factor `18.7 / 97.5 = 0.1918` come
from the measured synthetic/real pixels-per-foot and saturation statistics above. The downsampled
crops ended at Laplacian std 2.60, far below the synthetic target 9.9, so this is an intentionally
strong stress test rather than a calibrated final augmentation. It gives a small improvement, but
paired per-foot median changes are near zero and only about half the feet improve. Matching
saturation, blur, and denoise did not improve aggregate errors. This does not prove resolution is
the cause; it does rule out blindly applying the measured median transforms as a fix. Reproduce with
`uv run python -m footnet.ablate_real_domain`; per-view results are in ignored
`results/footnet/domain-ablation.csv`.

**MEASUREMENT.** A diagnostic partial-label fine-tune started from `v2-renders-7k.pt`, trained on
111 RTMPose seed crops from four clips, validated on 16 crops from `IMG_3248`, and kept
`IMG_3250` (26 crops) untouched for test. Real loss supervised only the big toe and heel; each
update also replayed a fully labelled synthetic batch. The 8-epoch, CPU run selected epoch 7 by
validation median.

| Split | Checkpoint | Median | p90 | Score ≥0.3 |
|---|---|---:|---:|---:|
| Validation, 31 visible points | 7k baseline | 0.1660 | 0.3504 | 54.8 % |
| Validation, 31 visible points | Fine-tuned | 0.1705 | 0.2970 | 45.2 % |
| Test, 48 visible points | 7k baseline | 0.1681 | 1.3899 | 43.8 % |
| Test, 48 visible points | Fine-tuned | 0.1380 | 1.5241 | 39.6 % |

The test median improved but p90 worsened and fewer points cleared the existing score threshold.
The single test clip is small, and its tail error is already above one foot length. Do not promote
this checkpoint. The pilot shows the current labels can drive partial-label training, but a larger,
more varied held-out set is needed before choosing a real fine-tune over annotation. The run log is
under ignored `assets/checkpoints/real-finetune-156/metrics.csv`; its checkpoints are kept there
as a diagnostic experiment and are not promoted for app export.

## 1. FootNet has not learned anything wrong. It has learned the wrong world

**MEASUREMENT.** Same protocol, same points, three sets:

| Test | Set | n | median | p90 |
|---|---|---|---|---|
| A | Synthetic renders **seen during training** | 380 | **0.017** | 0.052 |
| B | Synthetic renders **held out** | 380 | **0.017** | 0.048 |
| C | **Real frames** | 378 | **0.136** | 0.292 |

**CONCLUSION, high confidence for these splits.** A and B are similar, so this test shows no
measurable overfitting on the sampled synthetic split. C is about eight times worse. The
synthetic-versus-real performance gap is measured; its individual causes are not isolated by this
comparison.

**MEASUREMENT.** The gap is scatter, not bias:

| | bias along | bias across | scatter along | scatter across |
|---|---|---|---|---|
| Synthetic holdout | +0.002 | −0.002 | 0.027 | 0.021 |
| Real | −0.056 | +0.007 | **0.177** | **0.096** |

**CONCLUSION, high.** A small bias would mean the human labeller and the renderer disagree about
where "the big toe" is. The bias is small and the scatter grew 6.6×. The model does not know where
the point is.

## 2. What the error actually does to a shoe

A pose has a direction, a position and a scale, and they fail differently. Splitting each foot's
error into the part both points share and the part that differs:

**MEASUREMENT**, n = 189 real feet:

| Quantity | median | p90 | on a 276 mm foot |
|---|---|---|---|
| Direction of the foot | **4.9°** | 17.2° | — |
| Common shift (both points together) | 0.103 | 0.223 | 28 mm |
| Differential (turns and stretches) | 0.160 | 0.345 | 44 mm |
| **Length of the foot** | **0.285** | 0.614 | **79 mm** |

**CONCLUSION, medium-high.** The three parts of the two-point estimate are not equally noisy on this
label set:

- **Direction is good.** 4.9° median, 6 % of feet worse than 30°. This matches the recording, where
  the shoe points the right way whenever it is drawn at all.
- **Position is moderate.** 28 mm of common drift is what "the shoe slides off the foot" is.
- **Scale is the worst thing in the system.** The predicted toe-to-heel distance is off by 28 % per
  frame. Almost all of the differential error is along the foot's axis, not across it — which is
  why the direction survives while the length does not.

**CONCLUSION, medium.** The shoe's length is derived from the same two points with the largest
frame-wise error. Temporal smoothing can reduce uncorrelated variation, but this point-error
comparison does not establish the remaining error after one second. Keep the smoothed-size result
as a separate device measurement.

## 3. Three things we believed that are not true

### RTMPose is not more accurate than FootNet

**MEASUREMENT.** RTMPose-m on the whole frame — the app's own configuration — on the **same 378
points**, same metric:

| | median | p90 | recall at score ≥ 0.3 | latency |
|---|---|---|---|---|
| FootNet, oracle crop | **0.136** | **0.292** | 55 % | 15 ms/foot (device) |
| RTMPose, whole frame | 0.149 | 0.705 | **73 %** | 20 ms (Mac CPU), 41–77 ms (device) |

**CONCLUSION, medium.** With a ground-truth crop, FootNet has lower measured p90 than RTMPose on a
whole frame. These are different crop conditions, so this does not establish a deployment
head-to-head. RTMPose's higher recall supports its detector role; comparing both models after the
same crop procedure would answer localization accuracy more cleanly.

Earlier in this project's own discussion the opposite was asserted, from watching the debug screen.
That was wrong, and it was stated with more confidence than the evidence supported. The
detector/tracker split is sound; the numbers say so.

### The confidence score is not broken

**MEASUREMENT.** Error against score bucket, real feet:

| score | n | share | median error |
|---|---|---|---|
| 0.0–0.1 | 89 | 24 % | 0.309 |
| 0.1–0.3 | 80 | 21 % | 0.205 |
| 0.3–0.5 | 34 | 9 % | 0.131 |
| 0.5–0.7 | 71 | 19 % | 0.149 |
| 0.7–1.0 | 104 | 28 % | 0.128 |

**CONCLUSION, medium.** Error is generally lower in higher-score buckets, but not strictly
monotonic (`0.131` at 0.3–0.5 versus `0.149` at 0.5–0.7). This supports a useful confidence signal;
it does not prove that 0.3 is the optimal threshold or rule out calibration issues.

### More information per frame does not help — the opposite

The obvious fix for two noisy points is to use more evidence. It was tested.

**MEASUREMENT.** Foot direction from three representations, same 189 real feet:

| Representation | median | p90 | worse than 30° |
|---|---|---|---|
| **Two keypoints (what we do now)** | **4.9°** | **17.2°** | **6 %** |
| All 8 keypoints, weighted by score | 6.3° | 40.4° | 16 % |
| Predicted mask, PCA of the silhouette | 10.2° | 52.0° | 22 % |

**CONCLUSION, high.** Redundancy helps when errors are independent. These are not: the eight points
are six toes clustered at one end plus two at the back, so their spread is mostly across the foot,
and the mask carries the same domain gap as the points. **Using more of the model's output makes the
direction worse.** The current choice of two points is not a shortcut — it is the best of the three.

This kills the "export the mask and fit the pose to it" plan that looked obvious before it was
measured. The mask may still be worth exporting for **occlusion**, which is a different job.

## 4. Train/inference mismatches

**MEASUREMENT.** The old Core ML contract returned sigmoid probabilities, while training refined
the 5×5 window with a softmax over logits. On 156 labelled oracle crops (1,248 joints) across the
7k, `best`, and `last` checkpoints, the actual Swift decoder differed from PyTorch by a median of
0.095 px, p90 0.232 px, and a maximum of 3.149 px. Score differences were much smaller: median
0.000048, p90 0.000184. The export check did not catch this because it ran the PyTorch decoder on
both outputs.

**FIXED AND MEASURED.** Core ML now exports float16 logits, and the shared Swift decoder applies
the same local softmax and sigmoid peak score as training. On the same 1,248 joints, comparing both
decoders on identical float16-rounded logits gives a coordinate p90 of 0.000020 px, maximum
0.000045 px, and score maximum delta 0.0000001. Against full-precision PyTorch, p90 is 0.000307 px;
rare half-precision ties can choose a different peak (maximum 4.769 px on the `last` checkpoint).
The existing Core ML export check on 16 SynFoot crops measured 0.010 px median and 0.985 px maximum
against full-precision PyTorch. The app's local compiled model was rebuilt with the original
`v2-renders-7k.pt` weights to isolate this decoder fix; no newer checkpoint was selected. This is
not an on-device measurement.

**FACT.** Crop borders differ. Training uses `BORDER_REPLICATE`; the app leaves the outside
**black**. Deliberate, but it is a mismatch for any foot near the frame edge.

**FACT.** [`ios/README.md`](../ios/README.md) says the crop is 1.45× the box. The code says
`context = 1.8`. The README is the stated contract for that layer and it is wrong.

**MEASUREMENT.** Added a Swift probe for crop coordinates → frame pixels → normalized frame
coordinates and a `frameToView` round-trip test for an off-centre point from a 1472×828 landscape
frame into a 393×852 portrait view. Both pass. These test the exact mapping helpers, but not the live
camera frame dimensions, bridge payload, preview orientation, or layout together. The landscape
misplacement in [state.md](state.md) therefore remains an on-device integration issue; the
coordinate formulas alone did not reproduce it.

## 5. The synthetic distribution does not cover production

**MEASUREMENT.** 300 synthetic crops against 50 real ones, both at 256×256:

| | synthetic | real | ratio |
|---|---|---|---|
| High-frequency detail (σ of Laplacian) | 9.9 | 28.2 | **2.9×** |
| Contrast | 21.2 | 30.5 | 1.4× |
| Saturation | 18.7 | 97.5 | **5.2×** |

**MEASUREMENT.** Source pixels per foot, against a 256×256 network input:

```
synthetic   median 243 px   ·  55 % of crops are SMALLER than the input
real        median 848 px   ·   8 % are smaller
frame:      render 720×960  ·  device 828×1472
```

Most training crops are **upsampled** into the network; real ones are **downsampled**. The render
frame has half the pixels of the frame the app sees.

*Caveat on method: real boxes were built from 2 labelled points and synthetic ones from 8, so the
multipliers differ and the crop medians carry that error. The frame resolutions do not.*

**FACT.** The augmentation moves the data away from reality on both axes it could address:
saturation is scaled by 0.6–1.4 (from a median of 18.7 that reaches ~26 at most, against 97.5 real),
and 30 % of samples are additionally blurred. It cannot close either gap and widens the larger one.

**HYPOTHESIS, medium.** `use_denoising = True` at `SAMPLES = 24` removes the high-frequency content
a real sensor has. Not isolated — see experiment 3.

**CONCLUSION, medium.** These are correlations with a plausible mechanism, not proven causes. No
re-render should be started until an ablation names which factor actually moves the error.

## 6. What the renderer actually needs

The floor-plane construction needs, per foot per frame: **two image points, gravity, and the camera
intrinsics.** Scale now comes from those same two points.

Given §3, the case for a richer representation is weaker than it looks: more of this model's output
makes the direction worse. The ranking that matters is instead:

1. **Direction** — already good enough (4.9°). Do not spend on it.
2. **Position** — 28 mm of common drift. Moderate.
3. **Scale** — 28 % per frame. The worst, and the most visible as "the shoe does not fit".
4. **Occlusion** — the person matte handles the leg; a foot mask would handle a trouser leg and the
   other foot. This is the one job the unused mask head is still a candidate for.

**CONCLUSION, medium-high.** The representation is not the bottleneck. The **quality of the two
points on real frames** is, and scale most of all.

## 7. Experiments, by information per hour

### No training needed

| # | Experiment | Answers | Status |
|---|---|---|---|
| 1 | Swift decoder vs `decode_points` on identical heatmaps | Is the soft-argmax mismatch real, and how large | **Done** — logits contract fixed; parity measured |
| 2 | Coordinate round-trip through Swift and `frameToView` | Are the mapping helpers wrong? | **Done** — helper tests pass; live landscape issue remains |
| 3 | **Ablation on real frames**: downsample→upsample, desaturate, blur, denoise, one factor at a time | Which synthetic property causes the gap | **Done** — only downsampling gave a small, non-conclusive gain |
| 4 | Ablation on synthetic: sharpen, saturate, sensor noise, JPEG | The same from the other side | Later |
| 5 | End-to-end latency on device, camera to drawn shoe | Never measured | Later |
| 6 | fps without a debugger attached | The 13–20 fps figure is confounded | Later |

### Short training

| # | Experiment | Answers | Status |
|---|---|---|---|
| 7 | **Fine-tune on the complete real feet**, split by clip and not by frame | Does partial real supervision improve held-out real frames? | **Pilot done** — median improved on test, p90 and confidence recall worsened; do not promote |
| 8 | Drop the ±180° rotation augmentation | Capacity spent on poses production never shows | Later |
| 9 | RTMPose pseudo-labels on the 557 prepared frames | Cheap real supervision — though at p90 0.705 the teacher is noisy | Later |

### Needs a new render — only after 3 and 4

Higher resolution, more samples, weaker denoising. 18+ hours, and only with a named cause.

### Needs annotation

Active learning: label the frames where confidence is low or where FootNet and RTMPose disagree.
Not random frames.

## 8. Target architecture, and how far it is from here

The structure is right. The changes are narrower than the earlier discussion assumed:

| | Now | Target | Why |
|---|---|---|---|
| Split | RTMPose detects, FootNet tracks | unchanged | Measured correct |
| Representation | 2 points of 8 | unchanged | The alternatives measured worse |
| Scale | measured per frame, smoothed | unchanged, but sized from a long average for any reported size | 28 % per-frame error |
| Filtering | per point, then per pose | filter the **pose** — position, scale, rotation as one coupled state | They are physically coupled and must not jump independently |
| Occlusion | person matte only | add the foot mask | Trouser legs, the other foot |
| Data | synthetic only | synthetic pre-training + real fine-tuning | Experiment 7 |

**Confidence: medium.** The filtering and occlusion changes are cheap and safe. The data change
depends on experiment 7. Nothing here calls for rewriting the pipeline.

## 9. The honest state of the criterion

| | Measured | Verdict |
|---|---|---|
| Direction correct | 4.9° median | good |
| Stays put | 28 mm common drift, 10–25 mm jitter after filtering | visible |
| Correct size | 28 % per frame, ~1 % after smoothing | usable only smoothed |
| Does not flicker | 55 % of points above threshold | the weak point |
| Survives brief loss | crop led by velocity, pose held 250 ms | added, unverified on device |
| fps | 13–30, confounded by the debugger | unmeasured properly |
| End-to-end latency | — | **never measured** |

## 10. What was tried in this audit and failed

Recorded because a disproven hypothesis is worth as much as a confirmed one, and because each of
these looked obviously right beforehand:

- *"The score is broken and throws away good points."* No — error falls monotonically with score.
- *"RTMPose is better, so FootNet's architecture is wrong."* No — FootNet is more accurate; RTMPose
  is more available.
- *"Two points are too few; use the mask or all 8."* No — both measured worse on real frames.
- *"More synthetic data will help."* Independently confirmed: A ≈ B, so the model already fits its
  distribution perfectly.
