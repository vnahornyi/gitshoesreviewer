# Spike: 3D foot pose from dense template correspondences

> The current mask-first model task is specified in [`FOOT_MASK_SPEC.md`](FOOT_MASK_SPEC.md): visible foot-surface masks are positive for bare skin and socks, and do not require a visible heel.

Can a network that maps every foot pixel to a point on a 3D foot template give the app a full 6-DoF foot pose? Two keypoints per foot, heel and toe from RTMPose, are not enough: from the front the heel is hidden, the model puts it on the ankle, and the shoe lands in the wrong place.

The spike uses the pretrained TOC model from [FOCUS](https://github.com/OllieBoyne/FOCUS) (Boyne & Cipolla, 3DV 2025). Its code is MIT, but it builds on [DSINE](https://github.com/baegwangbin/DSINE), whose licence allows **non-commercial research only**, so this model is for the spike only. The network is DenseDepth on EfficientNet-B5, 72M parameters, 277 MB. It predicts a foot mask, template coordinates (TOC) with their uncertainty, normals, and whether the foot is left or right. A TOC value maps to the [FIND](https://github.com/OllieBoyne/FIND) template foot (metres, x toe-ward, z up, 26.4 cm long) as `toc × (max − min) + min`. `cv2.solvePnPRansac` on 600 confident pixels gives the pose. The template is a left foot; for a right foot the fit mirrors y and keeps whichever variant fits better.

## Run

```bash
uv sync
git clone --recurse-submodules https://github.com/OllieBoyne/FOCUS.git vendor/FOCUS
# TOC model: the Google Drive link in the FOCUS README, saved to data/toc_model/densedepth_toc_predictor.pth
uv run python -m spike.viz <images…>        # mask and TOC as colour, results/viz/
uv run python -m spike.fit_close <images…>  # whole frame is one foot: fit the template, results/fit/
uv run python -m spike.fit_scene <images…>  # RTMPose finds each foot, the crop goes to the TOC model
```

`vendor/`, `data/` and `results/` are git-ignored. Frames come from `research/assets/capture/prepared/`.

## Result (2026-09-19, Mac, MPS, ~170 ms per 480×640 crop)

| Case | Result |
|---|---|
| Bare foot, close-up (`close`) | Very good: a clean mask, smooth TOC along the anatomy, and the template sits on the foot (toes, outline, heel) with a 6–8 px median error at 1080 px. The left/right call is right. |
| White sock, seen from above (`top`) | Partial: the TOC is plausible on the visible sock, but the mask spills onto the trousers when the crop is loose. |
| White sock, from the front in a mirror (`mirror-lower`) | Fails: scattered blobs, no usable fit. |
| RTMPose-driven crops | Loose, and cut off at the frame edge. A foot needs a tight crop to fill the input the way SynFoot training images do. |

The approach is right: dense correspondences give a full 3D foot pose, and the mask the app needs to hide the real foot. The pretrained model is not enough. It was trained on bare synthetic feet seen close from above (SynFoot), and the demo is a person in socks or shoes, often from the front in a mirror. The mobile path is a smaller network (e.g. MobileNet/EfficientNet-Lite U-Net, permissive licence) trained on synthetic renders that cover socks, shoes, trousers and frontal mirror views, with the TOC ground truth rendered from FIND or Foot3D feet in Blender.

# SynFoot V1: what the data holds

[SynFoot](https://github.com/OllieBoyne/SynFoot) V1 (MIT) goes to `assets/synfoot/V1/` (git-ignored): 50 000 renders, 480×640, each with an RGB image, a foot mask, a normals image and a label (8 keypoints, the Blender camera, the Foot3D foot ID). The foot scan is not included.

```bash
uv run python -m synfoot.check   # stats and a contact sheet, results/synfoot/sheet.jpg
```

`synfoot/data.py` reads a sample and converts the camera to OpenCV. `synfoot/foot_pose.py` triangulates each foot's 8 keypoints from 400 views and caches them in `assets/synfoot/foot_keypoints.json`. For each sample it then recovers the foot pose in the camera frame, and maps the FIND template onto the foot with a similarity transform.

| Finding (2026-09-22) | Consequence |
|---|---|
| The camera is Blender (XYZ Euler, looks down −Z), and the FOV spans the long 640 px side. | Checked by triangulation: foot lengths come out at 23–30 cm. |
| The 8 keypoints are fixed mesh vertices, projected whether visible or not. There is no visibility flag. | Per-sample PnP on the foot's own keypoints fits to 0.06 px median. The label camera is off by a hidden ~1.5 cm foot offset, so the pose is refit from the keypoints. That gives exact 6-DoF ground truth for every sample. |
| PnP on the generic FIND keypoints fits to 6 px median (p90 10 px) at 640 px. | This is the error the app pays for not knowing the person's foot shape. It is acceptable for placing a shoe. |
| Only 8 feet, all left. | Right feet come from mirroring the image and keypoints. Shape variety has to come from elsewhere (Foot3D scans or FIND samples). |
| The camera is always above the foot, 11–40 cm high and at most 41 cm away, with a 0–72° tilt to the side. It is never in front of the toes or behind the heel. The heel is off the silhouette in only 5 % of samples. | V1 covers the "on the feet" views (`top`, `close`), but not the mirror, where the camera looks at the toes from 1–2 m. |
| Bare feet only. Some renders show jeans or a bare shin above the mask cut. | Socks and shoes need B2. |

# FootNet: our own foot model (line B)

A U-Net on MobileNetV3-Large (`segmentation_models_pytorch` MIT, `timm` Apache-2.0, ImageNet weights). It takes a 256×256 crop around one foot and predicts:

- the foot mask;
- 8 keypoint heatmaps, decoded by argmax plus a local soft-argmax;
- whether it is a right foot.

The 3D pose is PnP of the 8 points on the FIND template keypoints.

```bash
uv run python -m footnet.train --epochs 20                 # SynFoot V1 only
uv run python -m footnet.train --renders --init assets/checkpoints/best.pt --epochs 8   # plus our renders
uv run python -m footnet.real                              # RTMPose crop → FootNet → PnP on real frames, results/footnet/real/
```

- **Training data.** SynFoot V1 with one scanned foot (`0033-A`) held out for validation. Augmentation mirrors half the feet into right feet, repaints the foot as a sock (plain or striped, keeping the shading), and applies a random crop scale and shift, a full rotation, colour jitter, motion blur and noise.
- **Losses:** mask BCE plus Dice, heatmap BCE, smooth-L1 on soft-argmax coordinates, and BCE on the side.
- **Why the coordinate term.** Without it, the heatmaps stay flat for the first epochs. After 400 steps it gives 41 px median against 170 px without it.
- **Result (2026-09-22).** 20 epochs on SynFoot V1 alone: 2.1 px median keypoint error on the held-out foot, 0.987 mask IoU, but only 11.7 px (p90 113 px) and 0.73 IoU on held-out renders. 8 more epochs with the renders mixed in (`--render-repeat 3`, lr 3e-4) bring the renders to 3.3 px (p90 16 px), 0.93 IoU and 98 % side accuracy, and SynFoot barely moves (2.3 px). On the developer's own frames the FIND template then fits to 0.3–2 px in the mirror and 1–3 px from above, bare or in socks. Shoes and the side view (`third`) still fail: neither dataset has footwear.
- **Result (2026-09-23).** A further 8-epoch fine-tune on the 20 000-frame render set finished at 20:42 EEST. Epoch 8 reached 2.938 px median / 11.824 px p90 / 0.942 IoU on held-out renders; SynFoot was 2.330 px median. `best.pt` is selected by SynFoot median, not render p90; the real-label comparison below did not support replacing the phone checkpoint. No on-device result is available.
- **Real-label checkpoint comparison (2026-09-23).** On 156 complete labels after excluding `IMG_3244` and `IMG_3246` (clips flagged for child appearances), the 20k checkpoints slightly improved the oracle-crop median but not p90. The RTMPose seed-crop proxy also showed no p90 improvement. This ran on Mac CPU and is not an on-device result. Reproduce from this directory with `uv run python -m footnet.audit_checkpoints`; the proxy does not reproduce the native tracker's crop history. The 19 incomplete label records are skipped automatically.
- **Swift decoder parity (2026-09-24).** The measured old app decoder differed from PyTorch by 0.095 px median / 0.232 px p90 on the 156 labelled oracle crops. Core ML export now returns float16 logits, so Swift and Python use the same 5×5 softmax refinement and sigmoid peak score. On identical float16-rounded heatmaps the coordinate delta is 0.000020 px p90 / 0.000045 px maximum. The app model was re-exported with the original `v2-renders-7k.pt` weights; the newer checkpoints were not adopted because real-label p90 did not improve. The Core ML CPU-only package check on 16 SynFoot crops was 0.010 px median / 0.985 px maximum against full-precision PyTorch. No iOS build or phone run has verified the new artifact.
- **Coordinate round trip (2026-09-24).** The shared Swift helper maps crop coordinates through 1472×828 frame pixels to normalized coordinates, and the JS `frameToView` test maps an off-centre point into a 393×852 portrait view. Both pass; this does not reproduce the live camera/preview landscape issue.
- **Real-image ablation (2026-09-24).** Downsampling real crops to the measured synthetic source scale gave a small, non-conclusive gain. Matching synthetic saturation, blur, or denoise did not help. Reproduce with `uv run python -m footnet.ablate_real_domain`; it writes per-view oracle and RTMPose-seed results to ignored `results/footnet/domain-ablation.csv`.
- **Partial-label real fine-tune (2026-09-24).** A pilot trained from `v2-renders-7k.pt` on 111 seed crops from four clips, with whole clips held out for validation and test and a fully labelled render batch replayed at each update. On the 26-crop test clip, median error improved from 0.1681 to 0.1380 foot lengths, but p90 worsened from 1.3899 to 1.5241 and score≥0.3 recall fell from 43.8 % to 39.6 %. Do not promote this checkpoint. Reproduce with `uv run python -m footnet.finetune_real`; metrics and unpromoted checkpoints are under ignored `assets/checkpoints/real-finetune-156/`.
- **Speed on the M1 Pro:** 0.45 s per batch of 32 in fp32. fp16 and bf16 autocast on MPS are slower, and `channels_last` fails in the decoder.

## Visible-foot mask experiment (2026-09-25)

The separate `footmask-v1` experiment trains from random initialization on the owned render masks;
it treats bare-foot and sock pixels as the same visible-foot class and does not require a visible
heel. Crops are centered from visible-mask bounds, with empty-background crops for rejection. It is
a separate optional diagnostic model: the example camera can preview masks inside crops already
seeded by RTMPose, but the model does not drive tracking or full-frame acquisition.

```bash
uv run python -m footnet.train_mask --epochs 20 --batch 32 --workers 8
uv run python -m footnet.evaluate_mask
uv run python -m footnet.export_mask
uv run python -m footnet.review_mask --captures --per-video 1
```

Weights and the Core ML package go under `assets/checkpoints/footmask-v1/`; validation and real
overlays go under the ignored `results/footmask-v1/`. The validation set is held out by render
frame. Synthetic mask scores and the real overlays cannot establish real-world boundary accuracy:
the repository has no real foot-instance mask labels. See [`FOOT_MASK_SPEC.md`](FOOT_MASK_SPEC.md)
for the target, scenario matrix, runtime boundaries and promotion criteria.

**Result (2026-09-25).** Trained from random initialization for 20 epochs on 45,600 crops from
19,000 render frames; the holdout is 2,400 crops from 1,000 other frames. The best checkpoint is
epoch 16, chosen by per-crop Dice at threshold 0.5. The last checkpoint is epoch 20.

| Threshold | Dice, all 2,400 crops | Dice, positive crops | Pixel precision | Pixel recall | Empty-crop false-positive area |
|---|---:|---:|---:|---:|---:|
| 0.3 | 0.8993 | 0.8870 | 0.8371 | 0.9611 | 0.38 % |
| 0.5 | 0.9048 | 0.8936 | 0.8672 | 0.9438 | 0.27 % |
| 0.7 | 0.9057 | 0.8942 | 0.8934 | 0.9206 | 0.19 % |

At threshold 0.5, positive-crop Dice is 0.914 mirror, 0.893 top, 0.854 third-person, 0.895 bare
and 0.893 sock; trouser and no-trouser subsets are 0.898 and 0.885. Visible-landmark error is 5.66 px median
/ 14.88 px p90 at 256×256, and side accuracy is 91.0 %. The last checkpoint scores 0.9031 Dice,
0.8635 pixel precision and 0.9449 pixel recall at 0.5, so it does not replace epoch 16.

The Core ML package and compiled model are each 2.8 MB. On the M1 Pro, Core ML measures 1.25
ms/crop with `.all`, 1.24 ms with `CPU_AND_NE`, and 3.82 ms with `CPU_ONLY`. This is a Mac timing,
not iPhone 11 latency or proof of Neural Engine placement. On 16 holdout crops, mask decisions
differ from PyTorch on 0.022 % of pixels. Of 78 labelled landmarks, 10 scored ≥0.3 in both
frameworks; those coordinates differ by at most 0.029 px. One low-confidence landmark (score
0.005) differs by 101.6 px, and is not a usable point.

The compiled `.mlmodelc` also loaded and returned all three expected output tensors through Swift
Core ML on the Mac. This was a package smoke check with a generated pixel buffer, not an iOS build,
an iPhone run or an execution-provider trace.

The qualitative pass sampled one middle frame from each of 23 real clips. RTMPose seeded 42 of 46
possible feet. The mask still misses a sock in some mirror crops and spills onto neighbouring
regions in some views at both 0.5 and 0.7. There are no dense real-foot labels to score these
overlays. The artifact remains a research candidate: the camera overlay is diagnostic only, it has
no iPhone 11 trace, and it does not remove RTMPose from initial crop acquisition. A lightweight
full-frame segmenter remains unmeasured; the existing 6 fps result belongs to RTMPose, not to such
a mask model.

# Synthetic renders without the Blender UI

SynFoot has no mirror views. We render our own with Blender as a Python module, in a separate Python 3.13 environment in `render/`. Nobody opens Blender.

```bash
uv run python -m synth.export_shapes --count 300      # FIND feet → assets/synth/shapes/
cd render && uv sync && .venv/bin/python render.py --total 20000   # → assets/synth/renders/, batches of 200 per process
```

- **Foot shapes.** `synth/find_model.py` re-implements the FIND displacement field (MIT) without pytorch3d, from `vendor/FOCUS/data/find/model.pth`. That checkpoint holds 8 fitted Foot3D feet, and they match the SynFoot feet to about 5 mm at the keypoints. New feet are random blends of those 8 latents with some noise and a 0.88–1.12 size. The ankle cap (`templ_masked_faces.npy`) is removed so a shin can be extruded from the opening.
- **Scene.** A left and a right foot with shins, sometimes a heel raise or a lifted foot. They are bare or in socks (plain or striped, at a random height), and trousers cover them in 70 % of frames. Everything stands on a procedural floor in front of a wall, lit by 1–3 random lights.
- **Camera.** It is one of three kinds:

  | Kind | Share | Placement |
  |---|---|---|
  | Mirror | 50 % | 0.8–2.6 m in front, 0.6–1.5 m high |
  | Top | 25 % | chest height, held out over the toes |
  | Third person | 25 % | anywhere |

  The field of view is either the iPhone wide lens or zoomed in.
- **Rendering.** Cycles on Metal, 24 samples with denoising, 720×960, about 6–9 s a frame while training shares the GPU.
- **Labels per frame.** The 8 keypoints of each foot, with visibility from a ray cast. The mask per foot counts visible pixels only and comes from a Workbench pass with flat colours. The file also stores each foot's pose in the FIND frame, and the camera's K and pose.
