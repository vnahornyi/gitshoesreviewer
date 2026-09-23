# Spike: 3D foot pose from dense template correspondences

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
- **Speed on the M1 Pro:** 0.45 s per batch of 32 in fp32. fp16 and bf16 autocast on MPS are slower, and `channels_last` fails in the decoder.

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
