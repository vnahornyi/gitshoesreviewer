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

`vendor/`, `data/` and `results/` are git-ignored. Frames come from `research/foot-tracking/data/prepared/`.

## Result (2026-09-19, Mac, MPS, ~170 ms per 480×640 crop)

| Case | Result |
|---|---|
| Bare foot, close-up (`close`) | Very good: a clean mask, smooth TOC along the anatomy, and the template sits on the foot (toes, outline, heel) with a 6–8 px median error at 1080 px. The left/right call is right. |
| White sock, seen from above (`top`) | Partial: the TOC is plausible on the visible sock, but the mask spills onto the trousers when the crop is loose. |
| White sock, from the front in a mirror (`mirror-lower`) | Fails: scattered blobs, no usable fit. |
| RTMPose-driven crops | Loose, and cut off at the frame edge. A foot needs a tight crop to fill the input the way SynFoot training images do. |

The approach is right: dense correspondences give a full 3D foot pose, and the mask the app needs to hide the real foot. The pretrained model is not enough. It was trained on bare synthetic feet seen close from above (SynFoot), and the demo is a person in socks or shoes, often from the front in a mirror. The mobile path is a smaller network (e.g. MobileNet/EfficientNet-Lite U-Net, permissive licence) trained on synthetic renders that cover socks, shoes, trousers and frontal mirror views, with the TOC ground truth rendered from FIND or Foot3D feet in Blender.
