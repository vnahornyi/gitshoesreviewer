# Visible foot mask: task definition

## Goal

Predict the visible pixels of one foot that a try-on shoe should cover. This is a visual segmentation target, not a biological skin classifier: bare skin and socks are the same positive class. Supervision tells the model to use visible texture, contour and local context; it does not require skin colour or an observed heel.

The mask follows the visible foot surface from the toes through the ankle opening. It includes the pixels of a sock when it covers that surface. It excludes the shin, trousers, floor, hands and other objects. It only labels pixels visible in the current frame. It does not fill hidden areas behind the other foot, a trouser cuff or the image edge.

The existing renderer's red/green mask pass already labels visible foot-mesh pixels separately for each side. This is the training target. The labels do not mark skin versus cloth.

## Scenario matrix

| View or condition | Positive target | Important failure to catch |
|---|---|---|
| Mirror, near frontal | Visible toes, forefoot and any visible midfoot | Requiring the hidden heel; moving the mask onto the ankle or trouser cuff |
| Mirror, oblique | The current visible side and toe surface | Swapping left/right when the camera or reflection reverses the view |
| Top-down, phone held over the feet | Visible toe, sole-facing surface and any visible heel pixels | Treating an occluded heel as a negative foot or shifting the crop toward an invented heel |
| Side / third-person | The visible side of the foot | Extending the mask up the shin or across the floor contact shadow |
| Bare foot | All visible pixels of the foot surface | Dependence on one skin tone, exposure or colour cast |
| Sock | All visible sock pixels on the foot, including plain, striped, light and dark material | Treating cloth texture or white socks as background |
| Partial frame or overlap | Only the part that is actually visible | Completing the hidden silhouette or dropping the visible part because one landmark is hidden |
| No foot in the crop / wrong crop | Empty mask | Painting a foot over wood, tile, carpet, trousers, a hand or another textured surface |

Current owned renders cover 20,000 frames: 10,020 mirror, 5,020 top and 4,960 third-person. They include bare/sock and trouser/no-trouser combinations. These counts describe the current synthetic distribution; they are not a product requirement. The four hand-selected real crops and captured footage are useful for visual review, but no current real file contains a ground-truth foot mask.

## Model and live pipeline

The first experiment is a per-foot crop model so mask quality can be measured while preserving the existing tracked-crop path. It still relies on RTMPose to provide an initial crop; at least two usable visible joints are needed. It therefore does not yet implement texture-only, full-frame search.

The measured 6 fps result was RTMPose run on every full frame. That does not establish the speed of a small, lower-resolution full-frame segmentation model. The renderer already has full-frame, per-foot visible masks, so a MobileNet full-frame segmenter is a valid competing design: it could discover foot-like texture without heel points or an initial body-model seed. Its recall, false-positive rate and iPhone 11 latency must be measured before choosing it over the crop model. Do not infer those results from the RTMPose timing or from Mac Core ML timing.

For this crop experiment, keep RTMPose as the low-frequency reacquisition path and run the crop model for tracked feet on incoming frames. The crop mask can later maintain a visible-region box when the heel heatmap is weak, but it cannot acquire a crop that was never seeded.

The crop model receives the existing 256×256 RGB crop. A compact MobileNetV3-Small U-Net predicts:

- one full-resolution mask logit map for the selected foot;
- the existing 8 landmark heatmaps as secondary outputs for shoe pose;
- the existing side logit.

The mask is the primary output for the new task. At runtime it can also supply a visible-region bounding box to keep the tracked crop alive when landmark confidence falls, including when the heel is absent. The initial RTMPose seed still needs two usable visible joints; the mask model does not solve failure to acquire any initial crop. This experiment exports the mask for evaluation; `FootNetRunner` does not consume it yet.

For native integration, run one crop inference for each active foot on every camera frame, keep logits in the native pipeline and use their visible-region bounds for tracking. Do not transfer the full 256×256 mask through the React Native bridge on every frame. Temporal filtering or flow-based propagation may be added only after motion clips show edge flicker; the 28–30 fps baseline leaves a 33 ms frame budget for both feet and the rest of the pipeline.

Core ML receives camera pixels and performs input scaling/normalization in the graph. Export mask and landmark logits directly and use Core ML `.all` so the graph can use the Neural Engine. Do not route through ONNX Runtime's Core ML provider: the measured FootNet graph was partitioned there and ran slower than direct Core ML. Report Mac timing separately from phone timing. A real iPhone 11 run is required to claim A13 latency, Neural Engine placement or 30 fps.

The mask is RGB-only. LiDAR depth can be an optional downstream boundary/occlusion cue on hardware that provides it; it cannot label RGB foot texture, is not required by this model and must never gate the iPhone 11 path.

## Training data and supervision

Train a new checkpoint from random initialization; do not initialize from the previous FootNet weights or the rejected real fine-tune. The first experiment uses the owned 20,000-frame Blender render set because it has visible-pixel masks, mirror/top/third views, socks, trousers, backgrounds, camera and pose labels. Keep the held-out frame split separate from training.

Create each positive training crop from the visible mask bounds, with jitter and context. This uses only pixels visible in the rendered frame and never the projected position of an invisible heel. Transform the mask with nearest-neighbour sampling. Train visible keypoint heatmaps only; invisible points have no positive heatmap and no coordinate loss. Mask supervision still applies to every visible foot pixel in the crop.

Add empty-mask crops from scene background and invalid crop locations so a mistaken seed can be rejected. Keep crops containing the other foot labeled only for the selected instance. Apply colour, exposure, blur and sensor-noise variation without changing the mask. Existing renderer and crop dimensions are the starting point; change their distributions only when an evaluation shows a specific gap.

The first run uses the trainer's current experiment settings: 20 epochs, batch 32, mask-loss multiplier 2.0, and one empty crop for every five positive-crop slots. These are implementation choices, not product requirements; record their effect in the validation results before changing them.

SynFoot V1 can be added as a later ablation for more bare-foot shape variation, with mask-only supervision. It has close overhead views, no mirror or socks, eight source feet and no landmark visibility flag; do not let it silently dominate the current scene distribution or supervise hidden landmarks.

## Evaluation

Report on held-out synthetic frames:

- per-crop mask IoU and Dice;
- pixel precision/recall and a threshold sweep, rather than hiding the threshold choice;
- results grouped by camera mode, bare/sock, and trouser/occlusion flags;
- landmark error only on visible labelled points, plus side accuracy.

Also export overlays on `assets/real-crops/` and selected real captures. These are qualitative checks only. `assets/capture/labels.json` contains toe/heel clicks, and `assets/capture/masks/` contains person/foreground masks, not foot-instance masks. Neither can be reported as ground truth for this model.

### First experiment result (2026-09-25)

`footmask-v1` trained from random initialization for 20 epochs. The best checkpoint is epoch 16.
On 2,400 held-out crops from 1,000 held-out render frames, mean Dice at threshold 0.5 is 0.9048
when positives and empty crops are combined, and 0.8936 on positive crops only. Pixel precision /
recall across the combined set are 0.8672 / 0.9438; empty-crop false-positive area is 0.27 %.
Mode/material/clothing groups below are positive-crop only. The 0.3 / 0.5 / 0.7 threshold sweep is
in `results/footmask-v1/final-validation.json`. Third-person is the weakest mode at 0.8538 Dice;
mirror is 0.9144 and top is 0.8930.

The best checkpoint beats epoch 20 only slightly (0.9048 vs 0.9031 Dice at 0.5). Core ML converted
and compiled to 2.8 MB. On the M1 Pro it takes 1.25 ms/crop with `.all`, 1.24 ms with `CPU_AND_NE`
and 3.82 ms with `CPU_ONLY`; these Mac timings do not establish iPhone 11 performance or ANE
placement. On 16 held-out crops, mask threshold decisions differ from PyTorch on 0.022 % of pixels.
Of 78 labelled landmarks in that parity sample, 10 scored ≥0.3 in both frameworks; those differ by
at most 0.029 px. A low-confidence point (score 0.005) has a 101.6 px coordinate difference.
The compiled `.mlmodelc` loads and returns its mask, heatmap and side outputs in a Mac Swift Core ML
smoke check; it has not been built into the iOS app or run on an iPhone.

A qualitative review of one middle frame from each of 23 real clips found 42 RTMPose seeds among 46
possible feet. The predicted mask misses selected socks in some mirror views and leaks onto adjacent
regions in some crops at thresholds 0.5 and 0.7. No real foot-instance masks exist to compute
precision or recall. The best checkpoint is not integrated into the app, and the crop model still
needs RTMPose to acquire an initial crop. A full-frame MobileNet segmenter is a separate, unmeasured
candidate; the old 6 fps figure is not evidence about its speed.

Do not promote a checkpoint to the app based only on synthetic IoU or Mac Core ML timing. Promotion needs a manually labelled real mask set with clips held out by source video, a phone trace on iPhone 11 showing the graph's execution provider and latency, and live tracking checks for mirror, top-down and side views. The existing 28–30 fps and 15 ms/foot measurements are the comparison baseline, not a promise for the new graph.

## Known limits

- Similar foot-shaped texture in an unrelated region can be ambiguous from RGB alone. Empty-crop negatives reduce false positives; they cannot make identical visual evidence distinguishable.
- An offline qualitative run of the checkpoint currently bundled by the app (`v2-renders-7k.pt`; its export drops the mask head) on the four hand-picked real crops spills past sock boundaries onto the shin/background in some views. This is a visual observation, not a measured real-mask score; the crops have no ground-truth foot masks.
- The current synthetic mask ends at the rendered foot mesh. It does not supervise an arbitrary sock cuff extending up the shin, shoe removal, or a complete 3D surface behind an occluder.
- There are no real dense foot masks yet, so real-world mask precision, recall and boundary quality remain unknown until annotation.
- The task learns where a visible foot-like surface is. It does not by itself determine the full 3D shoe pose or replace the person matte used to show the real lower leg above a shoe collar.
