# 2. Finding keypoints

A keypoint model answers: *given this image, where is the big toe?* This chapter is how such a
model is built, trained and read, using ours as the worked example.

## The problem with asking directly

The naive design outputs two numbers per keypoint, x and y, and trains on the squared error. It
works badly, and it is worth understanding why, because the reason shapes everything else.

A convolutional network computes **local** features that keep their spatial arrangement. Asking it
for a global pair of coordinates forces the final layers to throw that arrangement away and encode
position in the *values* of a flat vector. It can be done, but the network has to learn the encoding
itself, and early in training it has no signal at all: every output is wrong by a similar amount, and
the gradient says "move everything a bit", which averages to nothing.

We measured this. Without a coordinate term, the heatmaps stay flat for the first epochs; with it,
after 400 steps the error is 41 px instead of 170 px. More on that below.

## Heatmaps: ask spatially

Instead, predict **one image per keypoint**. In the ground truth, the channel for the big toe is a
small Gaussian blob centred on the big toe and near zero everywhere else:

```python
heatmap[i] = exp(−‖pixel − point_i‖² / 2σ²)      # σ = 3 px, in dataset.py
```

Now the task is local — "is the big toe *here*?" — asked once per pixel, which is exactly what a
convolutional network is shaped to answer. Training is a per-pixel classification, and the gradient
is informative from the first step: the blob region says "more", everywhere else says "less".

This is the standard formulation, and ours is [`footnet/model.py`](../../research/foot-3d/footnet/model.py):
a U-Net whose output has 1 + 8 channels — one foot mask and eight keypoint heatmaps.

### Reading a point back out

Take the argmax — the brightest pixel. That is only accurate to the pixel, and the heatmap's
resolution may be lower than the image's, so a **sub-pixel refinement** follows: take a 5×5
neighbourhood around the peak, softmax its values into weights, and return the weighted mean of the
coordinates.

```python
weights = softmax(values in the 5×5 window)
point   = Σ weights · coordinates
```

That is `decode_points()`, and the Swift side reimplements exactly it in `FootNetRunner.decode`.
**The two must agree.** If training refines over 5×5 and the app over 3×3, every point is
systematically biased and nothing will tell you.

The peak value, squashed through a sigmoid, is the **score**: how sure the model is. Chapter 5
explains why ours comes out bimodal on real photographs.

> **Why not pure soft-argmax over the whole map?** Because a second, weaker blob elsewhere drags
> the mean towards itself. Argmax picks a mode; the local window refines *within* that mode. You
> get sub-pixel accuracy without the global averaging.

## U-Net: why this shape

A network that only downsamples ends up with strong semantics and no spatial precision — it knows
there is a foot, not where its toe is to the pixel. A [U-Net](https://arxiv.org/abs/1505.04597)
(Ronneberger et al., 2015) downsamples to understand, then upsamples back to full resolution, and
at each step **concatenates the matching encoder layer** via a skip connection. The decoder gets
both: what the thing is, from deep layers, and precisely where its edges are, from shallow ones.

Ours uses a [MobileNetV3](https://arxiv.org/abs/1905.02244) encoder — a backbone designed for
phones, built from depthwise-separable convolutions — with ImageNet weights. Starting from
ImageNet rather than random is **transfer learning**: edges, textures and shading are the same
problem everywhere, so only the last part has to be learned.

## The loss: four terms, and why each is there

From [`footnet/train.py`](../../research/foot-3d/footnet/train.py):

| Term | Weight | Why |
|---|---|---|
| Mask: BCE + Dice | 1 | Segmentation. Dice handles the class imbalance — a foot is a small part of the crop |
| Heatmaps: BCE | 20 | The main task, weighted up because its per-pixel values are tiny |
| Points: smooth-L1 on soft-argmax | 0.05 | The bootstrap, below |
| Side: BCE | 0.2 | Left or right foot |

**Binary cross-entropy** asks per pixel "should this be on?", and punishes confident mistakes hard.
**Dice** compares the overlap of two sets rather than counting pixels, which stops a model from
scoring well by predicting "background" everywhere ([Milletari et al., V-Net,
2016](https://arxiv.org/abs/1606.04797)).

The **coordinate term** is the interesting one. It decodes the predicted heatmaps with a
differentiable soft-argmax and compares the *resulting point* with the truth. Without it the model
has no reason to make its blobs sharp early on — flat maps are a local minimum of the heatmap loss.
With it, there is a gradient pulling the centre of mass towards the right place from step one.
41 px against 170 px after 400 steps, measured.

**Smooth-L1** (Huber) is squared error near zero and absolute error far from it, so one badly wrong
keypoint cannot dominate the batch. Choosing losses that are robust to outliers is a recurring
theme; chapter 6 has the geometric version.

### Multi-task training, then throwing heads away

We train the mask and side heads even though the app never uses them. They are **auxiliary
supervision**: forcing shared features to also support segmentation gives the keypoint head a
better representation to work from.

At export they are cut ([`footnet/export.py`](../../research/foot-3d/footnet/export.py) returns only
`sigmoid(heatmaps)`), which is most of the on-device speed win. Train with everything that helps;
ship only what is read. Chapter 4 covers the rest of the export.

## Augmentation: teaching invariance by construction

You want the model to be unmoved by things that should not matter. Rather than hoping it learns
that, you *construct* it — every training crop is randomly:

- mirrored (which also swaps left and right — SynFoot has only left feet, so right feet exist only
  because of this);
- rotated a full 360°, shifted ±15 % and scaled 1.15–1.7× around the keypoints;
- repainted as a sock — a random colour with optional stripes, **keeping the original shading**;
- jittered in hue/saturation/value, motion-blurred or Gaussian-blurred, and given noise.

Keeping the shading in the sock repaint matters: shading is the cue for 3D shape. Replacing a foot
with a flat colour would teach the model to find silhouettes, and silhouettes are exactly what a
sock changes.

> **The crop scale range is a design decision with a consequence.** Training saw 1.15–1.7×; the app
> ships 1.8×, because on 94 real feet the wider crop measured better (3.9 of 8 confident points
> against 3.0 at 1.3×). We are running slightly outside the training distribution on purpose,
> because the measurement said so — and that is the kind of number that must be written down rather
> than left in the code.

## A different formulation, which we also use

The body model, RTMPose, does not use heatmaps. It uses
[SimCC](https://arxiv.org/abs/2107.03332) (Li et al., ECCV 2022): classify the x coordinate and the
y coordinate *separately*, each into a row of fine bins. Two 1-D problems instead of one 2-D one —
much cheaper, no expensive upsampling, and sub-pixel precision from the bin width.

That is why `tools/model-convert/README.md` talks about `simcc_x` and `simcc_y` outputs, and why a
point's score is the *smaller* of the two maxima: it is only confident if both axes are.

It is also why converting that model to TFLite failed — the NCHW → NHWC layout rewrite broke the
SimCC head, silently, producing points up to 940 px off. See chapter 4.

## Try it

1. Read `decode_points()` in `model.py` and `decode` in
   [`FootNetRunner.swift`](../../ios/FootNetRunner.swift) side by side. Convince
   yourself they compute the same thing. Then consider what happens if someone changes one.
2. In `dataset.py`, change `SIGMA` from 3 to 1 and reason about it: sharper target, more precision
   if learned — but far fewer "on" pixels, so a weaker gradient. Which way would you bet?
3. Look at a heatmap for a hidden heel. The model has to put probability mass *somewhere*; where
   does it go when the answer is not visible?

## Further reading

- [U-Net](https://arxiv.org/abs/1505.04597) — the architecture, short and very readable.
- [SimCC](https://arxiv.org/abs/2107.03332) — coordinate classification.
- [RTMPose](https://arxiv.org/abs/2303.07399) — the body model we actually run, and a good tour of
  practical pose-estimation choices.
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — what a phone backbone is optimised for.
