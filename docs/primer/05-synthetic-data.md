# 5. Teaching a model with fake data

We have no dataset of feet with labelled keypoints. Producing one by hand — thousands of photos,
eight points each, from every angle — is months of work and the labels would be worse than a
renderer's. So we render.

This chapter is what that buys, what it costs, and the trap it comes with.

## Why synthetic data is tempting

A renderer knows the ground truth exactly, because it placed the foot. Per frame, for free:

- every keypoint's position, **including the ones no human could mark** — a hidden heel has an
  exact answer;
- a per-foot mask, visible pixels only;
- the full 6-DoF pose and the camera's parameters;
- a visibility flag per keypoint, from a ray cast.

No annotator error, no ambiguity, no cost per sample beyond compute. Our 20 000 frames took about
18 hours on one Mac.

The second gift is **control of the distribution**. Real datasets have whatever they happened to
capture. [SynFoot](https://github.com/OllieBoyne/SynFoot) is 50 000 renders of bare feet, and we
measured what it covers: the camera is always above the foot, 11–40 cm away, never in front of the
toes. Good for "looking down at your feet", useless for the mirror view — which is half our
product. So we built a renderer whose camera goes where we need it: mirror 50 %, top 25 %, third
person 25 %.

## The domain gap

A model trained on renders learns *renders*. Deployed on a phone camera it meets:

- sensor noise, rolling shutter, motion blur, compression;
- real skin, real fabric, real light bouncing off a real floor;
- lens distortion, auto-exposure, auto-white-balance;
- feet unlike any we modelled.

The mismatch between the training distribution and the deployment distribution is the **domain
gap**, and it is *the* central problem of synthetic training. Our model's symptom is precise:
per-point scores on real frames are **bimodal — 0.6–0.9 or 0.0–0.15**. It either recognises the
foot or it does not; there is no middle. That shape is characteristic. The features it learned are
present in some real frames and absent in others.

## Domain randomization

The main tool, and the idea is counter-intuitive: rather than making renders look realistic, make
them **varied enough that reality is just another variation**.

If floor texture, wall colour, light position, sock colour and lens are all random, the model
cannot rely on any of them, and is pushed towards what is invariant — the shape of a foot. Our
renderer randomizes shapes (blends of 8 scanned feet with noise, 0.88–1.12 size), socks and
trousers (70 % of frames), floor and wall materials, 1–3 lights, and the camera. The data loader
adds more on top: mirroring, rotation, crop jitter, sock repainting that keeps the shading, colour
jitter, motion blur, noise.

Keeping the shading when repainting a sock is the kind of detail that decides whether this works.
Shading carries 3D shape; a flat colour would teach silhouette-matching, and silhouette is exactly
what a sock changes.

## The trap: more data ≠ better

This project ran the experiment, so you do not have to guess.

We trained on 7 254 render frames, then re-rendered to 20 000 — **2.75× the data, same renderer,
same distribution** — and fine-tuned again. Watch the p90 column in
[measurements.md](../measurements.md#footnet-accuracy).

More samples from the same distribution buy you a better estimate of that distribution. They do
**not** add anything the distribution lacks. And what ours lacks is not subtle:

> **Neither dataset contains a single shoe.** SynFoot is bare feet; our renderer dresses feet in
> socks and trousers. The product is a shoe try-on.

No number of bare-foot frames teaches a model what a shod foot looks like. That is why
[state.md](../state.md) lists footwear in the renderer as the next real step and not a fifth
training run. When you are tempted to fix quality with volume, ask first: *is the thing I want the
model to learn present in the data at all?*

## Reading validation numbers honestly

Our training log carries every metric twice — once on held-out SynFoot, once on held-out renders:

- **median vs p90.** A good median with a large p90 is a bimodal model: mostly right, sometimes
  completely lost. The median flatters; p90 tells the truth.
- **A validation number is measured on synthetic data.** It says how well the model interpolates
  within its own world. It has repeatedly failed to predict device behaviour here, and it is not
  evidence about real frames. Only `footnet.real` and the on-device log are.
- **Hold out the right thing.** Ours holds out an entire scanned foot (`0033-A`), not random
  samples, because random samples of the same foot leak: views of one object are not independent.
  The render split holds out every 20th frame by index, which is stable as the set grows.

## Practical notes on training

**Fine-tuning** starts from a trained checkpoint rather than random weights — cheaper, and it keeps
what the model already knows. Ours starts from the SynFoot-only model and mixes renders in. SynFoot
metrics barely move during that, which is what you want: no **catastrophic forgetting**.

**Mixture ratio matters more than people expect.** `--render-repeat` weighs renders against
SynFoot's 44k. The validated recipe was roughly 45 % renders. When the render set grew 2.75×, the
old repeat factor would have made it ~72 % — a different experiment wearing the same command line.
Count your crops before trusting a flag.

**Learning-rate schedule.** We use [OneCycle](https://arxiv.org/abs/1708.07120) (Smith & Topin):
the rate rises, then falls to near zero. Restarting it for a fine-tune means the first epochs
*worsen* metrics while the rate is high. Judge a run by its last epochs, not its first — and do not
report epoch 1 as a result.

**Keep a named copy of the checkpoint you started from.** `best.pt` is overwritten in place. Without
a copy there is nothing to compare against and nothing to fall back to.

## Ways to narrow the gap, in order of effort

1. **Cover what is missing.** Footwear, here. Nothing else on this list competes with it.
2. **Stronger augmentation** at the edges of realism — noise, blur, colour, compression artefacts.
3. **Real data, even unlabelled**, for fine-tuning with self-training or consistency losses.
4. **Real data, labelled**, a few hundred frames. Expensive; usually worth more than another 20 000
   renders.

## Try it

1. Open a few frames in `research/foot-3d/data/synth/renders/` next to a photo of your own feet.
   List the differences you can see. That list is the domain gap, concretely.
2. Read the SynFoot findings table in `research/foot-3d/README.md`. Notice that the conclusion
   "V1 cannot cover the mirror view" came from *measuring the camera distribution*, not from
   looking at pictures.
3. Predict what the render p90 will do over eight epochs, then look at `log.csv`. Write your
   prediction down first.

## Further reading

- [Domain randomization](https://arxiv.org/abs/1703.06907) (Tobin et al., 2017) — the original
  idea, in robotics.
- [SynFoot / FOCUS](https://arxiv.org/abs/2502.06367) (Boyne et al.) — synthetic foot data and what
  it was built for.
- [Super-convergence / OneCycle](https://arxiv.org/abs/1708.07120) — the schedule we use.
- The `footnet` skill in this repository — the same material as a checklist for a training run.
