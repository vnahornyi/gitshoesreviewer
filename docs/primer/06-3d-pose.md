# 6. From points to a pose

We have eight 2D dots per foot. We need to place a 3D shoe: where it is, which way it points, how
it is tilted — **six degrees of freedom**, three of position and three of rotation, plus a scale.

This chapter is how that is normally done, how we do it, and two approaches that looked reasonable
and measured badly.

## What a pose is

A rigid transform: rotate, then translate.

```
P_camera = R · P_object + t
```

`R` is a 3×3 rotation (3 degrees of freedom, not 9 — its columns are unit length and mutually
perpendicular), `t` is a 3-vector. Add a uniform scale `s` and you have a **similarity transform**,
7 degrees of freedom, which is what you need when the object's real size is unknown — a foot's
length varies by person.

RealityKit wants this as a column-major 4×4. Getting the convention wrong (row vs column major, or
a left-handed vs right-handed frame) gives a confidently wrong result, which is harder to notice
than a crash. Every boundary in this project writes its convention down; copy that habit.

## The textbook method: PnP

**Perspective-n-Point**: given 3D points in the object's own frame, their 2D projections, and the
camera intrinsics, recover `R` and `t`.

The intuition is a constraint count. Each 2D observation gives two equations. Six unknowns means
three points suffice in principle (P3P, up to a four-fold ambiguity), and four or more resolve it.
More points over-determine the system and you solve in the least-squares sense.

Real detections contain outliers — a keypoint on the wrong foot, a heel placed on an ankle — and
least squares is not robust: one bad point drags the whole fit. Hence **RANSAC**: sample a minimal
set, fit, count how many other points agree, repeat, keep the best. `cv2.solvePnPRansac` does
this, and the research code uses it.

This is how `research/foot-3d` fits the FIND template to a foot, and it works well: 6 px median at
640 px on SynFoot using generic template keypoints. That 6 px is the price of **not knowing the
person's own foot shape** — a real, irreducible error you pay for using a generic template. It is
acceptable for placing a shoe, and knowing *which* error it is matters more than its size.

## Fitting a shape, not just a pose: Umeyama

When you have corresponding 3D points in two frames — a template and a measurement — and want the
similarity transform between them, there is a closed-form answer:
[Umeyama's algorithm](https://ieeexplore.ieee.org/document/88573) (1991). Centre both sets, take
the SVD of their cross-covariance, read the rotation off, derive scale and translation. No
iteration, no initial guess.

The catch is the same one: it is least-squares, so it is not robust to outliers, and it needs
**correspondences** — you must already know which point matches which.

## What we actually do, and why it is different

The app does **not** run PnP. It uses the floor-plane construction from chapter 1:

- a standing foot's landmarks are at known heights above the floor;
- gravity gives "up", an assumed camera height gives the offset;
- each image point back-projects to a 3D point on its own horizontal plane.

Two points — the toes and the back of the foot — then give position and direction directly, and
gravity gives the remaining rotation. No iterative solve, no ambiguity, no initial guess.

Why not PnP, given it is the textbook answer? Because with 2–3 reliable points in a near-degenerate
configuration (nearly collinear along the foot's axis) PnP is badly conditioned, and in the mirror
view it is worse. The floor-plane method substitutes a *different kind of knowledge* — the scene's
geometry — for the observations PnP would need.

That is the transferable idea: **when observations are weak, look for a constraint that does not
come from the observations.**

## The anchor and the direction

Read `shoeTransform()` in [`shoePose.ts`](../../src/pose/shoePose.ts). The structure is
deliberate:

- **the toes anchor the shoe** — they are what must line up in the image, and they are what FootNet
  is most confident about;
- **the ankle, or failing that the heel, gives direction**;
- gravity gives the rest.

Anchoring at the toes rather than the heel is a product decision as much as a geometric one. A
shoe misplaced at the heel looks wrong at the heel; misplaced at the toes it looks wrong
everywhere, because the toes are where the eye goes.

Landmark heights on the shoe (ankle ≈ 28 % of the length up and a quarter along, little toe ≈ 80 %
along) are **anatomical estimates, not measurements**. They are labelled as such in the code and
listed in [measurements.md](../measurements.md#unmeasured-and-known-to-be).

## Two approaches that failed measurement

Looking down at your own feet hides the heel behind the foot. FootNet is confident about it in only
about half of otherwise-good readings. So: could we *infer* it?

**Attempt 1 — extrapolate along the foot's axis.** The heel is roughly a fixed multiple of the
toe-to-extrema distance, back along the axis, with a correction for foot width. Fitted on SynFoot:

```
heel = extrema + 4.288·(extrema − toes) + 0.432·(littleToe − bigToe)
```

**54 mm mean error.** A foot is about 250 mm. Useless.

**Attempt 2 — fit the template in 2D.** Fit the FIND template's keypoints to the observed ones with
an image-space similarity transform, and read the heel off the fitted template. On 237 views:
**56° of direction error.** Worse than useless — confidently wrong.

**What did work — fit in 3D**, on the floor plane, using the back-projection above: **25 mm and
4.5°**. An order of magnitude better than the 2D fit.

The reason is worth internalising. A foot is a 3D object under perspective; its image outline
changes shape completely with viewpoint. No 2D similarity transform can model that, so the fit is
not slightly wrong, it is *the wrong model*. Fitting in the space where the object actually lives
works. **When a fit is bad, ask whether the model class can represent the phenomenon at all before
tuning it.**

Neither guess shipped. The app substitutes RTMPose's **ankle** when the heel is missing: a worse
landmark, but a real observation. A worse measurement beats a better guess.

## The template

The 3D foot model is [FIND](https://arxiv.org/abs/2210.12241) (Boyne, Charles & Cipolla, BMVC
2022) — a neural implicit foot model: a network that deforms a template mesh from a latent code, so
shapes interpolate meaningfully. `research/foot-3d/synth/find_model.py` reimplements its
displacement field without pytorch3d, and new feet for the renderer are random blends of 8 fitted
scans.

Our shoes obey a much simpler convention, which every layer assumes: **heel at the origin, sole on
`y = 0`, toe along `+Z`, length exactly 1**. The app scales to real millimetres. One normalization,
agreed everywhere, removes a whole class of bug — and when a shoe comes out rotated or backwards,
the fault is almost always at normalization time, not in the pose maths.

## Try it

1. Follow `shoeTransform()` from two image points to a 4×4. Identify where each of the six degrees
   of freedom is pinned down, and by what.
2. Work out why three nearly-collinear points make PnP ill-conditioned. Sketch it.
3. In `shoePose.test.ts`, feed a foot with only a toe and no back point. The function returns null.
   Argue for or against that choice versus guessing a direction.

## Further reading

- Hartley & Zisserman, *Multiple View Geometry*, chapters 6–7 — cameras and pose estimation.
- [OpenCV's solvePnP documentation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) — the
  methods, their minimum point counts, and their failure modes.
- [Umeyama 1991](https://ieeexplore.ieee.org/document/88573) — least-squares similarity between
  point sets, four pages.
- [FIND](https://arxiv.org/abs/2210.12241) and [FOCUS](https://arxiv.org/abs/2502.06367) — the foot
  model and the dense-correspondence approach our research spike evaluated.
