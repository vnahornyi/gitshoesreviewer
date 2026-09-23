# A primer on how this thing works

Written to be read in order, by someone who has not done computer vision before. It teaches the
theory this project actually uses — not a survey of the field — and every chapter ends on the code
in this repository where that theory lives.

The other documents answer *what we did*. This one answers *why any of it works*.

| # | Chapter | You will understand |
|---|---|---|
| 1 | [The camera](01-camera.md) | How a 3D world becomes pixels, and what you can and cannot invert |
| 2 | [Finding keypoints](02-keypoints.md) | How a network learns to say "the big toe is *here*" |
| 3 | [Doing it 30 times a second](03-realtime.md) | Detector/tracker splits, budgets, and why smoothing is not cheating |
| 4 | [Making it run on the phone](04-on-device.md) | Neural Engines, fp16, graph partitioning, and how to prove where a model ran |
| 5 | [Teaching a model with fake data](05-synthetic-data.md) | Synthetic training, the domain gap, and what more data cannot fix |
| 6 | [From points to a pose](06-3d-pose.md) | Getting 6 degrees of freedom out of a handful of 2D dots |

## How to read it

Each chapter is self-contained enough to read alone, but they build. Notation is introduced where
it is first needed and never assumed.

Two conventions throughout:

- **Bold claims carry their evidence.** Where a number appears, it comes from
  [measurements.md](../measurements.md) and says so.
- **Where we were wrong is written down.** Several chapters end with an approach that looked right
  and measured badly. Those sections are the most useful ones.

## Background you do not need

No machine-learning course, no linear-algebra course. You need to be comfortable reading code, and
willing to accept a matrix as "a table of numbers that transforms points" until chapter 1 makes it
concrete.

## Background that would help later

If you want to go deeper than this primer, in roughly this order:

- **Multiple View Geometry in Computer Vision**, Hartley & Zisserman — the standard reference for
  everything in chapter 1 and 6. Dense, but the first three chapters are worth the effort.
- **Dive into Deep Learning** ([d2l.ai](https://d2l.ai/)) — free, runnable, and covers the
  convolutional networks of chapter 2 properly.
- [OpenCV's camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
  — the same geometry as chapter 1, from the practical side.
