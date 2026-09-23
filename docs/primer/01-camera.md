# 1. The camera

Everything this project does rests on one idea: a camera destroys information in a very specific,
very predictable way, and if you know something else about the scene you can put the missing piece
back.

## A camera is a projection

Put the camera at the origin. It looks down the `+z` axis. A point in front of it,
`P = (X, Y, Z)`, lands on the image at

```
x = f · X / Z + cx
y = f · Y / Z + cy
```

That is the **pinhole model**, and it is essentially all there is. `f` is the *focal length in
pixels*, and `(cx, cy)` is the **principal point** — where the optical axis meets the sensor, near
the image centre. These three numbers (usually four: `fx` and `fy` separately) are the camera's
**intrinsics**, written as a matrix:

```
K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]
```

Everything about this is a division by `Z`. That single division is what makes the rest hard.

### What it destroys

Two different points on the same ray from the camera land on the same pixel. A small close foot
and a large distant foot are the same picture. **Depth is gone**, and no amount of cleverness gets
it back from one image alone.

What survives is the *direction*. Given a pixel, you can compute the ray it came along:

```
direction = ( (x − cx) / f,  (y − cy) / f,  1 )
```

This is `ray()` in [`src/foot-debug/shoePose.ts`](../../src/foot-debug/shoePose.ts). It says: the
point is *somewhere* along this line. Which is a lot — it is two of the three unknowns.

### Where our intrinsics come from

iOS hands them over. VisionCamera delivers `cameraIntrinsicMatrix` per frame, and
`intrinsicsFor()` adapts it: the matrix may describe the unrotated sensor at another resolution, so
the principal point is used to work out the scale. With no matrix, it falls back to about 65°
across the long side, which is roughly an iPhone wide lens.

> **Field of view and focal length are the same fact.** `fov = 2·atan(width / 2f)`. A "wider lens"
> is a shorter focal length. `verticalFovDegrees()` converts one to the other for RealityKit.

## Getting depth back by knowing something else

You cannot invert the projection. You can invert it *given one more equation*. Everything in
chapter 6 is a way of finding that extra equation. This project uses the simplest one available:

**A person standing on the floor has every foot landmark at a known height above that floor.**

The big toe is a couple of centimetres up. The ankle is about a quarter of the shoe's length up.
If you know which way is down, and how high the camera is, then the ray through a pixel meets that
landmark's horizontal plane at exactly one point — and now you have all three coordinates.

```
depth = (height − cameraHeight) / (direction · up)
```

That is `onPlane()`. The guard `rise > −GRAZING` rejects rays too close to horizontal, where a tiny
error in direction swings the intersection metres away. This is *conditioning*: the same equation
can be well or badly determined depending on the geometry, and a robust system refuses the badly
determined cases rather than returning nonsense.

### Which way is down

From the accelerometer. At rest it measures gravity, and CoreMotion separates gravity from user
acceleration for you. `createDeviceGravity()` exposes it; `gravityInBackCamera()` rotates it from
device axes into camera axes:

```ts
[gx, −gy, −gz]
```

Those sign flips are not magic — they are the difference between two conventions. An upright
back-camera frame has x right, **y down**, z forward; the device frame has y up and z out of the
screen. Getting a convention wrong produces a result that is confidently, consistently wrong, which
is much harder to spot than a crash.

## Why the mirror scenario forced this design

The obvious alternative: measure the foot's length in pixels, know it is ~26 cm, divide, get depth.

It works when the foot is side-on and large. It falls apart in the mirror view, where the foot is
seen nearly end-on and 1–2 m away — the apparent heel-to-toe distance is then small, it changes
slowly with depth, and every pixel of keypoint error becomes centimetres of depth error. The
problem is **ill-conditioned**: the thing you are solving for barely affects the thing you measure.

The floor-plane method does not care. The foot can be end-on; the ray still meets the plane at a
well-defined point, as long as it is not grazing. This is the general lesson: when a measurement is
ill-conditioned, do not estimate it better — find a different measurement.

## The assumption we are carrying

Camera height is **assumed at 1.3 m** and not measured. A wrong guess scales the whole scene: the
shoe comes out slightly too big or small, but stays *on* the feet in the image, because the error
is along the ray. For a prototype that is an acceptable trade; for a product it is not, and the way
out is measuring the floor plane directly (ARKit, or LiDAR where present) instead of assuming it.

Say "assumed" out loud whenever it matters. It is in
[measurements.md](../measurements.md#unmeasured-and-known-to-be) with the other assumptions.

## Try it

1. Open [`shoePose.ts`](../../src/foot-debug/shoePose.ts) and follow one image point through
   `ray()` → `onPlane()` → a 3D point. Everything above is those twenty lines.
2. Work out, on paper, what happens to `onPlane()` when `direction · up` approaches zero. That is
   the grazing guard, and you can now see why it exists.
3. In [`shoePose.test.ts`](../../src/foot-debug/shoePose.test.ts), change the assumed camera height
   and see which way the shoe's size moves. Predict the direction before running it.

## Further reading

- [OpenCV camera calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) —
  the pinhole model and distortion, with runnable code.
- Hartley & Zisserman, *Multiple View Geometry*, chapter 6 — camera models, done properly.
- [Apple: understanding camera intrinsics](https://developer.apple.com/documentation/avfoundation/avcameracalibrationdata)
  — what iOS actually gives you and in which frame.
