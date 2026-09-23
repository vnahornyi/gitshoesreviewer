import type { FootPoints, Point, Size } from '../track/footAxes';
import {
  add,
  cross,
  dot,
  length,
  normalize,
  scale,
  sub,
  type Vec3,
} from './vec3';

export type Intrinsics = { focal: number; cx: number; cy: number };

type ShoePoint = { height: number; forward: number };

// Where the pose model's points sit on a normalized shoe (heel at the origin, sole on y = 0, toe at z = 1).
// Ankle and toe midpoint are anatomical estimates (ankle joint ≈ 28 % of the shoe length up, a quarter of the way
// from the heel; little toe ≈ 80 % along), not measurements yet.
const HEEL_POINT: ShoePoint = { height: 0.08, forward: 0.035 };
const TOE_POINT: ShoePoint = { height: 0.08, forward: 0.95 };
const ANKLE_POINT: ShoePoint = { height: 0.28, forward: 0.26 };
const TOES_MIDPOINT: ShoePoint = { height: 0.08, forward: 0.86 };
const MIN_DEPTH_M = 0.1;
const MAX_DEPTH_M = 8;
// Rays closer to the horizon than this meet the floor too far away to trust.
const GRAZING = 0.05;
// Ankle and toes closer than this on the floor give no usable direction.
const MIN_SPAN_M = 0.02;

// iPhone wide cameras record 16:9 video with about 65° across the long side.
const FALLBACK_LONG_SIDE_FOV = (65 * Math.PI) / 180;

export function intrinsicsFor(
  frame: Size,
  cameraMatrix?: readonly number[],
): Intrinsics {
  const longSide = Math.max(frame.width, frame.height);
  const center = { cx: frame.width / 2, cy: frame.height / 2 };
  if (!cameraMatrix || cameraMatrix.length !== 9) {
    return {
      focal: longSide / 2 / Math.tan(FALLBACK_LONG_SIDE_FOV / 2),
      ...center,
    };
  }
  const rowMajor = cameraMatrix[2] > 1;
  const fx = cameraMatrix[0];
  const cx = rowMajor ? cameraMatrix[2] : cameraMatrix[6];
  const cy = rowMajor ? cameraMatrix[5] : cameraMatrix[7];
  // The matrix may describe the unrotated sensor at another resolution; its principal point tells the scale.
  const matrixLongSide = 2 * Math.max(cx, cy);
  return { focal: fx * (longSide / matrixLongSide), ...center };
}

export function verticalFovDegrees(
  frame: Size,
  intrinsics: Intrinsics,
): number {
  return (2 * Math.atan(frame.height / 2 / intrinsics.focal) * 180) / Math.PI;
}

// CoreMotion gravity is in device axes; an upright back-camera frame has x right, y down, z forward.
export function gravityInBackCamera(deviceGravity: readonly number[]): Vec3 {
  return [deviceGravity[0], -deviceGravity[1], -deviceGravity[2]];
}

function ray(point: Point, frame: Size, k: Intrinsics): Vec3 {
  return [
    (point.x * frame.width - k.cx) / k.focal,
    (point.y * frame.height - k.cy) / k.focal,
    1,
  ];
}

// Where a ray meets the horizontal plane `height` above the floor, the camera being `cameraHeight` above it.
function onPlane(
  direction: Vec3,
  height: number,
  up: Vec3,
  cameraHeight: number,
): Vec3 | null {
  const rise = dot(direction, up) / length(direction);
  if (rise > -GRAZING) {
    return null;
  }
  const depth = (height - cameraHeight) / dot(direction, up);
  if (depth < MIN_DEPTH_M || depth > MAX_DEPTH_M) {
    return null;
  }
  return scale(direction, depth);
}

// Every point of a standing foot sits at a known height above the floor, so with gravity and the camera height each
// image point becomes a 3D point on its own horizontal plane. That holds from any view, the frontal mirror one
// included, where solving depth from the heel-to-toe length is ill-conditioned. The toes anchor the shoe (they are
// what must line up in the image) and the ankle, or else the heel, gives its direction.
export function shoeTransform(
  foot: FootPoints,
  frame: Size,
  intrinsics: Intrinsics,
  gravity: Vec3,
  shoeLengthM: number,
  cameraHeightM: number,
): number[] | null {
  const up = normalize(scale(gravity, -1));
  const place = (point: Point, on: ShoePoint) =>
    onPlane(
      ray(point, frame, intrinsics),
      on.height * shoeLengthM,
      up,
      cameraHeightM,
    );

  const toes = foot.smallToe
    ? {
        x: (foot.toe.x + foot.smallToe.x) / 2,
        y: (foot.toe.y + foot.smallToe.y) / 2,
      }
    : foot.toe;
  const front = foot.smallToe ? TOES_MIDPOINT : TOE_POINT;
  const back = foot.ankle
    ? { point: foot.ankle, on: ANKLE_POINT }
    : foot.heel
    ? { point: foot.heel, on: HEEL_POINT }
    : null;
  if (!back) {
    return null;
  }
  const toe = place(toes, front);
  const behind = place(back.point, back.on);
  if (!toe || !behind) {
    return null;
  }
  const along = sub(toe, behind);
  const flat = sub(along, scale(up, dot(along, up)));
  if (length(flat) < MIN_SPAN_M) {
    return null;
  }
  return placement(toe, front, normalize(flat), up, shoeLengthM);
}

// The shoe's model matrix from one known point on it, its forward direction on the floor and up.
function placement(
  anchor: Vec3,
  anchorOnShoe: ShoePoint,
  forward: Vec3,
  up: Vec3,
  shoeLengthM: number,
): number[] {
  const side = cross(up, forward);
  const origin = sub(
    anchor,
    add(
      scale(up, anchorOnShoe.height * shoeLengthM),
      scale(forward, anchorOnShoe.forward * shoeLengthM),
    ),
  );

  // RealityKit's camera looks down -z with y up: flip y and z of the OpenCV camera frame.
  const flip = (v: Vec3): Vec3 => [v[0], -v[1], -v[2]];
  const [x, y, z, o] = [flip(side), flip(up), flip(forward), flip(origin)];
  const k = shoeLengthM;
  return [
    x[0] * k,
    x[1] * k,
    x[2] * k,
    0,
    y[0] * k,
    y[1] * k,
    y[2] * k,
    0,
    z[0] * k,
    z[1] * k,
    z[2] * k,
    0,
    o[0],
    o[1],
    o[2],
    1,
  ];
}
