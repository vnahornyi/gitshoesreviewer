import type { Point, Size } from './footAxes';
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

export type FootOnFloor = { heel: Point; toe: Point };

// Where the pose model's heel and big-toe points sit on a normalized shoe (heel at the origin, sole on y = 0, toe at z = 1).
const HEEL_POINT = { height: 0.08, forward: 0.035 };
const TOE_POINT = { height: 0.08, forward: 0.95 };
const MIN_DEPTH_M = 0.1;
const MAX_DEPTH_M = 8;
const GRAZING = 0.05;

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

// A standing foot is flat on the floor, so with gravity known the heel and toe share a height:
// that pins both depths from two image points and the shoe length, with no iterative solve.
export function shoeTransform(
  foot: FootOnFloor,
  frame: Size,
  intrinsics: Intrinsics,
  gravity: Vec3,
  shoeLengthM: number,
): number[] | null {
  const up = normalize(scale(gravity, -1));
  const heelRay = ray(foot.heel, frame, intrinsics);
  const toeRay = ray(foot.toe, frame, intrinsics);
  const keypointLength = (TOE_POINT.forward - HEEL_POINT.forward) * shoeLengthM;

  const heelUp = dot(heelRay, up);
  const toeUp = dot(toeRay, up);
  const toeOverHeel = Math.abs(toeUp) > GRAZING ? heelUp / toeUp : 1;
  const span = length(sub(scale(toeRay, toeOverHeel), heelRay));
  if (span === 0) {
    return null;
  }
  const heelDepth = keypointLength / span;
  const toeDepth = heelDepth * toeOverHeel;
  if (
    heelDepth < MIN_DEPTH_M ||
    heelDepth > MAX_DEPTH_M ||
    toeDepth < MIN_DEPTH_M ||
    toeDepth > MAX_DEPTH_M
  ) {
    return null;
  }

  const heel = scale(heelRay, heelDepth);
  const toe = scale(toeRay, toeDepth);
  const along = sub(toe, heel);
  const flat = sub(along, scale(up, dot(along, up)));
  if (length(flat) === 0) {
    return null;
  }
  const forward = normalize(flat);
  const side = cross(up, forward);
  const origin = sub(
    heel,
    add(
      scale(up, HEEL_POINT.height * shoeLengthM),
      scale(forward, HEEL_POINT.forward * shoeLengthM),
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
