import type { FootPoints, Point, Size } from './footAxes';
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

function inDepthRange(...depths: number[]): boolean {
  return depths.every(d => d >= MIN_DEPTH_M && d <= MAX_DEPTH_M);
}

// Seen from the front the model puts the "heel" on the ankle, so the ankle and the toes lead and the heel is the fallback.
export function shoeTransform(
  foot: FootPoints,
  frame: Size,
  intrinsics: Intrinsics,
  gravity: Vec3,
  shoeLengthM: number,
): number[] | null {
  const up = normalize(scale(gravity, -1));
  const rays = (p: Point) => ray(p, frame, intrinsics);
  if (foot.ankle) {
    const toes = foot.smallToe
      ? {
          x: (foot.toe.x + foot.smallToe.x) / 2,
          y: (foot.toe.y + foot.smallToe.y) / 2,
        }
      : foot.toe;
    const front = foot.smallToe ? TOES_MIDPOINT : TOE_POINT;
    return fromAnkle(rays(foot.ankle), rays(toes), front, up, shoeLengthM);
  }
  if (foot.heel) {
    return fromHeel(rays(foot.heel), rays(foot.toe), up, shoeLengthM);
  }
  return null;
}

// The ankle stands a known height above the toes and a known distance behind them along the floor. With gravity that
// is two equations for the two depths: the height fixes the toe depth given the ankle's, and the floor distance is a
// quadratic in the ankle depth with exactly one positive root while the height gap is smaller than the distance.
function fromAnkle(
  ankleRay: Vec3,
  toeRay: Vec3,
  front: ShoePoint,
  up: Vec3,
  shoeLengthM: number,
): number[] | null {
  const rise = (ANKLE_POINT.height - front.height) * shoeLengthM;
  const reach = (front.forward - ANKLE_POINT.forward) * shoeLengthM;
  const ankleUp = dot(ankleRay, up);
  const toeUp = dot(toeRay, up);
  if (Math.abs(toeUp) < GRAZING) {
    return null;
  }
  const flat = (v: Vec3) => sub(v, scale(up, dot(v, up)));
  // toeDepth = (ankleDepth · ankleUp − rise) / toeUp; the ankle-to-toe floor vector is ankleDepth · a + b.
  const a = flat(sub(scale(toeRay, ankleUp / toeUp), ankleRay));
  const b = flat(scale(toeRay, -rise / toeUp));
  const qa = dot(a, a);
  const qb = 2 * dot(a, b);
  const qc = dot(b, b) - reach * reach;
  const discriminant = qb * qb - 4 * qa * qc;
  if (qa === 0 || discriminant < 0) {
    return null;
  }
  const ankleDepth = (-qb + Math.sqrt(discriminant)) / (2 * qa);
  const toeDepth = (ankleDepth * ankleUp - rise) / toeUp;
  if (!inDepthRange(ankleDepth, toeDepth)) {
    return null;
  }
  const along = add(scale(a, ankleDepth), b);
  if (length(along) === 0) {
    return null;
  }
  return placement(
    scale(ankleRay, ankleDepth),
    ANKLE_POINT,
    normalize(along),
    up,
    shoeLengthM,
  );
}

// A standing foot is flat on the floor, so with gravity known the heel and toe share a height:
// that pins both depths from two image points and the shoe length, with no iterative solve.
function fromHeel(
  heelRay: Vec3,
  toeRay: Vec3,
  up: Vec3,
  shoeLengthM: number,
): number[] | null {
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
  if (!inDepthRange(heelDepth, toeDepth)) {
    return null;
  }
  const heel = scale(heelRay, heelDepth);
  const along = sub(scale(toeRay, toeDepth), heel);
  const flat = sub(along, scale(up, dot(along, up)));
  if (length(flat) === 0) {
    return null;
  }
  return placement(heel, HEEL_POINT, normalize(flat), up, shoeLengthM);
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
