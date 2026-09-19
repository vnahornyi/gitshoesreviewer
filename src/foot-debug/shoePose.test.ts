import {
  gravityInBackCamera,
  intrinsicsFor,
  shoeTransform,
  verticalFovDegrees,
  type Intrinsics,
} from './shoePose';
import { add, cross, normalize, scale, type Vec3 } from './vec3';

const frame = { width: 720, height: 1280 };
const k: Intrinsics = { focal: 1000, cx: 360, cy: 640 };
const SHOE = 0.29;

function project(p: Vec3) {
  return {
    x: (k.focal * (p[0] / p[2]) + k.cx) / frame.width,
    y: (k.focal * (p[1] / p[2]) + k.cy) / frame.height,
  };
}

// A camera 1.2 m above the floor, tilted 35° down, looking at a foot 1.5 m ahead, turned 30° to the right.
function scene() {
  const tilt = (35 * Math.PI) / 180;
  const gravity: Vec3 = [0, Math.cos(tilt), -Math.sin(tilt)];
  const up = scale(gravity, -1);
  const floorForward = normalize(add([0, 0, 1], scale(up, -up[2])));
  const floorRight = normalize(cross(floorForward, up));
  const yaw = (30 * Math.PI) / 180;
  const forward = normalize(
    add(scale(floorForward, Math.cos(yaw)), scale(floorRight, Math.sin(yaw))),
  );
  const floorPoint = add(scale(up, -1.2), scale(floorForward, 1.5));
  const heel = add(
    add(floorPoint, scale(up, 0.08 * SHOE)),
    scale(forward, 0.035 * SHOE),
  );
  const toe = add(heel, scale(forward, 0.915 * SHOE));
  return {
    gravity,
    up,
    forward,
    floorPoint,
    heel: project(heel),
    toe: project(toe),
  };
}

const flip = (v: Vec3): Vec3 => [v[0], -v[1], -v[2]];

describe('shoeTransform', () => {
  it('recovers the shoe on the floor from heel, toe and gravity', () => {
    const s = scene();
    const m = shoeTransform(
      { heel: s.heel, toe: s.toe },
      frame,
      k,
      s.gravity,
      SHOE,
    );
    expect(m).not.toBeNull();
    const [x, y, z] = [flip(s.up), flip(s.forward), flip(s.floorPoint)];
    expect(m![4] / SHOE).toBeCloseTo(x[0], 5);
    expect(m![5] / SHOE).toBeCloseTo(x[1], 5);
    expect(m![6] / SHOE).toBeCloseTo(x[2], 5);
    expect(m![8] / SHOE).toBeCloseTo(y[0], 5);
    expect(m![9] / SHOE).toBeCloseTo(y[1], 5);
    expect(m![10] / SHOE).toBeCloseTo(y[2], 5);
    expect(m![12]).toBeCloseTo(z[0], 5);
    expect(m![13]).toBeCloseTo(z[1], 5);
    expect(m![14]).toBeCloseTo(z[2], 5);
  });

  it('keeps a right-handed rotation', () => {
    const s = scene();
    const m = shoeTransform(
      { heel: s.heel, toe: s.toe },
      frame,
      k,
      s.gravity,
      SHOE,
    )!;
    const col = (i: number): Vec3 => [m[i * 4], m[i * 4 + 1], m[i * 4 + 2]];
    const handed = cross(col(0), col(1));
    expect(
      handed[0] * m[8] + handed[1] * m[9] + handed[2] * m[10],
    ).toBeGreaterThan(0);
  });

  it('rejects a heel and toe on opposite sides of the horizon', () => {
    const level: Vec3 = [0, 1, 0];
    const belowHorizon = { x: 0.5, y: 0.7 };
    const aboveHorizon = { x: 0.5, y: 0.3 };
    expect(
      shoeTransform(
        { heel: belowHorizon, toe: aboveHorizon },
        frame,
        k,
        level,
        SHOE,
      ),
    ).toBeNull();
  });
});

describe('intrinsicsFor', () => {
  it('rescales a landscape sensor matrix to the upright frame', () => {
    const sensor = [1500, 0, 0, 0, 1500, 0, 960, 540, 1];
    expect(intrinsicsFor(frame, sensor)).toEqual({
      focal: 1000,
      cx: 360,
      cy: 640,
    });
  });

  it('falls back to a typical iPhone field of view', () => {
    expect(verticalFovDegrees(frame, intrinsicsFor(frame))).toBeCloseTo(65, 5);
  });
});

describe('gravityInBackCamera', () => {
  it('points down the image when the phone is upright', () => {
    expect(gravityInBackCamera([0, -1, 0])).toEqual([0, 1, -0]);
  });
});
