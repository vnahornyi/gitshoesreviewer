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
const HEEL_SCENE_HEIGHT = 1.2;
const FRONTAL_SCENE_HEIGHT = 1.1;

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
  const floorPoint = add(
    scale(up, -HEEL_SCENE_HEIGHT),
    scale(floorForward, 1.5),
  );
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

// A mirror-like view: camera 1.1 m up, tilted 30° down, the foot 2 m ahead with its toes turned towards the camera.
function frontalScene(yawDegrees: number) {
  const tilt = (30 * Math.PI) / 180;
  const gravity: Vec3 = [0, Math.cos(tilt), -Math.sin(tilt)];
  const up = scale(gravity, -1);
  const floorForward = normalize(add([0, 0, 1], scale(up, -up[2])));
  const floorRight = normalize(cross(floorForward, up));
  const yaw = (yawDegrees * Math.PI) / 180;
  const forward = normalize(
    add(scale(floorForward, Math.cos(yaw)), scale(floorRight, Math.sin(yaw))),
  );
  const side = cross(up, forward);
  const origin = add(scale(up, -FRONTAL_SCENE_HEIGHT), scale(floorForward, 2));
  const onShoe = (height: number, along: number, across = 0) =>
    add(
      add(add(origin, scale(up, height * SHOE)), scale(forward, along * SHOE)),
      scale(side, across * SHOE),
    );
  const ankle = project(onShoe(0.28, 0.26));
  return {
    gravity,
    up,
    forward,
    origin,
    // What the pose model reports from the front: the "heel" lands on the ankle.
    foot: {
      ankle,
      heel: ankle,
      toe: project(onShoe(0.08, 0.86, 0.12)),
      smallToe: project(onShoe(0.08, 0.86, -0.12)),
    },
  };
}

function expectPlacement(
  m: number[] | null,
  s: { up: Vec3; forward: Vec3; origin: Vec3 },
  digits: number,
) {
  expect(m).not.toBeNull();
  const [y, z, o] = [flip(s.up), flip(s.forward), flip(s.origin)];
  [4, 5, 6].forEach((i, n) => expect(m![i] / SHOE).toBeCloseTo(y[n], digits));
  [8, 9, 10].forEach((i, n) => expect(m![i] / SHOE).toBeCloseTo(z[n], digits));
  [12, 13, 14].forEach((i, n) => expect(m![i]).toBeCloseTo(o[n], digits));
}

describe('shoeTransform', () => {
  it('recovers the shoe on the floor from heel, toe and gravity', () => {
    const s = scene();
    const m = shoeTransform(
      { heel: s.heel, toe: s.toe },
      frame,
      k,
      s.gravity,
      SHOE,
      HEEL_SCENE_HEIGHT,
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

  it('places the shoe from the ankle and toes when the heel sits on the ankle', () => {
    for (const yaw of [180, 150, 225, 90]) {
      const s = frontalScene(yaw);
      // The image midpoint of the two toes is not the projection of their 3D midpoint: allow half a centimetre.
      expectPlacement(
        shoeTransform(s.foot, frame, k, s.gravity, SHOE, FRONTAL_SCENE_HEIGHT),
        s,
        2,
      );
    }
  });

  it('uses the big toe alone when the little toe is not seen', () => {
    const s = frontalScene(180);
    const bigToeOnly = {
      ankle: s.foot.ankle,
      toe: project(
        add(
          add(
            add(s.origin, scale(s.up, 0.08 * SHOE)),
            scale(s.forward, 0.95 * SHOE),
          ),
          [0, 0, 0],
        ),
      ),
    };
    expectPlacement(
      shoeTransform(
        bigToeOnly,
        frame,
        k,
        s.gravity,
        SHOE,
        FRONTAL_SCENE_HEIGHT,
      ),
      s,
      5,
    );
  });

  it('keeps the toes on the image toes when the camera height is a guess', () => {
    const s = frontalScene(160);
    const toe = project(
      add(
        add(s.origin, scale(s.up, 0.08 * SHOE)),
        scale(s.forward, 0.95 * SHOE),
      ),
    );
    const m = shoeTransform(
      { ankle: s.foot.ankle, toe },
      frame,
      k,
      s.gravity,
      SHOE,
      FRONTAL_SCENE_HEIGHT * 1.25,
    )!;
    const local: Vec3 = [0, 0.08, 0.95];
    const row = (r: number) =>
      m[r] * local[0] + m[4 + r] * local[1] + m[8 + r] * local[2] + m[12 + r];
    const seen = project(flip([row(0), row(1), row(2)]));
    expect(seen.x).toBeCloseTo(toe.x, 6);
    expect(seen.y).toBeCloseTo(toe.y, 6);
  });

  it('keeps a right-handed rotation', () => {
    const s = scene();
    const m = shoeTransform(
      { heel: s.heel, toe: s.toe },
      frame,
      k,
      s.gravity,
      SHOE,
      HEEL_SCENE_HEIGHT,
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
        HEEL_SCENE_HEIGHT,
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
