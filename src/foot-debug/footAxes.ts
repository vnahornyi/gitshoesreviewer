import { FOOT_JOINTS, type FootJoint } from 'react-native-foot-pose';

export type Point = { x: number; y: number };
export type Size = { width: number; height: number };
export type FootSide = 'left' | 'right';
export type SceneMode = 'mirror' | 'direct';

// The big toe is always there; the rest is whatever the model saw well enough.
export type FootPoints = {
  toe: Point;
  smallToe?: Point;
  ankle?: Point;
  heel?: Point;
};

export type FootAxis = FootPoints & {
  side: FootSide;
  score: number;
};

type Joint = Point & { score: number };

function joint(points: readonly number[], name: FootJoint): Joint {
  const offset = FOOT_JOINTS.indexOf(name) * 3;
  return {
    x: points[offset],
    y: points[offset + 1],
    score: points[offset + 2],
  };
}

const SIDE_JOINTS: Record<
  FootSide,
  Record<'toe' | 'smallToe' | 'ankle' | 'heel', FootJoint>
> = {
  left: {
    toe: 'leftBigToe',
    smallToe: 'leftSmallToe',
    ankle: 'leftAnkle',
    heel: 'leftHeel',
  },
  right: {
    toe: 'rightBigToe',
    smallToe: 'rightSmallToe',
    ankle: 'rightAnkle',
    heel: 'rightHeel',
  },
};

// The model names feet by how they look in the image; a mirror shows your left foot as a right one.
export function sideInScene(modelSide: FootSide, scene: SceneMode): FootSide {
  if (scene === 'direct') {
    return modelSide;
  }
  return modelSide === 'left' ? 'right' : 'left';
}

export function footAxes(
  points: readonly number[],
  minScore: number,
): FootAxis[] {
  return (['left', 'right'] as const).flatMap(side => {
    const names = SIDE_JOINTS[side];
    const toe = joint(points, names.toe);
    const seen = (name: FootJoint): Point | undefined => {
      const j = joint(points, name);
      return j.score >= minScore ? { x: j.x, y: j.y } : undefined;
    };
    const ankle = joint(points, names.ankle);
    const heel = joint(points, names.heel);
    // A shoe needs the toe and something at the back of the foot: the ankle, or the heel.
    const score = Math.min(toe.score, Math.max(ankle.score, heel.score));
    if (score < minScore) {
      return [];
    }
    return [
      {
        side,
        toe: { x: toe.x, y: toe.y },
        smallToe: seen(names.smallToe),
        ankle: seen(names.ankle),
        heel: seen(names.heel),
        score,
      },
    ];
  });
}

export function frameToView(
  point: Point,
  frame: Size,
  view: Size,
  flipX: boolean,
): Point {
  const scale = Math.min(view.width / frame.width, view.height / frame.height);
  const offsetX = (view.width - frame.width * scale) / 2;
  const offsetY = (view.height - frame.height * scale) / 2;
  const x = flipX ? 1 - point.x : point.x;
  return {
    x: offsetX + x * frame.width * scale,
    y: offsetY + point.y * frame.height * scale,
  };
}
