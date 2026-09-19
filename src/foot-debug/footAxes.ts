import { FOOT_JOINTS, type FootJoint } from 'react-native-foot-pose';

export type Point = { x: number; y: number };
export type Size = { width: number; height: number };
export type FootSide = 'left' | 'right';
export type SceneMode = 'mirror' | 'direct';

export type FootAxis = {
  side: FootSide;
  heel: Point;
  toe: Point;
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

const SIDE_JOINTS: Record<FootSide, { toe: FootJoint; heel: FootJoint }> = {
  left: { toe: 'leftBigToe', heel: 'leftHeel' },
  right: { toe: 'rightBigToe', heel: 'rightHeel' },
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
    const toe = joint(points, SIDE_JOINTS[side].toe);
    const heel = joint(points, SIDE_JOINTS[side].heel);
    const score = Math.min(toe.score, heel.score);
    if (score < minScore) {
      return [];
    }
    return [
      {
        side,
        toe: { x: toe.x, y: toe.y },
        heel: { x: heel.x, y: heel.y },
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
