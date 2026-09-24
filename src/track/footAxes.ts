import {
  FOOT_JOINTS,
  FOOT_NET_JOINTS,
  type FootJoint,
  type FootNetJoint,
} from '../native/joints';

export type Point = { x: number; y: number };
export type Size = { width: number; height: number };
export type FootSide = 'left' | 'right';

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
  // FootNet's points, or the body model's. The body model's big toe, which is there either way, and which the frames
  // between two FootNet runs are carried by.
  refined: boolean;
  anchor: Point;
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

// FootNet's own score for a point; below it the peak is a guess (the 5th percentile on held-out SynFoot).
export const REFINED_MIN_SCORE = 0.3;

function refinedFoot(
  refined: readonly number[],
  index: number,
): FootPoints | undefined {
  const at = (name: FootNetJoint): Point | undefined => {
    const offset =
      (index * FOOT_NET_JOINTS.length + FOOT_NET_JOINTS.indexOf(name)) * 3;
    const score = refined[offset + 2];
    return score >= REFINED_MIN_SCORE
      ? { x: refined[offset], y: refined[offset + 1] }
      : undefined;
  };
  const toe = at('bigToe');
  // The toes are what the shoe is anchored to, so FootNet's are worth having on their own. Its heel comes with them
  // when it is sure of it — which is about half the time, because looking down at your own feet hides the heel behind
  // the foot — and the body model's ankle stands in for it otherwise.
  return toe ? { toe, smallToe: at('littleToe'), heel: at('heel') } : undefined;
}

export function footAxes(
  points: readonly number[],
  minScore: number,
  refined?: readonly number[],
): FootAxis[] {
  return (['left', 'right'] as const).flatMap((side, index): FootAxis[] => {
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
    // FootNet follows a foot on its own, so its points stand even on the frames where the body model did not run.
    const better = refined && refinedFoot(refined, index);
    if (better) {
      return [
        {
          side,
          ...better,
          ankle: better.heel ? undefined : seen(names.ankle),
          score: Math.max(score, REFINED_MIN_SCORE),
          refined: true,
          anchor: score >= minScore ? { x: toe.x, y: toe.y } : better.toe,
        },
      ];
    }
    if (score < minScore) {
      return [];
    }
    const anchor = { x: toe.x, y: toe.y };
    return [
      {
        side,
        toe: anchor,
        smallToe: seen(names.smallToe),
        ankle: seen(names.ankle),
        heel: seen(names.heel),
        score,
        refined: false,
        anchor,
      },
    ];
  });
}

export function frameToView(point: Point, frame: Size, view: Size): Point {
  const scale = Math.min(view.width / frame.width, view.height / frame.height);
  const offsetX = (view.width - frame.width * scale) / 2;
  const offsetY = (view.height - frame.height * scale) / 2;
  return {
    x: offsetX + point.x * frame.width * scale,
    y: offsetY + point.y * frame.height * scale,
  };
}
