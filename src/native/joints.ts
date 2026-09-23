/**
 * The joint orders the native side uses. They are here, away from `native.ts`, because everything
 * that reads a `FootPoseResult` needs them and `native.ts` builds the Nitro view at import time —
 * pulling that in from a pure module drags the native layer into plain unit tests.
 */

// RTMPose's 8 body joints, in `FootPoseResult.points` order.
export const FOOT_JOINTS = [
  'leftAnkle',
  'rightAnkle',
  'leftBigToe',
  'leftSmallToe',
  'leftHeel',
  'rightBigToe',
  'rightSmallToe',
  'rightHeel',
] as const;

export type FootJoint = (typeof FOOT_JOINTS)[number];

// FootNet's 8 points per foot, in `FootPoseResult.refined` order (left foot first, then right).
export const FOOT_NET_JOINTS = [
  'bigToe',
  'secondToe',
  'thirdToe',
  'fourthToe',
  'littleToe',
  'heel',
  'outerExtrema',
  'innerExtrema',
] as const;

export type FootNetJoint = (typeof FOOT_NET_JOINTS)[number];
