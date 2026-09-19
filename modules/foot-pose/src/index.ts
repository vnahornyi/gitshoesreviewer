import { NitroModules } from 'react-native-nitro-modules';
import type { FootPoseDetector } from './FootPoseDetector.nitro';

export type {
  FootModel,
  FootPoseDetector,
  FootPoseResult,
} from './FootPoseDetector.nitro';

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

export function createFootPoseDetector(): FootPoseDetector {
  return NitroModules.createHybridObject<FootPoseDetector>('FootPoseDetector');
}
