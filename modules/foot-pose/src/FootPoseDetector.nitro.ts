import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

export type FootModel = 'rtmw-x-l' | 'rtmpose-m';

export interface FootPoseResult {
  points: number[];
  // FootNet, when `refine` is on: left foot then right, 8 points × [x, y, score] normalized to the frame.
  // A foot RTMPose did not find, or FootNet was not run on, is all zeros.
  refined: number[];
  preprocessMs: number;
  inferenceMs: number;
  refineMs: number;
}

export interface FootPoseDetector extends HybridObject<{ ios: 'swift' }> {
  readonly status: string;
  // Run FootNet on a crop around each foot RTMPose found, for its 8 keypoints.
  refine: boolean;
  load(model: FootModel): void;
  detect(frame: Frame): FootPoseResult;
}
