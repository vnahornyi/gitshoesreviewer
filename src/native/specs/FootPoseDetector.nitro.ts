import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

export type FootModel = 'rtmw-x-l' | 'rtmpose-m';

export interface FootPoseResult {
  points: number[];
  // FootNet, when `refine` is on: left foot then right, 8 points × [x, y, score] normalized to the frame.
  // A foot RTMPose did not find, or FootNet was not run on, is all zeros.
  refined: number[];
  // The crop actually passed to FootNet this frame: left then right, [x, y, side, brightness], normalized to the
  // frame. A missing crop is all zeros; these are not the next crops maintained by the tracker.
  crops: number[];
  preprocessMs: number;
  inferenceMs: number;
  refineMs: number;
  maskPreviewMs: number;
  maskPreviewStatus: string;
}

export interface FootPoseDetector extends HybridObject<{ ios: 'swift' }> {
  readonly status: string;
  // Run FootNet on a crop around each foot RTMPose found, for its 8 keypoints.
  refine: boolean;
  // Run the optional visible-foot mask model on the already tracked crops for the debug preview only.
  maskPreview: boolean;
  maskThreshold: number;
  load(model: FootModel): void;
  detect(frame: Frame): FootPoseResult;
}
