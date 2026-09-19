import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

export type FootModel = 'rtmw-x-l' | 'rtmpose-m';

export interface FootPoseResult {
  points: number[];
  preprocessMs: number;
  inferenceMs: number;
}

export interface FootPoseDetector extends HybridObject<{ ios: 'swift' }> {
  readonly status: string;
  load(model: FootModel): void;
  detect(frame: Frame): FootPoseResult;
}
