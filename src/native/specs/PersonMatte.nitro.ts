import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

// Apple Vision person segmentation of the camera frame, kept for ShoeView to show the real leg over the shoe.
export interface PersonMatte extends HybridObject<{ ios: 'swift' }> {
  // Set from JS; read natively, so the frame processor needs no copy of it.
  enabled: boolean;
  // When enabled and idle, hands a small copy of the frame to a background segmentation; returns how long the last one took.
  update(frame: Frame): number;
}
