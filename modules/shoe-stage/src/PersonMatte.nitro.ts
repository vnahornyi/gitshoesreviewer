import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

// Apple Vision person segmentation of the camera frame, kept for ShoeView to show the real leg over the shoe.
export interface PersonMatte extends HybridObject<{ ios: 'swift' }> {
  // Segments the frame and returns the milliseconds it took.
  update(frame: Frame): number;
}
