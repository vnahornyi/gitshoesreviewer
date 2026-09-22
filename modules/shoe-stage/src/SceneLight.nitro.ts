import type { HybridObject } from 'react-native-nitro-modules';
import type { Frame } from 'react-native-vision-camera';

// The average colour of the lower half of the camera frame (floor and feet), kept for ShoeView to light the shoe like the scene.
export interface SceneLight extends HybridObject<{ ios: 'swift' }> {
  update(frame: Frame): void;
}
