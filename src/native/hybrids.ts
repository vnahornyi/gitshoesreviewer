/**
 * The native layer, as JavaScript sees it: five Nitro HybridObjects behind plain factory functions.
 *
 * All of them are iOS-only today. Each `create…` call reaches into the ShoeTryOn pod, so calling one
 * before `pod install` has run throws rather than returning a stub — the contract for each object is
 * in `ios/README.md`.
 */
import { getHostComponent, NitroModules } from 'react-native-nitro-modules';
import type { DeviceGravity } from './specs/DeviceGravity.nitro';
import type { FootPoseDetector } from './specs/FootPoseDetector.nitro';
import type { PersonMatte } from './specs/PersonMatte.nitro';
import type { SceneLight } from './specs/SceneLight.nitro';
import type { ShoeViewMethods, ShoeViewProps } from './specs/ShoeView.nitro';

export type {
  FootModel,
  FootPoseDetector,
  FootPoseResult,
} from './specs/FootPoseDetector.nitro';
export type { DeviceGravity } from './specs/DeviceGravity.nitro';
export type { PersonMatte } from './specs/PersonMatte.nitro';
export type { SceneLight } from './specs/SceneLight.nitro';
export type { ShoePose, ShoeSide, ShoeViewProps } from './specs/ShoeView.nitro';

export const ShoeView = getHostComponent<ShoeViewProps, ShoeViewMethods>(
  'ShoeView',
  () => require('../../nitrogen/generated/shared/json/ShoeViewConfig.json'),
);

export function createFootPoseDetector(): FootPoseDetector {
  return NitroModules.createHybridObject<FootPoseDetector>('FootPoseDetector');
}

export function createDeviceGravity(): DeviceGravity {
  return NitroModules.createHybridObject<DeviceGravity>('DeviceGravity');
}

export function createPersonMatte(): PersonMatte {
  return NitroModules.createHybridObject<PersonMatte>('PersonMatte');
}

export function createSceneLight(): SceneLight {
  return NitroModules.createHybridObject<SceneLight>('SceneLight');
}
