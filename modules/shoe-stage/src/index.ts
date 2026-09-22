import { getHostComponent, NitroModules } from 'react-native-nitro-modules';
import type { DeviceGravity } from './DeviceGravity.nitro';
import type { PersonMatte } from './PersonMatte.nitro';
import type { ShoeViewMethods, ShoeViewProps } from './ShoeView.nitro';

export type { DeviceGravity } from './DeviceGravity.nitro';
export type { PersonMatte } from './PersonMatte.nitro';
export type { ShoePose, ShoeSide, ShoeViewProps } from './ShoeView.nitro';

export const ShoeView = getHostComponent<ShoeViewProps, ShoeViewMethods>(
  'ShoeView',
  () => require('../nitrogen/generated/shared/json/ShoeViewConfig.json'),
);

export function createDeviceGravity(): DeviceGravity {
  return NitroModules.createHybridObject<DeviceGravity>('DeviceGravity');
}

export function createPersonMatte(): PersonMatte {
  return NitroModules.createHybridObject<PersonMatte>('PersonMatte');
}
