import type { HybridObject } from 'react-native-nitro-modules';

export interface DeviceGravity extends HybridObject<{ ios: 'swift' }> {
  readonly available: boolean;
  current(): number[];
}
