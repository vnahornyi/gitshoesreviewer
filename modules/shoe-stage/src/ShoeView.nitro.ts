import type {
  HybridView,
  HybridViewMethods,
  HybridViewProps,
} from 'react-native-nitro-modules';

export type ShoeSide = 'left' | 'right';

export interface ShoePose {
  id: number;
  side: ShoeSide;
  transform: number[];
}

export interface ShoeViewProps extends HybridViewProps {
  model: string;
  verticalFovDegrees: number;
  shoes: ShoePose[];
}

export interface ShoeViewMethods extends HybridViewMethods {}

export type ShoeView = HybridView<
  ShoeViewProps,
  ShoeViewMethods,
  { ios: 'swift' }
>;
