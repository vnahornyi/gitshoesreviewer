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
  // Cut the shoe where the person matte shows the leg above the shoe's collar; otherwise a fixed shin cylinder does it.
  legMatte: boolean;
}

export interface ShoeViewMethods extends HybridViewMethods {}

export type ShoeView = HybridView<
  ShoeViewProps,
  ShoeViewMethods,
  { ios: 'swift' }
>;
