import { useMemo } from 'react';
import { StyleSheet } from 'react-native';
import {
  createDeviceGravity,
  ShoeView,
  type ShoePose,
} from 'react-native-shoe-stage';
import { frameToView, type Size } from './footAxes';
import type { TrackedFoot } from './footTracker';
import {
  gravityInBackCamera,
  intrinsicsFor,
  shoeTransform,
  verticalFovDegrees,
} from './shoePose';
import type { FootPoseSample } from './useFootPose';

const SHOE_MODEL = 'placeholder-shoe';
// About EU 43. The catalog's sole length replaces this once real models land.
const SHOE_LENGTH_M = 0.29;

const gravity = createDeviceGravity();

type ShoeLayerProps = {
  feet: TrackedFoot[];
  sample: FootPoseSample;
  view: Size;
};

export function ShoeLayer({ feet, sample, view }: ShoeLayerProps) {
  const frame = useMemo(
    () => ({ width: sample.frameWidth, height: sample.frameHeight }),
    [sample.frameWidth, sample.frameHeight],
  );
  const intrinsics = intrinsicsFor(frame, sample.cameraMatrix);
  const topLeft = frameToView({ x: 0, y: 0 }, frame, view, false);
  const bottomRight = frameToView({ x: 1, y: 1 }, frame, view, false);
  const down = gravityInBackCamera(gravity.current());

  const shoes = feet.flatMap<ShoePose>(foot => {
    const transform = shoeTransform(
      foot,
      frame,
      intrinsics,
      down,
      SHOE_LENGTH_M,
    );
    return transform ? [{ id: foot.id, side: foot.side, transform }] : [];
  });

  return (
    <ShoeView
      style={[
        styles.stage,
        {
          left: topLeft.x,
          top: topLeft.y,
          width: bottomRight.x - topLeft.x,
          height: bottomRight.y - topLeft.y,
        },
      ]}
      model={SHOE_MODEL}
      verticalFovDegrees={verticalFovDegrees(frame, intrinsics)}
      shoes={shoes}
    />
  );
}

const styles = StyleSheet.create({
  stage: { position: 'absolute' },
});
