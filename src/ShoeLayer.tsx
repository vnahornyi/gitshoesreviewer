import { useMemo } from 'react';
import { StyleSheet } from 'react-native';
import { createDeviceGravity, ShoeView, type ShoePose } from './native/hybrids';
import { frameToView, type Size } from './track/footAxes';
import type { TrackedFoot } from './track/footTracker';
import {
  gravityInBackCamera,
  intrinsicsFor,
  shoeTransform,
  verticalFovDegrees,
} from './pose/shoePose';
import type { FootPoseSample } from './useFootPose';

const SHOE_MODEL = 'placeholder-shoe';
// About EU 43. The catalog's sole length replaces this once real models land.
const SHOE_LENGTH_M = 0.29;
// Assumed, not measured: a phone held at chest height, which a mirror keeps. A wrong guess scales the shoe a little
// but keeps it on the feet in the image.
const CAMERA_HEIGHT_M = 1.3;

const gravity = createDeviceGravity();

type ShoeLayerProps = {
  feet: TrackedFoot[];
  sample: FootPoseSample;
  view: Size;
  legMatte: boolean;
  matchCamera: boolean;
};

export function ShoeLayer({
  feet,
  sample,
  view,
  legMatte,
  matchCamera,
}: ShoeLayerProps) {
  const frame = useMemo(
    () => ({ width: sample.frameWidth, height: sample.frameHeight }),
    [sample.frameWidth, sample.frameHeight],
  );
  const intrinsics = intrinsicsFor(frame, sample.cameraMatrix);
  const topLeft = frameToView({ x: 0, y: 0 }, frame, view);
  const bottomRight = frameToView({ x: 1, y: 1 }, frame, view);
  const down = gravityInBackCamera(gravity.current());

  const shoes = feet.flatMap<ShoePose>(foot => {
    const transform = shoeTransform(
      foot,
      frame,
      intrinsics,
      down,
      SHOE_LENGTH_M,
      CAMERA_HEIGHT_M,
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
      legMatte={legMatte}
      matchCamera={matchCamera}
    />
  );
}

const styles = StyleSheet.create({
  stage: { position: 'absolute' },
});
