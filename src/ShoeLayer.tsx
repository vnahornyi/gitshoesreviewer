import { useEffect, useMemo, useRef } from 'react';
import { StyleSheet } from 'react-native';
import { createDeviceGravity, ShoeView, type ShoePose } from './native/hybrids';
import { frameToView, type Size } from './track/footAxes';
import {
  startOneEuro,
  stepOneEuro,
  type OneEuroParams,
  type OneEuroState,
} from './track/oneEuro';
import type { TrackedFoot } from './track/footTracker';
import {
  gravityInBackCamera,
  impliedShoeLengthM,
  intrinsicsFor,
  shoeTransform,
  verticalFovDegrees,
} from './pose/shoePose';
import type { FootPoseSample } from './useFootPose';

const SHOE_MODEL = 'placeholder-shoe';
// Only the starting point for the measurement below, and the fallback for a foot not measured yet. About EU 43.
const SHOE_LENGTH_M = 0.29;
// Assumed, not measured: a phone held at chest height, which a mirror keeps. Drawing the measured length makes the
// rendered image independent of this number — a wrong camera height scales the reconstruction and the shoe by the
// same factor, and a similarity about the camera projects to the same pixels. It still has to be right to report a
// real size in millimetres, which is a separate feature.
const CAMERA_HEIGHT_M = 1.3;

// No adult foot is outside this. A pose that implies one came from landmarks that are wrong or from a ray near the
// horizon, where the floor intersection runs away: in this project's own recording every bad frame read 385–2174 mm
// while every good one read 281–306. Cheaper and sharper than a confidence threshold, and it needs nothing the
// tracker does not already hand over.
const MIN_SHOE_M = 0.18;
const MAX_SHOE_M = 0.33;

// FootNet's confidence on real frames is bimodal and flickers between frames of a scene that has not changed, so a
// foot that is plainly there loses its pose for a frame or two at a time. A shoe is a physical object and does not
// vanish, so the last pose is kept for this long rather than the shoe being deleted and drawn again. It is a hold,
// not a guess: the shoe stays exactly where it was. That only works because the crop is led by the foot's speed on
// the native side — holding a stale pose through a stride without that would slide the shoe off the foot.
const HOLD_MS = 250;

// A person's shoe length does not change, so this is a plain low-pass: no speed adaptation, unlike the points.
const LENGTH_SMOOTHING: OneEuroParams = {
  minCutoff: 0.25,
  beta: 0,
  derivativeCutoff: 1,
};

const gravity = createDeviceGravity();

const WINDOW_MS = 1000;

type Reading = {
  at: number;
  side: 'left' | 'right';
  raw: number | null;
  lengthM: number | null;
  held: boolean;
  origin: readonly [number, number, number] | null;
};

const mm = (metres: number) => (metres * 1000).toFixed(0);

function spread(values: number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

// Once a second, what the geometry says rather than what it was told. `довжина` is the smoothed length actually
// drawn, `сире` its spread before smoothing, and `дрож` how far the shoe's origin moves while standing still — the
// number behind "it wobbles", which no screenshot can settle. `відкинуто` counts the frames whose implied length was
// outside a human foot, where nothing was drawn.
function logPlacement(readings: Reading[]) {
  const line = (side: 'left' | 'right') => {
    const own = readings.filter(r => r.side === side);
    const drawn = own.filter(
      (r): r is Reading & { lengthM: number } => r.lengthM !== null,
    );
    const rejected = own.length - drawn.length;
    const held = own.filter(r => r.held).length;
    if (drawn.length === 0) {
      return own.length === 0
        ? 'none'
        : `відкинуто всі ${rejected}, утримано ${held}`;
    }
    const lengths = drawn.map(r => r.lengthM);
    const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const origins = drawn
      .map(r => r.origin)
      .filter((o): o is readonly [number, number, number] => o !== null);
    const jitter = [0, 1, 2].map(axis => spread(origins.map(o => o[axis])));
    return (
      `довжина ${mm(mean)} мм · сире ±${mm(
        spread(drawn.map(r => r.raw ?? 0)),
      )} ` +
      `· дрож x${mm(jitter[0])} y${mm(jitter[1])} z${mm(jitter[2])} мм ` +
      `· ${drawn.length} кадрів, відкинуто ${rejected}, утримано ${held}`
    );
  };
  console.log(
    `[shoe] камера ${mm(CAMERA_HEIGHT_M)} мм (не впливає на кадр), межі ${mm(
      MIN_SHOE_M,
    )}–${mm(MAX_SHOE_M)} мм` +
      `\n  ліва:  ${line('left')}\n  права: ${line('right')}`,
  );
}

type ShoeLayerProps = {
  feet: TrackedFoot[];
  sample: FootPoseSample;
  view: Size;
  legMatte: boolean;
  matchCamera: boolean;
  maskPreview?: boolean;
};

export function ShoeLayer({
  feet,
  sample,
  view,
  legMatte,
  matchCamera,
  maskPreview = false,
}: ShoeLayerProps) {
  const frame = useMemo(
    () => ({ width: sample.frameWidth, height: sample.frameHeight }),
    [sample.frameWidth, sample.frameHeight],
  );
  const intrinsics = intrinsicsFor(frame, sample.cameraMatrix);
  const topLeft = frameToView({ x: 0, y: 0 }, frame, view);
  const bottomRight = frameToView({ x: 1, y: 1 }, frame, view);
  const down = gravityInBackCamera(gravity.current());

  const smoothed = useRef(new Map<number, OneEuroState>());
  const now = Date.now();

  const placed = feet.map(foot => {
    const raw = impliedShoeLengthM(
      foot,
      frame,
      intrinsics,
      down,
      SHOE_LENGTH_M,
      CAMERA_HEIGHT_M,
    );
    if (raw === null || raw < MIN_SHOE_M || raw > MAX_SHOE_M) {
      return { foot, raw, lengthM: null, transform: null };
    }
    const previous = smoothed.current.get(foot.id);
    const state = previous
      ? stepOneEuro(previous, raw, now, LENGTH_SMOOTHING)
      : startOneEuro(raw, now);
    smoothed.current.set(foot.id, state);
    return {
      foot,
      raw,
      lengthM: state.value,
      transform: shoeTransform(
        foot,
        frame,
        intrinsics,
        down,
        state.value,
        CAMERA_HEIGHT_M,
      ),
    };
  });

  const lastPose = useRef(
    new Map<number, { transform: number[]; at: number }>(),
  );
  const drawn = placed.map(entry => {
    if (entry.transform) {
      return { ...entry, held: false };
    }
    const previous = lastPose.current.get(entry.foot.id);
    const fresh = previous && now - previous.at <= HOLD_MS;
    return {
      ...entry,
      transform: fresh ? previous.transform : null,
      held: !!fresh,
    };
  });

  const shoes = drawn.flatMap<ShoePose>(({ foot, transform }) =>
    transform ? [{ id: foot.id, side: foot.side, transform }] : [],
  );

  const window = useRef<Reading[]>([]);
  const loggedAt = useRef(0);
  useEffect(() => {
    const live = new Set(feet.map(foot => foot.id));
    smoothed.current.forEach((_, id) => {
      if (!live.has(id)) {
        smoothed.current.delete(id);
      }
    });
    lastPose.current.forEach((_, id) => {
      if (!live.has(id)) {
        lastPose.current.delete(id);
      }
    });
    drawn.forEach(({ foot, raw, lengthM, transform, held }) => {
      if (transform && !held) {
        lastPose.current.set(foot.id, { transform, at: now });
      }
      window.current.push({
        at: now,
        side: foot.side,
        raw,
        lengthM,
        held,
        origin: transform
          ? [transform[12], transform[13], transform[14]]
          : null,
      });
    });
    window.current = window.current.filter(r => now - r.at <= WINDOW_MS);
    if (now - loggedAt.current >= WINDOW_MS) {
      loggedAt.current = now;
      logPlacement(window.current);
    }
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
      model={maskPreview ? '' : SHOE_MODEL}
      verticalFovDegrees={verticalFovDegrees(frame, intrinsics)}
      shoes={shoes}
      legMatte={legMatte}
      matchCamera={matchCamera}
      maskPreview={maskPreview}
      maskPreviewVersion={sample.sampleVersion}
    />
  );
}

const styles = StyleSheet.create({
  stage: { position: 'absolute' },
});
