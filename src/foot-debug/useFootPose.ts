import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createFootPoseDetector, type FootModel } from 'react-native-foot-pose';
import { createPersonMatte } from 'react-native-shoe-stage';
import { useFrameOutput, type Frame } from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';
import {
  trackedFeet,
  updateTracks,
  type FootTrack,
  type TrackedFoot,
} from './footTracker';

export type FootPoseSample = {
  points: number[];
  preprocessMs: number;
  inferenceMs: number;
  matteMs?: number;
  frameWidth: number;
  frameHeight: number;
  isMirrored: boolean;
  cameraMatrix?: number[];
};

const FPS_WINDOW = 30;

// With `legMatte` the frame is also segmented for ShoeView, which then shows the real leg coming out of the shoe.
export function useFootPose(model: FootModel, legMatte: boolean) {
  const detector = useMemo(createFootPoseDetector, []);
  const matte = useMemo(createPersonMatte, []);
  const [sample, setSample] = useState<FootPoseSample | null>(null);
  const [status, setStatus] = useState(detector.status);
  const [fps, setFps] = useState(0);
  const [feet, setFeet] = useState<TrackedFoot[]>([]);
  const arrivals = useRef<number[]>([]);
  const tracks = useRef<FootTrack[]>([]);
  const lastId = useRef(0);

  useEffect(() => {
    detector.load(model);
    arrivals.current = [];
    tracks.current = [];
    setSample(null);
    setFeet([]);
    setStatus(detector.status);
  }, [detector, model]);

  const onSample = useCallback((next: FootPoseSample) => {
    const now = Date.now();
    const recent = [...arrivals.current, now].slice(-FPS_WINDOW);
    arrivals.current = recent;
    if (recent.length > 1) {
      setFps(((recent.length - 1) * 1000) / (now - recent[0]));
    }
    tracks.current = updateTracks(
      tracks.current,
      next.points,
      now,
      () => ++lastId.current,
    );
    setFeet(trackedFeet(tracks.current, now));
    setStatus('ready');
    setSample(next);
  }, []);

  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet';
      try {
        const result = detector.detect(frame);
        const matteMs = legMatte ? matte.update(frame) : undefined;
        scheduleOnRN(onSample, {
          points: result.points,
          preprocessMs: result.preprocessMs,
          inferenceMs: result.inferenceMs,
          matteMs,
          frameWidth: frame.width,
          frameHeight: frame.height,
          isMirrored: frame.isMirrored,
          cameraMatrix: frame.cameraIntrinsicMatrix,
        });
      } catch {
        scheduleOnRN(setStatus, detector.status);
      } finally {
        frame.dispose();
      }
    },
    [detector, matte, legMatte, onSample],
  );

  const frameOutput = useFrameOutput({
    pixelFormat: 'rgb',
    enablePhysicalBufferRotation: true,
    enableCameraMatrixDelivery: true,
    onFrame,
  });

  return { frameOutput, sample, feet, status, fps };
}
