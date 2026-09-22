import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createFootPoseDetector, type FootModel } from 'react-native-foot-pose';
import { createPersonMatte, createSceneLight } from 'react-native-shoe-stage';
import {
  CommonResolutions,
  useFrameOutput,
  type Frame,
} from 'react-native-vision-camera';
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

// What ShoeView needs from each frame besides the feet: the person matte for the leg, the average tone for the lights.
export type FrameExtras = { legMatte: boolean; sceneLight: boolean };

export function useFootPose(
  model: FootModel,
  { legMatte, sceneLight }: FrameExtras,
) {
  const detector = useMemo(createFootPoseDetector, []);
  const matte = useMemo(createPersonMatte, []);
  const light = useMemo(createSceneLight, []);
  const [sample, setSample] = useState<FootPoseSample | null>(null);
  const [status, setStatus] = useState(detector.status);
  const [fps, setFps] = useState(0);
  const [feet, setFeet] = useState<TrackedFoot[]>([]);
  const arrivals = useRef<number[]>([]);

  // The frame processor keeps the closure it started with, so the flags live on the native objects.
  useEffect(() => {
    matte.enabled = legMatte;
    light.enabled = sceneLight;
  }, [matte, light, legMatte, sceneLight]);
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
        // First, so the background segmentation overlaps with the pose model.
        const matteMs = matte.update(frame);
        light.update(frame);
        const result = detector.detect(frame);
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
    [detector, matte, light, onSample],
  );

  // The pose model shrinks the whole frame to 192×256, so a bigger frame only costs preprocessing time.
  const frameOutput = useFrameOutput({
    targetResolution: CommonResolutions.VGA_16_9,
    enablePreviewSizedOutputBuffers: true,
    pixelFormat: 'rgb',
    enablePhysicalBufferRotation: true,
    enableCameraMatrixDelivery: true,
    onFrame,
  });

  return { frameOutput, sample, feet, status, fps };
}
