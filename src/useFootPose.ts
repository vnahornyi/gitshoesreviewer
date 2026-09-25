import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FOOT_NET_JOINTS } from './native/joints';
import {
  createFootPoseDetector,
  createPersonMatte,
  createSceneLight,
} from './native/hybrids';
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
} from './track/footTracker';

export type FootPoseSample = {
  points: number[];
  refined: number[];
  // Where each foot was looked for and how bright that crop was: left then right, [x, y, side, brightness].
  crops: number[];
  refineMs: number;
  preprocessMs: number;
  inferenceMs: number;
  // The last whole-frame search, which does not happen on every frame any more.
  searchMs: number;
  matteMs?: number;
  maskPreviewMs: number;
  maskPreviewStatus: string;
  sampleVersion: number;
  frameWidth: number;
  frameHeight: number;
  isMirrored: boolean;
  cameraMatrix?: number[];
};

const FPS_WINDOW = 30;
const LOG_EVERY_MS = 1000;

// One line a second while we work out why FootNet sees nothing on the phone and everything on the Mac: where each foot
// was looked for, how bright that crop was, and what the model made of it.
function logSample(sample: FootPoseSample, fps: number) {
  const foot = (index: number) => {
    const [x, y, side, brightness] = sample.crops.slice(
      index * 4,
      index * 4 + 4,
    );
    if (side === 0) {
      return 'none';
    }
    const scores = Array.from({ length: FOOT_NET_JOINTS.length }, (_, joint) =>
      sample.refined[(index * FOOT_NET_JOINTS.length + joint) * 3 + 2].toFixed(
        2,
      ),
    );
    return (
      `at ${x.toFixed(2)},${y.toFixed(2)} side ${side.toFixed(2)} ` +
      `bright ${brightness.toFixed(2)} scores ${scores.join(' ')}`
    );
  };
  console.log(
    `[foot] ${fps.toFixed(1)} fps, frame ${sample.frameWidth}x${
      sample.frameHeight
    }, ` +
      `feet ${sample.refineMs.toFixed(0)} ms, search ${
        sample.inferenceMs > 0
          ? `${(sample.preprocessMs + sample.inferenceMs).toFixed(0)} ms`
          : 'skipped'
      }` +
      `\n  left:  ${foot(0)}\n  right: ${foot(1)}`,
  );
}

// What ShoeView needs from each frame besides the feet: the person matte for the leg, the average tone for the lights.
export type FrameExtras = {
  legMatte: boolean;
  sceneLight: boolean;
  maskPreview: boolean;
  maskThreshold: number;
};

// The body model. RTMW x-l sees more joints but is four times slower, and the foot joints are the same.
const MODEL = 'rtmpose-m';

export function useFootPose({
  legMatte,
  sceneLight,
  maskPreview,
  maskThreshold,
}: FrameExtras) {
  const detector = useMemo(createFootPoseDetector, []);
  const matte = useMemo(createPersonMatte, []);
  const light = useMemo(createSceneLight, []);
  const [sample, setSample] = useState<FootPoseSample | null>(null);
  const [status, setStatus] = useState(detector.status);
  const [fps, setFps] = useState(0);
  const [feet, setFeet] = useState<TrackedFoot[]>([]);
  const arrivals = useRef<number[]>([]);
  const searchMs = useRef(0);
  const loggedAt = useRef(0);
  const sampleVersion = useRef(0);

  // The frame processor keeps the closure it started with, so the flags live on the native objects.
  useEffect(() => {
    matte.enabled = legMatte;
    light.enabled = sceneLight;
    detector.refine = true;
    detector.maskPreview = maskPreview;
    detector.maskThreshold = maskThreshold;
  }, [
    detector,
    matte,
    light,
    legMatte,
    sceneLight,
    maskPreview,
    maskThreshold,
  ]);
  const tracks = useRef<FootTrack[]>([]);
  const lastId = useRef(0);

  useEffect(() => {
    detector.load(MODEL);
    arrivals.current = [];
    tracks.current = [];
    setSample(null);
    setFeet([]);
    setStatus(detector.status);
  }, [detector]);

  const onSample = useCallback((next: FootPoseSample) => {
    const now = Date.now();
    const recent = [...arrivals.current, now].slice(-FPS_WINDOW);
    arrivals.current = recent;
    const rate =
      recent.length > 1 ? ((recent.length - 1) * 1000) / (now - recent[0]) : 0;
    if (recent.length > 1) {
      setFps(rate);
    }
    if (now - loggedAt.current >= LOG_EVERY_MS) {
      loggedAt.current = now;
      logSample(next, rate);
    }
    tracks.current = updateTracks(
      tracks.current,
      next.points,
      next.refined,
      now,
      () => ++lastId.current,
    );
    setFeet(trackedFeet(tracks.current, now));
    setStatus('ready');
    if (next.inferenceMs > 0) {
      searchMs.current = next.preprocessMs + next.inferenceMs;
    }
    setSample({
      ...next,
      searchMs: searchMs.current,
      sampleVersion: ++sampleVersion.current,
    });
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
          refined: result.refined,
          crops: result.crops,
          refineMs: result.refineMs,
          maskPreviewMs: result.maskPreviewMs,
          maskPreviewStatus: result.maskPreviewStatus,
          preprocessMs: result.preprocessMs,
          inferenceMs: result.inferenceMs,
          searchMs: 0,
          sampleVersion: 0,
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
