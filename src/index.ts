/**
 * The package's public surface.
 *
 * This is the one file in the repository that re-exports rather than declares: it is the contract a
 * consumer installs against, so what is named here is what we promise to keep working. Everything
 * else imports the file that declares the symbol. If something is missing from this list it is
 * internal on purpose — ask before widening it rather than deep-importing.
 *
 * `src/README.md` explains how the pieces fit together.
 */

// The camera-to-poses pipeline, as one hook. This is what an app normally uses.
export {
  useFootPose,
  type FootPoseSample,
  type FrameExtras,
} from './useFootPose';

// The rendered shoe, placed from a `FootPoseSample`.
export { ShoeLayer } from './ShoeLayer';

// The pieces, for an app that needs to do its own thing with them.
export {
  footAxes,
  frameToView,
  REFINED_MIN_SCORE,
  type FootPoints,
  type FootSide,
  type Point,
  type Size,
} from './track/footAxes';
export {
  trackedFeet,
  updateTracks,
  type FootTrack,
  type TrackedFoot,
} from './track/footTracker';
export {
  gravityInBackCamera,
  intrinsicsFor,
  shoeTransform,
  verticalFovDegrees,
  type Intrinsics,
} from './pose/shoePose';
export { startOneEuro, stepOneEuro, type OneEuroState } from './track/oneEuro';
export type { Vec3 } from './pose/vec3';

// The native layer, for an app that wants the detector or the view directly.
export {
  createDeviceGravity,
  createFootPoseDetector,
  createPersonMatte,
  createSceneLight,
  ShoeView,
  type DeviceGravity,
  type FootModel,
  type FootPoseDetector,
  type FootPoseResult,
  type PersonMatte,
  type SceneLight,
  type ShoePose,
  type ShoeSide,
  type ShoeViewProps,
} from './native/hybrids';
export {
  FOOT_JOINTS,
  FOOT_NET_JOINTS,
  type FootJoint,
  type FootNetJoint,
} from './native/joints';
