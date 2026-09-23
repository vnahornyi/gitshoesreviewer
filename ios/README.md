# The native layer

Six Swift files behind five Nitro HybridObjects, all iOS 17+. They used to be two packages
(`react-native-foot-pose` and `react-native-shoe-stage`); they are one pod now, `ShoeTryOn`, because
npm installs a GitHub dependency from the repository root and a consumer should run `pod install`
once, not three times.

After changing anything in [`../src/native/specs/`](../src/native/specs), run `npm run codegen` **at the
repository root**, then `pod install` in the app. Forgetting the codegen is the most common way to
lose an hour here: the Swift side keeps compiling against the old generated spec.

| File | Hybrid object |
|---|---|
| `HybridFootPoseDetector.swift` | `FootPoseDetector` — RTMPose/RTMW through ONNX Runtime |
| `FootNetRunner.swift` | FootNet through Core ML, used by the detector when `refine` is on |
| `HybridShoeView.swift` | `ShoeView` — the RealityKit stage |
| `HybridPersonMatte.swift` | `PersonMatte` — Apple Vision person segmentation |
| `HybridSceneLight.swift` | `SceneLight` — frame exposure and colour cast |
| `HybridDeviceGravity.swift` | `DeviceGravity` — CoreMotion gravity |

## Models

The models (28 MB, 114 MB and 9 MB) are **not in git**, and `model/` is empty in a fresh clone. The
detector stays in its error state until they are there, and the podspec's globs are written to
tolerate an empty directory so the app still builds and reports the problem itself. Build them, then
copy them in before `pod install`:

```bash
cp assets/onnx/rtmpose-m-fp16.onnx assets/onnx/rtmw-x-l-fp16.onnx model/
cd research/foot-3d && uv run python -m footnet.export   # writes ../../model/footnet.mlmodelc
cd example/ios && bundle exec pod install
```

The RTMPose models run through ONNX Runtime, and Core ML compiles each into
`Caches/foot-pose-coreml/<model>` on its first load, so `status` stays `loading …` until that
finishes. FootNet is a Core ML model already compiled by the export, and loads straight away.

FootNet is not an ONNX model because ONNX Runtime's Core ML execution provider cut its graph into 22
partitions and copied the tensors out and back at each one, which left the Neural Engine slower than
the CPU (44 ms against 20). Converted from PyTorch by `coremltools` and run through Core ML the
whole network stays on the Neural Engine at 1.9 ms. The same is worth doing for RTMPose.

## FootPoseDetector

- `createFootPoseDetector()` creates the detector, and `load('rtmpose-m' | 'rtmw-x-l')` builds that
  model's session on a background queue. `detect` throws until `status` is `ready`.
- `detect(frame)` must receive an upright BGRA frame, so use
  `useFrameOutput({ pixelFormat: 'rgb', enablePhysicalBufferRotation: true })`. It returns `points`:
  8 joints × `[x, y, score]`, x and y normalized to the frame, in `FOOT_JOINTS` order — ankles, then
  the left big toe, small toe and heel, then the same for the right foot. It also returns the
  preprocessing and inference time in ms.
- With `refine` on, the whole-frame search is not what runs every frame: FootNet follows each foot in
  its own crop, and that crop comes from its own points on the previous frame. RTMPose only runs to
  pick up a foot FootNet is not following, and at most every 250 ms, so a foot that is out of frame
  cannot make it search on every one. On the frames where it did not run, `points` is all zeros and
  `inferenceMs` is 0 — read the feet from `refined` there. A crop that drifts onto the foot the other
  crop already follows is let go, and RTMPose picks that foot up again.
- `refine` turns on FootNet (`research/foot-3d`, `footnet.mlmodelc`), our own foot model. For each
  foot RTMPose found, the crop around its points (1.45× the box, at least 24 px) is scaled into a
  256×256 pixel buffer and handed to Core ML, which scales and normalises the pixels itself; its 8
  keypoint heatmaps come back as float16 and are decoded here: the peak of each, refined by a
  softmax-weighted mean over a 5×5 window. `refined` then holds the left foot's 8 points and the
  right foot's, each `[x, y, score]` normalized to the frame, in `FOOT_NET_JOINTS` order, and
  `refineMs` how long both feet took. A foot that was not found, or not refined, is all zeros.
- A foot FootNet did not refine comes back as zeros, so whoever reads the points has to cope without
  them — [`../src/track/footTracker.ts`](../src/track/footTracker.ts) carries the last ones over, moved by how
  far RTMPose's big toe moved.
- A foot seen from above hides its own heel, so FootNet is sure of that point about half the time
  even when it is sure of everything else. Its toes stand on their own, and
  [`../src/track/footAxes.ts`](../src/track/footAxes.ts) takes the ankle from RTMPose when the heel is missing.
- Point names follow how the foot looks in the image, which in a mirror is the other way round from
  the person's own left and right.

## The stage

- `<ShoeView model shoes verticalFovDegrees />`: `model` names a pair of bundled files
  `shoes/<model>-left.usdz` and `shoes/<model>-right.usdz`. Each entry in `shoes` is
  `{ id, side, transform }`, where `transform` is a column-major 4×4 matrix from the normalized shoe
  (heel at the origin, sole on `y = 0`, toe along `+Z`, length 1) to RealityKit camera space (x
  right, y up, looking down −z). Size the view to the camera frame's content rect so its aspect
  matches the frame, and pass the frame's vertical field of view.
- `legMatte`: when `true` and a fresh person matte exists (at most 0.3 s old), the view masks itself
  out where the camera shows the person above each shoe's collar, in a strip around the shin. The
  real leg from the preview then comes out of the shoe. Otherwise a fixed shin cylinder occludes the
  back of the shoe.
- `createPersonMatte()`: set `enabled` from JS and call `update(frame)` on every frame in the frame
  processor. When enabled and idle, it copies the frame down to 384 rows and runs Apple Vision person
  segmentation (`.balanced`) on that copy on a background queue. Frames that arrive meanwhile are
  skipped. It keeps the result, at most 256 rows, for `ShoeView`, and returns how long the last
  segmentation took. The frame must be upright (`enablePhysicalBufferRotation`), as for the detector.
  **The flags live on the native object** because a VisionCamera frame processor keeps the closure it
  started with.
- `matchCamera`: when `true` and a fresh scene tone exists, the key and fill lights take the frame's
  exposure (its luma against 0.45) and part of its colour cast, smoothed over frames. A RealityKit
  post-process then adds a slight blur and moving grain on the shoe only, stronger in dim scenes. It
  is a small Metal kernel compiled at runtime, because Core Image's kernels crash on the A13 GPU
  (iPhone 11) in this callback. All of these constants are estimates to tune on device.
- `createSceneLight()`: `enabled`, and `update(frame)`, which averages a 32×32 grid over the lower
  half of a BGRA frame (floor and feet) for `matchCamera`.
- `createDeviceGravity().current()` returns the latest CoreMotion gravity in device axes.

Shoe files come from [`../tools/asset-pipeline`](../tools/asset-pipeline) (`normalize.mjs`, then
`glb2usdz.py`, plus `--mirror` for the other foot). Their licences are in
[`../shoes/LICENSES.md`](../shoes/LICENSES.md).
