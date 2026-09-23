# react-native-foot-pose

A VisionCamera v5 frame processor plugin (a Nitro HybridObject). It finds foot keypoints with RTMPose-m (Halpe26) or RTMW whole-body x-l, running on ONNX Runtime with the Core ML execution provider. For why this model and this runtime, see `research/foot-tracking/README.md` and `tools/model-convert/README.md`.

## Models

The models (28 MB, 114 MB and 9 MB) are not in git. Build them, then copy them here before `pod install`:

```bash
cp tools/model-convert/work/rtmpose-m-fp16.onnx tools/model-convert/work/rtmw-x-l-fp16.onnx modules/foot-pose/model/
cd research/foot-3d && uv run python -m footnet.export   # writes model/footnet-fp16.onnx
cd ios && bundle exec pod install
```

The podspec ships `model/*.onnx` in the app bundle. On the first load of each model, Core ML compiles it into `Caches/foot-pose-coreml/<model>`, and `status` stays `loading …` until that finishes.

## API

- `createFootPoseDetector()` creates the detector, and `load('rtmpose-m' | 'rtmw-x-l')` builds that model's session on a background queue. `detect` throws until `status` is `ready`.
- `detect(frame)` must receive an upright BGRA frame, so use `useFrameOutput({ pixelFormat: 'rgb', enablePhysicalBufferRotation: true })`. It returns `points`: 8 joints × `[x, y, score]`, with x and y normalized to the frame. The joint order is `FOOT_JOINTS`: ankles, then left big toe, small toe and heel, then the same for the right foot. It also returns the preprocessing and inference time in ms.
- `refine` turns on FootNet (`research/foot-3d`, `footnet-fp16.onnx`), our own foot model. For each foot RTMPose found, the crop around its points (1.45× the box, at least 24 px) is resized to 256×256, normalised with ImageNet statistics on RGB 0…1, and its 8 keypoint heatmaps are decoded here: the peak of each, refined by a softmax-weighted mean over a 5×5 window. `refined` then holds the left foot's 8 points and the right foot's, each `[x, y, score]` normalized to the frame, in `FOOT_NET_JOINTS` order, and `refineMs` how long both feet took. A foot that was not found, or not refined, is all zeros.

  FootNet costs several times what RTMPose does, so with `refine` on it still only runs five times a second and the frames in between come back with `refined` all zeros; `refineMs` keeps reporting the last run. Whoever reads the points is expected to carry the last ones over those frames — `src/foot-debug/footTracker.ts` moves them by how far RTMPose's big toe moved.
- Point names follow how the foot looks in the image. In a mirror they are the other way round, and `src/foot-debug/footAxes.ts` swaps them.

After changing `src/FootPoseDetector.nitro.ts`, run `npm run codegen` in this directory.
