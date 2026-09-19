# react-native-foot-pose

A VisionCamera v5 frame processor plugin (a Nitro HybridObject). It finds foot keypoints with RTMPose-m (Halpe26) or RTMW whole-body x-l, running on ONNX Runtime with the Core ML execution provider. For why this model and this runtime, see `research/foot-tracking/README.md` and `tools/model-convert/README.md`.

## Model

The models (28 MB and 114 MB) are not in git. Build them with `tools/model-convert`, then copy them here before `pod install`:

```bash
cp tools/model-convert/work/rtmpose-m-fp16.onnx tools/model-convert/work/rtmw-x-l-fp16.onnx modules/foot-pose/model/
cd ios && bundle exec pod install
```

The podspec ships `model/*.onnx` in the app bundle. On the first load of each model, Core ML compiles it into `Caches/foot-pose-coreml/<model>`, and `status` stays `loading …` until that finishes.

## API

- `createFootPoseDetector()` creates the detector, and `load('rtmpose-m' | 'rtmw-x-l')` builds that model's session on a background queue. `detect` throws until `status` is `ready`.
- `detect(frame)` must receive an upright BGRA frame, so use `useFrameOutput({ pixelFormat: 'rgb', enablePhysicalBufferRotation: true })`. It returns `points`: 8 joints × `[x, y, score]`, with x and y normalized to the frame. The joint order is `FOOT_JOINTS`: ankles, then left big toe, small toe and heel, then the same for the right foot. It also returns the preprocessing and inference time in ms.
- Point names follow how the foot looks in the image. In a mirror they are the other way round, and `src/foot-debug/footAxes.ts` swaps them.

After changing `src/FootPoseDetector.nitro.ts`, run `npm run codegen` in this directory.
