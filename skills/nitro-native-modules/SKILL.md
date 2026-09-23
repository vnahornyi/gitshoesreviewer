---
name: nitro-native-modules
description: Changing this project's Swift native layer — the Nitro codegen step, what must be rerun after which change, how models get into the app bundle, and the build traps that cost hours here. Use when editing anything under ios/ or src/native/specs/, when a new field has to cross from Swift to JS, when a model file changes, or when the app builds but the change is not there.
---

# The native layer

One [Nitro](https://nitro.margelo.com) module, `ShoeTryOn`, iOS-only Swift, five HybridObjects, all
consumed from the frame processor worklet:

| Object | What it does |
|---|---|
| `FootPoseDetector` | RTMPose (ONNX Runtime) finds feet; FootNet (Core ML) refines each in its own crop |
| `ShoeView` | RealityKit draws the shoes |
| `PersonMatte`, `SceneLight`, `DeviceGravity` | The matte, the frame's light, and gravity |

The specs are in `src/native/specs/`, the Swift in `ios/`, and [`ios/README.md`](../../ios/README.md)
documents every contract. Update it in the same change — that README is the contract, not
decoration.

They were two pods until the repository became an installable library; one pod means a consumer runs
`pod install` once. If you are reading an old commit, `modules/foot-pose` and `modules/shoe-stage`
are where these files used to live.

## The one step that is always forgotten

After editing a `*.nitro.ts` spec, **run codegen at the repository root**:

```bash
npm run codegen
```

Without it the generated Swift protocol keeps the old shape. The symptom is not a clean error: the
build may succeed and the new field silently never arrive in JS. When a field you added is
undefined at runtime and the Swift side clearly sets it, this is the first thing to check.

The generated code under `nitrogen/generated/` is committed. Stage it with the change that caused
it; never edit it by hand.

## What to rerun after what

| Change | Then |
|---|---|
| `*.nitro.ts` spec | `npm run codegen` at the root, then rebuild |
| Swift implementation only | rebuild |
| podspec, new Swift file, new model resource | `cd example/ios && bundle exec pod install`, then rebuild |
| model file replaced in `modules/*/model/` | `pod install` (resources are copied at install time), then rebuild |

## Models are not in git

`model/` is git-ignored except for what `.gitignore` there allows. The RTMPose
ONNX files come from `tools/model-convert`, FootNet's `.mlmodelc` from
`research/foot-3d/footnet.export`. A fresh clone has no models and the detector's `status` stays an
error until they are copied in — see `ios/README.md`.

The podspec ships `model/*.onnx` and `model/*.mlmodelc`. A model added with a new extension needs
the podspec's `s.resources` updated, or it is simply absent at runtime.

## Build traps that cost real time here

- **`pod install` hides its own error** unless the locale is set. If it fails with something
  unhelpful, rerun as `LANG=en_US.UTF-8 bundle exec pod install` to see the actual message.
- **Prebuilt React Native flavor markers.** A Release build can leave release React/Hermes behind
  Debug markers, and then `Sealable`/`CDPDebugAPI` go missing at compile time. Write Release to the
  markers. (This is in project memory as `rn-prebuilt-flavor-markers`.)
- **`timeout` does not exist on macOS.** Do not reach for it in build scripts.
- **A frame processor keeps the closure it started with.** Flags that change at runtime
  (`matte.enabled`, `light.enabled`, `detector.refine`) must live on the native object and be set
  from an effect, never captured in the worklet. This is why `useFootPose` sets them the way it does.

## Memory safety, because this is where it bites

Both crashes this project has had were memory corruption in this layer, and both surfaced far from
their cause. When touching buffer code:

- Anything that hands out a pointer without copying (`tensorData()`,
  `dataWithBytesNoCopy:freeWhenDone:NO`) is only valid while the owning object is alive. ARC will
  release it at the end of the statement if nothing holds it. Bind the output buffer up front.
- vImage writes exactly where you point it. An inset computed from a rounded rect can go negative
  by a fraction of a pixel, and vImage will then write **before** the buffer and corrupt the heap.
  Clamp every inset and extent to the destination.
- An `EXC_BAD_ACCESS` line number is where the corruption was noticed, not where it happened. Say
  this out loud rather than "fixing" the line that crashed.
