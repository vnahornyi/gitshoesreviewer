---
name: apple-neural-engine
description: Getting a model to actually run on the Apple Neural Engine, and finding out when it does not — graph partitioning, the ops and shapes that force a CPU or GPU fallback, fp16 and static shapes, Core ML and ONNX Runtime CoreML execution provider settings, and how to verify where a model ran. Use when an iOS model is slower than its size suggests, when a log mentions nodes not assigned to the preferred execution provider, when exporting or converting a model for iOS, or when choosing between Core ML directly and ONNX Runtime.
---

> Copied from `ai-god/skills/apple-neural-engine`, which is the canonical copy: this skill is general, not about
> this project. Edit it there and copy it back, or the two drift apart.

# Making a model run on the Neural Engine

The Neural Engine is roughly two orders of magnitude faster than the CPU at convolutions. A model
that gets a tenth of the throughput of a similar-sized model on the same device is almost never
"a slower architecture" — it is running somewhere else, or being cut into pieces.

## First: find out where it ran

Never optimise before answering this.

- **Effective throughput.** Divide the model's multiply-accumulates by its measured time. Compare
  with another model on the same device. A gap of 3× or more on similar GMac means a different
  compute unit.
- **Compare compute units.** Run with `.all`, `.cpuAndGPU` and `.cpuOnly` (or the framework's
  equivalent). If `.all` is not clearly faster, the ANE contributed little or nothing.
- **Xcode Core ML performance report** shows per-operation compute-unit assignment for a `.mlmodel`
  / `.mlpackage`; `MLComputePlan` (iOS 17+) gives the same in code.
- A thread named `H11ANEServicesThread` in the process means the ANE is in use for at least part
  of the model.

## Partitioning is the usual culprit

Both Core ML and ONNX Runtime's CoreML execution provider split a graph at any operation they
cannot place. Each boundary is a copy out and back. A handful of unplaceable ops scattered through
a network turns into a dozen partitions and destroys the win — a documented case is a model with
14 reflect-padding nodes becoming 14 CoreML partitions with CPU round-trips between them.

So: a warning that "some nodes were not assigned to the preferred execution provider" is only
harmless if you check **how many partitions** resulted and **where** they sit. Shape-handling ops
pushed to the CPU at the edges of a graph are fine; a CPU op in the middle of the convolution
stack is not.

## What tends to fall back

Incomplete and version-dependent, but the recurring offenders:

- Custom layers; recurrent layers (LSTM/GRU); gather; dilated convolutions.
- Broadcast and "ND" ops — including a multiply of `C×H×W` by `C×1×1`. Anything that is not a
  plain 4D `(batch, channels, height, width)` tensor tends to be expressed with these.
- Pooling with a kernel larger than 13 or a stride above 2.
- Upsampling with a scale factor above 2 (so build 2× stages, not one 4× stage).
- Bilinear resize converts, but with subtly different results than PyTorch — check numerics after
  conversion, and prefer nearest-neighbour plus a convolution where accuracy permits.

## Design rules that keep a graph on the ANE

- **fp16 weights and activations.** The hardware is fp16; fp32 forces conversions.
- **Static shapes everywhere.** Dynamic dimensions disable the aggressive optimisations and, in
  ONNX Runtime, can push a node out of the CoreML partition entirely. Pad to a fixed size instead.
- **Rank 4, channels-first `(B, C, H, W)`.** Reshapes into rank 3 or 5 in the middle of a network
  are a common, avoidable fallback.
- **Sizes that are multiples of 16**, batch a power of two (1 is fine).
- **No control flow and no data-dependent shapes** in the exported graph. Argmax/top-k style
  decoding belongs in application code, not in the model — it splits the graph and blocks fp16.

## ONNX Runtime CoreML execution provider

Options worth setting explicitly:

- `ModelFormat: MLProgram` — required for fp16 activations and the modern converter (iOS 15+).
- `MLComputeUnits`: `ALL` by default; `CPUAndNeuralEngine` removes GPU scheduling noise and makes
  a fallback obvious in measurements rather than hidden behind a GPU path.
- `RequireStaticInputShapes: 1` — refuses to place dynamic-shape nodes rather than placing them
  badly. Combined with a fixed input size this is usually a speed-up.
- `ModelCacheDirectory` — without it the model is recompiled on every launch.

Bind the output buffers up front (`runWithInputs:outputs:`) rather than reading the session's own
tensors back: a tensor handed back by a run owns memory only as long as that run's output value
lives.

Core ML used directly (not through ONNX Runtime) additionally lets the model take a
`CVPixelBuffer` image input and apply scale and bias itself, which removes the application's own
preprocessing pass. When a model runs on every frame, that is a reason to prefer it.

## Sources

- [Does my Core ML model run on Apple's Neural Engine?](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md)
- [ANE unsupported layers](https://github.com/hollance/neural-engine/blob/master/docs/unsupported-layers.md)
- [ONNX Runtime — CoreML Execution Provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [CoreML EP partition round-trips from unsupported Pad nodes](https://github.com/microsoft/onnxruntime/issues/28022)
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Core ML Tools: Image Input and Output](https://apple.github.io/coremltools/docs-guides/source/image-inputs.html)
