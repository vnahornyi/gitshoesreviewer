# 4. Making it run on the phone

A phone has a chip specifically for neural networks. Getting your model onto it is not automatic,
failing to is silent, and the difference here was **16×**. This chapter is how that works and how
to check.

## Three processors, one chip

| Unit | Good at | Ours |
|---|---|---|
| CPU | Anything, one thing at a time | Fallback. Correct, slow |
| GPU | Wide parallel arithmetic | Middle ground |
| **Neural Engine (ANE)** | Convolutions in fp16, and nothing else | What we want |

The ANE is roughly two orders of magnitude faster than the CPU at convolutions — but it is fixed
function. It runs a specific set of operations, at a specific precision, on specific tensor shapes.
Step outside and the operation falls back to the CPU or GPU.

## Why a fallback is worse than it sounds

If 90 % of a network runs on the ANE and 10 % does not, the cost is not 10 %. Each boundary means
the tensor is copied out of the ANE's format and back, and those copies are not free.

Our FootNet through ONNX Runtime's Core ML execution provider was cut into **22 partitions**. The
result: on the ANE, 44 ms; on the CPU alone, 20 ms. *Enabling the accelerator made it twice as
slow*, because the model spent its time being shuttled rather than computed.

Converted directly with coremltools and run through Core ML, the whole graph stays on the ANE:
**1.9 ms on the Mac, and 65 → 15 ms per foot on the iPhone 11.**

The warning that said so — `VerifyEachNodeIsAssignedToAnEp` — was dismissed as noise twice before
being read. That is the actual lesson: **a warning is not noise until you have checked.**

## What forces a fallback

- **Precision.** The ANE is fp16. A model in fp32 is at best converted; at worst partitioned.
- **Dynamic shapes.** The ANE compiles for fixed shapes. A graph with a free batch dimension or a
  variable input size cannot be planned. `to_fp16.py` in `tools/model-convert` fixes every shape to
  batch 1 for exactly this reason.
- **Unsupported operations.** Exotic activations, broadcasting patterns, `ND` ops, upsample with a
  scale above 2, pooling with a large kernel. The lists ship with ONNX Runtime
  (`coreml_supported_mlprogram_ops.md`) and are worth reading before designing a decoder.
- **Layout.** The ANE wants rank-4 NCHW. Reshapes that break rank-4 cut the graph.

Design for this from the start; it is far cheaper than retrofitting. A decoder built from
`Upsample(2) + Conv` runs; one built from a clever reshape does not.

## fp16 is not just smaller

Half precision has ~3 decimal digits and a smaller exponent range. For inference that is almost
always fine — but "almost" needs checking, so both conversion tools here **verify against fp32**:

- RTMPose fp16 vs fp32: foot points within a few px above score 0.3. Below that, a point whose
  SimCC distribution has two peaks can flip between them, moving hundreds of px. Both precisions do
  it; the score threshold drops those points anyway.
- FootNet Core ML vs PyTorch: median **0.001 px**, max 0.974 px.

That second number is the bar. Anything above ~1 px means the conversion changed the model, and the
right response is to find out why, not to ship it and hope.

> **A conversion that produces a wrong model silently is the normal failure mode.** `onnx2tf`
> converted RTMW without a single error and produced points up to **940 px** off, because the
> NCHW → NHWC rewrite broke the SimCC head. It ran. It was wrong. That is why every conversion in
> this repo has a numeric parity check attached.

## Two routes, and when each applies

**Core ML directly** (`coremltools`): trace the PyTorch module, convert, compile. You control the
graph, so you can fold preprocessing in and drop heads you do not need. This is what FootNet does —
and both of those choices are why it is fast:

```python
ct.convert(traced,
           inputs=[ct.ImageType(name="image", shape=(1,3,256,256),
                                scale=1/255, color_layout=ct.colorlayout.RGB)],
           outputs=[ct.TensorType(name="heatmaps", dtype=np.float16)],
           compute_precision=ct.precision.FLOAT16,
           minimum_deployment_target=ct.target.iOS17)
```

`ImageType` means Core ML takes a `CVPixelBuffer` and does the scaling itself, on the accelerator —
no CPU pass over the pixels. ImageNet mean/std are folded into the graph's first operations. The
app hands over a pixel buffer and gets heatmaps; nothing in between touches the data.

**ONNX Runtime with the CoreML EP**: for models you cannot re-export from source. It partitions,
so check how many partitions you got. RTMPose still goes this way because coremltools cannot read
its ONNX, and converting it properly is open work.

## How to check where it actually ran

Do not assume. In order of effort:

1. **Compare compute units.** Run the model with `.cpuOnly`, `.cpuAndGPU`, `.all` and compare
   latency. If `.all` is not much faster, it is not on the ANE. Our numbers: 7.1 / 4.6 / 1.9 ms.
2. **Read the partition count** in the ONNX Runtime log. More than one or two is a red flag.
3. **Instruments → Core ML template** shows the actual unit per layer.

The first one is thirty seconds of work and would have saved this project a day.

## The app side

Some of this is Core ML specific and the rest is general buffer discipline, learned the hard way —
both crashes this project has had were here.

- **Reuse buffers.** `CVPixelBufferPool` for crops; a pre-allocated float array for heatmaps. At 30
  fps, allocation per frame is a real cost and a real source of jitter.
- **Convert fp16 → fp32 with vImage**, not a loop. `vImageConvert_Planar16FtoPlanarF` does it in
  one pass.
- **Anything handed out without copying is only valid while its owner lives.** `tensorData()`
  returns a no-copy pointer; ARC released the owning value at the end of the statement, and the
  read that followed was into freed memory. Bind output buffers up front.
- **vImage writes exactly where you point it.** An inset computed from a rounded rectangle went a
  fraction of a pixel negative, so it wrote *before* the buffer and corrupted the heap. Clamp every
  inset and extent to the destination.
- **An `EXC_BAD_ACCESS` line is where the damage was noticed**, not where it was done. Say so out
  loud, then go looking for who wrote outside a buffer.

## Try it

1. In `FootNetRunner.load`, change `computeUnits` to `.cpuOnly` and measure. You now know, for your
   own device, what the ANE is worth.
2. Open `modules/foot-pose/model/footnet.mlmodelc/model.mil` — the compiled model is readable. Find
   where the normalisation was folded in.
3. Read the export's parity check. Ask what you would do if it printed 4 px instead of 0.001.

## Further reading

- [coremltools guide](https://apple.github.io/coremltools/docs-guides/) — conversion, precision,
  `ImageType`, and the deployment targets.
- [Apple: Core ML performance](https://developer.apple.com/documentation/coreml/mlcomputeunits) —
  compute units and what they mean.
- [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
  — its options, and the supported-op lists that ship with the package.
- The `apple-neural-engine` skill in this repository — the same material as a checklist.
