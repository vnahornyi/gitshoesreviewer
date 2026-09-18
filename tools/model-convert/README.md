# Foot model for the app

The spike chose RTMW whole-body x-l (`research/foot-tracking/README.md`). The app runs it as **ONNX through ONNX Runtime with the Core ML execution provider**, not as TFLite.

## Why not TFLite

`onnx2tf` 2.6.9 converts RTMW without error, with both the `flatbuffer_direct` and `tf_converter` backends, but the result is wrong: even the float32 `.tflite` puts foot points up to ~940 px away from the ONNX model on the same input. The layout rewrite (NCHW → NHWC) breaks the SimCC head. Debugging that converter is open-ended, while ONNX Runtime runs the original graph unchanged, and the same `.onnx` serves Android later.

## Steps

```bash
uv sync
cp ~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx work/rtmw-x-l.onnx
uv run python to_fp16.py work/rtmw-x-l.onnx work/rtmw-x-l-fp16.onnx
uv run python verify.py work/rtmw-x-l.onnx work/rtmw-x-l-fp16.onnx --provider coreml <images…>
```

The model file comes from `rtmlib`'s cache after the spike has run once. `to_fp16.py` keeps the float32 inputs and outputs, so the app feeds float32 and reads float32. `verify.py` compares the foot keypoints (COCO-WholeBody 17, 19, 20, 22) of the candidate against the fp32 model on CPU, and reports the latency of both.

Model contract (same as `rtmlib`):
- input `input`, `1×3×256×192`, RGB, `(pixel − [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]`, letterboxed into 192×256;
- outputs `simcc_x` `1×133×384` and `simcc_y` `1×133×512`; a point is `argmax / 2` in input pixels, and its score is the smaller of the two maxima.

## Result (2026-09-18, Mac, 5 frames, one per scenario)

| | size | latency | foot points vs fp32 CPU |
|---|---|---|---|
| fp32, CPU | 229 MB | 66–68 ms | reference |
| fp16, CPU | 114 MB | 68 ms | ≤ 6 px |
| **fp16, Core ML EP** | **114 MB** | **18 ms** | 3–9 px; one point at score 0.22–0.26 flipped between two peaks (306 px) |

A point near the score threshold can flip, because its SimCC distribution has two peaks. Live mode has to smooth over time anyway. The iPhone 11 Neural Engine number is still to be measured in the app.
