# Where the prototype stands

Last updated 2026-09-24. This file is the honest status; the READMEs describe how things work, this
one says how well.

## Works

- **Live foot tracking at 28–30 fps** on an iPhone 11, with FootNet following each foot in its own
  crop and RTMPose searching only for a foot that was lost.
- **FootNet on the Neural Engine** through Core ML, 15 ms per foot on the previously measured
  device build. The current raw-logit export matches the native decoder on float16-rounded inputs;
  the updated native bundle has not yet been checked on a phone.
- **3D shoe placement** from the floor plane using device gravity, which holds in the mirror view
  where depth-from-size does not.
- **Person matte** cutting the real leg out of the shoe collar, and scene-light matching.
- **The shoe is sized from the image, not assumed.** Measured per foot, smoothed, and rejected when
  it implies a foot no human has — which is also what stops a shoe being drawn from garbage points.
- **Offline asset pipeline** from a product photo to a normalized USDZ.
- **Synthetic data**: a Blender renderer producing mirror / top / third-person views with socks and
  trousers, 20 000 frames rendered.
- **The repository is an installable library.** One npm package, one pod, one codegen; the app it
  used to be is now `example/`, and it consumes the library from the outside like anyone else.

## Does not work

- **Shoes.** Neither training dataset contains footwear, so a shod foot is where the model fails.
  This is the largest known gap and it is a data problem, not a tuning problem.
- **The side view** (`third`) for the same reason.
- **Confidence is bimodal on real frames** — 0.6–0.9 or 0.0–0.15. The synthetic-to-real gap.
- **The heel passes threshold about half the time** in otherwise good readings, because looking
  down at your own feet physically hides it. Mitigated by falling back to RTMPose's ankle; two
  attempts at guessing it geometrically were measured and rejected. The developer confirms the live
  model still flickers instead of recognizing the foot when the heel is not visible; the decoder
  parity and coordinate probes did not address this visibility/tracking failure.
- **Looking down at your own feet.** Measured 2026-09-23: the mirror view places both shoes
  correctly, the top-down view flickers between confident and blank on a static scene and draws
  nothing usable. Same model, same socks — this is the synthetic-to-real gap, not the geometry.
- **A landscape frame** (1472×828) draws points far from the feet. The Swift crop-to-frame and
  normalized-coordinate helpers, followed by the JS `frameToView` mapping, pass isolated round-trip
  probes. The live frame/preview composition still needs an on-device reproduction.
- **The models do not travel with the package.** They are 150 MB and not in git, so
  `npm install github:…` gives a library whose detector cannot load. A consumer has to copy three
  files into `node_modules/react-native-shoe-tryon/model/` by hand. git-lfs, a release asset, or a
  download step would each fix it; none has been chosen.
- **No Android**, no product UI, one debug screen.

## Completed

- FootNet fine-tune on the full 20 000-frame render set (started 2026-09-23 17:33, 8 epochs,
  completed by 20:42 EEST). Held-out render p90 improved to 11.824 px. This is a synthetic
  validation result; it does not establish improvement on real frames or fix the known bimodality.
  See [measurements.md](measurements.md) and `research/foot-3d/results/footnet/log.csv`.
- Compared `v2-renders-7k.pt`, `best.pt`, and `last.pt` on the same 156 fully labelled feet after
  excluding clips flagged for child appearances. The new checkpoints slightly improve the oracle
  crop median, but not p90; the RTMPose seed-crop proxy also shows no p90 improvement. This is a Mac
  CPU measurement, not an on-device result.
- Measured and fixed the Swift decoder mismatch. The old sigmoid-output contract caused a 0.232 px
  p90 decoder delta on 156 labelled crops. Core ML now exports logits; Swift and PyTorch agree to
  0.000020 px p90 on the same float16-rounded heatmaps. Re-exported the original 7k checkpoint to
  `model/footnet.mlmodelc`; the real-frame evaluation did not justify switching to a 20k checkpoint.
  The export check measured 0.010 px median / 0.985 px max against full-precision PyTorch on 16
  SynFoot crops. iOS project integration remains unverified because CocoaPods is unavailable here.
- Coordinate probes pass for crop → frame pixels → normalized frame coordinates at 1472×828 and
  for mapping an off-centre point into a 393×852 portrait view. The original live landscape report
  remains unresolved.
- Ran one-factor real-image ablations on 156 oracle and 153 RTMPose seed crops. Downsampling gave a
  small but non-conclusive gain; matching saturation, blur, or denoise did not help. A partial-label
  fine-tune on 111 seed crops improved the held-out test median but worsened its p90 and confidence
  recall, so the pilot checkpoint was not promoted.

## Next, in the order the evidence suggests

1. **Collect more varied real labels**, including shoes and independent source clips. The 156-label
   set is too small to establish generalization: its partial-label fine-tune helped the test median
   but hurt p90 and confidence recall.
2. **Add footwear to the renderer** for the separate, still-uncovered shod-foot case. Existing
   render validation contains no shoes.
3. RTMPose through Core ML, the 64×64 FootNet decoder, the floor-plane template fit, package model
   delivery, and Android remain later work.

## Things that are assumptions, not facts

Listed with the rest in [measurements.md](measurements.md): camera height, shoe length, the
anatomical landmark heights on the shoe, and the scene-light constants. Anything derived from them
inherits their uncertainty, and should say so.
