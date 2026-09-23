# Where the prototype stands

Last updated 2026-09-23. This file is the honest status; the READMEs describe how things work, this
one says how well.

## Works

- **Live foot tracking at 28–30 fps** on an iPhone 11, with FootNet following each foot in its own
  crop and RTMPose searching only for a foot that was lost.
- **FootNet on the Neural Engine** through Core ML, 15 ms per foot, exported with 0.001 px median
  parity against PyTorch.
- **3D shoe placement** from the floor plane using device gravity, which holds in the mirror view
  where depth-from-size does not.
- **Person matte** cutting the real leg out of the shoe collar, and scene-light matching.
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
  attempts at guessing it geometrically were measured and rejected.
- **A landscape frame** (1472×828) draws points far from the feet. Not diagnosed.
- **The models do not travel with the package.** They are 150 MB and not in git, so
  `npm install github:…` gives a library whose detector cannot load. A consumer has to copy three
  files into `node_modules/react-native-shoe-tryon/model/` by hand. git-lfs, a release asset, or a
  download step would each fix it; none has been chosen.
- **No Android**, no product UI, one debug screen.

## In progress

- FootNet fine-tune on the full 20 000-frame render set (started 2026-09-23 17:33, 8 epochs,
  ~4 hours). Expected to help p90 and generalisation; **not** expected to fix the bimodality, which
  is a real-vs-synthetic gap rather than a volume problem.

## Next, in the order the evidence suggests

1. **Footwear in the renderer.** Needs shoe meshes fitted to the FIND template. The repo has one
   placeholder shoe; training on it would teach the model that one placeholder, not footwear.
2. **RTMPose through Core ML**, the way FootNet was. coremltools cannot read its ONNX source, so
   this needs a different conversion route.
3. **FootNet decoder at 64×64** instead of 256×256 — less urgent since the Core ML switch.
4. **The floor-plane template fit** (25 mm, 4.5° direction) to replace the toe/ankle direction
   logic and make the hidden heel a non-issue.
5. **Ship the models with the package** — see above; it is what stands between "installable" and
   "usable by someone else".
6. Android.

## Things that are assumptions, not facts

Listed with the rest in [measurements.md](measurements.md): camera height, shoe length, the
anatomical landmark heights on the shoe, and the scene-light constants. Anything derived from them
inherits their uncertainty, and should say so.
