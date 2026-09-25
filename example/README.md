# The example app

A plain React Native app with one screen, and the only consumer of `react-native-shoe-tryon`. It is
also how the library is developed: every number in [`docs/measurements.md`](../docs/measurements.md)
was read off this screen or off its console log.

It depends on the library as `file:..`, so `node_modules/react-native-shoe-tryon` is a symlink to
the repository root and an edit in `../src` reaches the app on the next Metro reload. Metro watches
the root for exactly that reason — see [`metro.config.js`](metro.config.js).

## Running it

The library's models have to exist first; the root [README](../README.md) has that step.

```bash
npm install
bundle install
cd ios && bundle exec pod install && cd ..
npm run ios -- --list-devices
npm run ios -- --device "Your iPhone's name"
```

`npm run typecheck` and `npm run lint` are this app's own; the library's tests live at the root.

| File | What it is |
|---|---|
| `App.tsx` | Providers and the status bar, nothing else |
| `src/FootDebugScreen.tsx` | The screen: camera, detector stats, crop bounds, overlay and mask-threshold controls |
| `src/FootOverlay.tsx` | Draws the points and the axes over the preview |

The Xcode project is still named `aishoesreviewer`, from before the split. Renaming it is churn with
no benefit while there is one app.

## The debug screen is a measuring instrument

It shows every number the pipeline produces, raw, before any filtering — that is its job, and the
reason it is not a product UI. What the stats line and the per-second console log mean is in the
`device-diagnostics` skill and in [`../src/README.md`](../src/README.md).

Toggles for foot side, mirroring and model choice existed here and were removed once each question
had been settled by measurement. A toggle is a decision not yet made; keeping it after the decision
spreads branches through the code.

The **Маска** layer is a temporary diagnostic for `footmask-v1`, not a production shoe mode. To
enable it, copy `assets/checkpoints/footmask-v1/footmask-v1.mlmodelc` into `model/`, run
`cd example/ios && bundle exec pod install`, then launch the example on a real iPhone as above. The
screen lets you compare thresholds 0.3, 0.5 and 0.7, shows mask inference time alongside camera
FPS, and outlines the left and right crops used by the model. The outlines make it possible to tell
whether a misplaced mask follows a misplaced crop or comes from the prediction inside a correctly
placed crop; the bounds now match the inference input for that result rather than the tracker crop
for the next frame. The native overlay applies mask updates without implicit layer animations. It
also shows the last whole-frame RTMPose search's per-foot landmark count and age,
separate from FootNet's `N/16` confidence count. The stats line reports crop brightness separately.
The mask only predicts within crops already acquired by RTMPose; a missing local model is reported
without preventing the regular detector from loading. Use mirror, top-down and side views to inspect
the overlay. Simulator output cannot establish camera performance or iPhone 11 latency.
