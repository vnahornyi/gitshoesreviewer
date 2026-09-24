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
npm start
npm run ios          # a real device: the simulator has no usable camera
```

`npm run typecheck` and `npm run lint` are this app's own; the library's tests live at the root.

| File | What it is |
|---|---|
| `App.tsx` | Providers and the status bar, nothing else |
| `src/FootDebugScreen.tsx` | The screen: camera, the stats line, one layer toggle |
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
