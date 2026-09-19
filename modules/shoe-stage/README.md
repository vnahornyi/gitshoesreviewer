# react-native-shoe-stage

It draws shoe models at given poses over the camera preview with RealityKit (`ARView` in non-AR mode, clear background). It also exposes the device gravity from CoreMotion, which the app uses to put a standing foot flat on the floor.

- `<ShoeView model shoes verticalFovDegrees />`: `model` names a pair of bundled files `shoes/<model>-left.usdz` and `shoes/<model>-right.usdz`. Each entry in `shoes` is `{ id, side, transform }`, where `transform` is a column-major 4×4 matrix from the normalized shoe (heel at the origin, sole on `y = 0`, toe along `+Z`, length 1) to RealityKit camera space (x right, y up, looking down −z). Size the view to the camera frame's content rect so its aspect matches the frame, and pass the frame's vertical field of view.
- `createDeviceGravity().current()` returns the latest CoreMotion gravity in device axes.

Shoe files come from `tools/asset-pipeline` (`normalize.mjs`, then `glb2usdz.py`, plus `--mirror` for the other foot). Their licenses are in `shoes/LICENSES.md`.
