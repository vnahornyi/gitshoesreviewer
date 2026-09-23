# When it breaks

Only failures that have actually happened in this project, with what they turned out to be. Each
one wasted time; none of them looked like its cause.

## The app runs but the feet are wrong

**`feet 0 ms` in the log** — FootNet did not run, because it had no crop. A tracking problem, not a
model problem. Look at `crops` in the log: `none` means no crop at all.

**Scores all near zero with a sensible crop** (`at`/`side`/`bright` look right) — this is the model
failing, and on a shod foot it is the known data gap: no training dataset has footwear.

**The crop shrinks onto part of the foot over a few frames** — the next crop is being built only
from confident points. Fixed by clamping size change to 20 % per frame; if it returns, check
`sizeStep`.

**Points drawn far from the feet on a landscape frame** (seen at 1472×828) — open, not diagnosed.
Log `frameWidth`/`frameHeight` and `isMirrored` before theorising.

Read the `device-diagnostics` skill before drawing conclusions from any of these.

## The app crashes with EXC_BAD_ACCESS

Both crashes here were memory corruption in the native buffer code, and **the line in the crash
report was not the cause** — it was where the damage was noticed.

| Symptom | Cause |
|---|---|
| `EXC_BAD_ACCESS (code=257, …)` after enabling FootNet | `tensorData()` returns a no-copy pointer; ARC released the owning output value right after the guard, leaving a dangling read. Bind the output buffer up front. |
| `EXC_BAD_ACCESS (code=1, …)` in the pixel loop | A crop inset went a fraction of a pixel negative, so vImage wrote *before* the destination buffer and corrupted the heap. Clamp every inset and extent. |

When you see one of these, do not patch the crashing line. Find who wrote outside a buffer.

## A change to a native module does not appear

**A new field is undefined in JS** although Swift clearly sets it — codegen was not rerun.
`cd modules/<module> && npm run codegen`, then rebuild. The build does **not** fail; the field just
never arrives.

**A model file was replaced but the app uses the old one** — resources are copied at pod install
time. `cd ios && bundle exec pod install`.

**A new model extension is ignored at runtime** — the podspec's `s.resources` lists extensions
explicitly.

**A runtime flag has no effect inside the frame processor** — a VisionCamera frame processor keeps
the closure it started with. Flags must live on the native object.

## The build fails

**`pod install` fails with an unhelpful message** — it hides the real error unless the locale is
set:

```bash
cd ios && LANG=en_US.UTF-8 bundle exec pod install
```

**`Sealable` or `CDPDebugAPI` missing** — prebuilt React Native flavor markers: a Release build left
release React/Hermes behind Debug markers. Write Release to the markers.

**`timeout: command not found`** — macOS has no `timeout`.

## Research and tooling

**`.venv/bin/pip` is absent** — these environments are managed by `uv`. Use `uv add` / `uv run`.

**A Python probe silently finds nothing** — check the glob root. `Path(".").parent` is not the
current directory's parent in the way it reads; it globbed nothing and reported success.

**Training or export cannot find a checkpoint** — `best.pt` is overwritten in place by every run.
Keep a named copy of what you started from before training.

**cv2 fails on a video that was there a moment ago** — it may have been deleted from Downloads
after frames were extracted. Check the disk before assuming a decoder problem.
