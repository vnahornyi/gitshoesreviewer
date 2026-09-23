# Working in this repository

For any agent or developer. Runtime-specific notes are in the adapters listed at the bottom; this
file is the shared part.

## Read before you touch something

| Area | Read |
|---|---|
| Anything at all | [`docs/state.md`](docs/state.md) — what works and what does not, today |
| Placing or scaling a shoe, assets | skill [`shoe-try-on`](skills/shoe-try-on/SKILL.md) |
| The foot model, its data, its export | skill [`footnet`](skills/footnet/SKILL.md) |
| `modules/` — the Swift side | skill [`nitro-native-modules`](skills/nitro-native-modules/SKILL.md) |
| Anything about on-device behaviour | skill [`device-diagnostics`](skills/device-diagnostics/SKILL.md) |
| The pipeline as a whole | [`docs/architecture.md`](docs/architecture.md) |
| "Why is it like this?" | [`docs/decisions.md`](docs/decisions.md) |
| Any number | [`docs/measurements.md`](docs/measurements.md) |
| Something broke | [`docs/troubleshooting.md`](docs/troubleshooting.md) |

Each directory with real complexity also has its own `README.md`, and that README is the contract
for that layer. Update it in the same change that changes the contract.

## Git

**Never push.** Stage exact files with `git add -- <files>`, show the staged diff, propose a commit
message, and stop. The developer approves that staged set; a yes covers one staged set, not the
next. No amend, rebase, `reset --hard`, force push, or `--no-verify`. On rejection,
`git restore --staged --`, never discard the working tree.

## Before staging anything

```bash
npx tsc --noEmit
npm run lint
npm test
```

All three green. A simulator build compiles the native side; only a real device tells you whether
it works.

## Conventions

- **Language**: chat with the developer in Ukrainian; code, comments, commit messages and
  documentation in English.
- **Comments explain why**, and are written for someone who will read the code in six months. Match
  the density and idiom of the file you are in.
- **No barrel files** that only re-export. Import the file that declares the symbol.
- **Prototype, no ceremony**: no estimates, no epics, no approval gates. Just the work — the git
  rules above still apply.
- **Models are not in git.** A fresh clone needs them copied in; see the root README.
- `.claude/plans/` is git-ignored. Plan files are temporary and are deleted when their task is done.

## Verify the ask

There is no ticket and no design behind most of this. Treat a request as an input to check, not a
spec to transcribe:

- Say plainly when something is unbacked or contradicts what is written down, and get it confirmed.
- Every number traces to a document, a design, or a measurement. A number that traces only to
  existing code is an implementation artifact — label it and question it.
- If a better option exists, propose it with the trade-off in a line or two and mark it as needing
  agreement. Do not silently ship either the weaker option or your own.

## Report what happened

If tests fail, say so and show the output. If a step was skipped, say which. If a measurement
contradicts what you expected, lead with that. This project has lost a full day to a warning being
called noise twice without being checked — that is the failure mode to guard against.

## Runtime adapters

| Runtime | File |
|---|---|
| Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Codex and others | this file |

Skills in [`skills/`](skills/README.md) are agent-neutral by design: they say what and when, never
which tool to call.
