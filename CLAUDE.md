# Claude Code adapter

The shared rules for this repository are in [`AGENTS.md`](AGENTS.md) — read it. This file only adds
what is specific to running as Claude Code here.

## Skills

The skills live in [`skills/`](skills/README.md) as plain, agent-neutral files. Claude Code only
discovers skills in `.claude/skills/`, and there is no setting that points it elsewhere, so run
this once after cloning:

```bash
npm run skills
```

It symlinks each skill into `.claude/skills/`, which is git-ignored — the real files stay in
`skills/` and the per-agent plumbing stays local. They are then invokable by name: `shoe-try-on`,
`footnet`, `nitro-native-modules`, `device-diagnostics`, `realtime-vision-pipeline`,
`apple-neural-engine`.

Adding a skill: create `skills/<name>/SKILL.md` and rerun `npm run skills`. Never put the real file
under `.claude/` — it would be invisible to every other agent and excluded from git.

## Global rules still apply

The developer's `~/.claude/CLAUDE.md` governs git, workflow and memory, and outranks anything here
or in a plan file. In particular: never push, stage and stop for approval, and subagents never
stage or commit.

## This project's memory

Lives in `~/.claude/projects/-Users-vnahornyi-Developer-aishoesreviewer/memory/` and is **local by
the developer's decision** — it is not synced to the `ai-god` repository, unlike other projects.

## Long-running work

Training and rendering here run for hours. Start them in the background, and check them by reading
the output file rather than re-running anything:

- render → `research/foot-3d/results/render.out`, frames in `research/foot-3d/data/synth/renders/`
- training → `research/foot-3d/results/footnet/train-v3.out` and `results/footnet/log.csv`

`nohup` returns immediately; the wrapper "completing" does not mean the job finished. Check the
process before reporting anything about it.

## Diagnostics loop

The established loop for on-device questions: put the numbers in `console.log`, the developer runs
the app and pastes the Metro output. Prefer that over asking for screenshots when the question is
numeric. See the `device-diagnostics` skill.

## Slash-command note

Terminal-only dialogs (`/permissions`, `/hooks`, …) are unavailable in the desktop app session the
developer usually runs here; point at the app's own UI instead.
