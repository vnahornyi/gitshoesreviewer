# Skills

Knowledge about this project that an agent — or a new developer — needs before touching a given
area. One directory per skill, each holding a `SKILL.md` with YAML frontmatter:

```yaml
---
name: <kebab-case, matching the directory>
description: <what it covers, and when to reach for it — this is what gets matched against a task>
---
```

## The skills here

| Skill | Reach for it when |
|---|---|
| [shoe-try-on](shoe-try-on/SKILL.md) | Placing or scaling a shoe, adding an asset, deciding what "correct" means |
| [footnet](footnet/SKILL.md) | Training, exporting or shipping our foot model; judging whether a failure is the model's |
| [nitro-native-modules](nitro-native-modules/SKILL.md) | Editing anything under `modules/`, or a native change that does not appear |
| [device-diagnostics](device-diagnostics/SKILL.md) | Frame rate or keypoint quality is wrong on the phone; before proposing any unmeasured fix |
| [realtime-vision-pipeline](realtime-vision-pipeline/SKILL.md) | General: camera-rate vision on a phone, detector/tracker splits, budgets |
| [apple-neural-engine](apple-neural-engine/SKILL.md) | General: getting a model onto the Neural Engine and proving it got there |

The first four are about **this** project. The last two are general and their canonical copies live
in the developer's `ai-god` repository — the header of each says so. Edit them there and copy them
back, or the two drift.

## They are agent-neutral

A skill says **what** and **when**. It never names a runtime's tool API, a slash command, or how a
particular agent spawns subagents. Anything runtime-specific goes in that runtime's adapter:

| Runtime | Adapter |
|---|---|
| Claude Code | [`../CLAUDE.md`](../CLAUDE.md) |
| Codex, and anything else | [`../AGENTS.md`](../AGENTS.md) |

The files live here, once, and nowhere else. Claude Code only discovers skills in `.claude/skills/`
and has no setting for another path, so a one-line script links them into place:

```bash
npm run skills
```

`.claude/` is git-ignored, so that plumbing is local to whoever set it up and never lands in the
repository. Rerun the script after adding a skill.

## Writing one

- Say what was **measured**, with the number and the date, and what turned out to be false. A skill
  that only restates the code is noise; the code is already there.
- Name the trap, not the happy path. Most of the value here is "this looked like X and was Y".
- Point at the README or document that holds the detail rather than copying it, so there is one
  place to update.
