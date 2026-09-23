#!/bin/sh
# Point an agent runtime at the skills in this directory.
#
# The skills themselves are agent-neutral and live here, once. Claude Code only discovers skills in
# `.claude/skills/` — there is no setting for another path — so this links each one into place.
# `.claude/` is git-ignored, which keeps the per-agent plumbing out of the repository.
#
#     npm run skills
#
# Safe to rerun: it replaces its own links and leaves anything else in .claude/skills alone.

set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
target="$root/.claude/skills"
mkdir -p "$target"

for skill in "$root"/skills/*/; do
  name=$(basename "$skill")
  link="$target/$name"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    echo "skipping $name: $link exists and is not a symlink" >&2
    continue
  fi
  ln -sfn "../../skills/$name" "$link"
  echo "linked $name"
done
