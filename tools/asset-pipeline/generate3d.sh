#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
  echo "usage: generate3d.sh <shoe-id> <cutout.png> [--pipeline-type 512|1024|1024_cascade] [--seed N]" >&2
  exit 1
fi

SHOE_ID="$1"
INPUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
shift 2

OUT_DIR="$(pwd)/work/$SHOE_ID"
mkdir -p "$OUT_DIR"

cd vendor/trellis-mac
source .venv/bin/activate
python generate.py "$INPUT" --output "$OUT_DIR/raw" --texture-size 1024 "$@"
