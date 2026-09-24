#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TRELLIS_MAC_REPO="https://github.com/shivampkumar/trellis-mac.git"
TRELLIS_MAC_REF="d58628f4f5b9c3de8274cb110074154f4b31cef2"
VENDOR_DIR="vendor/trellis-mac"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Apple Silicon is required" >&2
  exit 1
fi

# trellis-mac only warns when the Metal backends fail to build, and texture baking then silently stops working
if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "Metal Toolchain is missing, run: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi

if [ ! -d "$VENDOR_DIR" ]; then
  git clone "$TRELLIS_MAC_REPO" "$VENDOR_DIR"
fi
git -C "$VENDOR_DIR" fetch --quiet origin "$TRELLIS_MAC_REF"
git -C "$VENDOR_DIR" checkout --quiet "$TRELLIS_MAC_REF"

EIGEN_INCLUDE="$(brew --prefix eigen 2>/dev/null)/include/eigen3"
if [ ! -f "$EIGEN_INCLUDE/Eigen/Dense" ]; then
  echo "Eigen is missing, run: brew install eigen" >&2
  exit 1
fi
export CPLUS_INCLUDE_PATH="$EIGEN_INCLUDE${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

# trellis-mac installs torch unpinned; torch 2.14+ requires C++20 and its Metal extensions build with -std=c++17
UV_CONSTRAINT="$(pwd)/constraints.txt" bash "$VENDOR_DIR/setup.sh"
npm ci
