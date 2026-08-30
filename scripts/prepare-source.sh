#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE_REPO=${SOURCE_REPO:-https://github.com/wtdcode/vllm-backport.git}
SOURCE_COMMIT=${SOURCE_COMMIT:-2674c8bb6d8799b32158c94bee33356d84772a2a}
SRC_DIR=${SRC_DIR:-$ROOT/build/vllm-src}

if [[ -e "$SRC_DIR" ]]; then
  echo "Refusing to overwrite existing source directory: $SRC_DIR" >&2
  echo "Remove or rename it explicitly, then rerun this script." >&2
  exit 1
fi

mkdir -p "$(dirname "$SRC_DIR")"
git clone --filter=blob:none --no-checkout "$SOURCE_REPO" "$SRC_DIR"
git -C "$SRC_DIR" fetch --depth 1 origin "$SOURCE_COMMIT"
git -C "$SRC_DIR" checkout --detach "$SOURCE_COMMIT"

git -C "$SRC_DIR" apply --check "$ROOT/patches/glm53-sm80.patch"
git -C "$SRC_DIR" apply --whitespace=nowarn "$ROOT/patches/glm53-sm80.patch"

# The overrides are the exact Python/Triton files used by the final tested
# runtime. They intentionally win over the earlier integration patch.
cp -a "$ROOT/overrides/." "$SRC_DIR/"

echo "Prepared source at $SRC_DIR"
echo "Baseline: $SOURCE_COMMIT"
