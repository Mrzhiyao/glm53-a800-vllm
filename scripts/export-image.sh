#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE=${IMAGE:-glm53-a800:sm80-v9-cudagraph}
OUT=${OUT:-./glm53-a800-sm80-v9.tar.zst}
ZSTD_LEVEL=${ZSTD_LEVEL:-10}

command -v zstd >/dev/null
docker image inspect "$IMAGE" >/dev/null
mkdir -p "$(dirname "$OUT")"

docker save "$IMAGE" | zstd -T0 "-$ZSTD_LEVEL" -o "$OUT"
out_dir=$(cd "$(dirname "$OUT")" && pwd)
out_name=$(basename "$OUT")
(cd "$out_dir" && sha256sum "$out_name" > "$out_name.sha256")

du -h "$OUT" "$OUT.sha256"
cat "$OUT.sha256"
