#!/usr/bin/env bash
set -Eeuo pipefail

ARCHIVE=${1:?Usage: import-image.sh <image.tar.zst>}

command -v zstd >/dev/null
test -s "$ARCHIVE"

if [[ -f "$ARCHIVE.sha256" ]]; then
  archive_dir=$(cd "$(dirname "$ARCHIVE")" && pwd)
  archive_name=$(basename "$ARCHIVE")
  (cd "$archive_dir" && sha256sum -c "$archive_name.sha256")
fi

zstd -dc "$ARCHIVE" | docker load
