#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIR=${SRC_DIR:-$ROOT/build/vllm-src}
IMAGE=${IMAGE:-glm53-a800:sm80-v9-cudagraph}
BASE_IMAGE=${BASE_IMAGE:-lazymio/vllm-backport:latest-sm80}
MAX_JOBS=${MAX_JOBS:-16}
NVCC_THREADS=${NVCC_THREADS:-1}

test -d "$SRC_DIR/.git"
test -f "$SRC_DIR/vllm/models/glm5next/nvidia/model.py"

DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  --build-arg NVCC_THREADS="$NVCC_THREADS" \
  -f "$ROOT/docker/Dockerfile.sm80" \
  -t "$IMAGE" \
  "$SRC_DIR"

docker image inspect "$IMAGE" --format 'Built {{.Id}} ({{.Size}} bytes)'
