#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_ID=${MODEL_ID:-ZhipuAI/GLM-5.3-Flash}
MODEL_DIR=${MODEL_DIR:-/opt/docker/models/GLM-5.3-Flash}

python3 -m pip install -U modelscope
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
test -s "$MODEL_DIR/model.safetensors.index.json"

echo "Downloaded $MODEL_ID to $MODEL_DIR"
