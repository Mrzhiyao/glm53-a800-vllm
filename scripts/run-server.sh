#!/usr/bin/env bash
set -Eeuo pipefail

MODEL=${MODEL:-/opt/docker/models/GLM-5.3-Flash}
IMAGE=${IMAGE:-glm53-a800:sm80-v9-cudagraph}
CONTAINER=${CONTAINER:-glm53-flash-a800}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-2}
PORT=${PORT:-8010}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-24,21}

docker image inspect "$IMAGE" >/dev/null
test -s "$MODEL/model.safetensors.index.json"

IFS=',' read -ra gpu_array <<< "$GPU_IDS"
if (( ${#gpu_array[@]} != TP_SIZE * PP_SIZE )); then
  echo "GPU count must equal TP_SIZE * PP_SIZE." >&2
  exit 1
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --stop-timeout 30 \
  --gpus "\"device=$GPU_IDS\"" \
  --ipc=host \
  --network host \
  --health-cmd "python3 -c 'import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:${PORT}/v1/models\", timeout=5).read()'" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  --health-start-period 30m \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_PP_LAYER_PARTITION="$PP_LAYER_PARTITION" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_ALGO=Ring \
  -e NCCL_PROTO=Simple \
  -v "$MODEL:$MODEL:ro" \
  "$IMAGE" \
  "$MODEL" \
  --served-model-name GLM-5.3-Flash glm-5.3-flash \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 \
  --port "$PORT"

echo "Started $CONTAINER on GPUs $GPU_IDS"
echo "TP=$TP_SIZE PP=$PP_SIZE context=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS batched_tokens=$MAX_NUM_BATCHED_TOKENS"
