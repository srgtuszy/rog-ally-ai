#!/usr/bin/env bash
# Usage: ./start-rocm.sh [model] [context_size] [port]
# Defaults: Q3_K_M model, 262144 context (256K), port 8000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export HIP_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=bin:rocm7-libs:${LD_LIBRARY_PATH:-}

MODEL="${1:-Qwen_Qwen3.6-27B-Q4_K_M.gguf}"
CTX="${2:-262144}"
PORT="${3:-8000}"
HOST="${HOST:-::}"
ALIAS="${MODEL_ALIAS:-qwen3.6-27b}"

# Wait for the ROCm GPU to be available (eGPU over USB4 may need time to enumerate)
MAX_RETRIES=30
RETRY_DELAY=4
GPU_READY=false
for ((i=0; i<MAX_RETRIES; i++)); do
    if ./llama-server-rocm2 --version >/dev/null 2>&1; then
        echo "ROCm GPU is ready (attempt $((i+1))/$MAX_RETRIES)"
        GPU_READY=true
        break
    fi
    echo "ROCm GPU not ready, waiting... (attempt $((i+1))/$MAX_RETRIES)"
    sleep "$RETRY_DELAY"
done

# Fail hard if GPU is not available so systemd restarts us
if [[ "$GPU_READY" != "true" ]]; then
    echo "ERROR: ROCm GPU failed to initialize after $((MAX_RETRIES * RETRY_DELAY)) seconds"
    exit 1
fi

# Verify model exists
if [[ ! -f "data/models/${MODEL}" ]]; then
    echo "ERROR: Model file not found: data/models/${MODEL}"
    exit 1
fi

# USB4/eGPU optimizations:
# - -ub 2048: larger physical batch = fewer kernel launches over USB4
# - --numa isolate: keep CPU threads on USB4 controller socket
# - Removed ngram spec (wasn't working, saves CPU overhead)
exec ./llama-server-rocm2 \
    -m "data/models/${MODEL}" \
    -ngl 999 \
    -sm none \
    -mg 0 \
    -c "${CTX}" \
    -np 1 \
    --ctx-checkpoints 0 \
    --cache-ram 4096 \
    -b 1024 \
    -ub 2048 \
    --no-warmup \
    -ctk q4_0 \
    -ctv q4_0 \
    --kv-offload \
    --no-host \
    --flash-attn on \
    --prio 2 \
    --prio-batch 2 \
    --poll 80 \
    --poll-batch 1 \
    --numa isolate \
    --alias "${ALIAS}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --jinja \
    --no-mmap
