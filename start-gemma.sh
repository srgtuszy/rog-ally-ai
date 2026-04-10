#!/usr/bin/env bash
# Usage: ./start-gemma.sh [context_size] [port]
# Default: 131072 context (128K), port 8001

cd /var/home/srgtuszy/gpu-setup
export ROCR_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH

CTX="${1:-131072}"
PORT="${2:-8001}"

exec ./llama-server-rocm2 \
    -m data/models/google_gemma-4-26B-A4B-it-Q5_K_M.gguf \
    -ngl 99 \
    -c "${CTX}" \
    -ctk q4_0 \
    -ctv q4_0 \
    --port "${PORT}" \
    --jinja \
    --no-warmup