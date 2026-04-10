#!/usr/bin/env bash
# Usage: ./start-rocm.sh [model] [context_size] [port]
# Defaults: Q3_K_M model, 262144 context (256K), port 8000

cd /var/home/srgtuszy/gpu-setup
export ROCR_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH

MODEL="${1:-Qwen3.5-27B-Opus-Reasoning.Q3_K_M.gguf}"
CTX="${2:-262144}"
PORT="${3:-8000}"

exec ./llama-server-rocm2 \
    -m "data/models/${MODEL}" \
    -ngl 99 \
    -c "${CTX}" \
    -ctk q4_0 \
    -ctv q4_0 \
    --port "${PORT}" \
    --jinja \
    --no-warmup