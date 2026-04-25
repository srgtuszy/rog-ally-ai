#!/usr/bin/env bash
# Usage: ./start-vulkan-qwen.sh [context_size] [port]
# Default: 262144 context (256K), port 8000
# Uses Vulkan backend

cd "$HOME/gpu-setup"
export GGML_VK_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=llama.cpp/build/bin:$LD_LIBRARY_PATH

CTX="${1:-262144}"
PORT="${2:-8000}"

exec ./llama.cpp/build/bin/llama-server \
    -m data/models/Qwen3.5-27B-Opus-Reasoning.Q3_K_M.gguf \
    -ngl 99 \
    -c "${CTX}" \
    --port "${PORT}" \
    --jinja