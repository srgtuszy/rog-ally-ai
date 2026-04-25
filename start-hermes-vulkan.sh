#!/usr/bin/env bash
# Usage: ./start-hermes-vulkan.sh [context_size] [port]
# Default: 65536 context (64K), port 8002
# Uses Vulkan backend - works when ROCm fails

cd "$HOME/gpu-setup"
export GGML_VK_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=llama.cpp/build/bin:$LD_LIBRARY_PATH

CTX="${1:-65536}"
PORT="${2:-8002}"

exec ./llama.cpp/build/bin/llama-server \
    -m data/models/Hermes-3-8B.Q4_K_M.gguf \
    -ngl 99 \
    -c "${CTX}" \
    --port "${PORT}" \
    --jinja