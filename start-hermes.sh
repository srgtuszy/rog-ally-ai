#!/usr/bin/env bash
# Usage: ./start-hermes.sh [context_size] [port]
# Default: 262144 context (256K), port 8002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export HIP_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH

CTX="${1:-262144}"
PORT="${2:-8002}"

exec ./llama-server-rocm2 \
    -m data/models/Hermes-3-8B.Q4_K_M.gguf \
    -ngl 99 \
    -c "${CTX}" \
    -ctk q4_0 \
    -ctv q4_0 \
    --port "${PORT}" \
    --jinja \
    --no-warmup
