#!/usr/bin/env bash

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Server engine: "mainline" or "turboquant"
SERVER_ENGINE="${LLAMA_ENGINE:-mainline}"

if [ "${SERVER_ENGINE}" = "turboquant" ]; then
  LLAMA_CPP_DIR="${BASE_DIR}/llama.cpp-turboquant"
  log "Using TurboQuant engine (llama.cpp fork with TurboQuant KV cache)"
else
  LLAMA_CPP_DIR="${BASE_DIR}/llama.cpp"
fi

LLAMA_SERVER="${LLAMA_CPP_DIR}/build/bin/llama-server"
MODEL_DIR="${BASE_DIR}/data/ollama/models/blobs"
PIDFILE="${BASE_DIR}/data/llama-server.pid"
LOGFILE="${BASE_DIR}/data/llama-server.log"

# Default model
DEFAULT_MODEL="${BASE_DIR}/data/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
MODEL="${LLAMA_MODEL:-${DEFAULT_MODEL}}"

# Server settings
PORT="${LLAMA_PORT:-8000}"
CTX="${LLAMA_CTX:-131072}"
GPU_DEVICE="${LLAMA_GPU_DEVICE:-1}"  # 0=iGPU, 1=R9700

log() { printf '[gpu-setup] %s\n' "$*"; }
warn() { printf '[gpu-setup] WARNING: %s\n' "$*" >&2; }
err()  { printf '[gpu-setup] ERROR: %s\n' "$*" >&2; }

check_gpu() {
  # NOTE: llama-server --list-devices and vulkaninfo both hang on this system
  # due to _amdgpu_device_initialize on the iGPU. The server works fine regardless.
  log "Skipping GPU check (use GGML_VK_VISIBLE_DEVICES to select GPU)"
  return 0
}

check_deps() {
  local missing=0
  if [ ! -x "${LLAMA_SERVER}" ]; then
    err "llama-server not found at ${LLAMA_SERVER}"
    err "Run: ./stack.sh install"
    missing=1
  fi
  if [ ! -f "${MODEL}" ]; then
    err "Model not found at ${MODEL}"
    missing=1
  fi
  return ${missing}
}

install_stack() {
  log "Installing llama.cpp with Vulkan backend..."

  # Check for build deps
  for cmd in cmake git glslc; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
      log "Installing ${cmd} via brew..."
      brew install "${cmd}" 2>/dev/null || {
        err "Failed to install ${cmd}. Install it manually."
        return 1
      }
    fi
  done

  # Ensure Vulkan headers
  if ! find /home/linuxbrew -path "*/vulkan/vulkan.h" -print -quit 2>/dev/null | grep -q .; then
    log "Installing Vulkan headers..."
    brew install vulkan-headers shaderc
  fi

  # Clone or update llama.cpp
  if [ -d "${LLAMA_CPP_DIR}/.git" ]; then
    log "Updating llama.cpp..."
    git -C "${LLAMA_CPP_DIR}" pull --ff-only 2>/dev/null || true
  else
    log "Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_DIR}"
  fi

  # Build with Vulkan
  log "Building with Vulkan backend..."
  cd "${LLAMA_CPP_DIR}"
  rm -rf build
  cmake -B build \
    -DGGML_VULKAN=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DVulkan_INCLUDE_DIR=/home/linuxbrew/.linuxbrew/include \
    -DVulkan_LIBRARY=/usr/lib64/libvulkan.so.1

  cmake --build build -j"$(nproc)" --target llama-server llama-cli

  if [ -x build/bin/llama-server ]; then
    log "Build successful: ${LLAMA_CPP_DIR}/build/bin/llama-server"
  else
    err "Build failed"
    return 1
  fi
}

run_server() {
  local kv_type
  if [ "${SERVER_ENGINE}" = "turboquant" ]; then
    kv_type="turbo3"
    log "KV cache type: turbo3 (3.25 bits/val, ~5x compression vs f16)"
  else
    kv_type="q4_0"
  fi

  LD_LIBRARY_PATH="${LLAMA_CPP_DIR}/build/bin:${LD_LIBRARY_PATH}" GGML_VK_VISIBLE_DEVICES="${GPU_DEVICE}" "${LLAMA_SERVER}" \
    -m "${MODEL}" \
    -ngl 99 \
    --flash-attn on \
    -c "${CTX}" \
    --swa-full \
    -np 1 \
    -b 2048 \
    -ub 512 \
    -ctk "${kv_type}" \
    -ctv "${kv_type}" \
    --cache-ram 0 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --jinja \
    >> "${LOGFILE}" 2>&1
}

wait_for_port_free() {
  local i
  for i in $(seq 1 30); do
    if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
      return 0
    fi
    sleep 1
  done
  warn "Port ${PORT} still in use after 30s, forcing cleanup..."
  pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null
  sleep 2
}

start_server() {
  if ! check_deps; then
    return 1
  fi

  # Stop existing
  stop_server 2>/dev/null
  wait_for_port_free

  check_gpu

  log "Starting llama-server on port ${PORT}..."
  log "Model: $(basename "${MODEL}")"
  log "Context: ${CTX}, GPU device: Vulkan${GPU_DEVICE}"
  log "Watchdog enabled — server will auto-restart on crash"

  # Watchdog loop: restart server on crash
  while true; do
    run_server &
    local pid=$!
    echo "${pid}" > "${PIDFILE}"
    log "Server PID: ${pid}"

    # Wait for health
    local i
    local ready=0
    for i in $(seq 1 60); do
      if curl -sf http://localhost:"${PORT}"/health >/dev/null 2>&1; then
        ready=1
        break
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 2
    done

    if [ "${ready}" -eq 1 ]; then
      log "Server is ready at http://localhost:${PORT}"
      log "OpenAI-compatible API: http://localhost:${PORT}/v1"
    else
      err "Server crashed during startup. Check: ${LOGFILE}"
      rm -f "${PIDFILE}"
      return 1
    fi

    # Monitor running server — restart if it dies
    while kill -0 "${pid}" 2>/dev/null; do
      sleep 5
    done
    wait "${pid}" 2>/dev/null
    local exit_code=$?

    # If we're stopping (no pidfile), exit cleanly
    if [ ! -f "${PIDFILE}" ]; then
      return 0
    fi

    warn "Server exited with code ${exit_code}, restarting..."
    wait_for_port_free
  done
}

stop_server() {
  if [ -f "${PIDFILE}" ]; then
    local pid
    pid=$(cat "${PIDFILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      log "Stopping llama-server (PID ${pid})..."
      kill "${pid}"
      sleep 2
      kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null
    fi
    rm -f "${PIDFILE}"
    log "Server stopped."
  else
    # Try to find by process name
    pkill -f "llama-server.*--port ${PORT}" 2>/dev/null && log "Server stopped." || log "No server running."
  fi
}

status_server() {
  if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
    log "Server is running (PID $(cat "${PIDFILE}"))"
    curl -sf http://localhost:"${PORT}"/health && echo ""
    log "API: http://localhost:${PORT}/v1"
  else
    log "Server is not running."
  fi
}

verify_server() {
  if ! curl -sf http://localhost:"${PORT}"/health >/dev/null 2>&1; then
    err "Server not reachable at http://localhost:${PORT}"
    return 1
  fi

  log "Testing inference..."
  local resp
  resp=$(curl -sf http://localhost:"${PORT}"/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}' 2>&1)

  if echo "${resp}" | grep -q "choices"; then
    log "Inference working!"
    echo "${resp}" | python3 -m json.tool 2>/dev/null | grep "content" || true
    return 0
  else
    err "Inference failed"
    echo "${resp}"
    return 1
  fi
}

list_models() {
  log "Available model blobs:"
  find "${MODEL_DIR}" -size +500M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}'
  echo ""
  log "Set LLAMA_MODEL=/path/to/model.gguf to use a different model"
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") <command>

Commands:
  install   Build llama.cpp with Vulkan backend
  start     Start the inference server
  stop      Stop the inference server
  restart   Stop then start
  status    Show server status
  verify    Test inference end-to-end
  models    List available model files
  logs      Tail the server log

Environment:
  LLAMA_MODEL       Path to GGUF model file (default: nemotron-cascade-2)
  LLAMA_PORT        Server port (default: 8000)
  LLAMA_CTX         Context length (default: 8192)
  LLAMA_GPU_DEVICE  Vulkan device index (default: 1 = R9700)
USAGE
}

cmd="${1:-start}"
case "${cmd}" in
  install)  install_stack ;;
  start)    start_server ;;
  stop)     stop_server ;;
  restart)  stop_server && start_server ;;
  status)   status_server ;;
  verify)   verify_server ;;
  models)   list_models ;;
  logs)     tail -f "${LOGFILE}" ;;
  *)        usage; exit 1 ;;
esac
