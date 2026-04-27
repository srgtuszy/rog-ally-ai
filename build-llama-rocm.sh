#!/usr/bin/env bash
# Build llama.cpp with ROCm support for gfx1201 (R9700) + gfx1103 (Z1 iGPU)
# Uses a Podman Ubuntu 24.04 container with ROCm 7.2.2
#
# Usage:
#   ./build-llama-rocm.sh              # Build from latest master
#   ./build-llama-rocm.sh <commit>     # Build specific commit
#
# NOTE: First run builds the ROCm container image (~6GB download, 10-20 min)
# Subsequent builds only compile llama.cpp (~5-10 min)
#
# Output: llama-server-rocm2 + bin/*.so + rocm7-libs/ in ~/gpu-setup/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE_NAME="llama-rocm-build-env"
CONTAINER_NAME="llama-rocm-build"
COMMIT="${1:-master}"
CMAKE_ARCHS="gfx1201;gfx1103"

# ---- Step 0: Build image if needed ----
if ! podman images --format '{{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}:latest$"; then
    echo "=== Building ROCm container image (first time, downloads ~6GB) ==="
    podman build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile.rocm-build" .
    echo ""
    echo "Image built successfully."
    echo ""
fi

echo "=== Building llama.cpp ROCm for ${CMAKE_ARCHS} (commit: ${COMMIT}) ==="
echo ""

# ---- Step 1: Start container ----
echo "[1/5] Starting container..."
podman rm -f "${CONTAINER_NAME}" 2>/dev/null || true
podman run -d --name "${CONTAINER_NAME}" \
    --privileged \
    -v "${SCRIPT_DIR}/rocm-build:/workspace" \
    "${IMAGE_NAME}" \
    sleep infinity

# ---- Step 2: Clone/update llama.cpp ----
echo "[2/5] Cloning llama.cpp at commit ${COMMIT}..."
podman exec "${CONTAINER_NAME}" bash -euo pipefail <<REMOTE
set -ex
cd /workspace

if [ -d llama.cpp/.git ]; then
    cd llama.cpp
    git fetch --depth=50 origin
    git checkout ${COMMIT}
else
    rm -rf llama.cpp
    git clone --depth 50 --branch ${COMMIT} \
        https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
fi
REMOTE

# ---- Step 3: Build ----
echo "[3/5] Building llama.cpp (this takes ~10 min)..."
podman exec "${CONTAINER_NAME}" bash -euo pipefail <<REMOTE
set -ex
cd /workspace/llama.cpp
rm -rf build
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES="${CMAKE_ARCHS}" \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_NO_VMM=ON \
    -DGGML_CUDA_FA=ON

cmake --build . -j\$(nproc)

echo "Build complete"
ls -lh bin/llama-server
REMOTE

# ---- Step 4: Copy artifacts to host ----
echo "[4/5] Copying binaries to host..."

# Backup old binary
if [ -f "${SCRIPT_DIR}/llama-server-rocm2" ]; then
    BACKUP="${SCRIPT_DIR}/llama-server-rocm2.bak"
    cp "${SCRIPT_DIR}/llama-server-rocm2" "${BACKUP}"
    echo "  Old binary backed up to ${BACKUP}"
fi

# Copy new binary
podman cp "${CONTAINER_NAME}:/workspace/llama.cpp/build/bin/llama-server" \
    "${SCRIPT_DIR}/llama-server-rocm2"
chmod +x "${SCRIPT_DIR}/llama-server-rocm2"

# Copy shared libs
mkdir -p "${SCRIPT_DIR}/bin"
podman exec "${CONTAINER_NAME}" bash -c \
    "cp /workspace/llama.cpp/build/bin/lib*.so /workspace/"
podman cp "${CONTAINER_NAME}:/workspace/lib*.so" "${SCRIPT_DIR}/bin/"

# Copy ROCm runtime libs
mkdir -p "${SCRIPT_DIR}/rocm7-libs"
podman exec "${CONTAINER_NAME}" bash -c \
    "cp /opt/rocm/lib/lib*so* /workspace/"
podman cp "${CONTAINER_NAME}:/workspace/lib*.so*" "${SCRIPT_DIR}/rocm7-libs/"

# ---- Step 5: Verify ----
echo "[5/5] Verifying new binary..."
cd "${SCRIPT_DIR}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/bin:${SCRIPT_DIR}/rocm7-libs"
VERSION_OUTPUT=$(./llama-server-rocm2 --version 2>&1 | head -5)
echo "${VERSION_OUTPUT}"

echo ""
echo "=== Build complete ==="
echo "New binary: ${SCRIPT_DIR}/llama-server-rocm2"
echo "Backup:     ${SCRIPT_DIR}/llama-server-rocm2.bak"
echo ""
echo "To use the new build, restart the server:"
echo "  ./llama-ctl.sh restart"
