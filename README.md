# ROG Ally X + R9700 eGPU Inference Server

Turn an ASUS ROG Ally X into a local LLM inference server by connecting an AMD Radeon AI PRO R9700 over USB4 and running llama.cpp with ROCm backend.

## Hardware

- **Host**: ASUS ROG Ally X (Ryzen Z1 Extreme, 8GB iGPU)
- **eGPU**: AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM) via USB4
- **OS**: Bazzite (Fedora Atomic)

## Quick Start

```bash
# Start the ROCm server (64K context, Q4 model)
./start-rocm.sh

# Or with custom context
./start-rocm.sh Qwen3.5-27B.Q4_K_M.gguf 65536 8000
```

Server runs on `http://localhost:8000` with OpenAI-compatible API at `/v1`.

## Systemd Service

```bash
sudo cp llama-rocm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-rocm
```

## The Setup Process

### 1. Fix the SMU issue

The R9700 over USB4 hits an SMU firmware version mismatch that causes the GPU to hang after boot.

```bash
sudo rpm-ostree kargs \
  --replace=amdgpu.ppfeaturemask=0xfffd7fff \
  --append=amdgpu.runpm=0
```

Reboot. Verify the GPU:

```bash
cat /sys/bus/pci/devices/0000:08:00.0/power_state  # Must show D0
rocm-smi  # Must show both GPUs
```

### 2. Build llama.cpp with ROCm (the hard part)

The pre-built ROCm binaries don't work with ROCm 6.4 on Bazzite. Build from source using a container:

```bash
# Start an Ubuntu 24.04 container with ROCm 7.2
podman run -d --privileged -v $(pwd)/rocm-build:/workspace docker.io/ubuntu:24.04 sleep infinity

# In the container:
apt-get update && apt-get install -y wget gnupg2
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
dpkg -i amdgpu-install_7.2.1.70201-1_all.deb || apt --fix-broken install -y
amdgpu-install -y --rocm

apt-get install -y git cmake make

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
cd /workspace/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1103"
cmake --build . -j$(nproc)

# Copy binary and ROCm libraries back to host
podman cp container:/workspace/llama.cpp/build/bin/llama-server /path/to/host/llama-server-rocm2
podman cp container:/opt/rocm-7.2.1/lib/. /path/to/host/rocm7-libs/
podman cp container:/workspace/llama.cpp/build/bin/lib*.so /path/to/host/bin/
```

### 3. Run with ROCR_VISIBLE_DEVICES=1

This hides the iGPU and forces 100% of the model to the dGPU:

```bash
export ROCR_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH
./llama-server-rocm2 -m data/models/Qwen3.5-27B.Q4_K_M.gguf -ngl 99 -c 65536 --port 8000 --jinja --no-warmup
```

## Why ROCm instead of Vulkan?

**Vulkan** works but has issues:
- Hangs on higher context (128K fails, 64K works)
- Model splits across both GPUs, causing USB4 bottleneck

**ROCm** (after the build effort):
- Full model on dGPU only (no USB4 bottleneck)
- 64K context works reliably
- Better VRAM management

## Context vs VRAM

| Model | Context | VRAM | Notes |
|---|---|---|---|
| Qwen3.5-27B Q3_K_M | 32K | ~12 GB | Smaller model, lower quality |
| Qwen3.5-27B Q4_K_M | 64K | ~15 GB | Good balance (default) |
| Qwen3.5-27B Q4_K_M | 32K | ~15 GB | More headroom |

## Troubleshooting

**Server won't start** - Check ROCm libs are in LD_LIBRARY_PATH:
```bash
LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH ./llama-server-rocm2 --version
```

**Model loads but server exits** - Try lower context or different model quant.

**Using both GPUs** - Must set `ROCR_VISIBLE_DEVICES=1` to hide iGPU.

## Files

| File | Purpose |
|---|---|
| `start-rocm.sh` | Launch script |
| `llama-server-rocm2` | ROCm binary built from source |
| `bin/` | GGML shared libraries |
| `rocm7-libs/` | ROCm 7.2 runtime |
| `llama-rocm.service` | Systemd unit |

## Performance

- Qwen3.5-27B Q4_K_M at 64K context: ~25 tok/s
- 100% on dGPU (12GB model) + minimal CPU (682MB)
- No USB4 bottleneck between GPUs