# ROG Ally X + R9700 eGPU Inference Server

Turn an ASUS ROG Ally X into a local LLM inference server by connecting an AMD Radeon AI PRO R9700 over USB4 and running llama.cpp with ROCm backend.

## Hardware

- **Host**: ASUS ROG Ally X (Ryzen Z1 Extreme, 8GB iGPU - gfx1103)
- **eGPU**: AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM) via USB4
- **OS**: Bazzite (Fedora Atomic)

## GPU Layout

| Device | ROCm Index | Name | VRAM |
|---|---|---|---|
| dGPU | 0 | Radeon AI PRO R9700 | 32GB |
| iGPU | 1 | Ryzen Z1 Extreme (Phoenix) | 8GB |

> **Important**: The `llama-server-rocm2` binary uses the AMD HIP runtime (`libamdhip64.so.7`), which respects `HIP_VISIBLE_DEVICES`, **not** `ROCR_VISIBLE_DEVICES`. The dGPU is device `0`.

## Quick Start

```bash
# Start the ROCm server (256K context, Q4 model, port 8000)
./start-rocm.sh

# Or with custom model/context/port
./start-rocm.sh Qwen3.5-27B.Q4_K_M.gguf 262144 8000
```

Server listens on `http://[::]:8000` (IPv6 any-address, includes IPv4 via mapped addresses) with OpenAI-compatible API at `/v1`.

## Systemd Service

```bash
# User service (no sudo needed)
cp llama-rocm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-rocm
```

## Hermes Agent Integration

The [Hermes agent](https://github.com/NousResearch/hermes-agent) is configured to use this local server. In `~/.hermes/config.yaml`:

```yaml
model:
  default: qwen3.5-27b
  provider: custom
  base_url: http://localhost:8000/v1
  context_length: 262144
```

All auxiliary services (vision, compression, web extract, etc.) also route to `http://localhost:8000/v1`.

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

### 3. Run with HIP_VISIBLE_DEVICES=0

This hides the iGPU and forces 100% of the model to the dGPU:

```bash
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH
./llama-server-rocm2 \
    -m data/models/Qwen3.5-27B.Q4_K_M.gguf \
    -ngl 99 \
    -c 262144 \
    -ctk q4_0 \
    -ctv q4_0 \
    --alias qwen3.5-27b \
    --host :: \
    --port 8000 \
    --jinja \
    --no-warmup
```

## Why ROCm instead of Vulkan?

**Vulkan** works but has issues:
- Hangs on higher context (128K fails, 64K works)
- Model splits across both GPUs, causing USB4 bottleneck

**ROCm** (after the build effort):
- Full model on dGPU only (no USB4 bottleneck)
- 256K context works reliably
- Better VRAM management

## Context vs VRAM

| Model | Context | VRAM | Notes |
|---|---|---|---|
| Qwen3.5-27B Q4_K_M | 64K | ~15 GB | Conservative |
| Qwen3.5-27B Q4_K_M | 128K | ~18 GB | Good balance |
| Qwen3.5-27B Q4_K_M | **256K** | ~21 GB | **Default** |
| Hermes-3-8B Q4_K_M | 256K | ~6 GB | Small model, separate port |

## Files

| File | Purpose |
|---|---|
| `start-rocm.sh` | Launch ROCm server (Qwen, port 8000) |
| `start-hermes.sh` | Launch ROCm Hermes server (port 8002) |
| `start-hermes-vulkan.sh` | Launch Vulkan Hermes server (fallback) |
| `start-vulkan-qwen.sh` | Launch Vulkan Qwen server (fallback) |
| `llama-server-rocm2` | ROCm binary built from source |
| `bin/` | GGML shared libraries |
| `rocm7-libs/` | ROCm 7.2 runtime |
| `llama-rocm.service` | Systemd unit for Qwen server |
| `llama-hermes.service` | Systemd unit for Hermes server |
| `stack.sh` | Docker Compose stack management |

## Troubleshooting

**Server won't start** - Check ROCm libs are in LD_LIBRARY_PATH:
```bash
LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH ./llama-server-rocm2 --version
```

**Model loads but server exits** - Try lower context or different model quant.

**Out of memory errors** - Check no other ROCm processes are using GPU 0:
```bash
rocm-smi --showpidgpus
rocm-smi --showmeminfo vram
```

**Hermes agent can't connect** - Verify the server is listening on the expected interface:
```bash
ss -tlnp | grep 8000
curl http://127.0.0.1:8000/health
curl http://[::1]:8000/health
```

## Performance

- Qwen3.5-27B Q4_K_M at 256K context: ~25 tok/s
- 100% on dGPU (21GB model) + minimal CPU
- No USB4 bottleneck between GPUs
