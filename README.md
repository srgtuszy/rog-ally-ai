# ROG Ally X + Radeon AI PRO R9700 eGPU: llama.cpp ROCm

Local LLM inference on an ASUS ROG Ally X via AMD Radeon AI PRO R9700 over USB4, using llama.cpp with ROCm backend on Bazzite (Fedora Atomic).

## Hardware

- **Host**: ASUS ROG Ally X (Ryzen Z1 Extreme, 8GB iGPU - gfx1103)
- **eGPU**: AMD Radeon AI PRO R9700 (gfx1201, 32GB VRAM) via USB4

## GPU Layout

| Device | ROCm Index | Name | VRAM |
|---|---|---|---|
| dGPU | 1 | Radeon AI PRO R9700 | 32GB |
| iGPU | 0 | Ryzen Z1 Extreme | 8GB |

> **Important**: `HIP_VISIBLE_DEVICES` is the variable that matters. `1` selects the dGPU, `0` selects the iGPU.

## Quick Start

```bash
# Qwen 3.5 27B on port 8000 (dGPU)
./start-rocm.sh

# Hermes 3 8B on port 8002 (iGPU)
./start-hermes.sh
```

## Systemd Service

The service uses `systemd-inhibit` to prevent sleep and display dimming while the server is running.

```bash
cp llama-rocm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-rocm
```

## Setup

### 1. Fix the SMU hang (R9700 over USB4)

The R9700 hits an SMU firmware version mismatch that causes the GPU to hang after boot.

```bash
sudo rpm-ostree kargs \
  --replace=amdgpu.ppfeaturemask=0xfffd7fff \
  --append=amdgpu.runpm=0
```

Reboot, then verify:

```bash
cat /sys/bus/pci/devices/0000:08:00.0/power_state  # Must show D0
rocm-smi  # Must show both GPUs
```

### 2. Build llama.cpp with ROCm

Pre-built ROCm binaries from upstream do not work on Bazzite's ROCm 6.4. Build from source in a container:

```bash
# Start Ubuntu 24.04 container with ROCm 7.2
podman run -d --privileged -v $(pwd)/rocm-build:/workspace docker.io/ubuntu:24.04 sleep infinity

# In the container:
apt-get update && apt-get install -y wget gnupg2 git cmake make
wget https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
dpkg -i amdgpu-install_7.2.1.70201-1_all.deb || apt --fix-broken install -y
amdgpu-install -y --rocm

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp
cd /workspace/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx1201;gfx1103"
cmake --build . -j$(nproc)

# Copy back to host
podman cp container:/workspace/llama.cpp/build/bin/llama-server /path/to/host/llama-server-rocm2
podman cp container:/opt/rocm-7.2.1/lib/. /path/to/host/rocm7-libs/
podman cp container:/workspace/llama.cpp/build/bin/lib*.so /path/to/host/bin/
```

### 3. Verify

```bash
LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH ./llama-server-rocm2 --version
```

## Files

| File | Purpose |
|---|---|
| `start-rocm.sh` | Launch Qwen server (dGPU, port 8000) |
| `start-hermes.sh` | Launch Hermes server (iGPU, port 8002) |
| `llama-server-rocm2` | ROCm binary built from source |
| `bin/` | GGML shared libraries |
| `rocm7-libs/` | ROCm 7.2 runtime |
| `llama-rocm.service` | Systemd unit for Qwen (with sleep inhibitor) |
| `llama-hermes.service` | Systemd unit for Hermes (with sleep inhibitor) |

## Troubleshooting

**Server won't start** - Check ROCm libs are found:
```bash
LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH ./llama-server-rocm2 --version
```

**Out of memory** - Check VRAM usage:
```bash
rocm-smi --showpidgpus
rocm-smi --showmeminfo vram
```

**Verify server is listening**:
```bash
ss -tlnp | grep 8000
curl http://127.0.0.1:8000/health
```
