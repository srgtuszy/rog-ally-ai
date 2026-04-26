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

> **Important**: `HIP_VISIBLE_DEVICES=1` selects the dGPU. `0` selects the iGPU. The server is hardcoded to use the dGPU only.

## Quick Start

```bash
# Start/stop/restart the Qwen server
./llama-ctl.sh start    # or: stop, restart, status, logs
```

The server auto-starts at login. To disable:
```bash
./llama-ctl.sh disable
```

## Default Model

- **Model**: `Qwen_Qwen3.6-27B-Q5_K_M.gguf`
- **Context**: 262,144 tokens (256K)
- **Port**: 8000
- **Offload**: 63/63 layers on AI PRO R9700
- **KV cache**: q4_0 quantization on GPU
- **Flash Attention**: auto-enabled (critical for USB4 eGPU)
- **Parallel slots**: `-np 1` (requests queue, no concurrent decoding)
- **Context checkpoints**: `--ctx-checkpoints 4` (capped at ~600 MB instead of ~4.8 GB)
- **Batch size**: `-b 512` (reduced from 2048 to lower peak VRAM during long prompts)
- **Warmup**: `--no-warmup` (saves ~600 MB at startup)

## Helper Script

`llama-ctl.sh` wraps systemd for easy management:

```bash
./llama-ctl.sh {start|stop|restart|status|logs|enable|disable}
```

| Command | Action |
|---|---|
| `start` | Start the server |
| `stop` | Stop the server |
| `restart` | Restart the server |
| `status` | Show systemd status |
| `logs` | Show last 50 journal lines |
| `enable` | Auto-start at login (default) |
| `disable` | Manual start only |

## Systemd Service

The service uses `systemd-inhibit` to prevent sleep and display dimming while the server is running.

```bash
# Install service (one-time)
cp llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server
```

> **Note**: The old `llama-rocm.service` has been deprecated. Use `llama-server.service` instead.

## Memory Requirements for 256K Context

Running 256K context with a 27B model requires significant memory:

| Component | Size | Location |
|---|---|---|
| Model weights (Q3_K_M) | ~12.1 GB | AI Pro GPU |
| KV cache (256K × q4_0) | ~5.2 GB | AI Pro GPU |
| Recurrent state | ~0.6 GB | AI Pro GPU |
| Compute buffers | ~0.8 GB | AI Pro GPU |
| Token embeddings | ~0.5 GB | Host (required) |
| ROCm dispatch | ~0.5 GB | Host (required) |

**Total GPU**: ~18.8 GB / 32 GB VRAM
**Total Host**: ~1.0 GB (unavoidable llama.cpp overhead)

### Why Flash Attention Matters for USB4 eGPU

The Radeon AI PRO R9700 connects over USB4, which provides ~40 Gbps bandwidth (roughly 5 GB/s usable). This is significantly less than a native PCIe x16 slot (~32 GB/s). Flash Attention is critical here because:

| Without Flash Attention | With Flash Attention |
|---|---|
| Materializes full N×N score matrix in VRAM | Computes attention in tiles on-chip |
| For 256K context: ~137 GB of HBM traffic per layer | Only output vectors written to HBM |
| Saturates USB4 bandwidth, causing stalls | Minimal HBM traffic, USB4 stays uncongested |
| Model weights may get evicted due to memory pressure | More VRAM headroom for weights + KV cache |

Standard attention creates an intermediate matrix of size `seq_len² × heads × bytes`. At 256K context, that's approximately 137 GB of read/write traffic per attention layer. Over USB4's 5 GB/s, just materializing this matrix once would take ~27 seconds.

Flash Attention avoids this by loading small tiles into fast on-chip SRAM, computing softmax and the output incrementally, and writing only the final result back to VRAM. This keeps the memory traffic bounded by the model weights and KV cache rather than exploding quadratically with sequence length.

For this setup, Flash Attention is the difference between "usable 256K context" and "GPU constantly thrashing across the USB4 link."

### Swap Configuration

For stability with 256K context, ensure adequate swap:

```bash
# Check current swap
swapon --show

# The system should have at least 32GB total swap (zram + disk)
# Bazzite uses zram by default. To resize:
sudo swapoff /dev/zram0
echo 1 | sudo tee /sys/block/zram0/reset
echo 34359738368 | sudo tee /sys/block/zram0/disksize  # 32GB
sudo mkswap /dev/zram0
sudo swapon /dev/zram0
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
| `llama-ctl.sh` | Helper script to start/stop/restart/status the server |
| `start-rocm.sh` | Launch script (called by systemd and llama-ctl.sh) |
| `llama-server.service` | Systemd user unit (auto-start, sleep inhibitor, restart) |
| `llama-rocm.service` | **Deprecated** - old service file, do not use |
| `llama-server-rocm2` | ROCm binary built from source |
| `bin/` | GGML shared libraries |
| `rocm7-libs/` | ROCm 7.2 runtime |

## Startup Behavior

`start-rocm.sh` waits up to 2 minutes for the ROCm GPU to be available before starting the server. This handles the eGPU USB4 enumeration delay on boot. If the GPU isn't ready, the script exits with an error so systemd will retry.

## Troubleshooting

**Server won't start** - Check ROCm libs are found:
```bash
LD_LIBRARY_PATH=bin:rocm7-libs:$LD_LIBRARY_PATH ./llama-server-rocm2 --version
```

**Out of memory / system freezes** - The Qwen3.6 hybrid architecture creates large compute buffers during prompt processing. Applied fixes:
- `-np 1` prevents concurrent requests from allocating multiple slot buffers
- `--ctx-checkpoints 4` caps recurrent state snapshots to ~600 MB
- `-b 512` reduces peak VRAM during long-prompt prefill
- `--no-warmup` skips the empty warmup run

Also ensure 32GB+ swap:
```bash
free -h && swapon --show
```

**Check VRAM usage**:
```bash
rocm-smi --showmeminfo vram
```

**Verify server is listening**:
```bash
ss -tlnp | grep 8000
curl http://127.0.0.1:8000/health
```

**View server logs**:
```bash
./llama-ctl.sh logs
```

**Check GPU layer allocation**:
```bash
journalctl --user -u llama-server | grep "offloaded"
```
