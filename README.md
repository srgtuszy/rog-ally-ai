# ROG Ally X + R9700 eGPU Inference Server

Turn an ASUS ROG Ally X into a local LLM inference server by connecting an AMD Radeon AI PRO R9700 over USB4 and running llama.cpp with the Vulkan backend.

## What you need

- ASUS ROG Ally X (or any USB4/Thunderbolt host)
- AMD Radeon AI PRO R9700 in a USB4/Thunderbolt eGPU enclosure
- [Bazzite](https://bazzite.gg/) installed (Fedora Atomic — works out of the box with the Ally X)

## Step 1: Fix the kernel parameters

The R9700 over USB4 hits an SMU firmware version mismatch that causes the GPU to hang after boot. The amdgpu driver's runtime power management triggers a fatal D3cold resume failure.

Fix it by disabling the overdrive power feature and runtime PM:

```bash
sudo rpm-ostree kargs \
  --replace=amdgpu.ppfeaturemask=0xfffd7fff \
  --append=amdgpu.runpm=0
```

Reboot. Then verify the GPU is alive:

```bash
# Must show D0 (not D3cold)
cat /sys/bus/pci/devices/0000:08:00.0/power_state

# Must show both GPUs with utilization
rocm-smi
```

## Step 2: Install Homebrew dependencies

Bazzite is immutable — you can't `dnf install` into the base OS. Use Homebrew (pre-installed on Bazzite) for build tools:

```bash
brew install cmake vulkan-headers shaderc
```

## Step 3: Build llama.cpp with Vulkan

```bash
cd ~/gpu-setup
./install.sh
```

This clones llama.cpp, builds it with `-DGGML_VULKAN=ON` targeting the RADV Mesa driver, and produces `llama.cpp/build/bin/llama-server`.

## Step 4: Download a model

Grab any GGUF from [HuggingFace](https://huggingface.co/models?sort=trending&search=gguf). The R9700 has 32 GB VRAM — models up to ~28 GB in Q4 quantization fit comfortably.

```bash
mkdir -p ~/gpu-setup/data/models
cd ~/gpu-setup/data/models
wget https://huggingface.co/bartowski/Nemotron-Cascade-2-30B-A3B-GGUF/resolve/main/Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf
```

Then point the server at it:

```bash
LLAMA_MODEL=~/gpu-setup/data/models/Nemotron-Cascade-2-30B-A3B-Q4_K_M.gguf ./start.sh
```

Or set it as the default by editing the `DEFAULT_MODEL` path in `stack.sh`.

## Step 5: Start the server

```bash
./start.sh
```

The server starts on `http://localhost:8000` with an OpenAI-compatible API at `/v1`. Any tool that speaks the OpenAI protocol works — Open WebUI, Hermes Agent, Continue, SillyTavern, etc.

The start script runs a **watchdog** — if the server crashes (e.g. GPU ring timeout), it automatically restarts. No manual intervention needed.

```bash
./stack.sh verify
```

## Step 6 (optional): Set up Hermes Agent

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a self-improving AI agent by Nous Research that connects to any OpenAI-compatible endpoint.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc

hermes model
# Select: Custom endpoint
# API base URL: http://localhost:8000/v1
# API key: none
# Model name: (leave default)
# Context length: 8192

hermes
```

## Scripts

| Script | What it does |
|---|---|
| `./install.sh` | Build llama.cpp with Vulkan |
| `./start.sh` | Start the inference server |
| `./stop.sh` | Stop it |
| `./stack.sh status` | Check if it's running |
| `./stack.sh verify` | Test inference end-to-end |
| `./stack.sh logs` | Tail the server log |
| `./stack.sh models` | List downloaded model files |

## Environment variables

| Variable | Default | What it controls |
|---|---|---|
| `LLAMA_MODEL` | (built-in default) | Path to the GGUF model file |
| `LLAMA_PORT` | `8000` | Server port |
| `LLAMA_CTX` | `131072` | Context window size (Gemma-4 supports up to 262144) |
| `LLAMA_GPU_DEVICE` | `1` | Vulkan device index (`0` = iGPU, `1` = R9700) |

## Stability tuning

The server ships with flags tuned for Gemma-4's hybrid architecture (Gated Delta Net + SWA attention):

| Flag | Value | Why |
|---|---|---|
| `--swa-full` | full-size SWA KV cache | Prevents checkpoint invalidation on hybrid models |
| `-np 1` | 1 parallel slot | Single concurrent request, minimizes VRAM pressure |
| `-b 2048` | batch size 2048 | Default prefill batch |
| `-ub 512` | ubatch size 512 | Default physical batch |
| `--cache-ram 0` | disabled prompt cache | SWA layers make checkpoints useless; saves 2-4s per request |

## Why Vulkan instead of ROCm

As of April 2026, every ROCm-based inference stack fails on the R9700 (gfx1201):

- **Ollama** — The HIP runtime hangs during GPU discovery for 30 seconds then falls back to CPU. Open issue: [ollama#13236](https://github.com/ollama/ollama/issues/13236). Tested with `ollama/ollama:rocm` and the community `rjmalagon/ollama-linux-amd-apu` image (ROCm 7.2). Both timeout identically.
- **vLLM** — The `kyuz0/vllm-therock-gfx1201` container (built on TheRock ROCm nightlies for R9700) crashes with `double free or corruption` during engine core initialization. Tested with gpt-oss-20b, Qwen3-14B, and Llama 3.1 8B — all crash the same way.
- **Ollama in containers** — Any container that passes `/dev/kfd` to the GPU causes the process to hang and become an unkillable zombie. The container cannot be stopped or removed — only a host reboot clears it.

**llama.cpp with Vulkan** uses the open-source RADV Mesa driver and avoids the entire ROCm/HIP stack. It works immediately with no driver installation beyond what Bazzite ships.

Community benchmarks on the R9700 ([source](https://github.com/ggml-org/llama.cpp/discussions/21043)):

| Model | Decode | Prefill |
|---|---|---|
| Qwen3.5-35B-A3B MoE (Q4_K_XL) | 148 tok/s | 2,400 tok/s |
| Qwen3.5-27B Dense (Q4_K_M) | 29 tok/s | 800 tok/s |

## Known issues

**GPU ring timeout during long inference** — The R9700 can hit a GFX ring timeout during sustained inference. The kernel logs `ring gfx_0.0.0 timeout` followed by a failed GPU reset. This is an amdgpu kernel driver issue with gfx12 over USB4. The start script runs a watchdog that automatically restarts the server when this happens.

**iGPU blocks Vulkan enumeration** — The Ryzen Z1 Extreme's integrated GPU (Phoenix, gfx1103) has a broken `amdgpu_device_initialize` call that hangs any process that enumerates Vulkan devices (including `vulkaninfo` and `llama-server --list-devices`). The start script skips device enumeration and uses `GGML_VK_VISIBLE_DEVICES` to target the R9700 directly. If you need to run `vulkaninfo`, it will hang — this is a known libdrm issue with the Phoenix iGPU on Bazzite.

**Container GPU passthrough** — Running llama-server inside a container requires the container's Mesa/libdrm to support gfx1201. The official `ghcr.io/ggml-org/llama.cpp:server-vulkan` image ships an older Mesa that doesn't recognize the R9700. A custom Fedora 43 container with the host-matching `mesa-vulkan-drivers` works, but adds complexity. Native execution is simpler.

## Troubleshooting

**GPU not detected after boot** — Check `cat /sys/bus/pci/devices/0000:08:00.0/power_state`. If it shows `D3cold`, the kernel params from Step 1 aren't applied. Verify with `cat /proc/cmdline | grep amdgpu`.

**GPU stuck after crash** — If `rocm-smi` shows the GPU but inference hangs or produces garbage, check `journalctl -k -b | grep "amdgpu.*08:00"` for ring timeout or GPU reset messages. Reboot to recover.

**Server won't start / "cannot open shared library"** — Run `./install.sh` to rebuild. The server depends on shared libraries in `llama.cpp/build/bin/` — the start script sets `LD_LIBRARY_PATH` automatically.

**Server picks the wrong GPU** — The iGPU is Vulkan0, the R9700 is Vulkan1. Set `LLAMA_GPU_DEVICE=1` (default).

**"error loading model hyperparameters"** — The model architecture is too new for the current llama.cpp build. Run `./install.sh` to rebuild from latest source.
