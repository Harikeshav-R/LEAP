# LEAP (Loss-tolerant, Energy-aware Asymmetric Pipeline)

LEAP is a high-performance, distributed Large Language Model (LLM) inference engine written in C++20. It enables the execution of massive models (e.g., Llama 3 70B) on clusters of consumer-grade devices or heterogeneous servers by splitting the model layer-wise across a **Ring Topology**.

## Core Achievement

LEAP solves the "VRAM Wall" problem. Instead of requiring a single expensive GPU with massive memory (like an H100), LEAP pipelines the inference process across multiple smaller devices connected via standard Ethernet. It achieves near-native performance through aggressive optimizations:
*   **Zero-Copy Networking**: A custom Linux Kernel Module (`leap_kmod`) bypasses the OS network stack.
*   **SIMD Acceleration**: Hand-written AVX2 (x86) and NEON (ARM) kernels for maximum CPU throughput.
*   **Asymmetric Pipelining**: Supports nodes with different compute capabilities working in unison.
*   **Dynamic Load Balancing**: Runtime `/resize` commands to redistribute layers across workers without restart.

## Use Cases
*   **Home Lab Clusters**: Chain together spare hardware (MacBooks, Gaming PCs, Mini PCs) to run state-of-the-art 70B+ models that no single device could fit.
*   **Edge AI**: Deploy powerful inference on a stack of embedded devices (e.g., NVIDIA Jetson, Rockchip) in environments with limited power and connectivity.
*   **Privacy-Focused Local Inference**: Run sensitive workloads entirely on-premise without relying on cloud APIs or expensive enterprise hardware.
*   **Cost-Effective Scaling**: Utilize fragmented, heterogeneous resources in a data center without needing expensive interconnects like InfiniBand or NVLink.

---

## Architecture

LEAP consists of five components that form an end-to-end pipeline:

```
┌──────────────┐    ┌──────────────┐    ┌────────────────────────────────────┐
│   Exporter   │    │  Tokenizer   │    │        Inference Engine            │
│              │    │              │    │                                    │
│ Safetensors  │───▶│ tiktoken     │───▶│  FloatTransformer (FP32)          │
│ → LEAP .bin  │    │ → tokenizer  │    │  QuantizedTransformer (W8A8 INT8) │
│ (FP32/INT8)  │    │   .bin       │    │  Transport (TCP/UDP/Kernel)       │
└──────────────┘    └──────────────┘    └────────────────────────────────────┘
                                                        │
                                                 Ring Topology
                                                        │
                                         ┌──────────────┴──────────────┐
                                         │     Worker Nodes (N-1)      │
                                         │  Each runs assigned layers  │
                                         └─────────────────────────────┘
```

1. **Exporter** (`src/export/`) — Converts PyTorch/Safetensors models into LEAP's optimized binary format (FP32 or INT8 quantized).
2. **Model Library** (`src/model/`) — LibTorch-based Llama architecture used by the exporter for weight loading and manipulation.
3. **Tokenizer** (`src/tokenizer/`) — Standalone tokenizer export tool (Tiktoken → binary). The inference engine has its own dependency-free BPE implementation.
4. **Inference Engine** (`src/inference/`) — The core runtime with SIMD-optimized transformer implementations, sampling, and interactive chat mode.
5. **Transport Layer** (`src/inference/Transport*` & `src/kernel/`) — Modular networking supporting TCP, UDP, and a zero-copy Linux Kernel Module.

> 📖 **Detailed Documentation**: See the [docs/](docs/) directory for in-depth technical breakdowns of each component.

### Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | High-level system design, data flow, and component relationships |
| [Exporter](docs/exporter.md) | Model conversion pipeline, binary formats, and quantization scheme |
| [Inference Engine](docs/inference-engine.md) | SIMD kernels, forward pass, quantized inference, sampler, and chat mode |
| [Transport Layer](docs/transport.md) | TCP, UDP, and Kernel transport implementations and protocol details |
| [Kernel Module](docs/kernel-module.md) | Zero-copy Linux kernel module internals (Netfilter, mmap, ioctl) |
| [Model Library](docs/model-library.md) | LibTorch-based Llama architecture (Attention, FFN, RMSNorm) |
| [Tokenizer](docs/tokenizer.md) | BPE tokenizer export and inference implementations |
| [Build System](docs/build-system.md) | CMake configuration, dependencies, and platform-specific setup |

---

## Supported Models

LEAP is designed for **Llama Architecture** models:
*   **Supported**: Llama 2, Llama 3, Llama 3.1, CodeLlama, TinyLlama
*   **Key Features**: Grouped Query Attention (GQA), RoPE, RMSNorm, SwiGLU

### Precision Modes

| Mode | Memory per Param | SIMD Kernels | Use Case |
|------|-----------------|-------------|----------|
| FP32 | 4 bytes | AVX2 / NEON | Maximum accuracy |
| INT8 (W8A8) | ~1.25 bytes | `vdotq_s32`, `_mm256_madd_epi16` | 4× memory reduction, higher throughput |

---

## Quick Start

### Prerequisites
*   CMake 3.25+
*   C++20 Compiler (GCC 10+, Clang 12+, MSVC 19.30+)
*   OpenMP (`brew install libomp` on macOS)
*   **LibTorch** (Required for the `export` and `tokenizer` tools)

### 1. Clone

```bash
git clone --recursive https://github.com/Harikeshav-R/LEAP.git
cd LEAP
```

### 2. Obtain LibTorch

**Direct Download (x86_64):**
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip
unzip libtorch-*.zip -d third-party/
```

**Via Python (Recommended for ARM):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Build

```bash
# Direct download:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(pwd)/third-party/libtorch

# Via Python:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')

# Compile
cmake --build build --config Release -- -j$(nproc)
```

### 4. Build with Kernel Module (Linux Only)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_KERNEL_MODULE=ON [CMAKE_PREFIX_PATH args...]
cmake --build build --config Release -- -j$(nproc)
```

> See [Build System](docs/build-system.md) for detailed platform-specific instructions and troubleshooting.

---

## Usage

### Export a Model

```bash
# FP32 export
./export llama3-8b.bin --meta-llama /models/Meta-Llama-3-8B

# INT8 quantized export
./export llama3-8b-q8.bin --meta-llama /models/Meta-Llama-3-8B --version 2
```

### Export the Tokenizer

```bash
./tokenizer /models/Meta-Llama-3-8B/tokenizer.model -o tokenizer.bin
```

### Single-Node Inference

```bash
# Chat mode
./inference -c -p "Hello!" model.bin

# Completion mode
./inference -p "Once upon a time" --temp 0.8 --top-p 0.9 model.bin
```

### Distributed Inference (Ring Topology)

**2-Node Setup** (split at layer 16):

```bash
# Master (192.168.1.1) — runs layers 0-15
./inference --role master --split 16 --next-host 192.168.1.2 -c model.bin

# Worker (192.168.1.2) — runs layers 16-31
./inference --role worker --split 16 --next-host 192.168.1.1 model.bin
```

**3-Node Setup** (split at layers 11 and 22):

```bash
# Master (192.168.1.1) — layers 0-10
./inference --role master --split 11 --next-host 192.168.1.2 -c model.bin

# Worker 1 (192.168.1.2) — layers 11-21
./inference --role worker --split 11 --end 22 --next-host 192.168.1.3 model.bin

# Worker 2 (192.168.1.3) — layers 22-31 (tail)
./inference --role worker --split 22 --next-host 192.168.1.1 model.bin
```

### Transport Options
```bash
--transport tcp     # Default, reliable
--transport udp     # Lower latency, LAN only
--transport kernel  # Zero-copy, Linux only (requires insmod leap_transport.ko)
```

### Runtime Commands (Chat Mode)
```
/layers           — Show current layer distribution
/resize 8 16      — Redistribute layers (single worker)
/resize 8 12 16   — Redistribute layers (multi-worker chain)
/help             — Show available commands
exit              — Exit chat
```

---

## CLI Reference

### `export`

```
./export <output_path> --meta-llama <model_dir> [--version <1|2>]
```

### `tokenizer`

```
./tokenizer <model_path> [-o <output_path>]
```

### `inference`

```
./inference [options] <model_path>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --tokenizer` | `tokenizer.bin` | Path to tokenizer file |
| `-p, --prompt` | — | Input prompt |
| `-c, --chat` | off | Interactive chat mode |
| `-s, --system` | — | System prompt (chat mode) |
| `-n, --n-predict` | 4096 | Max tokens |
| `--temp` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--seed` | 0 (random) | RNG seed |
| `--role` | `single` | Node role: `single`, `master`, `worker` |
| `--split` | 0 | Start layer index |
| `--end` | 0 (all) | End layer index (exclusive) |
| `--transport` | `tcp` | Transport: `tcp`, `udp`, `kernel` |
| `--host` | `0.0.0.0` | Local bind address |
| `--port` | 9999 | Local bind port |
| `--next-host` | — | Next node IP (required for distributed) |
| `--next-port` | 9999 | Next node port |

---

## Performance Tuning

### OpenMP Thread Pinning

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
./inference ...
```

> On hybrid CPUs (Intel 12th+ Gen), bind threads to P-cores for best results.

### Kernel Module Tuning

```bash
# Increase busy-wait limit for ultra-low latency
insmod leap_transport.ko busy_wait_limit=10000
```

---

## Benchmarking

To ensure LEAP performs efficiently across heterogeneous hardware, we benchmarked the inference engine using a distributed ring topology.

**Test Setup:**
*   **Model:** Llama 3.2 11B Instruct
*   **Master Node:** MacBook Pro (M3 Pro (11 CPU/14 GPU), 18GB RAM) – macOS
*   **Worker 1:** Linux VM on Host (ARM64, 4 vCPUs, 4GB RAM)
*   **Worker 2:** Raspberry Pi 3B+ (Cortex-A53, 1GB RAM) – Connected via Ethernet

### 1. Kernel Transport (Linux-Only Zero-Copy)
*The most efficient mode, bypassing the kernel network stack overhead.*

| METRIC | MEAN | MEDIAN | MIN | MAX | STD DEV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Throughput | 22.20 | 22.19 | 15.22 | 24.44 | 1.41 |
| Latency (s) | 2.47 | 2.45 | 2.24 | 3.49 | 0.17 |

### 2. UDP Transport (User-Space)
*Lower overhead than TCP, but subject to context switching costs.*

| METRIC | MEAN | MEDIAN | MIN | MAX | STD DEV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Throughput | 18.50 | 18.48 | 14.10 | 19.95 | 1.25 |
| Latency (s) | 2.96 | 2.95 | 2.75 | 3.88 | 0.22 |

### 3. TCP Transport (Default)
*Standard reliable delivery, incurs highest protocol overhead.*

| METRIC | MEAN | MEDIAN | MIN | MAX | STD DEV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Throughput | 17.80 | 17.75 | 13.50 | 18.90 | 1.10 |
| Latency (s) | 3.08 | 3.06 | 2.90 | 4.10 | 0.18 |

> 📖 **Full Guide**: See [docs/benchmarking.md](docs/benchmarking.md) for instructions on how to replicate these tests.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| LibTorch ABI errors (`std::__cxx11::basic_string`) | Use the **cxx11 ABI** version of LibTorch |
| Kernel module `Operation not permitted` | Disable Secure Boot or sign the module |
| Kernel module `Exec format error` | Rebuild for current kernel: `sudo apt install linux-headers-$(uname -r)` |
| Network unreachable (UDP/Kernel) | Check firewall: `ufw status`, ensure UDP port 9999 is open |
| OpenMP not found (macOS) | `brew install libomp` |

---

## License

See [LICENSE](LICENSE) for details.
