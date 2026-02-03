# LEAP (Loss-tolerant, Energy-aware Asymmetric Pipeline)

LEAP is a high-performance, distributed Large Language Model (LLM) inference engine written in C++20. It enables the execution of massive models (e.g., Llama 3 70B) on clusters of consumer-grade devices or heterogeneous servers by splitting the model layer-wise across a **Ring Topology**.

## Core Achievement
LEAP solves the "VRAM Wall" problem. Instead of requiring a single expensive GPU with massive memory (like an H100), LEAP pipelines the inference process across multiple smaller devices connected via standard Ethernet. It achieves near-native performance through aggressive optimizations:
*   **Zero-Copy Networking**: A custom Linux Kernel Module (`leap_kmod`) bypasses the OS network stack.
*   **SIMD Acceleration**: Hand-written AVX2 (x86) and NEON (ARM) kernels for maximum CPU throughput.
*   **Asymmetric Pipelining**: Supports nodes with different compute capabilities working in unison.

## Use Cases
*   **Home Lab Clusters**: Chain together spare hardware (MacBooks, Gaming PCs, Mini PCs) to run state-of-the-art 70B+ models that no single device could fit.
*   **Edge AI**: Deploy powerful intelligence on a stack of embedded devices (e.g., NVIDIA Jetson, Rockchip) in environments with limited power and connectivity.
*   **Privacy-Focused Local Inference**: Run sensitive workloads entirely on-premise without relying on cloud APIs or expensive enterprise hardware.
*   **Cost-Effective Scaling**: Utilize fragmented, heterogeneous resources in a data center without needing expensive interconnects like InfiniBand or NVLink.

---

LEAP prioritizes low latency and high throughput through aggressive optimizations, including SIMD (AVX2/NEON), custom quantization, and a specialized Linux Kernel Module for zero-copy networking.

## 1. Project Architecture

LEAP consists of four main components:

1. **Exporter**: Converts PyTorch/Safetensors models into LEAP's optimized binary format.
2. **Tokenizer**: A standalone, pure C++ tokenizer (BPE) compatible with Llama 3 / Tiktoken.
3. **Inference Engine**: The core runtime supporting Float32 and Int8 Quantized execution.
4. **Transport Layer**: A modular networking interface supporting TCP, UDP, and Kernel-Bypass modes.

---

## 2. Exporter (`src/export/`)

The exporter is responsible for preparing models for inference. It reads `.safetensors` checkpoints, transforms weights
into the correct memory layout, and serializes them.

### Features

* **Source Format**: Supports HuggingFace-style `.safetensors` (sharded or single file).
* **Weight Transformation**: Automatically permutes weights for Rotary Positional Embeddings (RoPE) to match the
  inference engine's expected interleaved format.
* **Quantization**: Supports post-training quantization to Int8 (Symmetric).

### Output Formats

* **Float32 (`float32_export`)**:
    * **Magic**: `0x616B3432` ("ak42")
    * **Version**: `1`
    * **Content**: Full precision weights.
* **Int8 (`int8_export`)**:
    * **Magic**: `0x616B3432`
    * **Version**: `2`
    * **Quantization Scheme**: Symmetric Block-wise Quantization.
        * **Group Size**: Configurable (default 64). If `dim` is not divisible, it backs off to 32.
        * **Storage**: Weights stored as `int8` + `float` scale per block.
        * **Norms**: Layer norms and output norms are kept in FP32 for stability.
    * **Pipeline**: Uses a lookahead pipeline (Async Quantize + Sync Write) to maximize export speed.

---

## 3. Tokenizer (`src/tokenizer/` & `src/inference/Tokenizer.cpp`)

LEAP uses a custom BPE tokenizer implementation designed for zero-dependency inference.

### Training/Export (`src/tokenizer/`)

* Wraps the `tiktoken` C++ library to load BPE models (like Llama 3's `tokenizer.model`).
* **Special Tokens**: Explicitly handles Llama 3 special tokens (`<|begin_of_text|>`, `<|eot_id|>`, etc.) and reserves
  slots for fine-tuning tokens.
* **Output Format**: A simple binary format containing the vocabulary, scores, and token strings.

### Inference (`src/inference/`)

* **Pure C++**: No runtime dependency on Python or HuggingFace libraries.
* **Algorithm**: Implements BPE merge using a **Priority Queue** (O(N log N)) to greedily merge adjacent token pairs
  based on score/rank.
* **Byte Fallback**: Supports raw byte tokens (`<0xHH>`) for robust handling of arbitrary binary data/utf-8 fragments.

---

## 4. Inference Engine (`src/inference/`)



The core engine implements a standard Llama-2/3 architecture: RMSNorm, SwiGLU FeedForward, and Rotary Positional Embeddings (RoPE). It is built for maximum efficiency on CPU, utilizing two distinct transformer implementations.



### 4.1. FloatTransformer (Pure FP32)

The `FloatTransformer` is the baseline implementation offering maximum precision. It is designed around zero-copy memory mapping and rigorous loop optimization.



#### Architecture & Memory Layout

*   **Zero-Copy Loading**: The model weights are accessed directly via `mmap`. The file is mapped into the process's virtual address space, and pointers (`FloatTransformerWeights`) are set to specific offsets. This eliminates parsing overhead and allows the OS to manage page caching.

*   **State Management**: To prevent dynamic memory allocation during the inference loop, a `FloatRunState` structure pre-allocates all necessary buffers:

    *   `x`, `xb`: Activation buffers (size `dim`).

    *   `q`, `k`, `v`: Query/Key/Value vectors.

    *   `key_cache`, `value_cache`: KV Cache for past tokens (size `n_layers * seq_len * kv_dim`).

    *   `att`, `logits`: Attention scores and output probabilities.



#### Execution Flow

For every token, the engine performs the following operations per layer:

1.  **RMSNorm**: Normalizes the input vector `x`.

    *   *Optimization*: Uses SIMD (AVX2/NEON) to compute the sum of squares. A Newton-Raphson iteration is applied to refine the reciprocal square root estimate (`vrsqrteq_f32` / `_mm_rsqrt_ss`) for high precision without the cost of a full `sqrt` call.

2.  **QKV Projection**: FP32 Matrix Multiplication to generate Query, Key, and Value vectors.

3.  **RoPE (Rotary Positional Embeddings)**:

    *   Rotates `Q` and `K` vectors to encode position information.

    *   *Optimization*: Pairs of elements `(x, y)` are transformed into `(x*cos - y*sin, x*sin + y*cos)`. This is fully vectorized using SIMD shuffles (`vrev64q_f32` on NEON, `_mm256_permute_ps` on AVX2) to process multiple heads simultaneously.

4.  **Multi-Head Attention (MHA/GQA)**:

    *   **Score**: `Attn = Q @ K.T` (Dot product).

    *   **Softmax**: Exponentiates and normalizes scores.

    *   **Weighted Sum**: `Output = Attn @ V`.

5.  **Feed Forward (SwiGLU)**:

    *   Computes `Gate = w1(x)` and `Up = w3(x)`.

    *   Applies **SiLU** activation: `val = val * (1 / (1 + exp(-val)))`.

    *   Computes `Down = w2(Gate * Up)`.

    *   Residual connection: `x += Down`.



### 4.2. QuantizedTransformer (W8A8 Dynamic)

The `QuantizedTransformer` reduces memory usage by 4x and increases compute throughput by utilizing integer arithmetic. It implements a **Weight-Int8, Activation-Int8 (W8A8)** scheme with dynamic activation quantization.



#### Quantization Scheme

*   **Block-wise Quantization**: To maintain accuracy, weights are not quantized globally but in small groups (default `group_size=64`).

*   **QuantizedTensor Structure**:

    *   `int8_t* q`: Pointer to the raw compressed 8-bit integer weights.

    *   `float* s`: Pointer to the scaling factors (one float per `group_size` weights).

*   **Dynamic Activation Quantization**:

    *   Before every matrix multiplication, the input activation vector `x` (FP32) is dynamically quantized to Int8.

    *   The range of `x` is measured, and a scale factor is computed to map floats to `[-127, 127]`.



#### Integer Matrix Multiplication (GEMM)

The critical performance driver is the integer dot product.

1.  **Operation**: The dot product of a quantized weight vector `w` and activation `x` is computed as:

    $ \text{Result} = \sum (w_i \cdot x_i) \cdot s_w \cdot s_x $

    Where $w_i, x_i$ are `int8`, and $s_w, s_x$ are `float` scales.

2.  **SIMD Kernels**:

    *   **ARM NEON**: Uses the `vdotq_s32` instruction (Dot Product), which performs 4-way int8 multiplication and accumulation into a 32-bit integer in a single cycle.

    *   **AVX2**: Uses `_mm256_madd_epi16` (Multiply-Add). It expands int8s to int16s, multiplies them, and adds adjacent pairs, reducing register pressure.

3.  **Dequantization**: The accumulated 32-bit integer result is converted back to FP32 and multiplied by the combined scale factor (`weight_scale * activation_scale`) to recover the true value.



#### Performance Benefits

*   **Memory Bandwidth**: Fetching 1 byte per weight instead of 4 bytes relieves the bottleneck on memory-bound CPUs.

*   **Cache Efficiency**: 4x more parameters fit into L1/L2/L3 caches.

*   **Compute**: Integer MAC (Multiply-Accumulate) instructions often have higher throughput and lower latency than their FP32 counterparts.

---

## 5. Distributed Transport (`src/inference/Transport*`)

LEAP supports distributed inference via a **Ring Topology**. The model is split layer-wise across multiple nodes.

* **Flow**: `Prev Node -> [Receive] -> Compute Layers -> [Send] -> Next Node`.
* **Protocol**: A lightweight binary protocol sending a `PacketHeader` followed by the raw activation tensor (embedding
  size * float32).

### Transport Modes

#### 1. TCP (`TcpTransport`)

* **Reliability**: Guaranteed delivery via standard TCP/IP.
* **Optimization**:
    * `TCP_NODELAY`: Nagle's algorithm disabled for lowest latency.
    * **Large Buffers**: 8MB Socket buffers (`SO_RCVBUF`/`SO_SNDBUF`).
    * **Multipart Send**: Uses `writev` to send Header + Data in a single syscall to avoid copying.

#### 2. UDP (`UdpTransport`)

* **Type**: Best-effort datagram transport.
* **Chunking**: Splits large tensors into `LEAP_CHUNK_SIZE` (1400 bytes) packets to avoid IP fragmentation.
* **Reassembly**: Receiver tracks `seq_id` and `chunk_id` to reassemble frames.
* **Note**: Currently lacks retransmission logic; assumes a high-quality LAN environment.

#### 3. Kernel Module (`src/kernel/` & `KernelTransport`)

The **LEAP Kernel Module (`leap_kmod`)** is the high-performance heart of the distributed pipeline. It implements a **Zero-Copy, Kernel-Bypass** networking stack specifically tailored for tensor transmission. By intercepting packets before the standard Linux network stack processes them, it drastically reduces latency and CPU context switches.

##### 3.1. Core Architecture
The module exposes a character device (`/dev/leap_tensor`) that userspace maps into memory.
*   **Split Ring Buffer (16MB)**:
    *   **Lower 8MB (RX Banks)**: Divided into 64 "Banks" (128KB each). This allows up to 64 concurrent packet streams (or out-of-order sequences) to be reassembled in parallel without locking the entire buffer.
    *   **Upper 8MB (TX Buffer)**: A contiguous staging area for outgoing tensors.
*   **Synchronization**:
    *   **Bitmaps & Spinlocks**: Each RX bank is protected by a fine-grained spinlock (`rx_locks`) and uses a bitmap to track received chunks. This ensures lock contention is minimized even under high packet loads.
    *   **Wait Queues**: Userspace threads sleep on a wait queue (`leap_wait_queue`) and are woken up *only* when a full bank is complete, avoiding busy-waiting.

##### 3.2. The Zero-Copy Data Path (RX)
1.  ** interception**: The module registers a Netfilter hook at `NF_INET_PRE_ROUTING` with highest priority (`NF_IP_PRI_FIRST`).
2.  **Packet Inspection**: Every incoming UDP packet is inspected. If it matches the configured `listening_port` and contains the `LEAP_MAGIC` header:
    *   The kernel extracts `seq_id` and `chunk_id`.
    *   It calculates the exact offset in the memory-mapped RX buffer: `Bank_Idx * Bank_Size + Chunk_Id * Chunk_Size`.
3.  **Direct DMA Copy**: The payload is copied *directly* from the socket buffer (`skb`) to the userspace-mapped RAM.
4.  **Verdict**: The hook returns `NF_STOLEN`, telling the OS to stop processing the packet immediately. The standard network stack never sees it.
5.  **Completion Notification**: Once all chunks for a bank are received, the module sets a bit in the `data_ready_map` and wakes up the userspace process.

##### 3.3. The Zero-Copy Data Path (TX)
1.  **Preparation**: Userspace writes the tensor data + header into the mapped **TX Buffer** (Upper 8MB).
2.  **Trigger**: Userspace calls `ioctl(LEAP_IOCTL_SEND)`.
3.  **Kernel Transmission**:
    *   The kernel constructs UDP/IP headers pointing to the data already in RAM.
    *   It uses `kernel_sendmsg` with a `kvec` pointing to the userspace buffer, allowing the NIC to DMA read directly from the application's memory view (Zero-Copy).

##### 3.4. IOCTL Interface
Control flow is managed via `ioctl`:
*   `LEAP_IOCTL_WAIT_DATA`: Sleep until a bank is ready. Returns the `bank_idx`.
*   `LEAP_IOCTL_SEND`: Trigger transmission of data in the TX buffer.
*   `LEAP_IOCTL_SET_DEST`: Configure the destination IP for the ring topology.
*   `LEAP_IOCTL_SET_PORT`: Bind to a specific local UDP port.
*   `LEAP_IOCTL_GET_BANK_SRC`: Retrieve the source IP/Port of the sender (for auto-discovery).

**Usage**: Requires `insmod src/kernel/leap_transport.ko`. This mode provides the lowest possible latency for Ethernet-based clusters.

---

## 6. Build Instructions

### Prerequisites
*   CMake 3.25+
*   C++20 Compiler (GCC 10+, Clang 12+, MSVC 19.30+)
*   OpenMP
*   **LibTorch** (Required for the `export` tool)

### 1. Clone the Repository
Clone the repository recursively to fetch all dependencies (e.g., `nlohmann_json`, `safetensors`).
```bash
git clone --recursive https://github.com/Harikeshav-R/LEAP.git
cd LEAP
```

### 2. Obtain LibTorch (C++ PyTorch)
The Exporter tool relies on LibTorch to read tensors. **Important: You only need the CPU version.**

#### Option A: Direct Download (x86_64)
Download the cxx11 ABI **CPU** version from the [PyTorch Website](https://pytorch.org/get-started/locally/). Extract it to `third-party/libtorch`.
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip
unzip libtorch-*.zip -d third-party/
```

#### Option B: Via Python (Recommended for systems that do not have a pre-built libtorch binary available)
If a pre-built LibTorch binary isn't available for your architecture (e.g., Raspberry Pi, Jetson Orin), install the **CPU** version of PyTorch via pip and point CMake to it.
```bash
# Create a virtual env (optional but recommended)
python3 -m venv venv
source venv/bin/activate
# Install CPU-only torch to save space and avoid CUDA dependency issues
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Build the Project

#### Configure CMake
If you used **Option A** (Direct Download):
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(pwd)/third-party/libtorch
```

If you used **Option B** (Python):
```bash
# Get the torch path
TORCH_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$TORCH_PATH
```

#### Compile
```bash
cmake --build build --config Release -- -j$(nproc)
```

### 4. Build with Kernel Module (Linux Only)
To enable the zero-copy kernel transport:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_KERNEL_MODULE=ON [CMAKE_PREFIX_PATH args...]
cmake --build build --config Release -- -j$(nproc)
```
*Note: You will need Linux kernel headers installed (e.g., `sudo apt install linux-headers-$(uname -r)`).*

## 7. Usage



### 1. Export Tool (`export`)

Converts PyTorch/Safetensors checkpoints into the LEAP binary format.



**Usage:**

```bash

./export <output_filepath> --meta-llama <path_to_model_dir> [options]

```



**Arguments:**

*   `output_filepath`: The destination path for the `.bin` model file.

*   `--meta-llama <path>`: (Required) Path to the directory containing the Llama model (`params.json` and `.safetensors` files).

*   `--version <1|2>`:

    *   `1`: Float32 export (Default).

    *   `2`: Int8 Quantized export.



**Example:**

```bash

# Export Llama-3-8B to Int8

./export llama3-8b-int8.bin --meta-llama /models/Meta-Llama-3-8B --version 2

```



### 2. Tokenizer Tool (`tokenizer`)

Converts a HuggingFace/Tiktoken tokenizer model (usually `tokenizer.model`) into LEAP's binary format.



**Usage:**

```bash

./tokenizer <path_to_tokenizer.model>

```

*   The output will be saved as `tokenizer.bin` in the same directory as the input.



### 3. Inference Engine (`inference`)

The main runtime for generating text. Supports single-node and distributed ring inference.



**Usage:**

```bash

./inference --model <path> [options]

```



#### General Options

*   `--model <path>`: (Required) Path to the exported LEAP model (`.bin`).

*   `-t, --tokenizer <path>`: Path to the tokenizer file (default: `tokenizer.bin`).

*   `-p, --prompt <text>`: Initial prompt to start generation.

*   `-c, --chat`: Enable interactive chat mode (Llama 3 instruction template).

*   `-s, --system <text>`: System prompt (only for chat mode).

*   `-n, --n-predict <int>`: Maximum number of tokens to generate (default: 4096).



#### Sampling Parameters

*   `--temp <float>`: Temperature for sampling (default: 1.0). Higher = more creative, Lower = more deterministic.

*   `--top-p <float>`: Top-P (Nucleus) sampling threshold (default: 0.9).

*   `--seed <int>`: Random seed for reproducibility (default: 0 = random).



#### Distributed Inference Options

*   `--role <single|master|worker>`: Node role (default: `single`).

    *   `single`: Runs the entire model locally.

    *   `master`: Runs the *first* part of the model and manages the prompt/generation loop.

    *   `worker`: Runs the *subsequent* parts of the model in a loop.

*   `--split <int>`:

    *   **Master**: Layer index where the model is split (Master runs 0 to `split`-1).

    *   **Worker**: Layer index to start processing from.

*   `--end <int>`: (Worker only) Layer index to stop processing at (exclusive). 0 = process until the last layer.

*   `--transport <tcp|udp|kernel>`: Transport protocol (default: `tcp`).

*   `--host <ip>`: Local interface to bind to (default: `0.0.0.0`).

*   `--port <int>`: Local port to listen on (default: `9999`).

*   `--next-host <ip>`: IP address of the *next* node in the ring (Required for distributed).

*   `--next-port <int>`: Port of the *next* node in the ring (default: `9999`).



#### Distributed Example (2 Nodes, split at layer 16)

**Master Node (192.168.1.1):**

```bash

./inference --model model.bin --role master --split 16 --next-host 192.168.1.2 --prompt "Once upon a time"

```



**Worker Node (192.168.1.2):**

```bash

./inference --model model.bin --role worker --split 16 --next-host 192.168.1.1

```