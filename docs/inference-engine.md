# Inference Engine (`src/inference/`)

The inference engine is the core runtime of LEAP. It loads exported models via `mmap`, runs the transformer forward pass with SIMD-optimized kernels, and supports both single-node and distributed execution across a ring of networked nodes.

## Source Files

| File | Purpose |
|------|---------|
| `main.cpp` | CLI entry point, chat/generate loops, slash commands |
| `FloatTransformer.h/cpp` | FP32 forward pass with AVX2/NEON SIMD |
| `QuantizedTransformer.h/cpp` | W8A8 INT8 forward pass with SIMD |
| `TransformerFactory.cpp` | Model loading, version detection, mmap management |
| `Transformer.h` | Base class, distributed config, control messages |
| `Config.h` | Model dimensions struct |
| `Tokenizer.h/cpp` | Pure C++ BPE tokenizer for inference |
| `Sampler.h/cpp` | Token sampling strategies |
| `Transport.h` | Abstract transport interface |
| `TcpTransport.h/cpp` | TCP ring transport |
| `UdpTransport.h/cpp` | UDP chunked transport |
| `KernelTransport.h/cpp` | Zero-copy kernel module transport |
| `Utils.h` | Timing, RNG, safe print utilities |

---

## Model Loading (`TransformerFactory.cpp`)

The factory method `Transformer::create()` handles:

1. **Header Parsing** — Reads the magic number (`0x616B3432`) and version to determine FP32 (v1) or INT8 (v2).
2. **Memory Mapping** — Uses `ScopedMmap` (RAII wrapper) to `mmap` the entire file with:
   - `MADV_SEQUENTIAL` — Hints for sequential page access.
   - `MADV_WILLNEED` — Triggers kernel readahead.
   - `MADV_HUGEPAGE` — Requests huge pages where available (Linux).
3. **Instantiation** — Creates `FloatTransformer` or `QuantizedTransformer`, passing the raw `mmap` pointer offset past the header.
4. **Legacy Support** — Files without the magic number are treated as legacy v0 (FP32).

---

## FloatTransformer (FP32)

### Memory Layout

The `FloatTransformerWeights` struct contains raw `float*` pointers into the mmapped file. No parsing or copying occurs — weights are accessed directly from the OS page cache.

The `FloatRunState` pre-allocates all runtime buffers at initialization:
- `x`, `xb`, `xb2` — Activation vectors (`dim`)
- `q`, `k`, `v` — QKV projection outputs
- `key_cache`, `value_cache` — KV cache (`n_layers × seq_len × kv_dim`)
- `att` — Attention scores (`n_heads × seq_len`)
- `logits` — Output logits (`vocab_size`)
- `hb`, `hb2` — FFN hidden buffers (`hidden_dim`)

### Forward Pass — `run_layer()`

For each layer, the following operations are performed:

#### 1. RMSNorm

Normalizes the input activation vector:

$$\text{out}_i = x_i \cdot w_i / \sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}$$

**SIMD Implementation:**
- **NEON**: Uses `vld1q_f32` for vectorized load, accumulates sum-of-squares in `float32x4_t`, reduces with `vaddvq_f32`. The reciprocal square root is computed via `vrsqrteq_f32` with a Newton-Raphson refinement step.
- **AVX2**: Uses `_mm256_loadu_ps` for 8-wide loads, horizontal sum via `_mm256_hadd_ps` + `_mm_add_ss`, and `_mm_rsqrt_ss` for fast reciprocal square root.
- Scalar fallback for non-SIMD platforms.

#### 2. QKV Projection

Standard FP32 matrix multiplication (`matmul`) to compute Query, Key, and Value vectors. The matmul uses OpenMP parallelization over the output rows.

**SIMD Matmul:**
- **NEON**: 4-wide `float32x4_t` FMA operations (`vfmaq_f32`).
- **AVX2**: 8-wide `_mm256_fmadd_ps` for fused multiply-add.
- The inner loop processes 4× unrolled iterations for pipeline utilization.

#### 3. RoPE (Rotary Positional Embeddings)

Applies position-dependent rotation to Q and K vectors:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$$

**SIMD Implementation:**
- **NEON**: Uses `vrev64q_f32` to swap pairs, `vnegq_f32` + `vbslq_f32` for sign manipulation, then `vfmaq_f32` for the rotation.
- **AVX2**: Uses `_mm256_permute_ps` for pair swapping, `_mm256_xor_ps` with a sign mask, then `_mm256_fmadd_ps` / `_mm256_fmsub_ps`.

Frequencies are precomputed in `precompute_freqs()` using the standard RoPE formula:
$$\theta_i = 10000^{-2i/\text{dim}}$$

#### 4. Multi-Head Attention

- **Score**: Dot product of Q with cached K vectors.
- **Softmax**: Numerically stable (max subtraction + exp + normalize).
- **Weighted Sum**: Attention weights applied to V vectors.
- **GQA Support**: When `n_kv_heads < n_heads`, multiple query heads share the same KV head via index mapping.

#### 5. Feed Forward (SwiGLU)

```
Gate = W1 @ x
Up   = W3 @ x
Gate = Gate * σ(Gate)     // SiLU activation
x    = W2 @ (Gate * Up)  // Output projection
x    += residual          // Residual connection
```

### Forward Pass — `forward()`

Orchestrates the full forward pass:
1. Embeds the input token.
2. Runs all assigned layers (respects `split_layer` / `end_layer` for distributed mode).
3. In distributed mode:
   - **Master**: Runs layers 0 to `split_layer-1`, sends activation via transport, receives back from tail.
   - **Worker**: Receives activation, runs assigned layers, sends forward.
4. Applies final RMSNorm and output projection (only on the tail/single node).
5. Returns logits for sampling.

### Worker Loop — `worker_loop()`

The blocking event loop for worker nodes:
1. `recv_prev()` — Receive activation + header from previous node.
2. **Control Message Check** — Inspects first 2 bytes for `CONTROL_MAGIC` to detect resize commands.
3. Process all assigned layers via `run_layer()`.
4. **Tail node**: `send_prev()` back to master (completing the ring).
5. **Intermediate node**: `send_next()` to next worker.
6. On `RESIZE` commands: updates layer config, clears KV cache, sends ACK.

---

## QuantizedTransformer (W8A8)

### Quantization Scheme

Uses **Weight-Int8, Activation-Int8** with dynamic activation quantization:

- **QuantizedTensor**: Stores `int8_t* q` (quantized values) and `float* s` (block-wise scales).
- **Block size**: Inherited from the exported file (default 64).
- **Dynamic Activation Quantization**: Before each matmul, the FP32 activation vector is quantized to INT8 on-the-fly:
  1. Find `max(|x|)`.
  2. Compute scale: `s = 127.0 / max_abs`.
  3. Quantize: `q[i] = round(x[i] * s)`.

### Integer GEMM

The core INT8 matrix multiplication:

```
result[j] = Σ(w_q[i] × x_q[i]) × w_scale × x_scale
```

**SIMD Kernels:**
- **ARM NEON (`vdotq_s32`)**: Performs 4-way int8 multiply-accumulate into int32 in a single instruction. Processes 16 elements per iteration.
- **AVX2 (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`)**: Expands int8 to int16 via unsigned-signed multiply, then adds adjacent pairs. Processes 32 elements per iteration with 4× loop unrolling.

### Dequantize

Reconstructs FP32 values from quantized data, used for normalization layers:
- **NEON**: `vcvtq_f32_s32` for int32→float conversion, then multiply by scale.
- **AVX2**: `_mm256_cvtepi32_ps` for vectorized int-to-float conversion.

---

## Tokenizer (`Tokenizer.cpp`)

A standalone BPE tokenizer for inference (no Python/HuggingFace dependency):

- **Loading**: Reads the binary `tokenizer.bin` file containing vocabulary strings, scores, and metadata.
- **Encoding**: Implements BPE merge via a **priority queue** (O(N log N)):
  1. Start with character-level tokens.
  2. Greedily merge the highest-scoring adjacent pair.
  3. Repeat until no more merges are possible.
- **Decoding**: Direct lookup from token ID to string, with byte-fallback support (`<0xHH>`).
- **Special Tokens**: Handles Llama 3 control tokens (`<|begin_of_text|>`, `<|eot_id|>`, `<|start_header_id|>`, etc.).

---

## Sampler (`Sampler.cpp`)

Implements three sampling strategies, all with SIMD acceleration:

### 1. Argmax (Temperature = 0)
Finds the token with highest probability. SIMD-vectorized with:
- **AVX2**: 8-wide comparison and blend (`_mm256_cmp_ps`, `_mm256_blendv_ps`).
- **NEON**: 4-wide comparison and blend (`vcgtq_f32`, `vbslq_f32`).

### 2. Multinomial Sampling
Draws from the full probability distribution using the CDF inversion method.

### 3. Top-P (Nucleus) Sampling
1. Filters tokens below a cutoff threshold.
2. Sorts remaining by probability (descending).
3. Accumulates probabilities until the top-p threshold is reached.
4. Samples from the truncated distribution.

**Temperature Scaling** is applied before softmax using vectorized multiply:
- **AVX2**: `_mm256_mul_ps` with broadcast inverse temperature.
- **NEON**: `vmulq_f32` with `vdupq_n_f32`.

---

## Chat Mode

The `chat()` function implements the Llama 3 instruction template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{generated_response}<|eot_id|>
```

### Slash Commands (Distributed Mode)
- `/layers` — Display current layer distribution.
- `/resize <boundaries...>` — Dynamically redistribute layers across workers.
- `/help` — Show available commands.

---

## CLI Reference

```bash
./inference [options] <model_path>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t, --tokenizer` | `tokenizer.bin` | Path to tokenizer file |
| `-p, --prompt` | — | Input prompt |
| `-c, --chat` | off | Enable interactive chat mode |
| `-s, --system` | — | System prompt (chat mode only) |
| `-n, --n-predict` | 4096 | Max tokens to generate |
| `--temp` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--seed` | 0 (random) | RNG seed for reproducibility |
| `--role` | `single` | Node role: `single`, `master`, `worker` |
| `--split` | 0 | Start layer index for this node |
| `--end` | 0 (all) | End layer index (exclusive) |
| `--transport` | `tcp` | Transport: `tcp`, `udp`, `kernel` |
| `--host` | `0.0.0.0` | Local bind address |
| `--port` | 9999 | Local bind port |
| `--next-host` | — | Next node IP (required for distributed) |
| `--next-port` | 9999 | Next node port |
