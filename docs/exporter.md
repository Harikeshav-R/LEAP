# Exporter (`src/export/`)

The exporter converts HuggingFace Llama-format model checkpoints into LEAP's optimized binary format for the inference engine. It supports both full-precision (FP32) and quantized (INT8) export.

## Source Files

| File | Purpose |
|------|---------|
| `main.cpp` | CLI entry point — parses args, invokes loader + exporter |
| `Loader.h/cpp` | Reads `.safetensors` files → LibTorch tensors, applies RoPE permutation |
| `Export.h/cpp` | Serializes model to LEAP binary format (FP32 or INT8) |
| `Utils.h/cpp` | Conversion utilities (BF16→FP32, byte-swapping) |

---

## Loading Pipeline

### `Loader.cpp` — `load_meta_model()`

This function orchestrates the full model loading process:

1. **Parse `params.json`** — Reads model hyperparameters (`dim`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`, etc.) from the model directory.

2. **Discover Shards** — Scans for `*.safetensors` files and sorts them, supporting both single-file and multi-shard checkpoints.

3. **Load Tensors** — For each shard:
   - Uses `safetensors::mmap_from_file()` for zero-copy reads.
   - Maps safetensors dtypes to `torch::ScalarType`.
   - Constructs `torch::Tensor` views over the mmapped data.
   - Automatically converts BF16 tensors to FP32 via `Utils::bf16_to_fp32()`.

4. **Weight Assignment** — Maps tensor names to model parameters:
   - `model.embed_tokens.weight` → `tok_embeddings`
   - `model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` → Attention weights
   - `model.layers.{L}.mlp.{gate,up,down}_proj.weight` → FFN weights
   - `model.layers.{L}.input_layernorm.weight` → Attention norm
   - `model.layers.{L}.post_attention_layernorm.weight` → FFN norm
   - `model.norm.weight` → Final RMSNorm
   - `lm_head.weight` → Output projection

5. **RoPE Permutation** — Applies `permute_reverse()` to WQ and WK weight matrices to match the inference engine's expected interleaved RoPE layout:
   ```
   [dim, dim] → reshape to [n_heads, dim/n_heads/2, 2, dim]
                → permute(0, 2, 1, 3) → reshape back
   ```

### `load_file()` — Safetensors Reader

Loads a single `.safetensors` file using the `safetensors-cpp` library:
- Memory-maps the file for zero-copy access.
- Iterates over all tensors in the file.
- Returns a `std::map<std::string, torch::Tensor>`.

---

## Export Formats

### FP32 Export (Version 1) — `float32_export()`

Writes the model in full 32-bit floating point precision.

**Binary Layout:**
```
Offset 0x00:   Magic (int32)     = 0x616B3432 ("ak42")
Offset 0x04:   Version (int32)   = 1
Offset 0x08:   Config (28 bytes) = {dim, hidden_dim, n_layers, n_heads,
                                     n_kv_heads, vocab_size, seq_len}
Offset 0x100:  Weights start (256-byte aligned header)
```

**Weight Order:**
1. `tok_embeddings` — `[vocab_size × dim]`
2. Per-layer weights (repeated `n_layers` times):
   - `attention_norm` — `[dim]`
   - `wq, wk, wv, wo` — Attention weight matrices
   - `ffn_norm` — `[dim]`
   - `w1, w2, w3` — FFN weight matrices
3. `final_norm` — `[dim]`
4. `output` (classifier) — `[vocab_size × dim]` (omitted if shared with embeddings)

### INT8 Export (Version 2) — `int8_export()`

Quantizes weights to symmetric INT8 for 4× memory reduction.

**Binary Layout:**
```
Offset 0x00:   Magic (int32)           = 0x616B3432
Offset 0x04:   Version (int32)         = 2
Offset 0x08:   Config (28 bytes)
Offset 0x24:   SharedClassifier (u8)   = 0 or 1
Offset 0x25:   GroupSize (int32)       = 64 (default)
Offset 0x100:  Weights start (256-byte aligned)
```

**Quantization Scheme:**
- **Block-wise Symmetric Quantization** with configurable `group_size` (default: 64).
- If `dim % 64 != 0`, falls back to `group_size = 32`.
- Each block stores:
  - `int8_t[group_size]` — Quantized weights
  - `float` — Scale factor per block

**What Gets Quantized:**
- All weight matrices (Q, K, V, O, W1, W2, W3, embeddings, output)
- Normalization weights (RMSNorm) — kept in FP32 for numerical stability

**Pipeline Optimization:**
The INT8 export uses a **lookahead pipeline**:
1. **Async Quantize**: Fires off quantization of the next tensor on a background thread (`std::async`).
2. **Sync Write**: Writes the previously quantized tensor to disk.

This overlaps CPU-bound quantization with I/O-bound disk writes for maximum export speed.

---

## CLI Usage

```bash
./export <output_path> --meta-llama <model_dir> [--version <1|2>]
```

| Argument | Description |
|----------|-------------|
| `output_path` | Destination path for the `.bin` file |
| `--meta-llama` | Path to directory containing `params.json` and `.safetensors` |
| `--version 1` | FP32 export (default) |
| `--version 2` | INT8 quantized export |

**Example:**
```bash
# Export Llama-3-70B to INT8
./export llama3-70b-q8.bin --meta-llama /models/Meta-Llama-3-70B --version 2
```
