# Tokenizer (`src/tokenizer/` & `src/inference/Tokenizer.cpp`)

LEAP has two tokenizer implementations: an **export-time tokenizer** that wraps the Tiktoken library, and an **inference-time tokenizer** that is a pure C++ BPE implementation with zero external dependencies.

## Source Files

### Export Tokenizer (`src/tokenizer/`)

| File | Purpose |
|------|---------|
| `main.cpp` | CLI entry point for tokenizer export |
| `Tokenizer.h/cpp` | Tiktoken wrapper + binary export |

### Inference Tokenizer (`src/inference/`)

| File | Purpose |
|------|---------|
| `Tokenizer.h/cpp` | Pure C++ BPE encoder/decoder for inference |

---

## Export Tokenizer

### Loading

Wraps the `tokenizers::Tiktoken` C++ library to load BPE models:
1. Reads the tokenizer model file (typically `tokenizer.model` from HuggingFace).
2. Configures Llama 3 special tokens:
   - `<|begin_of_text|>` (128000)
   - `<|end_of_text|>` (128001)
   - `<|start_header_id|>` (128006)
   - `<|end_header_id|>` (128007)
   - `<|eot_id|>` (128009)
   - Reserved slots for fine-tuning tokens (128002–128005, 128008, 128010–128255)

### Binary Export — `export_tokenized_binary_file()`

Writes a compact binary file containing the vocabulary for the inference tokenizer:

**Binary Format:**
```
For each token (0 to vocab_size-1):
    score (float)         — BPE merge priority score
    token_len (int32)     — Length of token string
    token_str (bytes)     — Raw token string bytes
```

**Metadata:**
```
Header:
    max_token_length (int32) — Maximum token string length
```

### CLI Usage

```bash
./tokenizer <model_path> [-o <output_path>]
```

| Argument | Description |
|----------|-------------|
| `model_path` | Path to the tokenizer model file |
| `-o, --output` | Output path (default: `tokenizer.bin` in same directory) |

---

## Inference Tokenizer

A self-contained BPE tokenizer designed for inference — no Python, HuggingFace, or Tiktoken dependency.

### Loading

Reads the binary file produced by the export tokenizer:
1. Reads `max_token_length`.
2. For each token: reads score, length, and string.
3. Builds a `std::unordered_map<string_view, int>` lookup table.
4. Pre-allocates 256 `byte_pieces` for single-byte fallback decoding.

### Encoding (BPE Merge)

Converts a string into a sequence of token IDs using the BPE algorithm:

1. **Character-Level Initialization**: Start with one token per UTF-8 character. Each character is looked up in the vocabulary; if not found, byte-level fallback tokens (`<0xHH>`) are used.

2. **Greedy Merge Loop**: Repeatedly find and merge the highest-scoring adjacent token pair:
   ```
   Input:  [H, e, l, l, o]
   Merge:  [He, l, l, o]     ← "He" has highest BPE score
   Merge:  [He, ll, o]       ← "ll" merged
   Merge:  [Hell, o]         ← "Hell" merged
   Merge:  [Hello]           ← final token
   ```
   - Time complexity: O(N log N) per merge iteration (priority queue approach).
   - Stops when no adjacent pair exists in the vocabulary.

3. **Special Tokens**: Optionally prepends BOS token (128000) and appends EOS token (128001).

### Decoding

Maps token IDs back to strings:
- Direct vocabulary lookup.
- Handles byte-fallback tokens: `<0xHH>` → raw byte.
- Strips leading space from tokens following a BOS token.

### Data Structures

```cpp
class Tokenizer {
    std::vector<std::string> vocab;          // Token ID → string
    std::vector<float> vocab_scores;         // Token ID → BPE score
    std::unordered_map<string_view, int> vocab_lookup;  // string → Token ID
    int vocab_size;
    unsigned int max_token_length;
    std::vector<std::string> byte_pieces;    // 256 single-byte strings
};
```

---

## Design Rationale

| Aspect | Export Tokenizer | Inference Tokenizer |
|--------|-----------------|-------------------|
| **Dependencies** | LibTorch, Tiktoken C++ | None |
| **Use Case** | Convert model files | Runtime encoding/decoding |
| **Algorithm** | Tiktoken's implementation | Custom BPE from scratch |
| **Performance** | Not critical | Optimized for low latency |
| **Deployment** | Build machine only | Any target device |

This split design ensures the inference binary can be deployed on minimal embedded systems (ARM boards, Jetson Nanos) without requiring Python or heavy ML library dependencies.
