#include "Tokenizer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

// Note: negative lookahead \s+(?!\S) is removed/adapted if using RE2, but here we use the exact string
// and rely on the underlying engine (re2 in tiktoken cpp impl) to handle it or the user to accept limitations.
// However, since the C++ Tiktoken implementation explicitly mentions removing it for RE2 support, 
// we will use the one compatible with the C++ Tiktoken implementation to avoid runtime errors if that implementation is strict.
// The default pattern in Tiktoken C++ is:
// R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)"
static const std::string PAT_STR =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";

Tokenizer::Tokenizer(const std::string &model_path) : model_path_(model_path) {
    // Construct special tokens list matching Llama 3 specification
    std::vector<std::string> special_tokens;
    special_tokens.reserve(256);
    special_tokens.emplace_back("<|begin_of_text|>");
    special_tokens.emplace_back("<|end_of_text|>");
    special_tokens.emplace_back("<|reserved_special_token_0|>");
    special_tokens.emplace_back("<|reserved_special_token_1|>");
    special_tokens.emplace_back("<|reserved_special_token_2|>");
    special_tokens.emplace_back("<|reserved_special_token_3|>");
    special_tokens.emplace_back("<|start_header_id|>");
    special_tokens.emplace_back("<|end_header_id|>");
    special_tokens.emplace_back("<|reserved_special_token_4|>");
    special_tokens.emplace_back("<|eot_id|>"); // Index 9

    // Add reserved tokens 5 to 250 (inclusive of start, up to 256 items total special tokens)
    // Python: range(5, num_reserved_special_tokens - 5) -> range(5, 251) -> 5..250
    // Total tokens added here: 251 - 5 = 246.
    // 10 initial + 246 = 256.
    for (int i = 5; i <= 250; ++i) {
        special_tokens.push_back("<|reserved_special_token_" + std::to_string(i) + "|>");
    }

    // Indices for BOS and EOS in the special_tokens list
    size_t bos_idx = 0; // <|begin_of_text|>
    size_t eos_idx = 1; // <|end_of_text|>

    model_ = std::make_unique<tokenizers::Tiktoken>(
        PAT_STR,
        special_tokens,
        bos_idx,
        eos_idx
    );

    auto err = model_->load(model_path);
    if (err != tokenizers::Error::Ok) {
        throw std::runtime_error("Failed to load tokenizer model: " + model_path);
    }

    n_words_ = model_->vocab_size();
    bos_id_ = model_->bos_tok();
    eos_id_ = model_->eos_tok();
    pad_id_ = -1;

    stop_tokens_.insert(eos_id_);

    // Calculate eot_id
    // <|eot_id|> is at index 9 in special_tokens
    // The base vocabulary size is n_words_ - special_tokens.size()
    // special token ids start after base vocab
    uint64_t base_vocab_size = n_words_ - special_tokens.size();
    uint64_t eot_id = base_vocab_size + 9;
    stop_tokens_.insert(eot_id);
}

std::vector<int> Tokenizer::encode(const std::string &s, const bool bos, const bool eos) const {
    auto res = model_->encode(s, bos ? 1 : 0, eos ? 1 : 0);
    if (!res.ok()) {
        throw std::runtime_error("Failed to encode string");
    }
    std::vector<uint64_t> u64_tokens = res.get();
    // Convert to int
    std::vector<int> tokens(u64_tokens.begin(), u64_tokens.end());
    return tokens;
}

