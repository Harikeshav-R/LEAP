#include "Tokenizer.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace Tokenizer {
    // Note: negative lookahead \s+(?!\S) is removed/adapted if using RE2, but here we use the exact string
    // and rely on the underlying engine (re2 in tiktoken cpp impl) to handle it or the user to accept limitations.
    // However, since the C++ Tiktoken implementation explicitly mentions removing it for RE2 support,
    // we will use the one compatible with the C++ Tiktoken implementation to avoid runtime errors if that implementation is strict.
    // The default pattern in Tiktoken C++ is:
    // R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)"
    static const std::string PAT_STR =
            R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";

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

        if (auto err = model_->load(model_path); err != tokenizers::Error::Ok) {
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

    std::string Tokenizer::decode(const std::vector<int> &tokens) const {
        std::string result;
        for (const int t: tokens) {
            auto piece_res = model_->id_to_piece(static_cast<uint64_t>(t));
            if (piece_res.ok()) {
                result += piece_res.get();
            }
        }
        return result;
    }


    void Tokenizer::export_tokenized_binary_file() const {
        std::string output_path = model_path_.substr(0, model_path_.find_last_of('.')) + ".bin";
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + output_path);
        }

        std::vector<std::string> tokens(n_words_);
        std::vector<float> scores(n_words_);

        size_t max_token_length = 0;
        bool error_occurred = false;

#pragma omp parallel
        {
            size_t local_max_len = 0;

#pragma omp for
            for (uint64_t i = 0; i < n_words_; ++i) {
                if (error_occurred) continue;

                auto res = model_->id_to_piece(i);
                if (!res.ok()) {
#pragma omp critical
                    {
                        error_occurred = true;
                    }
                    continue;
                }

                const std::string &piece = res.get();
                tokens[i] = piece;
                scores[i] = static_cast<float>(i); // Score is index

                if (piece.length() > local_max_len) {
                    local_max_len = piece.length();
                }
            }

#pragma omp critical
            {
                if (local_max_len > max_token_length) {
                    max_token_length = local_max_len;
                }
            }
        }

        if (error_occurred) {
            throw std::runtime_error("Failed to get token piece for one or more ids during export.");
        }

        auto max_len_u32 = static_cast<uint32_t>(max_token_length);
        out.write(reinterpret_cast<const char *>(&max_len_u32), sizeof(max_len_u32));

        for (size_t i = 0; i < n_words_; ++i) {
            float score = scores[i];
            auto len = static_cast<uint32_t>(tokens[i].length());
            out.write(reinterpret_cast<const char *>(&score), sizeof(score));
            out.write(reinterpret_cast<const char *>(&len), sizeof(len));
            out.write(tokens[i].data(), len);
        }
        out.close();
    }
}